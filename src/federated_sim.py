# src/federated_sim.py
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from config import RANDOM_SEED
from fl_config import NUM_CLIENTS, SAMPLES_PER_CLIENT, ROUNDS, LOCAL_EPOCHS, LOCAL_BATCH, FEDPROX_MU, SEED
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy, weighted_average_weights, l2_norm_between_weights
from model_defs import build_hybrid
from tensorflow.keras import optimizers, losses

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Utilities to prepare client data
# ----------------------------
def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv").values
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")["label"].values
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv").values
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")["label"].values
    return X_train, y_train, X_test, y_test

def create_non_iid_partitions(X, y, num_clients, samples_per_client):
    """
    Create non-iid partitions by sampling class-skewed chunks.
    We'll sample for each client a subset of classes (2-4 classes per client)
    and draw samples for them.
    """
    df = pd.DataFrame(X)
    df['label'] = y
    clients_data = []
    classes = np.unique(y)
    num_classes = len(classes)

    rng = np.random.RandomState(SEED)
    for c in range(num_clients):
        # pick 2-4 classes for this client (non-iid)
        k = rng.randint(2, min(5, num_classes))
        chosen = rng.choice(classes, size=k, replace=False)
        # select samples for these classes
        df_c = df[df['label'].isin(chosen)]
        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=SEED + c)
        else:
            # if not enough, sample with replacement across chosen classes
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=SEED + c)
        Xc = df_sample.drop(columns=['label']).values
        yc = df_sample['label'].values
        clients_data.append((Xc, yc))
    return clients_data

# ----------------------------
# Client local training (with optional FedProx)
# ----------------------------
def client_update(global_weights, X, y, model_builder, local_epochs=1, batch_size=4, lr=1e-3, mu=0.0):
    """
    global_weights: list of numpy arrays
    returns: updated_weights (list), num_samples
    Implements FedProx by adding proximal term to gradients:
      grad = grad + mu * (w - w_global)
    """
    # build a fresh model and set global weights
    model = model_builder(input_shape=(X.shape[1], 1), num_classes=int(np.max(y)+1) if np.max(y)>=0 else 2)
    # But we want model with same output dimension as global; simpler: build model then set weights
    set_model_weights_from_numpy(model, global_weights)
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy()

    # training loop: small batches
    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=1024, seed=SEED).batch(batch_size)

    # convert global weights to list of tf tensors for proximal calculation
    gw = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]

    # training
    for epoch in range(local_epochs):
        for xb, yb in dataset:
            xb = tf.reshape(xb, (xb.shape[0], xb.shape[1], 1))  # (batch, timesteps, 1)
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss_value = loss_fn(yb, logits)
                # add proximal term (FedProx)
                if mu > 0.0:
                    prox = 0.0
                    # sum ||w - gw||^2
                    for v, g in zip(model.trainable_variables, gw):
                        prox += tf.nn.l2_loss(v - g)
                    loss_value = loss_value + (mu/2.0) * 2.0 * prox  # tf.nn.l2_loss returns sum(t**2)/2 so adjust
            grads = tape.gradient(loss_value, model.trainable_variables)
            # apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # return updated weights and number of samples
    return get_model_weights_as_numpy(model), X.shape[0]

# ----------------------------
# Evaluation helpers
# ----------------------------
def evaluate_model_on_global(model_weights, X_test, y_test, model_builder):
    # create model, set weights, evaluate accuracy
    model = model_builder(input_shape=(X_test.shape[1], 1), num_classes=int(np.max(y_test)+1))
    set_model_weights_from_numpy(model, model_weights)
    # predict in batches
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    probs = model.predict(X_in, batch_size=LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    return acc

# ----------------------------
# Federated simulation main
# ----------------------------
def run_federated_simulation(mu=0.0, save_prefix="fed"):
    print(f"Starting federated simulation: mu={mu}")

    # load processed data
    X_train, y_train, X_test, y_test = load_processed_data()

    # gather a subset for clients to simulate (we will sample client data from X_train)
    clients = create_non_iid_partitions(X_train, y_train, NUM_CLIENTS, SAMPLES_PER_CLIENT)
    print("Created client partitions:", [c[0].shape for c in clients])

    # initialize global model
    sample_input_shape = (X_train.shape[1], 1)
    # determine num_classes globally
    num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    global_model = build_hybrid(input_shape=sample_input_shape, num_classes=num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    # logs
    round_logs = {"round": [], "global_acc": [], "client_weighted_avg_acc": [], "client_weight_norms": []}
    per_client_weights = [None] * NUM_CLIENTS

    for r in range(1, ROUNDS + 1):
        print(f"\n--- Round {r}/{ROUNDS} ---")
        # clients perform local updates
        local_weights = []
        local_counts = []
        client_norms = []
        for i, (Xi, yi) in enumerate(clients):
            print(f" Client {i}: training on {Xi.shape[0]} samples (classes: {np.unique(yi)})")
            updated_w, count = client_update(global_weights, Xi, yi, build_hybrid,
                                             local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH, mu=mu)
            local_weights.append(updated_w)
            local_counts.append(count)
            # compute drift norm vs global
            norm = l2_norm_between_weights(global_weights, updated_w)
            client_norms.append(norm)
            # optionally save client weights
            per_client_weights[i] = updated_w

        # Aggregate (weighted average)
        new_global = weighted_average_weights(local_weights, local_counts)
        # compute client drift stats
        avg_drift = float(np.mean(client_norms))
        print(f" Avg client drift (L2) this round: {avg_drift:.6f}")

        # update global weights
        global_weights = new_global
        # evaluate globally
        global_acc = evaluate_model_on_global(global_weights, X_test, y_test, build_hybrid)
        print(f" Global test accuracy after round {r}: {global_acc:.4f}")

        # log
        round_logs["round"].append(r)
        round_logs["global_acc"].append(global_acc)
        round_logs["client_weighted_avg_acc"].append(None)
        round_logs["client_weight_norms"].append(client_norms)

        # optional: save intermediate global model
        # we will save at the end
    # save final global model weights to models/
    set_model_weights_from_numpy(global_model, global_weights)
    global_model.save(f"{MODELS_DIR}/{save_prefix}_global_mu{mu}.h5")
    print(f"Saved global model: {MODELS_DIR}/{save_prefix}_global_mu{mu}.h5")

    # save logs to CSV/npz
    import json
    with open(f"{RESULTS_DIR}/{save_prefix}_round_logs_mu{mu}.json", "w") as f:
        json.dump(round_logs, f, indent=2)
    print(f"Saved logs: {RESULTS_DIR}/{save_prefix}_round_logs_mu{mu}.json")

    # Plot global accuracy over rounds
    plt.figure()
    plt.plot(round_logs["round"], round_logs["global_acc"], marker="o", label=f"{save_prefix}_mu{mu}")
    plt.xlabel("Round")
    plt.ylabel("Global Test Accuracy")
    plt.title(f"Federated {save_prefix} Global Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{save_prefix}_global_acc_mu{mu}.png")
    plt.close()

    return round_logs

# ----------------------------
# Run both FedAvg (mu=0) and FedProx (mu=FEDPROX_MU)
# ----------------------------
if __name__ == "__main__":
    # First run FedAvg
    logs_fedavg = run_federated_simulation(mu=0.0, save_prefix="fedavg")
    # Then FedProx
    logs_fedprox = run_federated_simulation(mu=FEDPROX_MU, save_prefix="fedprox")
    print("Federated simulations complete.")
