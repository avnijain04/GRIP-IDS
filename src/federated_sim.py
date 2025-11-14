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
from config import RANDOM_SEED, NUM_CLIENTS, SAMPLES_PER_CLIENT, ROUNDS, LOCAL_EPOCHS, LOCAL_BATCH, FEDPROX_MU
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy, weighted_average_weights, l2_norm_between_weights
from model_defs import build_hybrid
from tensorflow.keras import optimizers, losses

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv").values
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")["label"].values
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv").values
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")["label"].values
    return X_train, y_train, X_test, y_test


def create_non_iid_partitions(X, y, num_clients, samples_per_client, seed=RANDOM_SEED):
    df = pd.DataFrame(X)
    df['label'] = y
    rng = np.random.RandomState(seed)
    clients_data = []
    classes = np.unique(y)
    num_classes = len(classes)
    for c in range(num_clients):
        k = rng.randint(2, min(5, num_classes))
        chosen = rng.choice(classes, size=k, replace=False)
        df_c = df[df['label'].isin(chosen)]
        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=seed + c)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=seed + c)
        Xc = df_sample.drop(columns=['label']).values
        yc = df_sample['label'].values
        clients_data.append((Xc, yc))
    return clients_data


# ----------------------------
# Client local training (with optional FedProx)
# ----------------------------
def client_update(global_weights, X, y, model_builder, global_num_classes,
                  local_epochs=1, batch_size=4, lr=1e-3, mu=0.0):
    """
    global_weights: list of numpy arrays
    global_num_classes: use this to build client's model so shapes match global model
    returns: updated_weights (list), num_samples
    Implements FedProx by adding proximal term to gradients:
      grad = grad + mu * (w - w_global)
    """
    # build a fresh model (use global_num_classes) and set global weights
    model = model_builder(input_shape=(X.shape[1], 1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy()

    # training loop: small batches
    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=1024, seed=RANDOM_SEED).batch(batch_size)

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
                    for v, g in zip(model.trainable_variables, gw):
                        prox += tf.nn.l2_loss(v - g)
                    loss_value = loss_value + (mu/2.0) * 2.0 * prox
            grads = tape.gradient(loss_value, model.trainable_variables)
            # apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # return updated weights and number of samples
    return get_model_weights_as_numpy(model), X.shape[0]


def evaluate_model_on_global(model_weights, X_test, y_test, model_builder):
    model = model_builder(input_shape=(X_test.shape[1], 1), num_classes=int(np.max(y_test) + 1))
    set_model_weights_from_numpy(model, model_weights)
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    probs = model.predict(X_in, batch_size=LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, preds)
    return acc


def run_federated_simulation(mu=0.0, save_prefix="fed"):
    print(f"Starting federated simulation: mu={mu}")
    X_train, y_train, X_test, y_test = load_processed_data()
    clients = create_non_iid_partitions(X_train, y_train, NUM_CLIENTS, SAMPLES_PER_CLIENT)
    print("Created client partitions:", [c[0].shape for c in clients])

    # global number of classes (important to build client models with same output dim)
    num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    print("Global num_classes:", num_classes)

    global_model = build_hybrid(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    round_logs = {"round": [], "global_acc": [], "client_weight_norms": []}

    for r in range(1, ROUNDS + 1):
        print(f"\n--- Round {r}/{ROUNDS} ---")
        local_weights = []
        local_counts = []
        client_norms = []
        for i, (Xi, yi) in enumerate(clients):
            print(f" Client {i}: training on {Xi.shape[0]} samples (classes: {np.unique(yi)})")
            updated_w, count = client_update(global_weights, Xi, yi, build_hybrid, num_classes,
                                             local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH, mu=mu)
            local_weights.append(updated_w)
            local_counts.append(count)
            # compute drift norm vs global
            norm = l2_norm_between_weights(global_weights, updated_w)
            client_norms.append(norm)

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

        # logs
        round_logs["round"].append(r)
        round_logs["global_acc"].append(global_acc)
        round_logs["client_weight_norms"].append(client_norms)

    # debug: print some final shapes for sanity
    print("global_weights layer shapes:", [w.shape for w in global_weights])
    # build a sample client model (using the same global num_classes) and show its weight shapes
    sample_model = build_hybrid(input_shape=(Xi.shape[1], 1), num_classes=num_classes)
    print("sample client model layer shapes:", [w.shape for w in sample_model.get_weights()])

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


if __name__ == "__main__":
    logs_fedavg = run_federated_simulation(mu=0.0, save_prefix="fedavg")
    logs_fedprox = run_federated_simulation(mu=FEDPROX_MU, save_prefix="fedprox")
    print("Federated simulations complete.")
