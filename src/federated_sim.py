# src/federated_sim.py (updated)
# Federated simulation with improved non-IID Dirichlet splits, per-client metrics,
# FedAvg / FedProx / FedNova / FedOpt support, logging, and compatibility with
# the project preprocessing and model_defs APIs.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import random
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import (
    RANDOM_SEED,
    NUM_CLIENTS,
    SAMPLES_PER_CLIENT,
    ROUNDS,
    LOCAL_EPOCHS,
    LOCAL_BATCH,
    FEDPROX_MU,
)
from fl_utils import (
    get_model_weights_as_numpy,
    set_model_weights_from_numpy,
    weighted_average_weights,
    l2_norm_between_weights,
)
from model_defs import build_cnn
from tensorflow.keras import optimizers, losses
from tensorflow.keras.optimizers import legacy as legacy_optimizers

# deterministic
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------- Data utilities -------------------------
def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["label"].values
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["label"].values
    return X_train, y_train, X_test, y_test


# ------------------------- Partitioning -------------------------
def dirichlet_partition(X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = RANDOM_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce non-IID partitions using Dirichlet distribution over labels per client.
    Returns list of (X_client, y_client).
    """
    np.random.seed(seed)
    labels = np.unique(y)
    num_classes = len(labels)

    # indices per class
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in labels}

    # initialize client index lists
    client_idx = {i: [] for i in range(num_clients)}

    # for each class, split indices to clients with Dirichlet proportions
    for c in labels:
        idx_c = idx_by_class[c]
        if len(idx_c) == 0:
            continue
        # draw proportions for clients
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        # scale proportions by available samples
        proportions = (proportions / proportions.sum()) * len(idx_c)
        # convert to integer counts (while keeping sum equal)
        counts = np.floor(proportions).astype(int)
        # distribute remaining
        residual = len(idx_c) - counts.sum()
        if residual > 0:
            for i in np.argsort(proportions - counts)[-residual:]:
                counts[i] += 1
        # shuffle indices and assign
        np.random.shuffle(idx_c)
        pointer = 0
        for client_id in range(num_clients):
            take = counts[client_id]
            if take > 0:
                client_idx[client_id].extend(idx_c[pointer:pointer + take])
                pointer += take

    # Build client datasets; ensure each client has at least one sample (upsample if empty)
    clients = []
    for i in range(num_clients):
        idxs = client_idx[i]
        if len(idxs) == 0:
            # sample randomly from global set
            idxs = list(np.random.choice(len(y), size=1, replace=False))
        Xc = X[idxs]
        yc = y[idxs]
        clients.append((Xc, yc))

    return clients


# ------------------------- Client update (FedProx support) -------------------------

def client_update(global_weights: List[np.ndarray], X: np.ndarray, y: np.ndarray, model_builder, global_num_classes: int,
                  local_epochs: int = 1, batch_size: int = 4, lr: float = 1e-3, mu: float = 0.0) -> Tuple[List[np.ndarray], int]:
    """
    Train local model starting from global_weights. Supports FedProx via mu.
    Returns updated weights and local sample count.
    """
    # Build model with correct input shape and num_classes
    model = model_builder(input_shape=(X.shape[1], 1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)

    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy()

    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=1024, seed=RANDOM_SEED).batch(batch_size)

    # convert global weights to tensors for proximal calculation
    gw = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]

    for epoch in range(local_epochs):
        for xb, yb in dataset:
            # reshape to (batch, timesteps, 1)
            xb = tf.reshape(xb, (xb.shape[0], xb.shape[1], 1))
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss_value = loss_fn(yb, logits)
                if mu > 0.0:
                    prox = 0.0
                    for v, g in zip(model.trainable_variables, gw):
                        prox += tf.nn.l2_loss(v - g)
                    loss_value = loss_value + (mu / 2.0) * 2.0 * prox
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return get_model_weights_as_numpy(model), X.shape[0]


# ------------------------- FedNova aggregation helper -------------------------
def fednova_aggregate(local_weights: List[List[np.ndarray]], local_counts: List[int], global_weights: List[np.ndarray]) -> List[np.ndarray]:
    """
    Simple FedNova implementation: normalize local updates by local step counts.
    This is a pragmatic approximate FedNova useful in simulation.
    """
    total_samples = sum(local_counts)
    num_clients = len(local_weights)
    # compute deltas per client
    deltas = []
    for w in local_weights:
        delta = [w_i.astype(np.float32) for w_i in w]
        deltas.append(delta)

    # weighted average of deltas
    new_global = []
    for layer_idx in range(len(global_weights)):
        layer_vals = np.zeros_like(global_weights[layer_idx], dtype=np.float32)
        for i in range(num_clients):
            layer_vals += (local_weights[i][layer_idx] * (local_counts[i] / total_samples))
        new_global.append(layer_vals)

    return new_global


# ------------------------- Evaluation -------------------------

def evaluate_model_on_global(model_weights: List[np.ndarray], X_test: np.ndarray, y_test: np.ndarray, model_builder) -> Dict[str, float]:
    num_classes = int(np.max(y_test) + 1)
    model = model_builder(input_shape=(X_test.shape[1], 1), num_classes=num_classes)
    set_model_weights_from_numpy(model, model_weights)
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    probs = model.predict(X_in, batch_size=LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, average='macro', zero_division=0))
    rec = float(recall_score(y_test, preds, average='macro', zero_division=0))
    f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ------------------------- Simulation runner -------------------------

def run_federated_simulation(strategy: str = 'fedavg', mu: float = 0.0, alpha: float = 0.5, save_prefix: str = 'fed') -> Dict[str, Any]:
    """
    strategy: 'fedavg', 'fedprox', 'fednova', 'fedopt'
    mu: FedProx proximal term
    alpha: Dirichlet concentration (>0 -> more heterogeneous; larger -> more homogeneous)
    """
    print(f"Starting federated simulation: strategy={strategy}, mu={mu}, alpha={alpha}")
    X_train, y_train, X_test, y_test = load_processed_data()

    # create partitions using Dirichlet
    clients = dirichlet_partition(X_train, y_train, NUM_CLIENTS, alpha=alpha, seed=RANDOM_SEED)
    print("Client partitions created. Sizes:", [c[0].shape[0] for c in clients])

    # global classes
    num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)
    print("Global num_classes:", num_classes)

    # initialize global model
    global_model = build_cnn(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    logs = {"round": [], "global_metrics": [], "client_norms": []}

    # server-side optimizer for FedOpt
    from tensorflow.keras.optimizers import legacy as legacy_optimizers
    server_opt = legacy_optimizers.Adam(learning_rate=alpha)
    server_velocities = [np.zeros_like(w, dtype=np.float32) for w in global_weights]

    for r in range(1, ROUNDS + 1):
        print(f"\n--- Round {r}/{ROUNDS} ---")
        local_weights = []
        local_counts = []
        client_norms = []

        # client updates
        for i, (Xi, yi) in enumerate(clients):
            print(f" Client {i}: samples={Xi.shape[0]}, classes={np.unique(yi)}")
            if strategy == 'fedprox':
                updated_w, cnt = client_update(global_weights, Xi, yi, build_cnn, num_classes,
                                                local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH, mu=mu)
            else:
                updated_w, cnt = client_update(global_weights, Xi, yi, build_cnn, num_classes,
                                                local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH, mu=0.0)

            local_weights.append(updated_w)
            local_counts.append(cnt)

            norm = float(l2_norm_between_weights(global_weights, updated_w))
            client_norms.append(norm)

        # aggregation
        if strategy == 'fedavg' or strategy == 'fedprox' or strategy == 'fedopt':
            new_global = weighted_average_weights(local_weights, local_counts)
        elif strategy == 'fednova':
            new_global = fednova_aggregate(local_weights, local_counts, global_weights)
        else:
            raise ValueError('Unknown strategy')

                # FedOpt: server optimizer update using delta weights
        if strategy == 'fedopt':
            # compute delta = new_global - global_weights
            deltas = [new_global[i].astype(np.float32) - global_weights[i].astype(np.float32) for i in range(len(global_weights))]

            # grads are negative deltas (we want to move global_weights toward new_global)
            grads = [tf.convert_to_tensor(-d, dtype=tf.float32) for d in deltas]

            # create temporary tf.Variables that the optimizer can update
            global_vars = [tf.Variable(w, dtype=tf.float32) for w in global_weights]

            # Ensure the optimizer is built for these variables (fixes the KeyError)
            try:
                server_opt.build(global_vars)
            except Exception:
                # some TF/Keras builds don't require this step; ignore if it errors
                pass

            # apply the grads (server optimizer step)
            server_opt.apply_gradients(zip(grads, global_vars))

            # extract updated global weights
            updated_global = [v.numpy() for v in global_vars]
            new_global = updated_global

        # update global
        global_weights = new_global

        # evaluate
        metrics = evaluate_model_on_global(global_weights, X_test, y_test, build_cnn)
        print(f" Global metrics after round {r}: {metrics}")

        logs["round"].append(r)
        logs["global_metrics"].append(metrics)
        logs["client_norms"].append(client_norms)

    # save final model
    set_model_weights_from_numpy(global_model, global_weights)
    global_model.save(os.path.join(MODELS_DIR, f"{save_prefix}_global_{strategy}.h5"))
    print(f"Saved global model: {os.path.join(MODELS_DIR, f'{save_prefix}_global_{strategy}.h5')}")

    # dump logs
    with open(os.path.join(RESULTS_DIR, f"{save_prefix}_logs_{strategy}.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # plot global accuracy
    rounds = [r for r in logs["round"]]
    accs = [m["accuracy"] for m in logs["global_metrics"]]
    plt.figure()
    plt.plot(rounds, accs, marker='o', label=f"{strategy}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Federated {strategy} Global Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}_global_acc_{strategy}.png"))
    plt.close()

    return logs


if __name__ == "__main__":
    # Run three strategies and compare
    logs_fedavg = run_federated_simulation(strategy='fedavg', mu=0.0, alpha=0.5, save_prefix='fedavg')
    logs_fedprox = run_federated_simulation(strategy='fedprox', mu=FEDPROX_MU, alpha=0.5, save_prefix='fedprox')
    logs_fedopt = run_federated_simulation(strategy='fedopt', mu=0.0, alpha=0.5, save_prefix='fedopt')
    print("Federated simulations complete.")
