# src/federated_sim.py
# Robust federated simulation helper (drop into src/)
import os
import sys

# Ensure src/ is on sys.path so local imports work whether run as "python src/..." or "python -m src.federated_sim"
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_this_dir, ".."))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import warnings
warnings.filterwarnings("ignore")

import json
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# -------------------------------------------
# FIXED IMPORTS (consistent: from src.*)
# -------------------------------------------
from src.config import (
    RANDOM_SEED,
    NUM_CLIENTS,
    SAMPLES_PER_CLIENT,
    ROUNDS,
    LOCAL_EPOCHS,
    LOCAL_BATCH,
    FEDPROX_MU,
    SEQUENCE_LENGTH,
)
from src.fl_utils import (
    get_model_weights_as_numpy,
    set_model_weights_from_numpy,
    weighted_average_weights,
    l2_norm_between_weights,
)
from src.model_defs import build_cnn, build_hybrid
from tensorflow.keras import optimizers, losses

# optional comm utils
try:
    from src.comm_utils import serialized_size_bytes
except Exception:
    def serialized_size_bytes(wts) -> int:
        s = 0
        for arr in wts:
            a = np.array(arr, dtype=np.float32)
            s += a.nbytes
        return int(s)


# determinism
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------
# FIXED: Unified model builder for whole sim
# -------------------------------------------
MODEL_BUILDER = build_hybrid   # or build_cnn, but must stay consistent


# ------------------------- Data utilities -------------------------
def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads NPY arrays (saved by preprocess) and returns arrays.
    Ensures X arrays are 3D: (n, seq_len, features).
    """
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    def ensure3d(X):
        if X.ndim == 2:
            if SEQUENCE_LENGTH == 1:
                return X.reshape((X.shape[0], 1, X.shape[1]))
            else:
                feats = X.shape[1]
                if feats % SEQUENCE_LENGTH != 0:
                    raise ValueError(f"Cannot reshape features {feats} into seq_len={SEQUENCE_LENGTH}")
                return X.reshape((X.shape[0], SEQUENCE_LENGTH, feats // SEQUENCE_LENGTH))
        elif X.ndim == 3:
            return X
        else:
            raise ValueError(f"Unexpected ndim: {X.ndim}")

    X_train = ensure3d(X_train)
    X_test = ensure3d(X_test)

    return (
        X_train.astype(np.float32),
        y_train.astype(np.int32),
        X_test.astype(np.float32),
        y_test.astype(np.int32),
    )


# ------------------------- Partitioning -------------------------
def dirichlet_partition(X: np.ndarray, y: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = RANDOM_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID partition using Dirichlet proportions.
    Ensures each client has >=1 sample (fixed).
    """
    rng = np.random.RandomState(seed)
    labels = np.unique(y)
    idx_by_class = {int(c): np.where(y == c)[0].tolist() for c in labels}
    client_idx = {i: [] for i in range(num_clients)}

    for c, idxs in idx_by_class.items():
        if len(idxs) == 0:
            continue

        props = rng.dirichlet([alpha] * num_clients)
        props = (props / props.sum()) * len(idxs)
        counts = np.floor(props).astype(int)

        residual = int(len(idxs) - counts.sum())
        if residual > 0:
            frac = props - counts
            chosen = np.argsort(frac)[-residual:]
            for ch in chosen:
                counts[ch] += 1

        rng.shuffle(idxs)
        ptr = 0
        for cid in range(num_clients):
            take = int(counts[cid])
            if take > 0:
                client_idx[cid].extend(idxs[ptr:ptr+take])
                ptr += take

    # ------------------------------------
    # FIXED: final safeguard (empty client)
    # ------------------------------------
    n_total = X.shape[0]
    for cid in range(num_clients):
        if len(client_idx[cid]) == 0:
            client_idx[cid] = [int(rng.choice(n_total))]

    clients = []
    for cid in range(num_clients):
        idcs = client_idx[cid]
        Xc = X[idcs]
        yc = y[idcs]
        clients.append((Xc.astype(np.float32), yc.astype(np.int32)))

    return clients


# ------------------------- Client update -------------------------
def client_update(global_weights: List[np.ndarray], X: np.ndarray, y: np.ndarray, model_builder, global_num_classes: int,
                  local_epochs: int = 1, batch_size: int = 8, lr: float = 1e-3, mu: float = 0.0) -> Tuple[List[np.ndarray], int]:

    # normalize shape
    if X.ndim == 2:
        if SEQUENCE_LENGTH == 1:
            X_in = X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            feats = X.shape[1]
            assert feats % SEQUENCE_LENGTH == 0
            X_in = X.reshape((X.shape[0], SEQUENCE_LENGTH, feats // SEQUENCE_LENGTH))
    else:
        X_in = X

    seq_len = X_in.shape[1]
    features = X_in.shape[2]

    model = model_builder(input_shape=(seq_len, features), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)

    opt = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)

    ds = tf.data.Dataset.from_tensor_slices((X_in.astype(np.float32), y.astype(np.int32)))
    ds = ds.shuffle(1024, seed=RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    gw = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]

    for epoch in range(local_epochs):
        for xb, yb in ds:
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss_value = loss_fn(yb, logits)

                # -----------------------------------------------------
                # FIXED: Correct FedProx penalty (no double counting)
                # -----------------------------------------------------
                if mu > 0.0:
                    prox = tf.constant(0.0, dtype=tf.float32)
                    for v, g in zip(model.trainable_variables, gw):
                        prox += tf.nn.l2_loss(v - g)  # ½||v-g||²
                    loss_value += (mu / 2.0) * prox  # correct FedProx

            grads = tape.gradient(loss_value, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

    return get_model_weights_as_numpy(model), int(X_in.shape[0])


# ------------------------- FedNova aggregator -------------------------
def fednova_aggregate(local_weights: List[List[np.ndarray]], local_counts: List[int], global_weights: List[np.ndarray]) -> List[np.ndarray]:
    total = sum(local_counts)
    num_clients = len(local_weights)
    new_global = []
    for layer_idx in range(len(global_weights)):
        layer_vals = np.zeros_like(global_weights[layer_idx], dtype=np.float32)
        for i in range(num_clients):
            layer_vals += local_weights[i][layer_idx] * (local_counts[i] / total)
        new_global.append(layer_vals)
    return new_global


# ------------------------- Evaluation -------------------------
def evaluate_model_on_global(model_weights: List[np.ndarray], X_test: np.ndarray, y_test: np.ndarray, model_builder) -> Dict[str, Any]:

    if X_test.ndim == 2 and SEQUENCE_LENGTH == 1:
        X_in = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    else:
        X_in = X_test

    # --------------------------------------------------------
    # FIXED: correct class count (no assumption y starts at 0)
    # --------------------------------------------------------
    num_classes = len(np.unique(y_test))

    model = model_builder(input_shape=(X_in.shape[1], X_in.shape[2]), num_classes=num_classes)
    set_model_weights_from_numpy(model, model_weights)

    probs = model.predict(X_in, batch_size=LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, average='macro', zero_division=0)),
        "recall": float(recall_score(y_test, preds, average='macro', zero_division=0)),
        "f1": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "y_probs": probs.tolist(),
        "y_pred": preds.tolist()
    }


# ------------------------- Simulation runner -------------------------
def run_federated_simulation(strategy: str = 'fedavg', mu: float = 0.0, alpha: float = 0.5, save_prefix: str = 'fed') -> Dict[str, Any]:

    print(f"Starting federated simulation: strategy={strategy}, mu={mu}, alpha={alpha}")
    X_train, y_train, X_test, y_test = load_processed_data()

    clients = dirichlet_partition(X_train, y_train, NUM_CLIENTS, alpha=alpha, seed=RANDOM_SEED)
    print("Client partitions sizes:", [int(c[0].shape[0]) for c in clients])

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print("Global num_classes:", num_classes)

    global_model = MODEL_BUILDER(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    logs = {"round": [], "global_metrics": [], "client_norms": [], "comm_bytes": []}

    for r in range(1, ROUNDS + 1):
        print(f"\n--- Round {r}/{ROUNDS} ---")
        local_weights = []
        local_counts = []
        client_norms = []
        client_comm = []

        for i, (Xi, yi) in enumerate(clients):
            print(f" Client {i}: samples={int(Xi.shape[0])}, classes={np.unique(yi)}")

            updated_w, cnt = client_update(
                global_weights, Xi, yi, MODEL_BUILDER, num_classes,
                local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH,
                mu=(mu if strategy == 'fedprox' else 0.0)
            )

            local_weights.append(updated_w)
            local_counts.append(cnt)
            norm = float(l2_norm_between_weights(global_weights, updated_w))
            client_norms.append(norm)
            client_comm.append(int(serialized_size_bytes(updated_w)))

        if strategy in ('fedavg', 'fedprox', 'fedopt'):
            new_global = weighted_average_weights(local_weights, local_counts)
        elif strategy == 'fednova':
            new_global = fednova_aggregate(local_weights, local_counts, global_weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # --------------------------------------------
        # FIXED: ensure all weights are float32
        # --------------------------------------------
        global_weights = [w.astype(np.float32) for w in new_global]

        metrics = evaluate_model_on_global(global_weights, X_test, y_test, MODEL_BUILDER)
        print(f" Global metrics after round {r}: {metrics}")

        logs["round"].append(r)
        logs["global_metrics"].append({
            k: metrics[k] for k in ("accuracy", "precision", "recall", "f1")
        })
        logs["client_norms"].append(client_norms)
        logs["comm_bytes"].append(client_comm)

    # apply weights to model before saving
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = os.path.join(MODELS_DIR, f"{save_prefix}_global_{strategy}.h5")

    try:
        global_model.save(model_path)
    except Exception:
        np.savez_compressed(
            os.path.join(MODELS_DIR, f"{save_prefix}_global_{strategy}_weights.npz"),
            *[np.array(w, dtype=np.float32) for w in global_weights]
        )

    print(f"Saved global model: {model_path}")

    with open(os.path.join(RESULTS_DIR, f"{save_prefix}_logs_{strategy}.json"), "w") as f:
        json.dump(logs, f, indent=2)

    # ------------------------------------------------------
    # FIXED: timestamp to avoid overwriting plots
    # ------------------------------------------------------
    timestamp = int(time.time())
    rounds = logs["round"]
    accs = [m["accuracy"] for m in logs["global_metrics"]]

    plt.figure()
    plt.plot(rounds, accs, marker='o', label=f"{strategy}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Federated {strategy} Global Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}_global_acc_{strategy}_{timestamp}.png"))
    plt.close()

    return logs


# ------------------------- Unified API (kept same except imports) -------------------------
def run_federated_rounds(
        strategy="fedavg",
        mu=0.0,
        alpha=0.5,
        rounds=5,
        batch_size=32,
        local_epochs=1,
        byzantine_ratio=0.0,
        byzantine_mode="none",
        seed=42,
        num_clients=None,
        samples_per_client=None
    ):

    from src.config import NUM_CLIENTS as CFG_NUM_CLIENTS, SAMPLES_PER_CLIENT as CFG_SAMPLES_PER_CLIENT

    if num_clients is None:
        num_clients = CFG_NUM_CLIENTS
    if samples_per_client is None:
        samples_per_client = CFG_SAMPLES_PER_CLIENT

    np.random.seed(seed)

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    if X_train.ndim == 2:
        if SEQUENCE_LENGTH == 1:
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        else:
            feats = X_train.shape[1]
            X_train = X_train.reshape((X_train.shape[0], SEQUENCE_LENGTH, feats // SEQUENCE_LENGTH))
            X_test = X_test.reshape((X_test.shape[0], SEQUENCE_LENGTH, feats // SEQUENCE_LENGTH))

    clients = []
    idxs = np.arange(X_train.shape[0])
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, num_clients)

    for s in splits:
        if s.size == 0:
            s = np.array([int(np.random.choice(X_train.shape[0]))])
        Xi = X_train[s]
        yi = y_train[s]

        if Xi.shape[0] >= samples_per_client:
            choose = np.random.choice(Xi.shape[0], size=samples_per_client, replace=False)
        else:
            choose = np.random.choice(Xi.shape[0], size=samples_per_client, replace=True)
        Xi = Xi[choose]
        yi = yi[choose]

        clients.append((Xi, yi))

    num_classes = len(np.unique(y_train))

    seq_len = int(X_train.shape[1])
    features = int(X_train.shape[2])

    from src.model_defs import build_hybrid as _build_hybrid
    model = _build_hybrid(input_shape=(seq_len, features), num_classes=num_classes)
    global_weights = [np.array(w, dtype=np.float32) for w in model.get_weights()]

    logs = []

    for r in range(1, rounds + 1):
        local_updates = []
        local_sizes = []

        for cid, (Xc, yc) in enumerate(clients):

            X_local = Xc.copy()
            y_local = yc.copy()

            if byzantine_ratio > 0 and cid < int(num_clients * byzantine_ratio):
                if byzantine_mode == "label_flip":
                    n_classes = int(np.max(y_local) + 1)
                    if n_classes > 1:
                        for ii in range(len(y_local)):
                            choices = [int(c) for c in range(n_classes) if c != int(y_local[ii])]
                            y_local[ii] = np.random.choice(choices)

            try:
                new_w, n_s = client_update(global_weights, X_local, y_local, _build_hybrid, num_classes,
                                           local_epochs=local_epochs, batch_size=batch_size, mu=mu)
            except Exception as e:
                print(f"WARNING: client {cid} update failed: {e}. Using global weights for this client.")
                new_w = global_weights
                n_s = X_local.shape[0]

            local_updates.append(new_w)
            local_sizes.append(int(n_s))

        total = float(sum(local_sizes))
        new_global = []
        for layer_idx in range(len(global_weights)):
            stacked = np.stack(
                [local_updates[i][layer_idx].astype(np.float64) * (local_sizes[i] / total)
                 for i in range(len(local_sizes))],
                axis=0
            )
            layer_avg = np.sum(stacked, axis=0).astype(np.float32)
            new_global.append(layer_avg)

        global_weights = new_global

        metrics = evaluate_model_on_global(global_weights, X_test, y_test, _build_hybrid)
        metrics["round"] = r
        logs.append(metrics)

    return logs


# When running this file directly, run full simulation
if __name__ == "__main__":
    for strat, mu in [("fedavg", 0.0), ("fedprox", FEDPROX_MU), ("fednova", 0.0)]:
        try:
            run_federated_simulation(strategy=strat, mu=mu, alpha=0.5, save_prefix=f"fed_{strat}")
        except Exception as e:
            print("Simulation", strat, "failed:", e)
