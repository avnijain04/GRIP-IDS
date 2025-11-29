import os
import time
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from config import (
    RANDOM_SEED,
    BYZANTINE_RATIO,
    BYZANTINE_MODE,
    BYZANTINE_PARAMS,
    MULTIKRUM_F,
    MULTIKRUM_M,
    NUM_CLIENTS,
    SAMPLES_PER_CLIENT,
    ROUNDS,
    LOCAL_EPOCHS,
    LOCAL_BATCH,
)
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy
from model_defs import build_hybrid
from attacks import label_flip_attack, weight_scaling_attack, sign_attack
from multi_krum import multi_krum
from comm_utils import serialized_size_bytes

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_processed_data():
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

    def ensure3d(X):
        if X.ndim == 2:
            return X.reshape((X.shape[0], 1, X.shape[1]))
        return X

    return (
        ensure3d(X_train).astype(np.float32),
        y_train.astype(np.int32),
        ensure3d(X_test).astype(np.float32),
        y_test.astype(np.int32)
    )

def create_non_iid_partitions(X, y, num_clients, samples_per_client, seed=RANDOM_SEED):
    df = pd.DataFrame(X.reshape(X.shape[0], -1))
    df["label"] = y

    rng = np.random.RandomState(seed)
    clients = []

    classes = np.unique(y)
    C = len(classes)

    for cid in range(num_clients):

        if C >= 2:
            k = rng.randint(1, C + 1)
        else:
            k = 1

        chosen = rng.choice(classes, size=k, replace=False)

        df_c = df[df["label"].isin(chosen)]

        # FIX: fallback if empty selection
        if len(df_c) == 0:
            idx = rng.choice(len(df))
            df_c = df.iloc[[idx]]

        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=seed + cid)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=seed + cid)

        Xc = df_sample.drop(columns=["label"]).values
        yc = df_sample["label"].values

        # reshape back into 3D: (n,1,f)
        Xc = Xc.reshape((Xc.shape[0], 1, Xc.shape[1]))

        clients.append((Xc.astype(np.float32), yc.astype(np.int32)))

    return clients

def local_train(
    model_builder, global_weights, global_num_classes, X, y,
    local_epochs=1, batch_size=LOCAL_BATCH, lr=1e-3, mu=0.0
):
    # Ensure proper shape
    if X.ndim == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))

    input_shape = (X.shape[1], X.shape[2])
    model = model_builder(input_shape=input_shape, num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)

    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    try:
        ds = ds.cache()
    except Exception:
        pass
    ds = ds.shuffle(1024, seed=RANDOM_SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if mu == 0.0:
        model.fit(ds, epochs=local_epochs, verbose=0)
        return get_model_weights_as_numpy(model)

    # FedProx loop
    gw = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss_value = loss_fn(yb, logits)

            prox = tf.constant(0.0, dtype=tf.float32)
            for v, g in zip(model.trainable_variables, gw):
                prox += tf.nn.l2_loss(v - g)     # ½||v-g||²
            loss_value += (mu / 2.0) * prox     # correct FedProx

        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    for epoch in range(local_epochs):
        for xb, yb in ds:
            train_step(xb, yb)

    return get_model_weights_as_numpy(model)

def evaluate_with_metrics(global_weights, X_test, y_test, global_num_classes):
    if X_test.ndim == 2:
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = build_hybrid(input_shape=(X_test.shape[1], X_test.shape[2]), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)

    probs = model.predict(X_test, batch_size=LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average='macro', zero_division=0)),
        "precision_weighted": float(precision_score(y_test, preds, average='weighted', zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average='macro', zero_division=0)),
        "recall_weighted": float(recall_score(y_test, preds, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average='macro', zero_division=0)),
        "f1_weighted": float(f1_score(y_test, preds, average='weighted', zero_division=0)),
    }

    cm = confusion_matrix(y_test, preds)
    return metrics, cm, probs, preds

def simulate(run_name="run", use_multikrum=False, byzantine_clients_idx=None, byzantine_mode="label_flip"):
    X_train, y_train, X_test, y_test = load_processed_data()

    global_num_classes = len(np.unique(np.concatenate([y_train, y_test])))

    clients = create_non_iid_partitions(X_train, y_train, NUM_CLIENTS, SAMPLES_PER_CLIENT)
    print("Clients created:", [c[0].shape for c in clients])

    global_model = build_hybrid(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=global_num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    logs = {
        "round": [],
        "metrics": [],
        "cms": [],
        "client_comm": [],
        "byz_count": len(byzantine_clients_idx) if byzantine_clients_idx else 0
    }

    for r in range(1, ROUNDS + 1):
        t_round = time.time()
        print(f"\n--- Round {r}/{ROUNDS} ---")

        local_updates = []
        local_comm = []

        for i, (Xi, yi) in enumerate(clients):
            Xi_local = Xi.copy()
            yi_local = yi.copy()

            # Byzantine injections
            if byzantine_clients_idx and i in byzantine_clients_idx:
                print(f" Client {i} is BYZANTINE, mode={byzantine_mode}")

                if byzantine_mode == "label_flip":
                    yi_local = label_flip_attack(yi_local, flip_to=None, seed=RANDOM_SEED + i)

                upd = local_train(
                    build_hybrid,
                    global_weights,
                    global_num_classes,
                    Xi_local,
                    yi_local,
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=LOCAL_BATCH
                )

                if byzantine_mode == "scale":
                    upd = weight_scaling_attack(upd, scale=BYZANTINE_PARAMS.get("scale", 10.0))

                elif byzantine_mode == "sign":
                    upd = sign_attack(upd, magnitude=BYZANTINE_PARAMS.get("sign_mag", 1.0))

            else:
                upd = local_train(
                    build_hybrid,
                    global_weights,
                    global_num_classes,
                    Xi_local,
                    yi_local,
                    local_epochs=LOCAL_EPOCHS,
                    batch_size=LOCAL_BATCH
                )

            # ensure clean dtype
            upd = [u.astype(np.float32) for u in upd]

            local_updates.append(upd)
            local_comm.append(int(serialized_size_bytes(upd)))

        # ------------------------------
        # Aggregation
        # ------------------------------
        if use_multikrum:
            agg = multi_krum(local_updates, f=MULTIKRUM_F, m=MULTIKRUM_M)
        else:
            num_layers = len(local_updates[0])
            agg = []
            for layer_idx in range(num_layers):
                layer_stack = np.stack([u[layer_idx].astype(np.float64) for u in local_updates], axis=0)
                agg.append(np.mean(layer_stack, axis=0).astype(np.float32))

        global_weights = agg

        metrics, cm, probs, preds = evaluate_with_metrics(global_weights, X_test, y_test, global_num_classes)

        print(f" Global metrics after round {r}: acc={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}")

        # Save confusion matrix
        cm_path = f"{RESULTS_DIR}/{run_name}_cm_round{r}.csv"
        np.savetxt(cm_path, cm, delimiter=",")

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"{run_name} Confusion Matrix Round {r}")
        plt.colorbar()
        plt.xlabel("pred")
        plt.ylabel("true")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{run_name}_cm_round{r}.png")
        plt.close()

        logs["round"].append(r)
        logs["metrics"].append(metrics)
        logs["cms"].append(cm.tolist())
        logs["client_comm"].append(local_comm)

        print(f" Round {r} time: {time.time() - t_round:.2f}s")

    # Save final model
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = f"{MODELS_DIR}/{run_name}_global.h5"
    global_model.save(model_path)

    # Save logs
    log_path = f"{RESULTS_DIR}/{run_name}_logs.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    # Final metrics plot
    rounds = logs["round"]
    accs = [m["accuracy"] for m in logs["metrics"]]
    f1s = [m["f1_macro"] for m in logs["metrics"]]

    plt.figure()
    plt.plot(rounds, accs, marker='o', label="accuracy")
    plt.plot(rounds, f1s, marker='o', label="f1_macro")
    plt.legend()
    plt.title(run_name + " Metrics")
    plt.xlabel("round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{run_name}_metrics.png")
    plt.close()

    print("Saved:", model_path, log_path)
    return logs

if __name__ == "__main__":
    byz_count = max(1, int(BYZANTINE_RATIO * NUM_CLIENTS))
    byz_idx = list(range(byz_count))
    print("Byzantine indices:", byz_idx)

    print("\nRunning FedAvg under attack (label-flip)")
    simulate(
        run_name="fedavg_attack_labelflip",
        use_multikrum=False,
        byzantine_clients_idx=byz_idx,
        byzantine_mode=BYZANTINE_MODE
    )

    print("\nRunning Multi-Krum under attack (label-flip)")
    simulate(
        run_name="multikrum_attack_labelflip",
        use_multikrum=True,
        byzantine_clients_idx=byz_idx,
        byzantine_mode=BYZANTINE_MODE
    )

    print("Robust federated experiments complete.")
