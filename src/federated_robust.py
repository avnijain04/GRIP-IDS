# src/federated_robust.py
import os
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy
from model_defs import build_hybrid
from attacks import label_flip_attack, weight_scaling_attack, sign_attack
from multi_krum import multi_krum
import robust_config as cfg
from tensorflow.keras import optimizers, losses

# seeds
SEED_GLOBAL = 42
np.random.seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)
tf.random.set_seed(SEED_GLOBAL)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# Data helpers
# ----------------------------
def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv").values
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv")["label"].values
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv").values
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv")["label"].values
    return X_train, y_train, X_test, y_test

def create_non_iid_partitions(X, y, num_clients, samples_per_client):
    df = pd.DataFrame(X)
    df['label'] = y
    rng = np.random.RandomState(SEED_GLOBAL)
    clients = []
    classes = np.unique(y)
    for c in range(num_clients):
        k = rng.randint(2, min(5, len(classes)))
        chosen = rng.choice(classes, size=k, replace=False)
        df_c = df[df['label'].isin(chosen)]
        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=SEED_GLOBAL + c)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=SEED_GLOBAL + c)
        Xc = df_sample.drop(columns=['label']).values
        yc = df_sample['label'].values
        clients.append((Xc, yc))
    return clients

# ----------------------------
# Local training helper
# ----------------------------
def local_train(model_builder, global_weights, global_num_classes, X, y,
                local_epochs=1, batch_size=4, lr=1e-3, mu=0.0):
    model = model_builder(input_shape=(X.shape[1],1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    optimizer = optimizers.Adam(learning_rate=lr)
    loss_fn = losses.SparseCategoricalCrossentropy()

    dataset = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    dataset = dataset.shuffle(buffer_size=1024, seed=SEED_GLOBAL).batch(batch_size)

    gw = [tf.convert_to_tensor(w, dtype=tf.float32) for w in global_weights]
    for epoch in range(local_epochs):
        for xb, yb in dataset:
            xb = tf.reshape(xb, (xb.shape[0], xb.shape[1], 1))
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                loss_value = loss_fn(yb, logits)
                if mu > 0.0:
                    prox = 0.0
                    for v, g in zip(model.trainable_variables, gw):
                        prox += tf.nn.l2_loss(v - g)
                    loss_value = loss_value + (mu/2.0)*2.0*prox
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return get_model_weights_as_numpy(model)

# ----------------------------
# Evaluation helper
# ----------------------------
def evaluate(global_weights, X_test, y_test, global_num_classes):
    model = build_hybrid(input_shape=(X_test.shape[1],1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    probs = model.predict(X_in, batch_size=cfg.LOCAL_BATCH, verbose=0)
    preds = np.argmax(probs, axis=1)
    return accuracy_score(y_test, preds)

# ----------------------------
# Simulation: FedAvg vs Multi-Krum under attack
# ----------------------------
def simulate(run_name="run", use_multikrum=False, byzantine_clients_idx=None, byzantine_mode="label_flip"):
    X_train, y_train, X_test, y_test = load_processed_data()
    global_num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    clients = create_non_iid_partitions(X_train, y_train, cfg.NUM_CLIENTS, cfg.SAMPLES_PER_CLIENT)
    print("Clients created:", [c[0].shape for c in clients])

    # init global model
    global_model = build_hybrid(input_shape=(X_train.shape[1],1), num_classes=global_num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    logs = {"round": [], "global_acc": [], "byz_count": len(byzantine_clients_idx) if byzantine_clients_idx else 0}

    for r in range(1, cfg.ROUNDS + 1):
        print(f"\n--- Round {r}/{cfg.ROUNDS} ---")
        local_updates = []
        for i, (Xi, yi) in enumerate(clients):
            Xi_local = Xi.copy()
            yi_local = yi.copy()
            if byzantine_clients_idx and i in byzantine_clients_idx:
                print(f" Client {i} is BYZANTINE, mode={byzantine_mode}")
                if byzantine_mode == "label_flip":
                    yi_local = label_flip_attack(yi_local, flip_to=None, seed=SEED_GLOBAL+i)
                    upd = local_train(build_hybrid, global_weights, global_num_classes, Xi_local, yi_local,
                                      local_epochs=cfg.LOCAL_EPOCHS, batch_size=cfg.LOCAL_BATCH, lr=1e-3, mu=0.0)
                elif byzantine_mode == "scale":
                    upd = local_train(build_hybrid, global_weights, global_num_classes, Xi_local, yi_local,
                                      local_epochs=cfg.LOCAL_EPOCHS, batch_size=cfg.LOCAL_BATCH, lr=1e-3, mu=0.0)
                    upd = weight_scaling_attack(upd, scale=cfg.BYZANTINE_PARAMS.get("scale", 10.0))
                elif byzantine_mode == "sign":
                    upd = local_train(build_hybrid, global_weights, global_num_classes, Xi_local, yi_local,
                                      local_epochs=cfg.LOCAL_EPOCHS, batch_size=cfg.LOCAL_BATCH, lr=1e-3, mu=0.0)
                    upd = sign_attack(upd, magnitude=cfg.BYZANTINE_PARAMS.get("sign_mag", 1.0))
                else:
                    yi_local = label_flip_attack(yi_local, flip_to=None, seed=SEED_GLOBAL+i)
                    upd = local_train(build_hybrid, global_weights, global_num_classes, Xi_local, yi_local,
                                      local_epochs=cfg.LOCAL_EPOCHS, batch_size=cfg.LOCAL_BATCH, lr=1e-3, mu=0.0)
            else:
                upd = local_train(build_hybrid, global_weights, global_num_classes, Xi_local, yi_local,
                                  local_epochs=cfg.LOCAL_EPOCHS, batch_size=cfg.LOCAL_BATCH, lr=1e-3, mu=0.0)
            local_updates.append(upd)

        # Aggregate
        if use_multikrum:
            agg = multi_krum(local_updates, f=cfg.MULTIKRUM_F, m=cfg.MULTIKRUM_M)
        else:
            # simple average
            num_layers = len(local_updates[0])
            agg = []
            for layer_idx in range(num_layers):
                layer_stack = np.stack([u[layer_idx].astype(np.float64) for u in local_updates], axis=0)
                agg.append(np.mean(layer_stack, axis=0).astype(np.float32))

        global_weights = agg
        acc = evaluate(global_weights, X_test, y_test, global_num_classes)
        print(f" Global test acc after round {r}: {acc:.4f}")
        logs["round"].append(r)
        logs["global_acc"].append(acc)

    # save model & logs
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = f"{MODELS_DIR}/{run_name}_global.h5"
    global_model.save(model_path)
    log_path = f"{RESULTS_DIR}/{run_name}_logs.json"
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print("Saved:", model_path, log_path)
    # plot
    plt.figure()
    plt.plot(logs["round"], logs["global_acc"], marker="o")
    plt.title(run_name + " Global Acc")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{run_name}_acc.png")
    print("Saved plot:", f"{RESULTS_DIR}/{run_name}_acc.png")
    return logs

# ----------------------------
# Run experiments
# ----------------------------
if __name__ == "__main__":
    # compute how many Byzantine clients to set
    byz_count = max(1, int(cfg.BYZANTINE_RATIO * cfg.NUM_CLIENTS))
    byz_idx = list(range(byz_count))
    print("Byzantine clients indices:", byz_idx)

    print("\nRunning FedAvg under attack (label-flip)")
    logs1 = simulate(run_name="fedavg_attack_labelflip", use_multikrum=False,
                     byzantine_clients_idx=byz_idx, byzantine_mode=cfg.BYZANTINE_MODE)

    print("\nRunning Multi-Krum under attack (label-flip)")
    logs2 = simulate(run_name="multikrum_attack_labelflip", use_multikrum=True,
                     byzantine_clients_idx=byz_idx, byzantine_mode=cfg.BYZANTINE_MODE)

    print("Robust federated experiments complete.")
