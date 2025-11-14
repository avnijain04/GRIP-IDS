# src/federated_secure.py
import os
import time
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_defs import build_hybrid
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy
from crypto_utils import (
    serialize_weights, deserialize_weights,
    generate_aes_key, aes_encrypt, aes_decrypt,
    generate_ed25519_keypair, sign_bytes, verify_signature,
    serialize_public_key, deserialize_public_key
)

# --------------------
# Small config for 4GB demo
# --------------------
NUM_CLIENTS = 3
SAMPLES_PER_CLIENT = 2000
ROUNDS = 3
LOCAL_EPOCHS = 1
LOCAL_BATCH = 8
SEED = 42

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# seeds
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# data helpers
# ----------------------------
def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["label"].values
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["label"].values
    return X_train, y_train, X_test, y_test

def create_non_iid_partitions(X, y, num_clients, samples_per_client, seed=SEED):
    df = pd.DataFrame(X)
    df['label'] = y
    rng = np.random.RandomState(seed)
    clients = []
    classes = np.unique(y)
    for c in range(num_clients):
        k = rng.randint(2, min(5, len(classes)))
        chosen = rng.choice(classes, size=k, replace=False)
        df_c = df[df['label'].isin(chosen)]
        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=seed + c)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=seed + c)
        Xc = df_sample.drop(columns=['label']).values
        yc = df_sample['label'].values
        clients.append((Xc, yc))
    return clients

# ----------------------------
# local training
# ----------------------------
def local_train_simple(global_weights, X_local, y_local, global_num_classes, local_epochs=1, batch_size=8):
    """
    Build a new local model, set global_weights, train for few epochs, return updated weights.
    """
    model = build_hybrid(input_shape=(X_local.shape[1],1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    # compile and train quietly
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    X_in = X_local.astype(np.float32).reshape((X_local.shape[0], X_local.shape[1], 1))
    model.fit(X_in, y_local, epochs=local_epochs, batch_size=batch_size, verbose=0)
    return get_model_weights_as_numpy(model)

# ----------------------------
# aggregation helper
# ----------------------------
def weighted_average_weights_simple(weight_list):
    num_layers = len(weight_list[0])
    agg = []
    for li in range(num_layers):
        stacked = np.stack([w[li].astype(np.float64) for w in weight_list], axis=0)
        agg.append(np.mean(stacked, axis=0).astype(np.float32))
    return agg

# ----------------------------
# evaluation helper
# ----------------------------
def evaluate_global(weights_list, X_test, y_test, global_num_classes):
    m = build_hybrid(input_shape=(X_test.shape[1],1), num_classes=global_num_classes)
    set_model_weights_from_numpy(m, weights_list)
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    preds = np.argmax(m.predict(X_in, batch_size=LOCAL_BATCH, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test, preds)

# ----------------------------
# main secure federated simulation
# ----------------------------
def run_secure_federated_simulation():
    # symmetric AES key (demo). In real deployment use secure KEX.
    aes_key = generate_aes_key()

    # generate per-client Ed25519 keypairs (private,public) and serialize public keys for server
    client_keypairs = []
    server_pubkeys = {}
    for i in range(NUM_CLIENTS):
        priv, pub = generate_ed25519_keypair()
        client_keypairs.append((priv, pub))
        server_pubkeys[i] = serialize_public_key(pub)

    # load data and create partitions
    X_train, y_train, X_test, y_test = load_processed_data()
    clients = create_non_iid_partitions(X_train, y_train, NUM_CLIENTS, SAMPLES_PER_CLIENT)

    # compute global num classes
    global_num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    # initialize a global model object once and keep it (so saving shapes consistent)
    global_model = build_hybrid(input_shape=(X_train.shape[1],1), num_classes=global_num_classes)
    global_weights = get_model_weights_as_numpy(global_model)

    round_logs = {"round": [], "time_plain": [], "time_crypto": [], "rejected_updates": [], "accepted_updates": []}

    for r in range(1, ROUNDS + 1):
        print(f"\n--- Secure Round {r}/{ROUNDS} ---")

        # ---------- Plain (no crypto) local updates ----------
        t0 = time.time()
        plain_updates = []
        for i, (Xc, yc) in enumerate(clients):
            upd = local_train_simple(global_weights, Xc, yc, global_num_classes, local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH)
            plain_updates.append(upd)
        plain_agg = weighted_average_weights_simple(plain_updates)
        t1 = time.time()

        # ---------- Crypto-enabled: sign + encrypt ----------
        t2 = time.time()
        encrypted_msgs = []
        # each client locally trains, serializes its update, signs plaintext, encrypts plaintext
        for i, (Xc, yc) in enumerate(clients):
            upd = local_train_simple(global_weights, Xc, yc, global_num_classes, local_epochs=LOCAL_EPOCHS, batch_size=LOCAL_BATCH)
            raw = serialize_weights(upd)  # bytes
            priv = client_keypairs[i][0]
            signature = sign_bytes(priv, raw)
            nonce, ct = aes_encrypt(aes_key, raw)
            encrypted_msgs.append({"client": i, "nonce": nonce, "ct": ct, "sig": signature})
        # server receives and processes
        accepted_updates = []
        rejected = 0
        for msg in encrypted_msgs:
            cid = msg["client"]
            pub_raw = server_pubkeys[cid]
            pub = deserialize_public_key(pub_raw)
            nonce = msg["nonce"]
            ct = msg["ct"]
            sig = msg["sig"]
            # try decrypt
            try:
                plaintext = aes_decrypt(aes_key, nonce, ct)
            except Exception:
                rejected += 1
                continue
            # verify signature over plaintext
            ok = verify_signature(pub, sig, plaintext)
            if not ok:
                rejected += 1
                continue
            # deserialize weights and accept
            upd_weights = deserialize_weights(plaintext)
            accepted_updates.append(upd_weights)
        # aggregate accepted updates (or fallback)
        if len(accepted_updates) == 0:
            crypto_agg = global_weights[:]  # keep unchanged
        else:
            crypto_agg = weighted_average_weights_simple(accepted_updates)
        t3 = time.time()

        # choose crypto_agg as the new global weights (we're demonstrating secure flow)
        global_weights = crypto_agg

        # evaluate
        acc = evaluate_global(global_weights, X_test, y_test, global_num_classes)
        print(f" Global test acc: {acc:.4f} | rejected updates: {rejected} | accepted: {len(accepted_updates)}")

        # log times
        round_logs["round"].append(r)
        round_logs["time_plain"].append(t1 - t0)
        round_logs["time_crypto"].append(t3 - t2)
        round_logs["rejected_updates"].append(rejected)
        round_logs["accepted_updates"].append(len(accepted_updates))

    # --- Save final global model (use the global_model object to avoid shape inference issues) ---
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = os.path.join(MODELS_DIR, "secure_global.h5")
    global_model.save(model_path)
    log_path = os.path.join(RESULTS_DIR, "secure_round_logs.json")
    with open(log_path, "w") as f:
        json.dump(round_logs, f, indent=2)
    print(f"Saved model: {model_path}")
    print(f"Saved logs: {log_path}")

    # also save a small timing plot
    plt.figure()
    rounds = round_logs["round"]
    plt.plot(rounds, round_logs["time_plain"], marker="o", label="plain")
    plt.plot(rounds, round_logs["time_crypto"], marker="o", label="crypto")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.title("Plain vs Crypto time per round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "secure_time_comparison.png"))
    print("Saved plot:", os.path.join(RESULTS_DIR, "secure_time_comparison.png"))

    return round_logs

# ----------------------------
# run
# ----------------------------
if __name__ == "__main__":
    logs = run_secure_federated_simulation()
