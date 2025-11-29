"""
Simulated secure aggregation: clients locally train, then send AES-encrypted + signed updates.
Server verifies signatures and deserializes safely before aggregation.
Produces timing comparison plots and accepted/rejected counts.
"""
import os, time, json, random
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from config import (
    RANDOM_SEED, SECURE_NUM_CLIENTS, SECURE_SAMPLES_PER_CLIENT, SECURE_ROUNDS,
    SECURE_LOCAL_EPOCHS, SECURE_LOCAL_BATCH
)
from model_defs import build_hybrid
from fl_utils import get_model_weights_as_numpy, set_model_weights_from_numpy
from crypto_utils import (
    serialize_weights, deserialize_weights,
    generate_aes_key, aes_encrypt, aes_decrypt,
    generate_ed25519_keypair, sign_bytes, verify_signature,
    pack_signed_update, unpack_signed_update, serialize_public_key, deserialize_public_key
)
from sklearn.metrics import accuracy_score

# seeds
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PROCESSED = "data/processed"
RESULTS = "results"
MODELS = "models"
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


def load_processed_flat():
    X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED, "y_test.npy"))
    return X_train, y_train, X_test, y_test


def create_non_iid_partitions(X, y, num_clients, samples_per_client, seed=RANDOM_SEED):
    # Ensure X is 2D for DataFrame (flatten time axis)
    if X.ndim == 3:
        X_flat = X.reshape((X.shape[0], -1))
    else:
        X_flat = X
    df = pd.DataFrame(X_flat)
    df["label"] = y
    rng = np.random.RandomState(seed)
    clients = []
    classes = np.unique(y)
    C = len(classes)

    for c in range(num_clients):
        if C >= 2:
            k = rng.randint(1, min(4, C) + 1)
        else:
            k = 1
        chosen = rng.choice(classes, size=k, replace=False)
        df_c = df[df["label"].isin(chosen)]

        # fallback if empty selection
        if len(df_c) == 0:
            idx = rng.choice(len(df))
            df_c = df.iloc[[idx]]

        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=seed + c)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=seed + c)

        Xc = df_sample.drop(columns=["label"]).values
        yc = df_sample["label"].values

        # restore 3D shape (n,1,features)
        Xc = Xc.reshape((Xc.shape[0], 1, Xc.shape[1]))

        clients.append((Xc.astype(np.float32), yc.astype(np.int32)))
    return clients


def local_train_simple(global_weights, X_local, y_local, global_num_classes, local_epochs=1, batch_size=8):
    # Ensure 3D (n, seq_len, features)
    if X_local.ndim == 2:
        X_in = X_local.reshape((X_local.shape[0], 1, X_local.shape[1])).astype(np.float32)
    else:
        X_in = X_local.astype(np.float32)
    input_shape = (X_in.shape[1], X_in.shape[2])
    model = build_hybrid(input_shape=input_shape, num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(X_in, y_local, epochs=local_epochs, batch_size=batch_size, verbose=0)
    return get_model_weights_as_numpy(model)



def weighted_average_weights_simple(weight_list):
    num_layers = len(weight_list[0])
    agg = []
    for li in range(num_layers):
        stacked = np.stack([w[li].astype(np.float64) for w in weight_list], axis=0)
        agg.append(np.mean(stacked, axis=0).astype(np.float32))
    return agg


def evaluate_global(weights_list, X_test, y_test, global_num_classes):
    if X_test.ndim == 2:
        X_in = X_test.reshape((X_test.shape[0], 1, X_test.shape[1])).astype(np.float32)
    else:
        X_in = X_test.astype(np.float32)
    input_shape = (X_in.shape[1], X_in.shape[2])
    m = build_hybrid(input_shape=input_shape, num_classes=global_num_classes)
    set_model_weights_from_numpy(m, weights_list)
    preds = np.argmax(m.predict(X_in, batch_size=SECURE_LOCAL_BATCH, verbose=0), axis=1)
    return float(accuracy_score(y_test, preds))


def shapes_template_from_weights(weights_list):
    return [tuple(w.shape) for w in weights_list]

def weight_shapes_match(candidate_weights, template_shapes):
    if len(candidate_weights) != len(template_shapes):
        return False
    for w, s in zip(candidate_weights, template_shapes):
        if tuple(w.shape) != tuple(s):
            return False
    return True


def run_secure_federated_simulation():
    # generate per-client crypto keys
    client_aes_keys = [generate_aes_key() for _ in range(SECURE_NUM_CLIENTS)]
    client_keypairs = [generate_ed25519_keypair() for _ in range(SECURE_NUM_CLIENTS)]
    client_pubkeys = {i: serialize_public_key(kp[1]) for i, kp in enumerate(client_keypairs)}
    X_train, y_train, X_test, y_test = load_processed_flat()
    clients = create_non_iid_partitions(X_train, y_train, SECURE_NUM_CLIENTS, SECURE_SAMPLES_PER_CLIENT)
    global_num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    global_model = build_hybrid(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=global_num_classes
    )
    global_weights = get_model_weights_as_numpy(global_model)
    template_shapes = shapes_template_from_weights(global_weights)
    logs = {"round": [], "time_plain": [], "time_crypto": [], "accepted_updates": [], "rejected_updates": []}
    for r in range(1, SECURE_ROUNDS + 1):
        print(f"\n--- Secure Round {r}/{SECURE_ROUNDS} ---")
        t0 = time.time()
        plain_updates = []
        for i, (Xc, yc) in enumerate(clients):
            upd = local_train_simple(global_weights, Xc, yc, global_num_classes, local_epochs=SECURE_LOCAL_EPOCHS, batch_size=SECURE_LOCAL_BATCH)
            plain_updates.append(upd)
        t1 = time.time()
        encrypted_msgs = []
        for i, upd in enumerate(plain_updates):
            upd = [np.array(w, dtype=np.float32) for w in upd]
            raw = serialize_weights(upd)
            signature = sign_bytes(client_keypairs[i][0], raw)
            packed = pack_signed_update(signature, raw)
            nonce, ct = aes_encrypt(client_aes_keys[i], packed)
            encrypted_msgs.append({"client": i, "nonce": nonce, "ct": ct})
        t2 = time.time()
        accepted_updates = []
        rejected = 0
        for msg in encrypted_msgs:
            cid = msg["client"]; nonce = msg["nonce"]; ct = msg["ct"]
            pub_raw = client_pubkeys[cid]
            pub = deserialize_public_key(pub_raw)
            try:
                packed = aes_decrypt(client_aes_keys[cid], nonce, ct)
            except Exception:
                rejected += 1; print(f" decrypt failed for client {cid}"); continue
            try:
                signature, raw = unpack_signed_update(packed)
            except Exception:
                rejected += 1; print(f" unpack failed for client {cid}"); continue
            if not verify_signature(pub, signature, raw):
                rejected += 1; print(f" signature invalid for client {cid}"); continue
            try:
                upd = deserialize_weights(raw)
            except Exception:
                rejected += 1; print(f" deserialization failed for client {cid}"); continue
            if not weight_shapes_match(upd, template_shapes):
                rejected += 1; print(f" shape mismatch for client {cid}"); continue
            accepted_updates.append(upd)
        t3 = time.time()
        if len(accepted_updates) == 0:
            crypto_agg = [np.array(w, dtype=np.float32) for w in global_weights]
        else:
            crypto_agg = weighted_average_weights_simple(accepted_updates)
        global_weights = crypto_agg
        acc = evaluate_global(global_weights, X_test, y_test, global_num_classes)
        print(f" Global test acc: {acc:.4f} | rejected updates: {rejected} | accepted: {len(accepted_updates)}")
        logs["round"].append(r); logs["time_plain"].append(t1 - t0); logs["time_crypto"].append(t3 - t2); logs["accepted_updates"].append(len(accepted_updates)); logs["rejected_updates"].append(rejected)
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = os.path.join(MODELS, "secure_global.h5")
    try:
        global_model.save(model_path)
    except Exception:
        np.savez_compressed(os.path.join(MODELS, "secure_global_weights.npz"), *[np.array(w, dtype=np.float32) for w in global_weights])
    with open(os.path.join(RESULTS, "secure_round_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)
    # timing plot
    rounds = logs["round"]
    # plot crypto vs plain time
    plt.figure()
    plt.plot(rounds, logs["time_plain"], marker="o", label="plain")
    plt.plot(rounds, logs["time_crypto"], marker="o", label="crypto")
    plt.xlabel("Round")
    plt.ylabel("Time (s)")
    plt.title("Plain vs Crypto time per round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ts = int(time.time())
    plt.savefig(os.path.join(RESULTS, f"secure_time_comparison_{ts}.png"))
    plt.close()

    print("Saved model:", model_path)
    print("Saved logs:", os.path.join(RESULTS, "secure_round_logs.json"))


if __name__ == "__main__":
    run_secure_federated_simulation()
