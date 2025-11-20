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
    serialize_weights,
    deserialize_weights,
    generate_aes_key,
    aes_encrypt,
    aes_decrypt,
    generate_ed25519_keypair,
    sign_bytes,
    verify_signature,
    serialize_public_key,
    deserialize_public_key,
    pack_signed_update,
    unpack_signed_update,
)
from config import (
    RANDOM_SEED,
    SECURE_NUM_CLIENTS,
    SECURE_SAMPLES_PER_CLIENT,
    SECURE_ROUNDS,
    SECURE_LOCAL_EPOCHS,
    SECURE_LOCAL_BATCH,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))["label"].values
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))["label"].values
    return X_train, y_train, X_test, y_test


def create_non_iid_partitions(X, y, num_clients, samples_per_client, seed=RANDOM_SEED):
    df = pd.DataFrame(X)
    df["label"] = y
    rng = np.random.RandomState(seed)
    clients = []
    classes = np.unique(y)
    for c in range(num_clients):
        k = rng.randint(2, min(5, len(classes)))
        chosen = rng.choice(classes, size=k, replace=False)
        df_c = df[df["label"].isin(chosen)]
        if len(df_c) >= samples_per_client:
            df_sample = df_c.sample(samples_per_client, random_state=seed + c)
        else:
            df_sample = df_c.sample(samples_per_client, replace=True, random_state=seed + c)
        Xc = df_sample.drop(columns=["label"]).values
        yc = df_sample["label"].values
        clients.append((Xc, yc))
    return clients


def local_train_simple(global_weights, X_local, y_local, global_num_classes, local_epochs=1, batch_size=8):
    model = build_hybrid(input_shape=(X_local.shape[1], 1), num_classes=global_num_classes)
    set_model_weights_from_numpy(model, global_weights)
    # compile without metrics (consistent with model_defs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    X_in = X_local.astype(np.float32).reshape((X_local.shape[0], X_local.shape[1], 1))
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
    m = build_hybrid(input_shape=(X_test.shape[1], 1), num_classes=global_num_classes)
    set_model_weights_from_numpy(m, weights_list)
    X_in = X_test.astype(np.float32).reshape((X_test.shape[0], X_test.shape[1], 1))
    preds = np.argmax(m.predict(X_in, batch_size=SECURE_LOCAL_BATCH, verbose=0), axis=1)
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_test, preds)


def shapes_template_from_weights(weights_list):
    """Return list of shapes for weights_list (used to validate updates)."""
    return [tuple(w.shape) for w in weights_list]


def weight_shapes_match(candidate_weights, template_shapes):
    """Return True if candidate_weights has same number of layers and matching shapes."""
    if len(candidate_weights) != len(template_shapes):
        return False
    for w, s in zip(candidate_weights, template_shapes):
        if tuple(w.shape) != tuple(s):
            return False
    return True


def run_secure_federated_simulation():
    # per-client AES keys (simulation)
    client_aes_keys = [generate_aes_key() for _ in range(SECURE_NUM_CLIENTS)]

    # generate signing keys and store serialized public keys
    client_keypairs = []
    client_pubkeys = {}
    for i in range(SECURE_NUM_CLIENTS):
        priv, pub = generate_ed25519_keypair()
        client_keypairs.append((priv, pub))
        client_pubkeys[i] = serialize_public_key(pub)

    # load data and create clients
    X_train, y_train, X_test, y_test = load_processed_data()
    clients = create_non_iid_partitions(X_train, y_train, SECURE_NUM_CLIENTS, SECURE_SAMPLES_PER_CLIENT)
    global_num_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    # build initial global model and get template shapes
    global_model = build_hybrid(input_shape=(X_train.shape[1], 1), num_classes=global_num_classes)
    global_weights = get_model_weights_as_numpy(global_model)
    template_shapes = shapes_template_from_weights(global_weights)

    # debug: show template shapes once
    print("[MODEL SHAPE TEMPLATE] layers:", len(template_shapes))
    for i, s in enumerate(template_shapes):
        print(f" layer {i}: {s}")

    round_logs = {"round": [], "time_plain": [], "time_crypto": [], "accepted_updates": [], "rejected_updates": []}

    for r in range(1, SECURE_ROUNDS + 1):
        print(f"\n--- Secure Round {r}/{SECURE_ROUNDS} ---")

        # Local training once per client (plain updates)
        t0 = time.time()
        plain_updates = []
        for i, (Xc, yc) in enumerate(clients):
            upd = local_train_simple(global_weights, Xc, yc, global_num_classes, local_epochs=SECURE_LOCAL_EPOCHS, batch_size=SECURE_LOCAL_BATCH)
            plain_updates.append(upd)

        # Validate plain_updates shapes and drop any that don't match (shouldn't happen normally)
        validated_plain = []
        plain_rejected = 0
        for i, upd in enumerate(plain_updates):
            if not weight_shapes_match(upd, template_shapes):
                plain_rejected += 1
                print(f" WARNING: plain update from client {i} has mismatched shapes; rejecting.")
            else:
                validated_plain.append(upd)

        if len(validated_plain) == 0:
            # fallback to global weights (rare)
            plain_agg = global_weights[:]
        else:
            plain_agg = weighted_average_weights_simple(validated_plain)

        t1 = time.time()

        # Build encrypted messages from the already computed (and validated) plain_updates
        t2 = time.time()
        encrypted_msgs = []
        for i, upd in enumerate(plain_updates):
            # still allow sending even if plain was invalid; but we'll verify later and reject
            raw = serialize_weights(upd)
            signature = sign_bytes(client_keypairs[i][0], raw)
            packed = pack_signed_update(signature, raw)
            nonce, ct = aes_encrypt(client_aes_keys[i], packed)
            encrypted_msgs.append({"client": i, "nonce": nonce, "ct": ct})
        t3 = time.time()

        # Verify/decrypt messages and collect accepted updates (with shape validation)
        accepted_updates = []
        rejected = 0
        for msg in encrypted_msgs:
            cid = msg["client"]
            nonce = msg["nonce"]
            ct = msg["ct"]
            pub_raw = client_pubkeys[cid]
            pub = deserialize_public_key(pub_raw)

            # decrypt
            try:
                packed = aes_decrypt(client_aes_keys[cid], nonce, ct)
            except Exception:
                rejected += 1
                print(f"  decrypt failed for client {cid}; rejected.")
                continue

            # unpack signature+raw safely
            try:
                signature, raw = unpack_signed_update(packed)
            except Exception:
                rejected += 1
                print(f"  unpack failed for client {cid}; rejected.")
                continue

            # verify signature
            if not verify_signature(pub, signature, raw):
                rejected += 1
                print(f"  signature invalid for client {cid}; rejected.")
                continue

            # deserialize weights and validate shapes
            try:
                upd = deserialize_weights(raw)
            except Exception:
                rejected += 1
                print(f"  deserialization failed for client {cid}; rejected.")
                continue

            if not weight_shapes_match(upd, template_shapes):
                rejected += 1
                print(f"  deserialized weights shapes mismatch for client {cid}; expected {len(template_shapes)} layers; got {len(upd)}; rejected.")
                # optionally dump shapes for debugging
                for j, w in enumerate(upd):
                    print(f"    client {cid} layer {j} shape: {tuple(w.shape)}")
                continue

            # accepted
            accepted_updates.append(upd)

        # Aggregate accepted updates (or keep global if none accepted)
        if len(accepted_updates) == 0:
            crypto_agg = global_weights[:]
        else:
            crypto_agg = weighted_average_weights_simple(accepted_updates)

        # Update global weights and evaluate
        global_weights = crypto_agg
        acc = evaluate_global(global_weights, X_test, y_test, global_num_classes)
        print(f" Global test acc: {acc:.4f} | rejected updates: {rejected} | accepted: {len(accepted_updates)}")

        round_logs["round"].append(r)
        round_logs["time_plain"].append(t1 - t0)
        round_logs["time_crypto"].append(t3 - t2)
        round_logs["accepted_updates"].append(len(accepted_updates))
        round_logs["rejected_updates"].append(rejected)

    # Save final model & logs
    set_model_weights_from_numpy(global_model, global_weights)
    model_path = os.path.join(MODELS_DIR, "secure_global.h5")
    global_model.save(model_path)
    log_path = os.path.join(RESULTS_DIR, "secure_round_logs.json")
    with open(log_path, "w") as f:
        json.dump(round_logs, f, indent=2)
    print(f"Saved model: {model_path}")
    print(f"Saved logs: {log_path}")

    # Timing plot
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


if __name__ == "__main__":
    run_secure_federated_simulation()
