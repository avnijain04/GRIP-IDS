import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE, RANDOM_STATE, PROCESSED_DIR, SEQUENCE_LENGTH,
    X_TRAIN_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_TEST_NPY,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE
)
from load_data import load_ciciot

# ----------------------------
# Build sequence windows
# ----------------------------
def build_sequences(arr, labels, seq_len):
    n = arr.shape[0]
    if n < seq_len:
        pad = np.repeat(arr[-1:, :], seq_len - n, axis=0)
        seq = np.vstack([arr, pad])[None, ...]
        return seq, np.array([labels[-1]])

    X = []
    y = []
    for start in range(n - seq_len + 1):
        X.append(arr[start:start + seq_len])
        y.append(labels[start + seq_len - 1])
    return np.array(X), np.array(y)


def preprocess(sequence_length=SEQUENCE_LENGTH):

    print("Loading CICIoT...")
    df = load_ciciot()

    # Make sure label column is lowercase
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column")

    # ----------------------------
    # MERGE RARE CLASSES (<5 samples)
    # ----------------------------
    vc = df["label"].value_counts()
    rare_classes = vc[vc < 5].index.tolist()

    if len(rare_classes) > 0:
        print("Merging rare classes:", rare_classes)
        df["label"] = df["label"].apply(
            lambda x: "RareAttack" if x in rare_classes else x
        )

    # ----------------------------
    # Select numeric features
    # ----------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "label"]
    print(f"Using {len(feature_cols)} numeric features.")

    X_raw = df[feature_cols].values
    y_raw = df["label"].values

    # ----------------------------
    # Encode labels (AFTER merging)
    # ----------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print("Final labels:", list(le.classes_))

    # ----------------------------
    # Build sequences with sliding window
    # ----------------------------
    print("Building sequences...")
    X_seq, y_seq = build_sequences(X_raw, y_encoded, sequence_length)

    print("Sequences:", X_seq.shape)
    print("Labels:", y_seq.shape)

    # ----------------------------
    # Standardize
    # ----------------------------
    n, t, f = X_seq.shape
    X_flat = X_seq.reshape(n * t, f)
    scaler = StandardScaler()
    X_scaled_flat = scaler.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n, t, f)

    # ----------------------------
    # Train/test split (NO STRATIFY)
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_seq, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save NPY (best)
    np.save(X_TRAIN_NPY, X_train)
    np.save(X_TEST_NPY,  X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_TEST_NPY,  y_test)

    # Save CSV (flattened)
    def save_flat(X, y, path):
        ns, sl, nf = X.shape
        cols = [f"t{t}_f{f}" for t in range(sl) for f in range(nf)]
        flat = X.reshape(ns, sl * nf)
        df_out = pd.DataFrame(flat, columns=cols)
        df_out["label"] = y
        df_out.to_csv(path, index=False)

    save_flat(X_train, y_train, X_TRAIN_FILE)
    save_flat(X_test,  y_test,  X_TEST_FILE)
    pd.DataFrame(y_train, columns=["label"]).to_csv(Y_TRAIN_FILE, index=False)
    pd.DataFrame(y_test,  columns=["label"]).to_csv(Y_TEST_FILE,  index=False)

    print("Preprocessing complete. Saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess()
