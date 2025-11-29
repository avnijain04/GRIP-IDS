import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE, RANDOM_STATE, PROCESSED_DIR, SEQUENCE_LENGTH,
    X_TRAIN_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_TEST_NPY,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE
)
from load_data import load_ciciot

HARDCODE_DROP = [
    "src_ip", "dst_ip",
    "http_uri", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "ssl_subject", "ssl_issuer",
    "dns_query",
    "type",              
    "weird_name",
    "weird_addl",
]


def preprocess(sequence_length=SEQUENCE_LENGTH):
    print("Loading CSV...")
    df = load_ciciot()

    # Normalize label name
    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    # DROP HEAVY/UNUSABLE COLUMNS
    keep_cols = [c for c in df.columns if c not in HARDCODE_DROP]
    df = df[keep_cols].copy()

    print(f"Columns after hard-drop: {len(df.columns)}")

    # Extract labels
    y_raw = df["label"].astype(str).values
    df = df.drop(columns=["label"])

   
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

     # DETECT numeric vs categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(cat_cols)}")

    # ENCODE CATEGORICAL COLUMNS
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("NA"))
        encoders[col] = le

    # after encoding, all columns are numeric
    df = df.fillna(0.0).astype(np.float32)
    
    # FEATURE MATRIX
    X = df.values

    # LABEL ENCODING
    le_label = LabelEncoder()
    y = le_label.fit_transform(y_raw).astype(np.int32)
    print("Final label classes:", list(le_label.classes_))

    # STANDARDIZE
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # SEQUENCE FORMATTING
    n, f = X.shape
    if sequence_length == 1:
        X_seq = X.reshape(n, 1, f)
    else:
        X_seq = np.repeat(X.reshape(n, 1, f), sequence_length, axis=1)

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # SAVE NPY
    np.save(X_TRAIN_NPY, X_train)
    np.save(X_TEST_NPY, X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_TEST_NPY, y_test)

    # SAVE CSV (flattened)
    def save_flat(X, y, outpath):
        ns, sl, nf = X.shape
        cols = [f"t{t}_f{f}" for t in range(sl) for f in range(nf)]
        flat = X.reshape(ns, sl * nf)
        df_out = pd.DataFrame(flat, columns=cols)
        df_out["label"] = y
        df_out.to_csv(outpath, index=False)

    save_flat(X_train, y_train, X_TRAIN_FILE)
    save_flat(X_test, y_test, X_TEST_FILE)
    pd.DataFrame(y_train, columns=["label"]).to_csv(Y_TRAIN_FILE, index=False)
    pd.DataFrame(y_test,  columns=["label"]).to_csv(Y_TEST_FILE, index=False)

    print("\n✔ Preprocessing DONE")
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)


if __name__ == "__main__":
    preprocess()
