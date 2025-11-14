# src/preprocess.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE, PROCESSED_DIR
from load_data import load_ciciot

def preprocess():
    print("Loading CICIoT dataset...")
    df = load_ciciot()

    if 'label' not in df.columns:
        raise ValueError("CSV must contain column 'label'")

    # Keep numeric features (and label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' not in numeric_cols:
        numeric_cols.append('label')
    df = df[numeric_cols]
    print("Using numeric columns:", len(numeric_cols))

    X = df.drop(columns=['label'])
    y = df['label']

    # Label encoding if needed
    if y.dtype.kind not in ('i', 'u'):
        le = LabelEncoder()
        y = le.fit_transform(y)
        print("Labels encoded:", list(le.classes_))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    pd.DataFrame(X_train).to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["label"]).to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["label"]).to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)
    print("Saved preprocessed files in", PROCESSED_DIR)

if __name__ == "__main__":
    preprocess()
