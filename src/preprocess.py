# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_STATE
from load_data import load_ciciot
import os


def preprocess():
    df = load_ciciot()

    # Identify target column
    if 'label' not in df.columns:
        raise ValueError("Your CSV must contain a column named 'label'.")

    # Drop non-numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' not in numeric_cols:
        numeric_cols.append('label')
    df = df[numeric_cols]

    print("Using numeric columns:", len(numeric_cols))

    # Separate features/labels
    X = df.drop(columns=['label'])
    y = df['label']

    # Encode label
    if y.dtype != np.int64 and y.dtype != np.int32:
        le = LabelEncoder()
        y = le.fit_transform(y)
        print("Labels encoded:", list(le.classes_))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Train shapes:", X_train.shape, y_train.shape)
    print("Test shapes:", X_test.shape, y_test.shape)

    # Save splits
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["label"]).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_test, columns=["label"]).to_csv("data/processed/y_test.csv", index=False)

    print("Saved preprocessed files in data/processed/")


if __name__ == "__main__":
    preprocess()
