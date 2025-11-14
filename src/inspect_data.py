# src/inspect_data.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from load_data import load_ciciot
from config import SMALL_SAMPLE, N_ROWS_SMALL

def inspect():
    print("Loading dataset for inspection...")
    df = load_ciciot()   # already handles SMALL_SAMPLE internally

    print("\n=== BASIC INFO ===")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    print("\n=== LABEL DISTRIBUTION ===")
    dist = df['label'].value_counts().sort_index()
    print(dist)

    os.makedirs("results", exist_ok=True)

    # Plot label distribution
    plt.figure(figsize=(8, 4))
    dist.plot(kind='bar')
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/label_dist.png")
    plt.close()
    print("Saved: results/label_dist.png")

    # Summary stats for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print("\n=== NUMERIC SUMMARY ===")
    if len(numeric_cols) == 0:
        print("No numeric columns found.")
    else:
        print(df[numeric_cols].describe())

    # Optional sample preview
    print("\n=== SAMPLE ROWS ===")
    print(df.head(5))

    if SMALL_SAMPLE:
        print(f"\nNOTE: SMALL_SAMPLE=True, dataset was limited to {N_ROWS_SMALL} rows.")

if __name__ == "__main__":
    inspect()
