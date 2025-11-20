import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from load_data import load_ciciot
from config import SMALL_SAMPLE, N_ROWS_SMALL, SEQUENCE_LENGTH

def inspect():
    df = load_ciciot()

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "Label" in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    dist = df["label"].value_counts()
    print("\nLabel Distribution:\n", dist)

    rare = dist[dist < 5]
    if len(rare) > 0:
        print("\nRare (<5) classes:", rare)

    # Plot
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(14, 8))  # wide enough for long labels
    dist.plot(kind='bar')

    plt.title("Label Distribution", fontsize=16)
    plt.xlabel("Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    plt.xticks(rotation=75, ha="right", fontsize=8)  # rotate to prevent cut-off
    plt.tight_layout()

    plt.savefig("results/label_dist.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nDefault SEQUENCE_LENGTH:", SEQUENCE_LENGTH)

    if SMALL_SAMPLE:
        print(f"NOTE: SMALL_SAMPLE enabled: using first {N_ROWS_SMALL} rows.")

if __name__ == "__main__":
    inspect()
