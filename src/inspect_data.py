# src/inspect_data.py

import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_ciciot
import os

def inspect():
    df = load_ciciot()

    print("Columns:", df.columns.tolist())

    # Label distribution
    print("Label distribution:")
    print(df['label'].value_counts())

    os.makedirs("results", exist_ok=True)
    df['label'].value_counts().sort_index().plot(kind='bar')
    plt.title("Label Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/label_dist.png")

    print("Saved plot to results/label_dist.png")

if __name__ == "__main__":
    inspect()
