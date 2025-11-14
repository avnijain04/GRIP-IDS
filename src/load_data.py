# src/load_data.py

import pandas as pd
from config import TRAIN_FILE, TEST_FILE, VALID_FILE, SMALL_SAMPLE, N_ROWS_SMALL

def load_ciciot():
    print("Loading CICIoT2023 dataset...")

    # Load CSVs
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    df_valid = pd.read_csv(VALID_FILE)

    print("Train:", df_train.shape, "Test:", df_test.shape, "Validation:", df_valid.shape)

    # Combine into single dataframe
    df = pd.concat([df_train, df_test, df_valid], ignore_index=True)
    print("Combined DF shape:", df.shape)

    # Reduce rows for 4GB laptop
    if SMALL_SAMPLE:
        df = df.sample(N_ROWS_SMALL, random_state=42)
        print(f"Reduced to {N_ROWS_SMALL} rows for 4GB laptop.")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Final loaded DF shape:", df.shape)
    return df

if __name__ == "__main__":
    df = load_ciciot()
