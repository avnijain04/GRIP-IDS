import pandas as pd
from config import TRAIN_FILE, TEST_FILE, VALID_FILE, SMALL_SAMPLE, N_ROWS_SMALL
import os

def load_ciciot():
    parts = []
    if os.path.exists(TRAIN_FILE):
        parts.append(pd.read_csv(TRAIN_FILE))
    if os.path.exists(TEST_FILE):
        parts.append(pd.read_csv(TEST_FILE))
    if os.path.exists(VALID_FILE):
        parts.append(pd.read_csv(VALID_FILE))

    if len(parts) == 0:
        raise FileNotFoundError("Place train/test/validation CSVs in data/")

    df = pd.concat(parts, ignore_index=True)

    # Small sample mode
    if SMALL_SAMPLE and len(df) > N_ROWS_SMALL:
        df = df.sample(N_ROWS_SMALL, random_state=42).reset_index(drop=True)

    return df
