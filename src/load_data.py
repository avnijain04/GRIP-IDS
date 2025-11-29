# load_data.py — robust single-CSV loader
import pandas as pd
import os
from config import TRAIN_FILE, TEST_FILE, VALID_FILE, SMALL_SAMPLE, N_ROWS_SMALL




def load_ciciot():
    files = []


    if TRAIN_FILE and os.path.exists(TRAIN_FILE):
        files.append(TRAIN_FILE)
    else:
        raise FileNotFoundError("Place train_test_network.csv inside data/ folder.")


    if TEST_FILE and os.path.exists(TEST_FILE):
        files.append(TEST_FILE)
    if VALID_FILE and os.path.exists(VALID_FILE):
        files.append(VALID_FILE)


    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)


    if SMALL_SAMPLE and len(df) > N_ROWS_SMALL:
        df = df.sample(N_ROWS_SMALL, random_state=42).reset_index(drop=True)


    return df