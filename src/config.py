# src/config.py

# Phase control
SMALL_SAMPLE = True         # True for 4GB dev, False for full runs on 32GB
N_ROWS_SMALL = 10000        # rows to sample when SMALL_SAMPLE=True

# Dataset files
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
VALID_FILE = "data/validation.csv"

# Processed data paths
PROCESSED_DIR = "data/processed"
X_TRAIN_FILE = PROCESSED_DIR + "/X_train.csv"
X_TEST_FILE = PROCESSED_DIR + "/X_test.csv"
Y_TRAIN_FILE = PROCESSED_DIR + "/y_train.csv"
Y_TEST_FILE = PROCESSED_DIR + "/y_test.csv"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model & training hyperparams (small for 4GB)
MODEL = {
    "cnn_filters": 16,     # change to 64/128 on 32GB
    "cnn_kernel": 3,
    "lstm_units": 16,      # change to 64/128 on 32GB
    "dense_units": 32,
    "dropout": 0.5,
    "batch_size": 2,       # increase on 32GB - 32
    "epochs": 1,           # small for prototype; increase on 32GB - 10
}

# Misc
RANDOM_SEED = 42
NUM_CLASSES = None  # will be inferred automatically

# SHAP settings
SHAP = {
    "use_kernel": True,    # True -> KernelExplainer (works everywhere). On 32GB set False to use DeepExplainer if you want speed.
    "background_size_4gb": 30,
    "explain_size_4gb": 100, #200
    "background_size_32gb": 500,
    "explain_size_32gb": 10000
}

