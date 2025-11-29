# config.py (suggested edits for 4GB; keep file structure same)

MACHINE = "4GB"
SMALL_SAMPLE = True            # keep sample mode for 4GB
N_ROWS_SMALL = 10_000          # use full 10k sample

TRAIN_FILE = "data/train_test_network.csv"
TEST_FILE = ""
VALID_FILE = ""

PROCESSED_DIR = "data/processed"

# CSV paths
X_TRAIN_FILE = f"{PROCESSED_DIR}/X_train.csv"
X_TEST_FILE  = f"{PROCESSED_DIR}/X_test.csv"
Y_TRAIN_FILE = f"{PROCESSED_DIR}/y_train.csv"
Y_TEST_FILE  = f"{PROCESSED_DIR}/y_test.csv"

# NPY paths — REQUIRED by preprocess.py + train.py + SHAP
X_TRAIN_NPY = f"{PROCESSED_DIR}/X_train.npy"
X_TEST_NPY  = f"{PROCESSED_DIR}/X_test.npy"
Y_TRAIN_NPY = f"{PROCESSED_DIR}/y_train.npy"
Y_TEST_NPY  = f"{PROCESSED_DIR}/y_test.npy"

# sequence = 1 (connection-level). If you later extract time windows, increase to 20
SEQUENCE_LENGTH = 1

TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_SEED = 42

# Model sizes kept small for memory
MODEL = {
    "cnn_filters": 32,        # slightly larger filters for representational power
    "cnn_kernel": 1,
    "lstm_units": 32,
    "dense_units": 64,
    "dropout": 0.4,
    "batch_size": 64,         # keep 64 to utilize CPU RAM efficiently
    "epochs": 8               # bump epochs for meaningful training
}

# SHAP: keep kernel but small background/explain sizes on 4GB
SHAP = {"use_kernel": True, "background_size": 20, "explain_size": 100}

# Federated: keep small but run more rounds and repeat seeds for statistics
NUM_CLIENTS = 8             # increase number of clients to test heterogeneity
SAMPLES_PER_CLIENT = 600
ROUNDS = 8                    # more rounds to observe convergence
LOCAL_EPOCHS = 2
LOCAL_BATCH = 32

FEDPROX_MU = 0.01

BYZANTINE_RATIO = 0.33
MULTIKRUM_F = 1
MULTIKRUM_M = 1
BYZANTINE_MODE = "label_flip"
BYZANTINE_PARAMS = {"scale": 10.0, "sign_mag": 1.0}

SECURE_NUM_CLIENTS = 8
SECURE_SAMPLES_PER_CLIENT = 2000
SECURE_ROUNDS = 8
SECURE_LOCAL_EPOCHS = 1
SECURE_LOCAL_BATCH = 16
