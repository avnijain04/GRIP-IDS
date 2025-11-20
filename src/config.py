MACHINE = "4GB"
#MACHINE = "32GB"

############ DATASET SETTINGS ############
SMALL_SAMPLE = (MACHINE == "4GB")
N_ROWS_SMALL = 10_000

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
VALID_FILE = "data/validation.csv"

PROCESSED_DIR = "data/processed"
X_TRAIN_FILE = f"{PROCESSED_DIR}/X_train.csv"
X_TEST_FILE = f"{PROCESSED_DIR}/X_test.csv"
Y_TRAIN_FILE = f"{PROCESSED_DIR}/y_train.csv"
Y_TEST_FILE = f"{PROCESSED_DIR}/y_test.csv"

# New: preferred numpy save paths (recommended for DL)
X_TRAIN_NPY = f"{PROCESSED_DIR}/X_train.npy"
X_TEST_NPY = f"{PROCESSED_DIR}/X_test.npy"
Y_TRAIN_NPY = f"{PROCESSED_DIR}/y_train.npy"
Y_TEST_NPY = f"{PROCESSED_DIR}/y_test.npy"

# How many timesteps per sequence (tunable)
SEQUENCE_LENGTH = 20

TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_SEED = 42

############ MODEL SETTINGS ############
if MACHINE == "4GB":
    MODEL = {
        "cnn_filters": 16,
        "cnn_kernel": 3,
        "lstm_units": 16,
        "dense_units": 32,
        "dropout": 0.5,
        "batch_size": 2,
        "epochs": 1
    }
else:
    MODEL = {
        "cnn_filters": 32,
        "cnn_kernel": 3,
        "lstm_units": 32,
        "dense_units": 64,
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 5
    }

############ SHAP SETTINGS ############
if MACHINE == "4GB":
    SHAP = {"use_kernel": True, "background_size": 30, "explain_size": 100}
else:
    SHAP = {"use_kernel": False, "background_size": 300, "explain_size": 800}

############ FEDERATED SETTINGS ############
if MACHINE == "4GB":
    NUM_CLIENTS = 3
    SAMPLES_PER_CLIENT = 800
    ROUNDS = 3
    LOCAL_EPOCHS = 1
    LOCAL_BATCH = 32
else:
    NUM_CLIENTS = 10
    SAMPLES_PER_CLIENT = 8_000
    ROUNDS = 5
    LOCAL_EPOCHS = 2
    LOCAL_BATCH = 64

FEDPROX_MU = 0.01

############ ROBUST FED SETTINGS ############
if MACHINE == "4GB":
    BYZANTINE_RATIO = 0.33
    MULTIKRUM_F = 1
    MULTIKRUM_M = 1
else:
    BYZANTINE_RATIO = 0.05
    MULTIKRUM_F = 2
    MULTIKRUM_M = 2

BYZANTINE_MODE = "label_flip"
BYZANTINE_PARAMS = {"scale": 10.0, "sign_mag": 1.0}

############ SECURE-FL SETTINGS ############
if MACHINE == "4GB":
    SECURE_NUM_CLIENTS = 3
    SECURE_SAMPLES_PER_CLIENT = 2000
else:
    SECURE_NUM_CLIENTS = 5
    SECURE_SAMPLES_PER_CLIENT = 10000

SECURE_ROUNDS = 3
SECURE_LOCAL_EPOCHS = 1
SECURE_LOCAL_BATCH = 16
