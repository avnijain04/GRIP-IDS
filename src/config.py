# src/config.py

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
        "cnn_filters": 128,
        "cnn_kernel": 3,
        "lstm_units": 128,
        "dense_units": 128,
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 10
    }

############ SHAP SETTINGS ############
if MACHINE == "4GB":
    SHAP = {"use_kernel": True, "background_size": 30, "explain_size": 100}
else:
    SHAP = {"use_kernel": False, "background_size": 1000, "explain_size": 10000}

############ FEDERATED SETTINGS ############
if MACHINE == "4GB":
    NUM_CLIENTS = 3
    SAMPLES_PER_CLIENT = 2000
    ROUNDS = 5
    LOCAL_EPOCHS = 1
    LOCAL_BATCH = 4
else:
    NUM_CLIENTS = 20
    SAMPLES_PER_CLIENT = 200_000
    ROUNDS = 20
    LOCAL_EPOCHS = 5
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
    SECURE_NUM_CLIENTS = 20
    SECURE_SAMPLES_PER_CLIENT = 200000

SECURE_ROUNDS = 3
SECURE_LOCAL_EPOCHS = 1
SECURE_LOCAL_BATCH = 8
