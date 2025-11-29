MACHINE = "4GB"
SMALL_SAMPLE = True            
N_ROWS_SMALL = 10_000         

TRAIN_FILE = "data/train_test_network.csv"
TEST_FILE = ""
VALID_FILE = ""

PROCESSED_DIR = "data/processed"

X_TRAIN_FILE = f"{PROCESSED_DIR}/X_train.csv"
X_TEST_FILE  = f"{PROCESSED_DIR}/X_test.csv"
Y_TRAIN_FILE = f"{PROCESSED_DIR}/y_train.csv"
Y_TEST_FILE  = f"{PROCESSED_DIR}/y_test.csv"

X_TRAIN_NPY = f"{PROCESSED_DIR}/X_train.npy"
X_TEST_NPY  = f"{PROCESSED_DIR}/X_test.npy"
Y_TRAIN_NPY = f"{PROCESSED_DIR}/y_train.npy"
Y_TEST_NPY  = f"{PROCESSED_DIR}/y_test.npy"

SEQUENCE_LENGTH = 1

TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_SEED = 42

MODEL = {
    "cnn_filters": 32,        
    "cnn_kernel": 1,
    "lstm_units": 32,
    "dense_units": 64,
    "dropout": 0.4,
    "batch_size": 64,         
    "epochs": 8               
}

SHAP = {"use_kernel": True, "background_size": 20, "explain_size": 100}

NUM_CLIENTS = 8           
SAMPLES_PER_CLIENT = 600
ROUNDS = 8                    
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
