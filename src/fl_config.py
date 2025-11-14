# src/fl_config.py
# Federated hyperparameters and simulation settings

# Simulated clients
NUM_CLIENTS = 3            # 3 for 4GB; increase on 32GB
SAMPLES_PER_CLIENT = 2000  # 2k samples per client on 4GB

# Rounds and local training
ROUNDS = 5                 # number of communication rounds (small for 4GB)
LOCAL_EPOCHS = 1           # local epochs per client
LOCAL_BATCH = 4            # small batch size for low RAM

# FedProx proximal term strength
FEDPROX_MU = 0.01          # 0 -> FedAvg; >0 -> FedProx

# Random seed
SEED = 42
