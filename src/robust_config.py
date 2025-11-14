# src/robust_config.py
# Central config for robust federated experiments (works for 4GB and 32GB)

# Simulation size
NUM_CLIENTS = 3            # number of simulated clients (3 for 4GB)
SAMPLES_PER_CLIENT = 2000  # samples per client on 4GB; increase on 32GB

# Federated rounds / local training
ROUNDS = 5
LOCAL_EPOCHS = 1
LOCAL_BATCH = 4

# FedProx (if used elsewhere)
FEDPROX_MU = 0.01

# Byzantine attack configuration
BYZANTINE_RATIO = 0.33      # fraction of clients to make malicious (0.33 => 1 of 3)
BYZANTINE_MODE = "label_flip"  # "label_flip", "scale", "sign"
BYZANTINE_PARAMS = {
    "scale": 10.0,      # for weight scaling attack
    "sign_mag": 1.0     # for sign attack magnitude
}

# Multi-Krum parameters
MULTIKRUM_F = 1   # assume up to 1 Byzantine
MULTIKRUM_M = 1   # number of winners to average (1 => Krum)
