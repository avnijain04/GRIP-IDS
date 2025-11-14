# src/fl_utils.py
import numpy as np

def get_model_weights_as_numpy(model):
    # returns list of numpy arrays (copy)
    return [w.numpy() if hasattr(w, "numpy") else np.array(w) for w in model.get_weights()]

def set_model_weights_from_numpy(model, weights_list):
    model.set_weights([np.array(w, dtype=np.float32) for w in weights_list])

def weighted_average_weights(weight_list, weights_counts):
    """
    weight_list: list of lists of numpy arrays; length = num_clients
    weights_counts: list of number_of_samples_per_client (for weighting)
    returns: averaged weights (list of numpy arrays)
    """
    total = float(sum(weights_counts))
    num_clients = len(weight_list)
    # initialize accumulator with zeros with same shapes
    avg = []
    for layer_idx in range(len(weight_list[0])):
        accum = np.zeros_like(weight_list[0][layer_idx], dtype=np.float64)
        for client_idx in range(num_clients):
            accum += weight_list[client_idx][layer_idx].astype(np.float64) * (weights_counts[client_idx] / total)
        avg.append(accum.astype(np.float32))
    return avg

def l2_norm_between_weights(w1, w2):
    """
    Compute L2 norm between two weight lists
    """
    s = 0.0
    for a, b in zip(w1, w2):
        diff = a.astype(np.float64) - b.astype(np.float64)
        s += np.sum(diff * diff)
    return float(np.sqrt(s))
