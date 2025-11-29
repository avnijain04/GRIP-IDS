import numpy as np

def get_model_weights_as_numpy(model):
    return [w.numpy() if hasattr(w, "numpy") else np.array(w) for w in model.get_weights()]

def set_model_weights_from_numpy(model, weights_list):
    cleaned = [np.asarray(w, dtype=np.float32) for w in weights_list]
    model.set_weights(cleaned)

def weighted_average_weights(weight_list, weights_counts):
    total = float(sum(weights_counts))
    num_clients = len(weight_list)
    avg = []
    for layer_idx in range(len(weight_list[0])):
        accum = np.zeros_like(weight_list[0][layer_idx], dtype=np.float64)
        for client_idx in range(num_clients):
            accum += weight_list[client_idx][layer_idx].astype(np.float64) * (weights_counts[client_idx] / total)
        avg.append(accum.astype(np.float32))
    return avg

def l2_norm_between_weights(w1, w2):
    s = 0.0
    for a, b in zip(w1, w2):
        diff = a.astype(np.float64) - b.astype(np.float64)
        s += np.sum(diff * diff)
    return float(np.sqrt(s))