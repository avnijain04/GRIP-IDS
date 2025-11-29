# src/multi_krum.py
import numpy as np

def flatten_weights(weight_list):
    flat = np.concatenate([w.flatten() for w in weight_list])
    return flat




def multi_krum(weight_list, f=1, m=1):
    n = len(weight_list)
    if n == 0:
        return []


    flats = [flatten_weights(w) for w in weight_list]
    flats = np.array(flats)


    k = max(0, n - f - 2)
    scores = []
    for i in range(n):
        dists = np.sqrt(((flats - flats[i])**2).sum(axis=1))
        dists[i] = np.inf
        nearest = np.sort(dists)[:k] if k > 0 else np.array([])
        score = np.sum(nearest) if nearest.size > 0 else 0.0
        scores.append(score)


    scores = np.array(scores)
    m = min(m, n)
    winners_idx = np.argsort(scores)[:m]


    num_layers = len(weight_list[0])
    agg = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([weight_list[i][layer_idx].astype(np.float64) for i in winners_idx], axis=0)
        layer_avg = np.mean(layer_stack, axis=0).astype(np.float32)
        agg.append(layer_avg)
    return agg