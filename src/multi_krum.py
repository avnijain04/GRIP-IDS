# src/multi_krum.py
import numpy as np

def flatten_weights(weight_list):
    """Concatenate all layer arrays into a 1D vector for a single client."""
    flat = np.concatenate([w.flatten() for w in weight_list])
    return flat

def unflatten_to_shape(flat_vec, template):
    """Not needed for Multi-Krum output since we return selected weight lists."""
    raise NotImplementedError

def multi_krum(weight_list, f=1, m=1):
    """
    Multi-Krum aggregator.
    weight_list: list of clients' weight lists. Each client's weights is a list of numpy arrays.
    f: number of Byzantine clients assumed.
    m: number of candidate updates to select (Multi-Krum selects m winners and average them).
    returns: aggregated_weights (list of numpy arrays) - averaged winners
    """
    n = len(weight_list)
    if n == 0:
        return []

    # Flatten each client's weights to vector
    flats = [flatten_weights(w) for w in weight_list]  # shape (n, D)
    flats = np.array(flats)
    D = flats.shape[1]

    # For each candidate i compute score = sum of distances to closest (n - f - 2) other vectors
    # Number of neighbors to consider:
    k = max(0, n - f - 2)
    scores = []
    for i in range(n):
        dists = np.sqrt(((flats - flats[i])**2).sum(axis=1))
        # exclude self
        dists[i] = np.inf
        # sort
        nearest = np.sort(dists)[:k] if k > 0 else np.array([])
        score = np.sum(nearest) if nearest.size > 0 else 0.0
        scores.append(score)

    scores = np.array(scores)
    # choose m indices with smallest scores
    m = min(m, n)
    winners_idx = np.argsort(scores)[:m]
    # average their weights elementwise (but in original shapes)
    # Convert winners' weight lists into float arrays and average per-layer
    # assume all clients have same layer shapes
    num_layers = len(weight_list[0])
    agg = []
    for layer_idx in range(num_layers):
        layer_stack = np.stack([weight_list[i][layer_idx].astype(np.float64) for i in winners_idx], axis=0)
        layer_avg = np.mean(layer_stack, axis=0).astype(np.float32)
        agg.append(layer_avg)
    return agg
