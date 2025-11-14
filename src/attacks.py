# src/attacks.py
import numpy as np

def label_flip_attack(y, flip_to=None, seed=0):
    """
    Simple label flip: change all labels to `flip_to` (if int) or to random wrong label.
    y: numpy array labels
    flip_to: int label to flip to. If None -> choose a label different from original (random).
    """
    rng = np.random.RandomState(seed)
    y_new = y.copy()
    n_classes = int(y.max()) + 1
    if flip_to is None:
        # pick a random label different than original per sample
        for i in range(len(y_new)):
            choices = list(range(n_classes))
            choices.remove(int(y_new[i]))
            y_new[i] = rng.choice(choices)
    else:
        y_new[:] = flip_to
    return y_new

def weight_scaling_attack(weights, scale=10.0):
    """
    Simple weight attack: scale weights by a large factor to push global model.
    weights: list of numpy arrays
    returns manipulated weights
    """
    return [w * scale for w in weights]

def sign_attack(weights, magnitude=1.0):
    """
    Sign-flipping attack: set each weight to magnitude * sign of weight
    """
    return [np.sign(w).astype(np.float32) * magnitude for w in weights]

def targeted_poisoning_replace(weights, layer_idx, delta):
    """
    Add delta to a specific layer (for targeted poisoning).
    """
    w_new = [np.array(w, copy=True) for w in weights]
    w_new[layer_idx] += delta
    return w_new
