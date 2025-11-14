# src/attacks.py
import numpy as np

def label_flip_attack(y, flip_to=None, seed=0):
    rng = np.random.RandomState(seed)
    y_new = y.copy()
    n_classes = int(y.max()) + 1
    if flip_to is None:
        for i in range(len(y_new)):
            choices = list(range(n_classes)); choices.remove(int(y_new[i]))
            y_new[i] = rng.choice(choices)
    else:
        y_new[:] = flip_to
    return y_new

def weight_scaling_attack(weights, scale=10.0):
    return [w * scale for w in weights]

def sign_attack(weights, magnitude=1.0):
    return [np.sign(w).astype(np.float32) * magnitude for w in weights]

def targeted_poisoning_replace(weights, layer_idx, delta):
    w_new = [np.array(w, copy=True) for w in weights]
    w_new[layer_idx] += delta
    return w_new
