import os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from config import SHAP

import warnings
warnings.filterwarnings("ignore")

PROCESSED = "data/processed"
RESULTS = "results"

MODEL_PATHS = [
    "models/hybrid_best.keras",
    "models/hybrid.keras",
    "models/hybrid_best.h5",
    "models/hybrid.h5",
]

MODEL_PATH = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError("Could not find hybrid model. Train the model first.")

os.makedirs(RESULTS, exist_ok=True)

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)

model_shape = model.input_shape
if model_shape is None or len(model_shape) < 3:
    raise ValueError(f"Unexpected model.input_shape: {model_shape}")

_, SEQ_LEN, FEATURES = model_shape
SEQ_LEN = int(SEQ_LEN)
FEATURES = int(FEATURES)
FLAT_LEN = SEQ_LEN * FEATURES

print(f"[MODEL SHAPE] seq_len={SEQ_LEN}, features={FEATURES}, flat_len={FLAT_LEN}")


# Data Loader
def load_data(limit=None):
    df = pd.read_csv(f"{PROCESSED}/X_test.csv")

    if df.shape[1] > FLAT_LEN:
        df = df.iloc[:, :FLAT_LEN]
    elif df.shape[1] < FLAT_LEN:
        missing = FLAT_LEN - df.shape[1]
        for i in range(missing):
            df[f"pad_{i}"] = 0.0

    X_flat = df.values.astype(np.float32)
    y = pd.read_csv(f"{PROCESSED}/y_test.csv")["label"].values

    if limit is not None:
        X_flat = X_flat[:limit]
        y = y[:limit]

    X_seq = X_flat.reshape((-1, SEQ_LEN, FEATURES))

    feat_names = [f"t{t}_f{f}" for t in range(SEQ_LEN) for f in range(FEATURES)]
    return X_seq, X_flat, y, feat_names


# Prediction wrapper for SHAP
def model_predict(flat_input):
    flat_input = np.array(flat_input, dtype=np.float32)
    X = flat_input.reshape((-1, SEQ_LEN, FEATURES))
    return model.predict(X, verbose=0)


# SHAP normalizer — handles ALL formats
def normalize_shap_output(shap_vals, flat_len):
    arr = np.array(shap_vals)

    # Case 1: list-of-arrays → OK
    if isinstance(shap_vals, list):
        normalized = []
        for sv in shap_vals:
            a = np.array(sv)
            if a.ndim == 2 and a.shape[1] >= flat_len:
                normalized.append(a[:, :flat_len])
            else:
                raise ValueError(f"Inconsistent SHAP array inside list: shape={a.shape}")
        return normalized

    # Case 2: multi-output shape (samples, features, classes)
    if arr.ndim == 3:
        n_samples, n_features, n_outputs = arr.shape
        if n_features < flat_len:
            raise ValueError(f"SHAP feature dimension mismatch: got {n_features}, expected ≥ {flat_len}")

        return [arr[:, :flat_len, i] for i in range(n_outputs)]

    # Case 3: normal shape (samples, features)
    if arr.ndim == 2:
        if arr.shape[1] < flat_len:
            raise ValueError(f"SHAP returned too few features: {arr.shape}")
        return arr[:, :flat_len]

    raise ValueError(f"Unsupported SHAP output format: shape={arr.shape}")


# Mean absolute SHAP importance
def to_mean_abs_feature_importance(shap_vals):
    if isinstance(shap_vals, list):
        per_class = [np.mean(np.abs(v), axis=0) for v in shap_vals]
        return np.mean(np.stack(per_class, axis=0), axis=0)
    arr = np.array(shap_vals)
    return np.mean(np.abs(arr), axis=0)


def safe_argsort_topk(arr, k=10):
    arr = np.array(arr).reshape(-1)
    order = np.argsort(-arr)
    return [int(x) for x in order[:k]]


def plot_beeswarm_safe(shap_vals, flat_samples, feat_names, outpath):
    try:
        shap.summary_plot(shap_vals, flat_samples, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
    except Exception as e:
        print("Beeswarm failed:", e)
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Beeswarm failed", ha='center')
        plt.axis("off")
        plt.savefig(outpath)
    plt.close()


def plot_classwise_bar(shap_vals, feat_names, outprefix, top_n=10):
    if isinstance(shap_vals, list):
        # multi-class
        for c, sv in enumerate(shap_vals):
            mean_abs = np.mean(np.abs(sv), axis=0)
            idx = safe_argsort_topk(mean_abs, top_n)
            names = [feat_names[i] for i in idx[::-1]]
            vals = mean_abs[idx][::-1]
            plt.figure(figsize=(7, 4))
            plt.barh(names, vals)
            plt.title(f"Class {c} — Top {top_n}")
            plt.tight_layout()
            plt.savefig(f"{outprefix}_class_{c}.png")
            plt.close()
    else:
        # single-output
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        idx = safe_argsort_topk(mean_abs, top_n)
        names = [feat_names[i] for i in idx[::-1]]
        vals = mean_abs[idx][::-1]
        plt.figure(figsize=(7, 4))
        plt.barh(names, vals)
        plt.title("Top Features")
        plt.tight_layout()
        plt.savefig(f"{outprefix}_single.png")
        plt.close()


def to_scalar(x):
    arr = np.array(x, dtype=float)
    return float(arr.reshape(-1)[0])


if __name__ == "__main__":
    bg_size = min(SHAP.get("background_size", 30), 50)
    explain_size = min(SHAP.get("explain_size", 100), 200)

    X_seq, X_flat, y, feat_names = load_data(limit=bg_size + explain_size)

    background = X_flat[:bg_size]
    to_explain = X_flat[bg_size:bg_size + explain_size]

    print("Background:", background.shape, "Explain:", to_explain.shape)

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals_raw = explainer.shap_values(to_explain, nsamples=SHAP.get("nsamples", 50))

    # Normalize to a guaranteed correct format
    shap_vals = normalize_shap_output(shap_vals_raw, FLAT_LEN)

    # Save raw SHAP
    np.savez_compressed(f"{RESULTS}/hybrid_shap_vals.npz", shap_vals=shap_vals)

    # Compute top features
    mean_abs = to_mean_abs_feature_importance(shap_vals)
    idx = safe_argsort_topk(mean_abs, 10)
    top10 = [(feat_names[i], to_scalar(mean_abs[i])) for i in idx]

    with open(f"{RESULTS}/hybrid_shap_top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    plot_beeswarm_safe(shap_vals, to_explain, feat_names, f"{RESULTS}/hybrid_shap_beeswarm.png")
    plot_classwise_bar(shap_vals, feat_names, f"{RESULTS}/hybrid_shap_bar", 10)

    print("SHAP step complete.")
