# src/shap_explain.py
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
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


PROCESSED = "data/processed"
RESULTS = "results"
MODEL_PATH = "models/hybrid.h5"

os.makedirs(RESULTS, exist_ok=True)

# -----------------------------------------------------------
# Load model and read its true expected input shape
# -----------------------------------------------------------
print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)

# model.input_shape = (None, seq_len, features)
_, SEQ_LEN, FEATURES = model.input_shape
FLAT_LEN = SEQ_LEN * FEATURES
print(f"[MODEL SHAPE] seq_len={SEQ_LEN}, features={FEATURES}, flat={FLAT_LEN}")

# -----------------------------------------------------------
def load_data(limit=None):
    df = pd.read_csv(f"{PROCESSED}/X_test.csv")

    # Fix extra columns (drop any column beyond expected FLAT_LEN)
    if df.shape[1] > FLAT_LEN:
        df = df.iloc[:, :FLAT_LEN]

    # If fewer columns (should not happen), pad with zeros
    if df.shape[1] < FLAT_LEN:
        missing = FLAT_LEN - df.shape[1]
        for i in range(missing):
            df[f"pad_{i}"] = 0.0

    X_flat = df.values.astype(np.float32)
    y = pd.read_csv(f"{PROCESSED}/y_test.csv")["label"].values

    if limit is not None and limit < len(X_flat):
        X_flat = X_flat[:limit]
        y = y[:limit]

    # reshape from flat → sequence
    X_seq = X_flat.reshape((-1, SEQ_LEN, FEATURES))

    # simple feature names f0..f45
    feat_names = [f"f{i}" for i in range(FEATURES)]

    return X_seq, X_flat, y, feat_names

# -----------------------------------------------------------
# Model prediction wrapper for Kernel SHAP
# -----------------------------------------------------------
def model_predict(flat_input):
    flat_input = np.array(flat_input, dtype=np.float32)

    # reshape from flat → 3D
    X = flat_input.reshape((-1, SEQ_LEN, FEATURES))

    return model.predict(X, verbose=0)

# -----------------------------------------------------------
# SHAP utilities
# -----------------------------------------------------------
def to_mean_abs_feature_importance(shap_vals):
    if isinstance(shap_vals, list):
        arr = np.array([np.mean(np.abs(v), axis=(0,1)) for v in shap_vals])
        return np.mean(arr, axis=0)
    else:
        arr = np.array(shap_vals)
        return np.mean(np.abs(arr), axis=(0,1))

def safe_argsort_topk(arr, k=10):
    arr = np.array(arr)
    return np.argsort(-arr)[:k].tolist()

def plot_beeswarm_safe(shap_vals, flat_samples, feat_names, outpath):
    try:
        shap.summary_plot(shap_vals, flat_samples, feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        print("Beeswarm failed:", e)
        plt.figure(figsize=(6,4))
        plt.text(0.5,0.5,"Beeswarm failed", ha='center')
        plt.axis("off")
        plt.savefig(outpath)
        plt.close()

def plot_classwise_bar(shap_vals, feat_names, outprefix, top_n=10):
    if isinstance(shap_vals, list):
        for c, sv in enumerate(shap_vals):
            arr = np.array(sv)
            mean_abs = np.mean(np.abs(arr), axis=(0,1))
            idx = safe_argsort_topk(mean_abs, top_n)
            names = [feat_names[i] for i in idx[::-1]]
            vals = mean_abs[idx][::-1]
            plt.figure(figsize=(6,4))
            plt.barh(names, vals)
            plt.title(f"Class {c} Top {top_n}")
            plt.tight_layout()
            plt.savefig(f"{outprefix}_class_{c}.png")
            plt.close()
    else:
        arr = np.array(shap_vals)
        mean_abs = np.mean(np.abs(arr), axis=(0,1))
        idx = safe_argsort_topk(mean_abs, top_n)
        names = [feat_names[i] for i in idx[::-1]]
        vals = mean_abs[idx][::-1]
        plt.figure(figsize=(6,4))
        plt.barh(names, vals)
        plt.title("Top Features")
        plt.tight_layout()
        plt.savefig(f"{outprefix}_single.png")
        plt.close()

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    bg_size = SHAP.get("background_size", 30)
    explain_size = SHAP.get("explain_size", 100)

    X_seq, X_flat, y, feat_names = load_data(limit=bg_size + explain_size)

    background = X_flat[:bg_size]
    to_explain = X_flat[bg_size:bg_size+explain_size]

    print("Background:", background.shape, "Explain:", to_explain.shape)

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals = explainer.shap_values(to_explain, nsamples=50)

    np.savez_compressed(f"{RESULTS}/hybrid_shap_vals.npz", shap_vals=shap_vals)

    mean_abs = to_mean_abs_feature_importance(shap_vals)
    idx = safe_argsort_topk(mean_abs, 10)
    top10 = [(feat_names[i], float(mean_abs[i])) for i in idx]

    with open(f"{RESULTS}/hybrid_shap_top10.json", "w") as f:
        json.dump(top10, f, indent=2)

    plot_beeswarm_safe(shap_vals, to_explain, feat_names, f"{RESULTS}/hybrid_shap_beeswarm.png")
    plot_classwise_bar(shap_vals, feat_names, f"{RESULTS}/hybrid_shap_bar", 10)

    print("SHAP step complete.")
