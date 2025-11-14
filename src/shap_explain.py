# src/shap_explain.py
import os, sys, json
import numpy as np, pandas as pd
import tensorflow as tf
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from config import SHAP

PROCESSED = "data/processed"
RESULTS = "results"
MODEL_PATH = "models/hybrid.h5"
os.makedirs(RESULTS, exist_ok=True)

def load_data(limit=None):
    X_test = pd.read_csv(f"{PROCESSED}/X_test.csv").astype(np.float32)
    y_test = pd.read_csv(f"{PROCESSED}/y_test.csv")["label"].values
    feat_names = X_test.columns.tolist()
    if limit is not None and limit < len(X_test):
        X_test = X_test.sample(limit, random_state=42).reset_index(drop=True)
    return X_test, y_test, feat_names

def model_predict(X):
    if isinstance(X, pd.DataFrame):
        arr = X.values.astype(np.float32)
    elif isinstance(X, np.ndarray):
        arr = X.astype(np.float32)
    else:
        arr = np.array(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0], 1))
    elif arr.ndim == 2:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
    return model.predict(arr, verbose=0)

def to_mean_abs_feature_importance(shap_vals):
    s = shap_vals
    if isinstance(s, list):
        try:
            s_arr = np.array(s)
        except:
            s_arr = np.stack([np.array(x) for x in s], axis=0)
    else:
        s_arr = np.array(s)
    if s_arr.ndim == 3:
        mean_abs = np.mean(np.abs(s_arr), axis=(0,1))
    elif s_arr.ndim == 2:
        mean_abs = np.mean(np.abs(s_arr), axis=0)
    else:
        raise ValueError(f"Unexpected shap_vals shape: {s_arr.shape}")
    return mean_abs

def safe_argsort_topk(arr, k=10):
    if arr.size == 0: return []
    k = min(k, arr.size)
    idx = np.argsort(-arr)[:k]
    return [int(i) for i in idx]

def plot_beeswarm_safe(shap_vals, to_explain_df, feat_names, outpath):
    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_vals, to_explain_df, feature_names=feat_names, show=False)
        plt.tight_layout(); plt.savefig(outpath); plt.close()
    except Exception as e:
        print("WARNING: beeswarm plot failed:", e)
        plt.figure(figsize=(6,4)); plt.text(0.5,0.5,"Beeswarm failed to render", ha='center', va='center'); plt.axis('off'); plt.savefig(outpath); plt.close()

def plot_classwise_bar(shap_vals, feat_names, outprefix, top_n=10):
    s = shap_vals
    if isinstance(s, list):
        for c, sv in enumerate(s):
            sv_arr = np.array(sv)
            if sv_arr.ndim != 2:
                print(f"Skipping class {c} plot, unexpected shape {sv_arr.shape}")
                continue
            mean_abs = np.mean(np.abs(sv_arr), axis=0)
            idx = safe_argsort_topk(mean_abs, top_n)
            names = [feat_names[i] for i in idx[::-1]]
            values = mean_abs[idx][::-1]
            plt.figure(figsize=(6,4)); plt.barh(names, values); plt.title(f"Top {top_n} features (class {c})"); plt.tight_layout(); plt.savefig(f"{outprefix}_class_{c}.png"); plt.close()
    else:
        s_arr = np.array(s)
        if s_arr.ndim == 2:
            mean_abs = np.mean(np.abs(s_arr), axis=0)
            idx = safe_argsort_topk(mean_abs, top_n)
            names = [feat_names[i] for i in idx[::-1]]; values = mean_abs[idx][::-1]
            plt.figure(figsize=(6,4)); plt.barh(names, values); plt.title(f"Top {top_n} features"); plt.tight_layout(); plt.savefig(f"{outprefix}_single.png"); plt.close()
        else:
            print("Skipping class-wise bar: unexpected shap array shape", s_arr.shape)

if __name__ == "__main__":
    print("Loading model:", MODEL_PATH)
    model = load_model(MODEL_PATH)

    bg_size = SHAP.get("background_size", 100)
    explain_size = SHAP.get("explain_size", 200)

    print(f"Loading test data (limit={bg_size + explain_size})")
    X_all, y_all, feat_names = load_data(limit=bg_size + explain_size)
    X_all = X_all.reset_index(drop=True)

    background = X_all.sample(n=bg_size, random_state=1).reset_index(drop=True)
    to_explain = X_all.drop(background.index).reset_index(drop=True).iloc[:explain_size]
    print("Background shape:", background.shape, "To_explain:", to_explain.shape)

    if SHAP.get("use_kernel", True):
        print("Initializing KernelExplainer... this may take time.")
        explainer = shap.KernelExplainer(model_predict, background)
        print("Computing SHAP values... (this can be slow)")
        try:
            shap_vals = explainer.shap_values(to_explain, nsamples=100)
        except Exception as e:
            print("KernelExplainer failed:", e)
            shap_vals = explainer.shap_values(to_explain, nsamples=50)
    else:
        print("Using DeepExplainer (faster on TF models with moderate size).")
        try:
            explainer = shap.DeepExplainer(model, background.values if hasattr(background, "values") else background)
            shap_vals = explainer.shap_values(to_explain.values if hasattr(to_explain, "values") else to_explain)
        except Exception as e:
            print("DeepExplainer failed:", e)
            print("Falling back to KernelExplainer")
            explainer = shap.KernelExplainer(model_predict, background)
            shap_vals = explainer.shap_values(to_explain, nsamples=50)

    np.savez_compressed(f"{RESULTS}/hybrid_shap_vals.npz", shap_vals=shap_vals)
    print("Saved shap values to results/hybrid_shap_vals.npz")

    mean_abs = to_mean_abs_feature_importance(shap_vals)
    idx = safe_argsort_topk(mean_abs, k=10)
    top10 = [(feat_names[i], float(mean_abs[i])) for i in idx]
    with open(f"{RESULTS}/hybrid_shap_top10.json", "w") as f:
        json.dump(top10, f, indent=2)
    print("Saved top10 to results/hybrid_shap_top10.json")

    plot_beeswarm_safe(shap_vals, to_explain, feat_names, f"{RESULTS}/hybrid_shap_beeswarm.png")
    print("Saved beeswarm to results/hybrid_shap_beeswarm.png")

    plot_classwise_bar(shap_vals, feat_names, f"{RESULTS}/hybrid_shap_bar", top_n=10)
    print("Saved class-wise bar plots (if applicable).")

    print("SHAP step complete.")
