import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score
)
from typing import Optional, Sequence, Dict

sns.set(style="whitegrid", rc={"figure.figsize": (6, 4)})


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray,
                          outpath: str,
                          normalize: bool = True,
                          annot: bool = True,
                          labels: Optional[Sequence[str]] = None):
    
    ensure_dir(outpath)
    cm = np.array(cm, dtype=np.float64)

    # Normalize (per-true-class)
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
        disp_cm = cm_norm
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        disp_cm = cm
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(disp_cm, annot=annot, fmt=fmt, cmap="Blues", cbar=True,
                     xticklabels=labels if labels is not None else range(disp_cm.shape[1]),
                     yticklabels=labels if labels is not None else range(disp_cm.shape[0]))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def _safe_onehot(y, num_classes):
    """Return one-hot (n, C) for labels y (n,)"""
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    for i, v in enumerate(y):
        oh[i, int(v)] = 1.0
    return oh


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, out_prefix: str):
    ensure_dir(out_prefix + "_roc_class0.png")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # If y_prob is 1D (binary probabilities for positive class), convert to (n,2)
    if y_prob.ndim == 1:
        y_prob_pos = y_prob
        y_prob = np.stack([1.0 - y_prob_pos, y_prob_pos], axis=1)

    n_samples, n_classes = y_prob.shape
    if len(np.unique(y_true)) == 1:
        # Single-class (degenerate) — skip plotting
        print("ROC/PR: only one class present in y_true; skipping ROC/PR plots.")
        return

    # one-hot target
    y_onehot = _safe_onehot(y_true, n_classes)

    # Per-class ROC & PR
    roc_aucs = {}
    pr_aps = {}

    # Combined ROC figure
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_onehot[:, c], y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            roc_aucs[c] = float(roc_auc)
            plt.plot(fpr, tpr, lw=2, label=f"Class {c} (AUC={roc_auc:.3f})")
            # save per-class ROC
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC class {c} (AUC={roc_auc:.3f})")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_roc_class{c}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning ROC class {c} failed: {e}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Precision-Recall
    plt.figure(figsize=(8, 6))
    for c in range(n_classes):
        try:
            precision, recall, _ = precision_recall_curve(y_onehot[:, c], y_prob[:, c])
            ap = average_precision_score(y_onehot[:, c], y_prob[:, c])
            pr_aps[c] = float(ap)
            plt.plot(recall, precision, lw=2, label=f"Class {c} (AP={ap:.3f})")

            # save per-class PR
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall class {c} (AP={ap:.3f})")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_pr_class{c}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Warning PR class {c} failed: {e}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision-Recall curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Optionally compute macro AUCs (one-vs-rest)
    try:
        auc_macro = roc_auc_score(y_onehot, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = None

    # Save a tiny JSON summary next to images
    summary = {
        "roc_auc_per_class": roc_aucs,
        "pr_ap_per_class": pr_aps,
        "roc_auc_macro_ovr": None if auc_macro is None else float(auc_macro),
    }
    try:
        import json
        json.dump(summary, open(f"{out_prefix}_rocpr_summary.json", "w"), indent=2)
    except Exception:
        pass

    return summary


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    res = {}
    for c in classes:
        mask = (y_true == c)
        total = mask.sum()
        if total == 0:
            res[int(c)] = None
            continue
        correct = int((y_pred[mask] == y_true[mask]).sum())
        res[int(c)] = float(correct) / float(total)
    return res
