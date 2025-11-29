# src/train.py — trains cnn / lstm / hybrid and saves metrics + plots (updated)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import (
    X_TRAIN_NPY, X_TEST_NPY,
    Y_TRAIN_NPY, Y_TEST_NPY,
    MODEL, RANDOM_SEED
)
from model_defs import build_cnn, build_lstm, build_hybrid
from plot_metrics import plot_roc_pr, plot_confusion_matrix, per_class_accuracy

# ===================================================
# Reproducibility
# ===================================================
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ===================================================
# Load processed NPY data
# ===================================================
def load_processed():
    return (
        np.load(X_TRAIN_NPY),
        np.load(X_TEST_NPY),
        np.load(Y_TRAIN_NPY),
        np.load(Y_TEST_NPY),
    )

# ===================================================
# Callbacks
# ===================================================
class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy"))
        self.val_acc.append(logs.get("val_accuracy"))

class PRFCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.precision = []
        self.recall = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        p = precision_score(self.y_val, y_pred, average="macro", zero_division=0)
        r = recall_score(self.y_val, y_pred, average="macro", zero_division=0)
        f = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        self.precision.append(p)
        self.recall.append(r)
        self.f1.append(f)
        print(f"— PRF epoch {epoch+1}: P={p:.4f} R={r:.4f} F1={f:.4f}")

# ===================================================
# Plotting utils
# ===================================================
def plot_history(hist, name):
    plt.figure(figsize=(6, 4))
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.legend()
    plt.title(f"{name} Loss")
    plt.savefig(f"results/{name}_loss.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(hist["accuracy"], label="Train Acc")
    plt.plot(hist["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title(f"{name} Accuracy")
    plt.savefig(f"results/{name}_acc.png")
    plt.close()

def plot_prf(cb, name):
    epochs = range(1, len(cb.f1) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, cb.precision, label="Precision")
    plt.plot(epochs, cb.recall, label="Recall")
    plt.plot(epochs, cb.f1, label="F1")
    plt.legend()
    plt.title(f"{name} PRF")
    plt.savefig(f"results/{name}_prf.png")
    plt.close()

# ===================================================
# Training
# ===================================================
def train_and_evaluate(build_fn, name):
    print(f"\n==============================")
    print(f"       TRAINING {name}")
    print(f"==============================")

    X_train, X_test, y_train, y_test = load_processed()

    unique = np.sort(np.unique(np.concatenate([y_train, y_test])))
    mapping = {old: i for i, old in enumerate(unique)}
    y_train = np.array([mapping[v] for v in y_train])
    y_test = np.array([mapping[v] for v in y_test])
    num_classes = len(unique)

    print("Classes:", unique.tolist())
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    class_w = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_train
    )
    class_w = {i: float(class_w[i]) for i in range(num_classes)}
    sample_weight = np.array([class_w[int(v)] for v in y_train], dtype=np.float32)

    model = build_fn(input_shape=X_train.shape[1:], num_classes=num_classes)

    hist_cb = HistoryCallback()
    prf_cb = PRFCallback(X_test, y_test)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        f"models/{name}_best.keras",
        monitor="val_loss",
        save_best_only=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=MODEL["epochs"],
        batch_size=MODEL["batch_size"],
        sample_weight=sample_weight,
        callbacks=[hist_cb, prf_cb, ckpt],
        verbose=1,
    )

    # Save final model in .keras and .h5 for compatibility
    try:
        model.save(f"models/{name}.keras")
    except Exception:
        pass
    try:
        model.save(f"models/{name}.h5")
    except Exception:
        pass

    # Evaluate
    y_prob = model.predict(X_test, batch_size=MODEL["batch_size"])
    y_pred = np.argmax(y_prob, axis=1)

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    json.dump(rep, open(f"results/{name}_report.json", "w"), indent=2)

    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(f"results/{name}_confusion.csv", cm, delimiter=",")
    plot_confusion_matrix(cm, f"results/{name}_confusion.png", normalize=True, annot=True)

    hist = {
        "loss": hist_cb.loss,
        "val_loss": hist_cb.val_loss,
        "accuracy": hist_cb.acc,
        "val_accuracy": hist_cb.val_acc,
    }
    json.dump(hist, open(f"results/{name}_history.json", "w"), indent=2)
    plot_history(hist, name)
    plot_prf(prf_cb, name)

    # ROC/PR per-class (use our utility)
    try:
        plot_roc_pr(y_test, y_prob, out_prefix=f"results/{name}")
    except Exception as e:
        print("ROC/PR plotting failed:", e)

    # Save per-class accuracy
    pca = per_class_accuracy(y_test, y_pred)
    json.dump(pca, open(f"results/{name}_per_class_accuracy.json", "w"), indent=2)

    print(f"→ {name} training complete.\n")

if __name__ == "__main__":
    train_and_evaluate(build_cnn, "cnn")
    train_and_evaluate(build_lstm, "lstm")
    train_and_evaluate(build_hybrid, "hybrid")
    print("All models trained successfully.")
