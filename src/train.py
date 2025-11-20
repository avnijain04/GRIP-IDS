import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_class_weight

from config import (
    X_TRAIN_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_TEST_NPY,
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    MODEL, RANDOM_SEED, SEQUENCE_LENGTH
)

from model_defs import build_cnn, build_lstm, build_hybrid

# -----------------------
# Basic setup
# -----------------------
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Data loader
# -----------------------
def load_processed():
    """Prefer .npy. Fallback to flattened CSV reconstruction."""
    if all([
        os.path.exists(X_TRAIN_NPY),
        os.path.exists(X_TEST_NPY),
        os.path.exists(Y_TRAIN_NPY),
        os.path.exists(Y_TEST_NPY)
    ]):
        X_train = np.load(X_TRAIN_NPY)
        X_test = np.load(X_TEST_NPY)
        y_train = np.load(Y_TRAIN_NPY)
        y_test = np.load(Y_TEST_NPY)
        return X_train, X_test, y_train, y_test

    X_train_df = pd.read_csv(X_TRAIN_FILE)
    X_test_df = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE)["label"].values
    y_test = pd.read_csv(Y_TEST_FILE)["label"].values

    seq_len = SEQUENCE_LENGTH

    def reconstruct(df):
        ncols = df.shape[1]
        if ncols % seq_len != 0:
            raise ValueError("Cannot reconstruct sequences from CSV: unexpected column count.")
        nfeat = ncols // seq_len
        return df.values.reshape((-1, seq_len, nfeat))

    X_train = reconstruct(X_train_df)
    X_test = reconstruct(X_test_df)
    return X_train, X_test, y_train, y_test


# -----------------------
# Callbacks
# -----------------------
class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss, self.val_loss = [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(float(logs.get("loss", np.nan)))
        self.val_loss.append(float(logs.get("val_loss", np.nan)))
        self.acc.append(float(logs.get("accuracy", logs.get("acc", np.nan))))
        self.val_acc.append(float(logs.get("val_accuracy", logs.get("val_acc", np.nan))))


class PRFCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.precision, self.recall, self.f1 = [], [], []

    def on_epoch_end(self, epoch, logs=None):
        preds_prob = self.model.predict(self.X_val, verbose=0)
        preds = np.argmax(preds_prob, axis=1)
        p = precision_score(self.y_val, preds, average="macro", zero_division=0)
        r = recall_score(self.y_val, preds, average="macro", zero_division=0)
        f = f1_score(self.y_val, preds, average="macro", zero_division=0)
        self.precision.append(float(p)); self.recall.append(float(r)); self.f1.append(float(f))
        print(f" — PRF epoch: P={p:.4f}  R={r:.4f}  F1={f:.4f}")


# -----------------------
# Plot utilities
# -----------------------
def plot_history(history, name):
    plt.figure(figsize=(6, 4))
    plt.plot(history["loss"], label="train_loss", marker="o")
    plt.plot(history["val_loss"], label="val_loss", marker="o")
    plt.title(f"{name} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/{name}_loss.png"); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["accuracy"], label="train_acc", marker="o")
    plt.plot(history["val_accuracy"], label="val_acc", marker="o")
    plt.title(f"{name} Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/{name}_acc.png"); plt.close()


def plot_prf(cb, name):
    epochs = range(1, len(cb.f1) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, cb.precision, marker="o", label="Precision")
    plt.plot(epochs, cb.recall, marker="o", label="Recall")
    plt.plot(epochs, cb.f1, marker="o", label="F1")
    plt.title(f"{name} PRF Metrics"); plt.xlabel("Epoch"); plt.ylabel("Metric")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"results/{name}_prf_metrics.png"); plt.close()


def plot_confusion(cm, classes, name):
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=90, fontsize=8)
    plt.yticks(ticks=np.arange(len(classes))+0.5, labels=classes, rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"results/{name}_confusion.png", dpi=300); plt.close()


# -----------------------
# Threshold tuning
# -----------------------
def tune_thresholds(y_true, y_probs, classes):
    results = {}
    y_true_onehot = np.eye(len(classes))[y_true]
    for i, cls in enumerate(classes):
        y_bin = y_true_onehot[:, i]
        y_score = y_probs[:, i]
        precision, recall, thresholds = precision_recall_curve(y_bin, y_score)
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        if len(thresholds) == 0:
            results[cls] = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        best_idx = np.argmax(f1[1:]) + 1
        results[cls] = {
            "threshold": float(thresholds[best_idx - 1]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1[best_idx])
        }
    return results


# -----------------------
# Main train/evaluate
# -----------------------
def train_and_evaluate(model_builder, name):
    print(f"\n--- Training {name} ---")

    X_train, X_test, y_train, y_test = load_processed()

    # --- remap classes to contiguous 0..C-1 based on labels present in data ---
    unique = np.unique(np.concatenate([y_train, y_test]))
    class_mapping = {old: new for new, old in enumerate(unique)}
    num_classes = len(unique)

    # remap arrays
    y_train = np.array([class_mapping[x] for x in y_train], dtype=np.int32)
    y_test = np.array([class_mapping[x] for x in y_test], dtype=np.int32)

    print("Final num_classes:", num_classes)
    print("Classes present (original labels):", unique.tolist())
    print("Shapes:", X_train.shape, X_test.shape)

    # --- compute class weights (balanced) and map to per-sample weights ---
    cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
    class_weights = {i: float(cw[i]) for i in range(num_classes)}
    print("Class weights sample:", dict(list(class_weights.items())[:6]))

    # create sample_weight array (per-sample weighting) to avoid tf.class_weight internals
    sample_weight = np.array([class_weights[int(lbl)] for lbl in y_train], dtype=np.float32)
    # validation sample weights optional (we don't need to pass them)
    # sample_weight_val = np.array([class_weights[int(lbl)] for lbl in y_test], dtype=np.float32)

    # --- build model ---
    model = model_builder(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # --- callbacks ---
    history_cb = HistoryCallback()
    prf_cb = PRFCallback(X_test, y_test)
    checkpoint_path = f"models/{name}_best.keras"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True)

    # --- train with sample_weight (no class_weight) ---
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MODEL["epochs"],
        batch_size=MODEL["batch_size"],
        sample_weight=sample_weight,
        callbacks=[history_cb, prf_cb, checkpoint],
        verbose=1
    )

    # --- save model ---
    final_path = f"models/{name}.keras"
    model.save(final_path)
    try:
        model.save(f"models/{name}.h5", include_optimizer=False)
        print(f"Saved models: {final_path} and models/{name}.h5")
    except Exception:
        print(f"Saved model: {final_path}")

    # --- evaluate ---
    y_probs = model.predict(X_test, batch_size=MODEL["batch_size"])
    y_pred = np.argmax(y_probs, axis=1)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    json.dump(report, open(f"results/{name}_report.json", "w"), indent=2)

    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(f"results/{name}_confusion.csv", cm, delimiter=",")
    plot_confusion(cm, [str(i) for i in range(num_classes)], name)

    # AUC macro (one-vs-rest)
    try:
        y_test_onehot = np.eye(num_classes)[y_test]
        auc_macro = float(roc_auc_score(y_test_onehot, y_probs, multi_class="ovr"))
    except Exception:
        auc_macro = None

    # thresholds
    thresholds = tune_thresholds(y_test, y_probs, list(range(num_classes)))
    json.dump(thresholds, open(f"results/{name}_thresholds.json", "w"), indent=2)

    # save history & plots
    history = {"loss": history_cb.loss, "val_loss": history_cb.val_loss, "accuracy": history_cb.acc, "val_accuracy": history_cb.val_acc}
    json.dump(history, open(f"results/{name}_history.json", "w"), indent=2)

    plot_history(history, name)
    plot_prf(prf_cb, name)

    summary = {
        "num_classes": num_classes,
        "auc_ovr": auc_macro,
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    }
    json.dump(summary, open(f"results/{name}_summary.json", "w"), indent=2)

    print(f"Saved plots and metrics for {name}.\n")


# -----------------------
# Run all models
# -----------------------
if __name__ == "__main__":
    train_and_evaluate(build_cnn, "cnn")
    train_and_evaluate(build_lstm, "lstm")
    train_and_evaluate(build_hybrid, "hybrid")
    print("All training completed.")
