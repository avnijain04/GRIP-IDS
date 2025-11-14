# src/train.py
import os
# Force non-interactive backend for robust saving on Windows
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd
import tensorflow as tf
# use Agg backend before pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE,
    MODEL, RANDOM_SEED
)
from model_defs import build_cnn, build_lstm, build_hybrid

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


class HistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss, self.val_loss = [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(float(logs.get("loss", None)))
        self.val_loss.append(float(logs.get("val_loss", None)))
        # some TF versions use 'accuracy' or 'acc' — handle both
        self.acc.append(float(logs.get("accuracy", logs.get("acc", None))))
        self.val_acc.append(float(logs.get("val_accuracy", logs.get("val_acc", None))))


def load_processed():
    X_train = pd.read_csv(X_TRAIN_FILE).values
    X_test = pd.read_csv(X_TEST_FILE).values
    y_train = pd.read_csv(Y_TRAIN_FILE)["label"].values
    y_test = pd.read_csv(Y_TEST_FILE)["label"].values
    return X_train, X_test, y_train, y_test


def reshape_for_sequence(X):
    return X.reshape((X.shape[0], X.shape[1], 1))


def safe_save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def train_and_evaluate(model_builder, name, save_history=True):
    print(f"\n--- Training {name} ---")
    X_train, X_test, y_train, y_test = load_processed()
    X_train = reshape_for_sequence(X_train)
    X_test = reshape_for_sequence(X_test)

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    print("Detected num_classes:", num_classes)
    print("Shapes:", X_train.shape, X_test.shape)

    model = model_builder(input_shape=X_train.shape[1:], num_classes=num_classes)

    history_cb = HistoryCallback()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MODEL["epochs"],
        batch_size=MODEL["batch_size"],
        verbose=1,
        callbacks=[history_cb]
    )

    # save model
    model_path = f"models/{name}.h5"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # predictions and metrics
    y_pred = np.argmax(model.predict(X_test, batch_size=MODEL["batch_size"]), axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    safe_save_json(f"results/{name}_report.json", report)
    np.savetxt(f"results/{name}_confusion.csv", cm, delimiter=",")

    # save history JSON so plotting is reproducible
    history_obj = {
        "loss": history_cb.loss,
        "val_loss": history_cb.val_loss,
        "accuracy": history_cb.acc,
        "val_accuracy": history_cb.val_acc
    }
    if save_history:
        safe_save_json(f"results/{name}_history.json", history_obj)
        print(f"Saved history JSON: results/{name}_history.json")

    # Create plots immediately (robust plotting)
    create_plots_from_history(history_obj, name)
    print(f"Saved plots for {name}.\n")


def create_plots_from_history(history_obj, name):
    # ensure lists exist and have numbers
    loss = history_obj.get("loss", []) or []
    val_loss = history_obj.get("val_loss", []) or []
    acc = history_obj.get("accuracy", []) or []
    val_acc = history_obj.get("val_accuracy", []) or []

    # convert to floats and replace None with nan
    def to_float_list(lst):
        out = []
        for x in lst:
            try:
                out.append(float(x))
            except Exception:
                out.append(np.nan)
        return out

    loss = to_float_list(loss)
    val_loss = to_float_list(val_loss)
    acc = to_float_list(acc)
    val_acc = to_float_list(val_acc)

    epochs = list(range(1, max(1, len(loss), len(acc)) + 1))

    # LOSS plot
    plt.figure(figsize=(6, 4))
    if len(loss) > 0:
        plt.plot(list(range(1, len(loss)+1)), loss, marker='o', label="train_loss")
    if len(val_loss) > 0:
        plt.plot(list(range(1, len(val_loss)+1)), val_loss, marker='o', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{name}_loss.png")
    plt.close()

    # ACC plot
    plt.figure(figsize=(6, 4))
    if len(acc) > 0:
        plt.plot(list(range(1, len(acc)+1)), acc, marker='o', label="train_acc")
    if len(val_acc) > 0:
        plt.plot(list(range(1, len(val_acc)+1)), val_acc, marker='o', label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{name}_acc.png")
    plt.close()


if __name__ == "__main__":
    # For quick debugging on low-RAM devices you can comment out the models you don't want to train
    train_and_evaluate(build_cnn, "cnn")
    train_and_evaluate(build_lstm, "lstm")
    train_and_evaluate(build_hybrid, "hybrid")
    print("All training completed.")
