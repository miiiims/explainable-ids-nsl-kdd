"""
Train Experiment 3b: LSTM sans attention.

Ce script utilise les mêmes données séquentielles que l'expérience LSTM + Attention,
mais sans la couche d'attention. Il permet de comparer directement le gain
apporté par l'attention sur la même architecture séquentielle.
"""

import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import LSTM_NAME, PROCESSED_DIR, RANDOM_SEED, lstm_dirs
from src.models.lstm_attention import build_lstm_model

DIRS = lstm_dirs()
MODELS_DIR = DIRS["models"]
METRICS_DIR = DIRS["metrics"]

BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
VAL_SIZE = 0.15
WEIGHT_MIN = 0.3
WEIGHT_MAX = 5.0


def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_seq.npy")).astype("float32")
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train_seq.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_test shape  : {y_test.shape}")
    return X_train, y_train, X_test, y_test, class_names


def print_distribution(y, class_names, split_name):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\nClass distribution - {split_name}:")
    for cls, count in zip(unique, counts):
        name = class_names[int(cls)] if cls < len(class_names) else f"class_{cls}"
        print(f"  {cls} ({name:8s}) : {count:6d} ({100 * count / total:5.2f}%)")


def compute_capped_class_weights(y_train, class_names):
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {}
    print("\nClass weights before / after capping:")

    for cls, raw_weight in zip(classes, raw_weights):
        capped = float(np.clip(raw_weight, WEIGHT_MIN, WEIGHT_MAX))
        class_weight[int(cls)] = capped
        name = class_names[int(cls)] if cls < len(class_names) else f"class_{cls}"
        print(f"  {cls} ({name:8s}) : {raw_weight:8.2f} -> {capped:.2f}")

    return class_weight


def main():
    set_global_seed(RANDOM_SEED)

    print("=" * 60)
    print("Training LSTM sans attention - NSL-KDD")
    print("=" * 60)

    X_train_full, y_train_full, X_test, y_test, class_names = load_data()
    num_classes = len(class_names)
    input_shape = X_train_full.shape[1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print_distribution(y_train, class_names, "train")
    print_distribution(y_val, class_names, "validation")
    print_distribution(y_test, class_names, "test")

    class_weight = compute_capped_class_weights(y_train, class_names)

    model = build_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=64,
        dense_units=64,
        dropout_rate_1=0.2,
        dropout_rate_2=0.1,
        l2_reg=1e-5,
        use_attention=False,
        return_attention=False,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    final_model_path = os.path.join(MODELS_DIR, "final_model.keras")

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)
    print(f"Saved final model -> {final_model_path}")
    print(f"Saved best model  -> {best_model_path}")

    history_pkl = os.path.join(METRICS_DIR, "training_history.pkl")
    with open(history_pkl, "wb") as f:
        pickle.dump(history.history, f)

    history_csv = os.path.join(METRICS_DIR, "training_history.csv")
    pd.DataFrame(history.history).to_csv(history_csv, index=False)

    np.save(os.path.join(METRICS_DIR, "class_names.npy"), class_names)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print("\nTraining summary")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.4f}")
    print(f"  Best val_accuracy: {history.history['val_accuracy'][best_epoch - 1]:.4f}")
    print(f"  Saved metrics in: {METRICS_DIR}")
    print(f"  Saved models in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
