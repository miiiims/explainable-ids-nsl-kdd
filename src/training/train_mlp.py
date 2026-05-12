"""
Train Experiment 2: lightweight tabular MLP on NSL-KDD.
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
from src.config import PROCESSED_DIR, RANDOM_SEED, mlp_dirs
from src.models.mlp_model import build_mlp_model


DIRS = mlp_dirs()
MODELS_DIR = DIRS["models"]
METRICS_DIR = DIRS["metrics"]

BATCH_SIZE = 128
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
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    return X_train, X_test, y_train, y_test, class_names


def print_distribution(y, class_names, split_name):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\nClass distribution - {split_name}:")
    for cls, count in zip(unique, counts):
        print(f"  {cls} ({class_names[int(cls)]:8s}): {count:6d} ({100 * count / total:5.2f}%)")


def compute_capped_class_weights(y_train, class_names):
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    class_weight = {}
    print("\nClass weights before/after capping:")
    for cls, raw_weight in zip(classes, raw_weights):
        capped = float(np.clip(raw_weight, WEIGHT_MIN, WEIGHT_MAX))
        class_weight[int(cls)] = capped
        print(f"  {cls} ({class_names[int(cls)]:8s}): {raw_weight:8.2f} -> {capped:.2f}")
    return class_weight


def main():
    set_global_seed(RANDOM_SEED)

    print("=" * 60)
    print("Training MLP - NSL-KDD tabular IDS")
    print("=" * 60)

    X_train_full, X_test, y_train_full, y_test, class_names = load_data()
    num_classes = len(class_names)

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

    model = build_mlp_model(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        dense_units_1=128,
        dense_units_2=64,
        dropout_rate=0.3,
        l2_reg=1e-5,
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
    print(f"  Best model: {best_model_path}")
    print(f"  Final model: {final_model_path}")
    print(f"  History CSV: {history_csv}")


if __name__ == "__main__":
    main()
