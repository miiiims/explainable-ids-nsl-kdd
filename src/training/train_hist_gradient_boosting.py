"""
Train a histogram gradient boosting baseline on NSL-KDD tabular features.

This model is included as a stronger tabular baseline than Random Forest. It
uses capped sample weights to give rare attack classes more influence without
letting the tiny U2R class dominate training.
"""

import os
import random
import sys

import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, hist_gradient_boosting_dirs


DIRS = hist_gradient_boosting_dirs()
MODELS_DIR = DIRS["models"]
METRICS_DIR = DIRS["metrics"]

VAL_SIZE = 0.15
MAX_ITER = 350
LEARNING_RATE = 0.05
MAX_LEAF_NODES = 31
L2_REGULARIZATION = 0.01
WEIGHT_MIN = 0.5
WEIGHT_MAX = 20.0


def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_test shape  : {y_test.shape}")
    return X_train, X_test, y_train, y_test, class_names


def make_sample_weights(y, class_names):
    classes = np.unique(y)
    raw_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {}

    print("\nClass weights before/after capping:")
    for cls, raw_weight in zip(classes, raw_weights):
        capped = float(np.clip(raw_weight, WEIGHT_MIN, WEIGHT_MAX))
        class_weight[int(cls)] = capped
        print(f"  {cls} ({class_names[int(cls)]:8s}): {raw_weight:8.2f} -> {capped:.2f}")

    return np.array([class_weight[int(label)] for label in y], dtype="float32")


def main():
    set_global_seed(RANDOM_SEED)

    print("=" * 60)
    print("Training HistGradientBoosting - NSL-KDD tabular IDS")
    print("=" * 60)

    X_train_full, X_test, y_train_full, y_test, class_names = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    sample_weight = make_sample_weights(y_train, class_names)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF_NODES,
        l2_regularization=L2_REGULARIZATION,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=RANDOM_SEED,
        verbose=1,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    print("\nValidation summary")
    print(f"  Accuracy : {val_accuracy:.4f}")
    print(f"  Macro F1 : {val_f1_macro:.4f}")
    print(f"  Iterations used: {model.n_iter_}")

    model_path = os.path.join(MODELS_DIR, "hist_gradient_boosting_model.pkl")
    dump(model, model_path)
    np.save(os.path.join(METRICS_DIR, "class_names.npy"), class_names)

    print(f"\nSaved HistGradientBoosting model -> {model_path}")
    print(f"Training completed. Outputs are stored in {DIRS['root']}")


if __name__ == "__main__":
    main()
