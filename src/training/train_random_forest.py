"""
Train Experiment 1: Random Forest baseline on NSL-KDD tabular features.

Ce script entraîne un Random Forest sur les mêmes features normalisées que le MLP.
Il sert de baseline classique à comparer avec LSTM sans attention et LSTM + Attention.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, RANDOM_FOREST_NAME, random_forest_dirs

DIRS = random_forest_dirs()
MODELS_DIR = DIRS["models"]
METRICS_DIR = DIRS["metrics"]

N_ESTIMATORS = 200
MAX_DEPTH = None
VAL_SIZE = 0.15
NUM_JOBS = -1


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


def main():
    set_global_seed(RANDOM_SEED)

    print("=" * 60)
    print("Training Random Forest baseline - NSL-KDD")
    print("=" * 60)

    X_train_full, X_test, y_train_full, y_test, class_names = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced_subsample",
        n_jobs=NUM_JOBS,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    dump(model, model_path)
    print(f"Saved Random Forest model -> {model_path}")

    np.save(os.path.join(METRICS_DIR, "class_names.npy"), class_names)

    feature_importances = model.feature_importances_
    np.save(os.path.join(METRICS_DIR, "feature_importances.npy"), feature_importances)

    print(f"Saved feature importances -> {os.path.join(METRICS_DIR, 'feature_importances.npy')}")
    print(f"Training completed. Model and metrics are stored in results/{RANDOM_FOREST_NAME}/")


if __name__ == "__main__":
    main()
