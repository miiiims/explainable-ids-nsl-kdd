"""
Stability analysis for the Random Forest baseline.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, random_forest_dirs


DIRS = random_forest_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "random_forest_model.pkl")
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

SIGMAS = [0.01, 0.05, 0.10]


def add_gaussian_noise(X, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape).astype("float32")
    return np.clip(X + noise, 0.0, 1.0)


def plot_stability(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["sigma"], df["prediction_stability"], marker="o")
    axes[0].set_title("Prediction stability")
    axes[0].set_xlabel("Gaussian noise sigma")
    axes[0].set_ylabel("Unchanged predictions")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["sigma"], df["confidence_drop"], marker="o", color="tab:orange")
    axes[1].set_title("Confidence drop")
    axes[1].set_xlabel("Gaussian noise sigma")
    axes[1].set_ylabel("Mean confidence drop")
    axes[1].grid(alpha=0.3)

    axes[2].plot(df["sigma"], df["f1_weighted_noisy"], marker="o", color="tab:green")
    axes[2].set_title("Weighted F1 under noise")
    axes[2].set_xlabel("Gaussian noise sigma")
    axes[2].set_ylabel("Weighted F1")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "stability_analysis_random_forest.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing Random Forest model: {MODEL_PATH}. Run src/training/train_random_forest.py first."
        )

    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)
    model = load(MODEL_PATH)

    print("=" * 60)
    print("Stability analysis - Random Forest")
    print("=" * 60)
    print(f"X_test shape: {X_test.shape}")
    print(f"Classes: {list(class_names)}")

    clean_proba = model.predict_proba(X_test)
    clean_pred = np.argmax(clean_proba, axis=1)
    clean_confidence = np.max(clean_proba, axis=1)

    rows = []
    for sigma in SIGMAS:
        print(f"\nPerturbation sigma={sigma}...")
        X_noisy = add_gaussian_noise(X_test, sigma, rng)
        noisy_proba = model.predict_proba(X_noisy)
        noisy_pred = np.argmax(noisy_proba, axis=1)
        noisy_confidence = np.max(noisy_proba, axis=1)

        prediction_stability = float(np.mean(clean_pred == noisy_pred))
        rows.append(
            {
                "sigma": sigma,
                "accuracy_noisy": accuracy_score(y_test, noisy_pred),
                "f1_macro_noisy": f1_score(y_test, noisy_pred, average="macro", zero_division=0),
                "f1_weighted_noisy": f1_score(y_test, noisy_pred, average="weighted", zero_division=0),
                "prediction_stability": prediction_stability,
                "prediction_change_rate": 1.0 - prediction_stability,
                "mean_clean_confidence": float(np.mean(clean_confidence)),
                "mean_noisy_confidence": float(np.mean(noisy_confidence)),
                "confidence_drop": float(np.mean(clean_confidence - noisy_confidence)),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(EXPLAIN_DIR, "stability_random_forest.csv")
    json_path = os.path.join(EXPLAIN_DIR, "stability_random_forest.json")
    plot_path = plot_stability(df)

    df.to_csv(csv_path, index=False)
    summary = {
        "model": "Random Forest",
        "sigmas": SIGMAS,
        "clean_accuracy": accuracy_score(y_test, clean_pred),
        "clean_f1_macro": f1_score(y_test, clean_pred, average="macro", zero_division=0),
        "clean_f1_weighted": f1_score(y_test, clean_pred, average="weighted", zero_division=0),
        "results": df.round(6).to_dict(orient="records"),
        "files": {"csv": csv_path, "plot": plot_path},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nStability results:")
    print(df.round(4).to_string(index=False))
    print(f"\nSaved stability outputs in {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
