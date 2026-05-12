"""
src/explainability/stability_analysis.py
========================================
Analyse de stabilite du modèle LSTM + Attention sous bruit gaussien.

Objectif académique:
  - mesurer la fiabilité des prédictions IDS sous petites perturbations;
  - mesurer la baisse de confiance;
  - mesurer la stabilité des poids d'attention comme explication.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, lstm_attention_dirs
from src.models.lstm_attention import TemporalAttention, build_lstm_model


DIRS = lstm_attention_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
LEGACY_MODEL_PATH = "results/models/best_lstm_attention_model.keras"
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

SIGMAS = [0.01, 0.05, 0.10]
RANDOM_SEED = 42
BATCH_SIZE = 512
ATTENTION_SAMPLE_SIZE = 1000

if not os.path.exists(MODEL_PATH) and os.path.exists(LEGACY_MODEL_PATH):
    MODEL_PATH = LEGACY_MODEL_PATH


def load_data():
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    class_names = np.load(
        os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True
    )
    return X_test, y_test, class_names


def load_models(input_shape, num_classes):
    prediction_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention},
    )

    attention_model = build_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=64,
        dense_units=64,
        dropout_rate_1=0.2,
        dropout_rate_2=0.1,
        l2_reg=1e-5,
        return_attention=True,
    )
    attention_model.set_weights(prediction_model.get_weights())
    return prediction_model, attention_model


def add_gaussian_noise(X, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape).astype("float32")
    X_noisy = X + noise
    return np.clip(X_noisy, 0.0, 1.0)


def mean_row_cosine_similarity(reference, perturbed):
    scores = []
    for i in range(reference.shape[0]):
        score = cosine_similarity(reference[i : i + 1], perturbed[i : i + 1])[0, 0]
        scores.append(score)
    return float(np.mean(scores))


def plot_stability(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(results_df["sigma"], results_df["prediction_stability"], marker="o")
    axes[0].set_title("Prediction stability")
    axes[0].set_xlabel("Gaussian noise sigma")
    axes[0].set_ylabel("Unchanged predictions")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)

    axes[1].plot(results_df["sigma"], results_df["confidence_drop"], marker="o", color="tab:orange")
    axes[1].set_title("Confidence drop")
    axes[1].set_xlabel("Gaussian noise sigma")
    axes[1].set_ylabel("Mean confidence drop")
    axes[1].grid(alpha=0.3)

    axes[2].plot(results_df["sigma"], results_df["attention_cosine_similarity"], marker="o", color="tab:green")
    axes[2].set_title("Attention stability")
    axes[2].set_xlabel("Gaussian noise sigma")
    axes[2].set_ylabel("Mean cosine similarity")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "stability_analysis_lstm_attention.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("Analyse de stabilite - LSTM+Attention")
    print("=" * 60)

    X_test, y_test, class_names = load_data()
    num_classes = len(class_names)
    prediction_model, attention_model = load_models(X_test.shape[1:], num_classes)

    print(f"X_test shape : {X_test.shape}")
    print(f"Classes      : {list(class_names)}")

    print("\nPredictions de reference...")
    clean_proba = prediction_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    clean_pred = np.argmax(clean_proba, axis=1)
    clean_confidence = np.max(clean_proba, axis=1)

    sample_size = min(ATTENTION_SAMPLE_SIZE, len(X_test))
    attention_indices = rng.choice(len(X_test), size=sample_size, replace=False)
    _, clean_attention = attention_model.predict(
        X_test[attention_indices], batch_size=BATCH_SIZE, verbose=0
    )

    rows = []
    for sigma in SIGMAS:
        print(f"\nPerturbation sigma={sigma}...")
        X_noisy = add_gaussian_noise(X_test, sigma, rng)
        noisy_proba = prediction_model.predict(X_noisy, batch_size=BATCH_SIZE, verbose=1)
        noisy_pred = np.argmax(noisy_proba, axis=1)
        noisy_confidence = np.max(noisy_proba, axis=1)

        _, noisy_attention = attention_model.predict(
            X_noisy[attention_indices], batch_size=BATCH_SIZE, verbose=0
        )

        prediction_stability = float(np.mean(clean_pred == noisy_pred))
        confidence_drop = float(np.mean(clean_confidence - noisy_confidence))
        attention_similarity = mean_row_cosine_similarity(clean_attention, noisy_attention)

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
                "confidence_drop": confidence_drop,
                "attention_cosine_similarity": attention_similarity,
            }
        )

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(EXPLAIN_DIR, "stability_lstm_attention.csv")
    json_path = os.path.join(EXPLAIN_DIR, "stability_lstm_attention.json")
    plot_path = plot_stability(results_df)

    results_df.to_csv(csv_path, index=False)
    summary = {
        "model": "LSTM+Attention",
        "sigmas": SIGMAS,
        "attention_sample_size": int(sample_size),
        "clean_accuracy": accuracy_score(y_test, clean_pred),
        "clean_f1_macro": f1_score(y_test, clean_pred, average="macro", zero_division=0),
        "clean_f1_weighted": f1_score(y_test, clean_pred, average="weighted", zero_division=0),
        "results": results_df.round(6).to_dict(orient="records"),
        "files": {
            "csv": csv_path,
            "plot": plot_path,
        },
        "interpretation_note": (
            "A robust IDS should keep predictions and explanations stable under "
            "small non-adversarial perturbations. Large drops indicate sensitivity "
            "that could become a security risk under adversarial manipulation."
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nResultats de stabilite :")
    print(results_df.round(4).to_string(index=False))
    print("\nFichiers sauvegardes :")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {plot_path}")
    print("\nAnalyse de stabilite terminee.")


if __name__ == "__main__":
    main()
