"""
src/explainability/attention_visualizer.py
==========================================
Visualisation des poids d'attention du modèle LSTM + Attention.

Objectif académique:
  - rendre les décisions IDS plus interprétables;
  - montrer quels pas temporels d'une fenêtre NSL-KDD influencent la décision;
  - produire des figures et résumés utilisables dans le rapport.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, lstm_attention_dirs
from src.models.lstm_attention import TemporalAttention, build_lstm_model


DIRS = lstm_attention_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
LEGACY_MODEL_PATH = "results/models/best_lstm_attention_model.keras"
METRICS_DIR = DIRS["metrics"]
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

SAMPLE_PER_CLASS = 2
RANDOM_SEED = 42
BATCH_SIZE = 512

if not os.path.exists(MODEL_PATH) and os.path.exists(LEGACY_MODEL_PATH):
    MODEL_PATH = LEGACY_MODEL_PATH


def load_data():
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy"))
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


def select_examples(y_true, y_pred, class_names, sample_per_class=2):
    selected = []
    rng = np.random.default_rng(RANDOM_SEED)

    for class_idx, class_name in enumerate(class_names):
        correct = np.where((y_true == class_idx) & (y_pred == class_idx))[0]
        candidates = correct if len(correct) > 0 else np.where(y_true == class_idx)[0]
        if len(candidates) == 0:
            continue

        chosen = rng.choice(
            candidates,
            size=min(sample_per_class, len(candidates)),
            replace=False,
        )
        for idx in chosen:
            selected.append(
                {
                    "index": int(idx),
                    "true_class": str(class_name),
                    "predicted_class": str(class_names[int(y_pred[idx])]),
                    "correct": bool(y_true[idx] == y_pred[idx]),
                }
            )

    return selected


def plot_attention_heatmap(attention_weights, examples, class_names):
    labels = [
        f"idx {ex['index']} | T={ex['true_class']} | P={ex['predicted_class']}"
        for ex in examples
    ]

    plt.figure(figsize=(10, max(4, 0.45 * len(examples))))
    sns.heatmap(
        attention_weights,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=[f"t-{attention_weights.shape[1] - 1 - i}" for i in range(attention_weights.shape[1])],
        yticklabels=labels,
        vmin=0,
        vmax=max(0.01, float(attention_weights.max())),
    )
    plt.title("Attention temporelle - exemples de test LSTM+Attention")
    plt.xlabel("Position dans la fenêtre séquentielle")
    plt.ylabel("Exemples")
    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "attention_heatmap_examples.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_average_attention(attention_by_class, class_names):
    plt.figure(figsize=(9, 5))
    for class_name in class_names:
        values = attention_by_class.get(str(class_name))
        if values is not None:
            plt.plot(
                np.arange(len(values)),
                values,
                marker="o",
                linewidth=2,
                label=str(class_name),
            )

    plt.title("Poids d'attention moyens par classe vraie")
    plt.xlabel("Position dans la fenêtre séquentielle")
    plt.ylabel("Poids d'attention moyen")
    plt.xticks(np.arange(len(next(iter(attention_by_class.values())))))
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "attention_average_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("=" * 60)
    print("Visualisation de l'attention - LSTM+Attention")
    print("=" * 60)

    X_test, y_test, class_names = load_data()
    num_classes = len(class_names)

    print(f"X_test shape : {X_test.shape}")
    print(f"Classes      : {list(class_names)}")

    prediction_model, attention_model = load_models(X_test.shape[1:], num_classes)

    print("\nPredictions...")
    y_pred_proba = prediction_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    examples = select_examples(y_test, y_pred, class_names, SAMPLE_PER_CLASS)
    example_indices = np.array([ex["index"] for ex in examples], dtype=int)

    print(f"\nExemples selectionnes : {len(example_indices)}")
    selected_probs, selected_attention = attention_model.predict(
        X_test[example_indices], batch_size=BATCH_SIZE, verbose=0
    )

    heatmap_path = plot_attention_heatmap(selected_attention, examples, class_names)

    print("\nCalcul attention moyenne par classe...")
    _, all_attention = attention_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

    attention_by_class = {}
    for class_idx, class_name in enumerate(class_names):
        mask = y_test == class_idx
        if np.any(mask):
            attention_by_class[str(class_name)] = all_attention[mask].mean(axis=0).tolist()

    avg_path = plot_average_attention(attention_by_class, class_names)

    summary = {
        "model": "LSTM+Attention",
        "num_examples": int(len(examples)),
        "sequence_length": int(X_test.shape[1]),
        "examples": examples,
        "attention_by_true_class": attention_by_class,
        "files": {
            "example_heatmap": heatmap_path,
            "average_attention_plot": avg_path,
        },
        "interpretation_note": (
            "Attention weights indicate which positions inside the local sequence "
            "received higher model focus. They are model-internal explanations and "
            "should be interpreted together with stability analysis."
        ),
    }

    with open(os.path.join(EXPLAIN_DIR, "attention_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    np.save(os.path.join(EXPLAIN_DIR, "attention_weights_test.npy"), all_attention)

    print("\nFichiers sauvegardes :")
    print(f"  {heatmap_path}")
    print(f"  {avg_path}")
    print(f"  {os.path.join(EXPLAIN_DIR, 'attention_summary.json')}")
    print(f"  {os.path.join(EXPLAIN_DIR, 'attention_weights_test.npy')}")
    print("\nAnalyse d'attention terminee.")


if __name__ == "__main__":
    main()
