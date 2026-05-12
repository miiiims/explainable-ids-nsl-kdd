"""
src/explainability/shap_explainer.py
====================================
SHAP explanation script for the LSTM + Attention IDS model.

This script uses Kernel SHAP on a small subset because the model input is a
3D sequence tensor. The explainer receives flattened windows, then the wrapper
reshapes them back to (samples, timesteps, features) before prediction.

Outputs are intentionally lightweight so the experiment remains compatible
with the assignment's compute constraints.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The 'shap' package is required for this script. Install it with:\n"
        "  .\\venv_dml\\Scripts\\python.exe -m pip install shap\n"
        "Then rerun:\n"
        "  .\\venv_dml\\Scripts\\python.exe src\\explainability\\shap_explainer.py"
    ) from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, lstm_attention_dirs
from src.models.lstm_attention import TemporalAttention


DIRS = lstm_attention_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
LEGACY_MODEL_PATH = "results/models/best_lstm_attention_model.keras"
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

RANDOM_SEED = 42
BACKGROUND_SIZE = 40
EXPLAIN_SIZE = 20
NSAMPLES = 120
BATCH_SIZE = 512

if not os.path.exists(MODEL_PATH) and os.path.exists(LEGACY_MODEL_PATH):
    MODEL_PATH = LEGACY_MODEL_PATH


def load_data():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_seq.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    feature_names = np.load(
        os.path.join(PROCESSED_DIR, "feature_names.npy"), allow_pickle=True
    )
    class_names = np.load(
        os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True
    )
    return X_train, X_test, y_test, feature_names, class_names


def flatten_sequences(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


def make_flat_feature_names(feature_names, timesteps):
    names = []
    for step in range(timesteps):
        for feature in feature_names:
            names.append(f"t{step}_{feature}")
    return np.array(names)


def normalize_shap_values(shap_values):
    """Return SHAP values in shape (classes, samples, flat_features)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values)

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] <= 10:
        return np.moveaxis(values, -1, 0)
    if values.ndim == 3:
        return values

    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def plot_global_feature_importance(global_importance):
    top = global_importance.head(20).iloc[::-1]

    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    plt.title("Top SHAP feature importance - LSTM+Attention")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "shap_global_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_class_feature_importance(class_importance, class_names):
    fig, axes = plt.subplots(len(class_names), 1, figsize=(9, 3 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]

    for ax, class_name in zip(axes, class_names):
        df = class_importance[class_importance["class"] == str(class_name)].head(10)
        df = df.iloc[::-1]
        ax.barh(df["feature"], df["mean_abs_shap"], color="#5a8f72")
        ax.set_title(f"Top SHAP features - {class_name}")
        ax.set_xlabel("Mean absolute SHAP value")

    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "shap_feature_importance_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 60)
    print("SHAP explanations - LSTM+Attention")
    print("=" * 60)

    X_train, X_test, y_test, feature_names, class_names = load_data()
    timesteps = X_test.shape[1]
    num_features = X_test.shape[2]

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"Classes       : {list(class_names)}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttention": TemporalAttention},
    )

    background_idx = rng.choice(len(X_train), size=BACKGROUND_SIZE, replace=False)
    explain_idx = rng.choice(len(X_test), size=EXPLAIN_SIZE, replace=False)

    X_background = flatten_sequences(X_train[background_idx])
    X_explain = flatten_sequences(X_test[explain_idx])
    flat_feature_names = make_flat_feature_names(feature_names, timesteps)

    def predict_flat(X_flat):
        X_seq = X_flat.reshape((-1, timesteps, num_features)).astype("float32")
        return model.predict(X_seq, batch_size=BATCH_SIZE, verbose=0)

    print("\nRunning Kernel SHAP...")
    print(
        f"Background={BACKGROUND_SIZE}, explained samples={EXPLAIN_SIZE}, "
        f"nsamples={NSAMPLES}"
    )

    explainer = shap.KernelExplainer(predict_flat, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES)
    shap_values = normalize_shap_values(shap_values)

    # Aggregate temporal attributions back to original NSL-KDD features.
    reshaped = shap_values.reshape(
        (len(class_names), EXPLAIN_SIZE, timesteps, num_features)
    )
    per_feature_by_class = np.mean(np.abs(reshaped), axis=(1, 2))
    global_per_feature = np.mean(per_feature_by_class, axis=0)

    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": global_per_feature,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    class_rows = []
    for class_idx, class_name in enumerate(class_names):
        order = np.argsort(per_feature_by_class[class_idx])[::-1]
        for feature_idx in order:
            class_rows.append(
                {
                    "class": str(class_name),
                    "feature": str(feature_names[feature_idx]),
                    "mean_abs_shap": float(per_feature_by_class[class_idx, feature_idx]),
                }
            )
    class_importance = pd.DataFrame(class_rows)

    global_csv = os.path.join(EXPLAIN_DIR, "shap_global_feature_importance.csv")
    class_csv = os.path.join(EXPLAIN_DIR, "shap_feature_importance_by_class.csv")
    values_path = os.path.join(EXPLAIN_DIR, "shap_values_lstm_attention.npy")
    summary_path = os.path.join(EXPLAIN_DIR, "shap_summary.json")

    global_importance.to_csv(global_csv, index=False)
    class_importance.to_csv(class_csv, index=False)
    np.save(values_path, shap_values)

    global_plot = plot_global_feature_importance(global_importance)
    class_plot = plot_class_feature_importance(class_importance, class_names)

    summary = {
        "model": "LSTM+Attention",
        "method": "Kernel SHAP on flattened sequence windows",
        "background_size": BACKGROUND_SIZE,
        "explained_samples": EXPLAIN_SIZE,
        "nsamples": NSAMPLES,
        "sequence_length": int(timesteps),
        "top_global_features": global_importance.head(10).to_dict(orient="records"),
        "explained_indices": explain_idx.astype(int).tolist(),
        "explained_true_classes": [str(class_names[int(y_test[i])]) for i in explain_idx],
        "files": {
            "global_importance_csv": global_csv,
            "class_importance_csv": class_csv,
            "shap_values": values_path,
            "global_plot": global_plot,
            "class_plot": class_plot,
        },
        "interpretation_note": (
            "Kernel SHAP estimates feature contributions by perturbing flattened "
            "sequence windows. Feature scores are aggregated across timesteps to "
            "recover NSL-KDD feature-level explanations."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nTop global SHAP features:")
    print(global_importance.head(10).to_string(index=False))
    print("\nFiles saved:")
    print(f"  {global_csv}")
    print(f"  {class_csv}")
    print(f"  {values_path}")
    print(f"  {summary_path}")
    print(f"  {global_plot}")
    print(f"  {class_plot}")
    print("\nSHAP explanation completed.")


if __name__ == "__main__":
    main()
