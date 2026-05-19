"""
SHAP explanations for the LSTM model without attention.

Ce script calcule des importances de features temporelles en utilisant Kernel SHAP.
Il agrége les contributions sur les timesteps pour produire une importance par feature.
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
    raise ImportError("Install shap before running this script.") from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, lstm_dirs

DIRS = lstm_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

BACKGROUND_SIZE = 20
EXPLAIN_SIZE = 10
NSAMPLES = 80
BATCH_SIZE = 128


def plot_global_importance(df):
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    plt.title("Top SHAP feature importance - LSTM sans attention")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "shap_global_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_class_importance(df, class_names):
    fig, axes = plt.subplots(len(class_names), 1, figsize=(9, 3 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]
    for ax, class_name in zip(axes, class_names):
        rows = df[df["class"] == str(class_name)].head(10).iloc[::-1]
        ax.barh(rows["feature"], rows["mean_abs_shap"], color="#5a8f72")
        ax.set_title(f"Top SHAP features - {class_name}")
        ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    path = os.path.join(VISUALIZATION_DIR, "shap_feature_importance_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values)
    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] <= 10:
        return np.moveaxis(values, -1, 0)
    if values.ndim == 3:
        return values
    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def flatten_sequences(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing LSTM model: {MODEL_PATH}. Run src/training/train_lstm.py first.")

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_seq.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    feature_names = np.load(os.path.join(PROCESSED_DIR, "feature_names.npy"), allow_pickle=True)
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    background_idx = rng.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)
    explain_idx = rng.choice(len(X_test), size=min(EXPLAIN_SIZE, len(X_test)), replace=False)
    X_background = X_train[background_idx]
    X_explain = X_test[explain_idx]
    timesteps = X_test.shape[1]
    num_features = X_test.shape[2]

    X_background_flat = flatten_sequences(X_background)
    X_explain_flat = flatten_sequences(X_explain)

    def predict_flat(X_flat):
        X_seq = X_flat.reshape((-1, timesteps, num_features)).astype("float32")
        return model.predict(X_seq, batch_size=BATCH_SIZE, verbose=0)

    print("=" * 60)
    print("SHAP explanations - LSTM sans attention")
    print("=" * 60)
    print(f"Background size: {len(X_background)} - Explain size: {len(X_explain)} - nsamples: {NSAMPLES}")

    explainer = shap.KernelExplainer(predict_flat, X_background_flat)
    shap_values = normalize_shap_values(
        explainer.shap_values(X_explain_flat, nsamples=NSAMPLES)
    )

    reshaped = shap_values.reshape(
        (len(class_names), len(X_explain), timesteps, num_features)
    )
    class_feature_importance = np.mean(np.abs(reshaped), axis=(1, 2))

    global_importance = np.mean(class_feature_importance, axis=0)
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": global_importance,
    }).sort_values("mean_abs_shap", ascending=False)

    rows = []
    for class_idx, class_name in enumerate(class_names):
        per_feature = class_feature_importance[class_idx]
        order = np.argsort(per_feature)[::-1]
        for feature_idx in order:
            rows.append({
                "class": str(class_name),
                "feature": str(feature_names[feature_idx]),
                "mean_abs_shap": float(per_feature[feature_idx]),
            })
    class_importance_df = pd.DataFrame(rows)

    global_csv = os.path.join(EXPLAIN_DIR, "shap_global_feature_importance.csv")
    class_csv = os.path.join(EXPLAIN_DIR, "shap_feature_importance_by_class.csv")
    values_path = os.path.join(EXPLAIN_DIR, "shap_values_lstm.npy")
    summary_path = os.path.join(EXPLAIN_DIR, "shap_summary.json")

    feature_importance_df.to_csv(global_csv, index=False)
    class_importance_df.to_csv(class_csv, index=False)
    np.save(values_path, shap_values)

    global_plot = plot_global_importance(feature_importance_df)
    class_plot = plot_class_importance(class_importance_df, class_names)

    summary = {
        "model": "LSTM sans attention",
        "background_size": int(len(X_background)),
        "explained_samples": int(len(X_explain)),
        "nsamples": NSAMPLES,
        "method": "Kernel SHAP on flattened sequence windows",
        "sequence_length": int(timesteps),
        "top_global_features": feature_importance_df.head(10).to_dict(orient="records"),
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
            "Kernel SHAP receives flattened LSTM windows because it expects 1D/2D "
            "instances. Scores are reshaped back to sequence form and averaged "
            "across timesteps to recover feature-level explanations."
        ),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Top global SHAP features:")
    print(feature_importance_df.head(10).to_string(index=False))
    print(f"Saved SHAP outputs in {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
