"""
SHAP explanations for Experiment 2: tabular MLP.
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
from src.config import PROCESSED_DIR, RANDOM_SEED, mlp_dirs


DIRS = mlp_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

BACKGROUND_SIZE = 60
EXPLAIN_SIZE = 30
NSAMPLES = 150
BATCH_SIZE = 512


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values)
    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] <= 10:
        return np.moveaxis(values, -1, 0)
    return values


def plot_global_importance(df):
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    plt.title("Top SHAP feature importance - MLP")
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


def main():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing MLP model: {MODEL_PATH}. Run src/training/train_mlp.py first."
        )

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    feature_names = np.load(os.path.join(PROCESSED_DIR, "feature_names.npy"), allow_pickle=True)
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    background_idx = rng.choice(len(X_train), size=BACKGROUND_SIZE, replace=False)
    explain_idx = rng.choice(len(X_test), size=EXPLAIN_SIZE, replace=False)
    X_background = X_train[background_idx]
    X_explain = X_test[explain_idx]

    def predict_fn(X):
        return model.predict(X.astype("float32"), batch_size=BATCH_SIZE, verbose=0)

    print("=" * 60)
    print("SHAP explanations - MLP")
    print("=" * 60)
    print(f"Background={BACKGROUND_SIZE}, explained samples={EXPLAIN_SIZE}, nsamples={NSAMPLES}")

    explainer = shap.KernelExplainer(predict_fn, X_background)
    shap_values = normalize_shap_values(
        explainer.shap_values(X_explain, nsamples=NSAMPLES)
    )

    per_feature_by_class = np.mean(np.abs(shap_values), axis=1)
    global_per_feature = np.mean(per_feature_by_class, axis=0)

    global_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": global_per_feature}
    ).sort_values("mean_abs_shap", ascending=False)

    rows = []
    for class_idx, class_name in enumerate(class_names):
        order = np.argsort(per_feature_by_class[class_idx])[::-1]
        for feature_idx in order:
            rows.append(
                {
                    "class": str(class_name),
                    "feature": str(feature_names[feature_idx]),
                    "mean_abs_shap": float(per_feature_by_class[class_idx, feature_idx]),
                }
            )
    class_importance = pd.DataFrame(rows)

    global_csv = os.path.join(EXPLAIN_DIR, "shap_global_feature_importance.csv")
    class_csv = os.path.join(EXPLAIN_DIR, "shap_feature_importance_by_class.csv")
    values_path = os.path.join(EXPLAIN_DIR, "shap_values_mlp.npy")
    summary_path = os.path.join(EXPLAIN_DIR, "shap_summary.json")

    global_importance.to_csv(global_csv, index=False)
    class_importance.to_csv(class_csv, index=False)
    np.save(values_path, shap_values)

    global_plot = plot_global_importance(global_importance)
    class_plot = plot_class_importance(class_importance, class_names)

    summary = {
        "model": "MLP",
        "method": "Kernel SHAP on tabular NSL-KDD features",
        "background_size": BACKGROUND_SIZE,
        "explained_samples": EXPLAIN_SIZE,
        "nsamples": NSAMPLES,
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
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nTop global SHAP features:")
    print(global_importance.head(10).to_string(index=False))
    print(f"\nSaved SHAP outputs in {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
