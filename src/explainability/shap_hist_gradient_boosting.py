"""
SHAP explanations for the HistGradientBoosting tabular baseline.

TreeExplainer support for sklearn histogram boosting varies across SHAP/sklearn
versions, so this script falls back to Kernel SHAP when needed.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

try:
    import shap
except ImportError as exc:
    raise ImportError("Install shap before running this script.") from exc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, hist_gradient_boosting_dirs


DIRS = hist_gradient_boosting_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "hist_gradient_boosting_model.pkl")
EXPLAIN_DIR = DIRS["explainability"]
VISUALIZATION_DIR = DIRS["explainability_visualizations"]

BACKGROUND_SIZE = 60
EXPLAIN_SIZE = 30
NSAMPLES = 150


def normalize_shap_values(shap_values, num_classes):
    """Return SHAP values in shape (classes, samples, features)."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values)

    values = np.asarray(shap_values)
    if values.ndim == 3 and values.shape[-1] == num_classes:
        return np.moveaxis(values, -1, 0)
    if values.ndim == 3 and values.shape[0] == num_classes:
        return values
    if values.ndim == 2:
        return np.expand_dims(values, axis=0)

    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def plot_global_importance(df):
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2f6f9f")
    plt.title("Top SHAP feature importance - HistGradientBoosting")
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


def explain_with_shap(model, X_background, X_explain, num_classes):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        return normalize_shap_values(shap_values, num_classes), "TreeExplainer"
    except Exception as exc:
        print(f"TreeExplainer unavailable ({exc}). Falling back to KernelExplainer.")

    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=NSAMPLES)
    return normalize_shap_values(shap_values, num_classes), "KernelExplainer"


def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing HistGradientBoosting model: {MODEL_PATH}. "
            "Run src/training/train_hist_gradient_boosting.py first."
        )

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy")).astype("float32")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    feature_names = np.load(os.path.join(PROCESSED_DIR, "feature_names.npy"), allow_pickle=True)
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    model = load(MODEL_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    background_idx = rng.choice(len(X_train), size=min(BACKGROUND_SIZE, len(X_train)), replace=False)
    explain_idx = rng.choice(len(X_test), size=min(EXPLAIN_SIZE, len(X_test)), replace=False)
    X_background = X_train[background_idx]
    X_explain = X_test[explain_idx]

    print("=" * 60)
    print("SHAP explanations - HistGradientBoosting")
    print("=" * 60)
    print(
        f"Background size: {len(X_background)} - Explain size: {len(X_explain)} - "
        f"Kernel nsamples: {NSAMPLES}"
    )

    shap_values_array, method = explain_with_shap(
        model, X_background, X_explain, len(class_names)
    )

    class_feature_importance = np.mean(np.abs(shap_values_array), axis=1)
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
    values_path = os.path.join(EXPLAIN_DIR, "shap_values_hist_gradient_boosting.npy")
    summary_path = os.path.join(EXPLAIN_DIR, "shap_summary.json")

    feature_importance_df.to_csv(global_csv, index=False)
    class_importance_df.to_csv(class_csv, index=False)
    np.save(values_path, shap_values_array)

    global_plot = plot_global_importance(feature_importance_df)
    class_plot = plot_class_importance(class_importance_df, class_names)

    summary = {
        "model": "HistGradientBoosting",
        "method": method,
        "background_size": int(len(X_background)),
        "explained_samples": int(len(X_explain)),
        "nsamples": NSAMPLES if method == "KernelExplainer" else None,
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
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Top global SHAP features:")
    print(feature_importance_df.head(10).to_string(index=False))
    print(f"Saved SHAP outputs in {EXPLAIN_DIR}")


if __name__ == "__main__":
    main()
