"""
Evaluate the LSTM model without attention.

This script loads the saved LSTM model from results/lstm/ and calcule les métriques
de classification et les visualisations pour comparaison avec LSTM+Attention.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, lstm_dirs

DIRS = lstm_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
METRICS_DIR = DIRS["metrics"]
REPORTS_DIR = DIRS["reports"]
VISUALIZATION_DIR = DIRS["visualizations"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load_data():
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)
    return X_test, y_test, class_names


def main():
    print("=" * 60)
    print("Evaluation - LSTM sans attention")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing LSTM model: {MODEL_PATH}. Run src/training/train_lstm.py first.")

    X_test, y_test, class_names = load_data()
    num_classes = len(class_names)

    print(f"X_test shape : {X_test.shape}")
    print(f"Classes      : {list(class_names)}")

    model = tf.keras.models.load_model(MODEL_PATH)
    y_pred_proba = model.predict(X_test, batch_size=512, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
    roc_auc_ovr = roc_auc_score(y_test_bin, y_pred_proba, average="weighted", multi_class="ovr")
    pr_auc_weighted = average_precision_score(y_test_bin, y_pred_proba, average="weighted")

    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}")
    print(f"ROC-AUC: {roc_auc_ovr:.4f}")

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
        "roc_auc_ovr_weighted": round(roc_auc_ovr, 4),
        "pr_auc_weighted": round(pr_auc_weighted, 4),
    }

    with open(os.path.join(METRICS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(METRICS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - LSTM sans attention")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix - LSTM sans attention")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "normalized_confusion_matrix.png"), dpi=150)
    plt.close()

    roc_auc_per_class = {}
    plt.figure(figsize=(9, 7))
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_score = auc(fpr, tpr)
        roc_auc_per_class[str(cls_name)] = round(auc_score, 4)
        plt.plot(fpr, tpr, color=COLORS[i], lw=2, label=f"{cls_name} (AUC={auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - LSTM sans attention")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "roc_curves.png"), dpi=150)
    plt.close()

    pr_auc_per_class = {}
    plt.figure(figsize=(9, 7))
    for i, cls_name in enumerate(class_names):
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        pr_auc_per_class[str(cls_name)] = round(ap, 4)
        plt.plot(recall_curve, precision_curve, color=COLORS[i], lw=2, label=f"{cls_name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - LSTM sans attention")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "precision_recall_curves.png"), dpi=150)
    plt.close()

    metrics["roc_auc_per_class"] = roc_auc_per_class
    metrics["pr_auc_per_class"] = pr_auc_per_class
    with open(os.path.join(METRICS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(METRICS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(os.path.join(METRICS_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(METRICS_DIR, "y_pred_lstm.npy"), y_pred)
    np.save(os.path.join(METRICS_DIR, "y_pred_proba_lstm.npy"), y_pred_proba)

    print(f"Saved LSTM results in {DIRS['root']}")


if __name__ == "__main__":
    main()
