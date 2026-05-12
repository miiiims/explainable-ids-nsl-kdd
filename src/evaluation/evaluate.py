"""
src/evaluation/evaluate.py
==========================
Évaluation complète du modèle LSTM + Attention sur NSL-KDD.

Métriques calculées :
  - Accuracy, Precision, Recall, F1 (macro + weighted)
  - Cohen's Kappa, Matthews Correlation Coefficient
  - ROC-AUC multiclasse (OvR)

Visualisations générées :
  - Confusion matrix (absolue + normalisée)
  - Courbes ROC (une par classe + moyenne)
  - Courbes Precision-Recall (une par classe)

Compatible : TensorFlow 2.10 + DirectML, chemins cohérents avec le projet.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

# ── Import du modèle custom ──────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, lstm_attention_dirs
from src.models.lstm_attention import TemporalAttention


# ============================================================
# Configuration des chemins
# ============================================================

DIRS = lstm_attention_dirs()
MODEL_PATH = os.path.join(DIRS["models"], "best_model.keras")
LEGACY_MODEL_PATH = "results/models/best_lstm_attention_model.keras"
METRICS_DIR = DIRS["metrics"]
REPORTS_DIR = DIRS["reports"]
VISUALIZATION_DIR = DIRS["visualizations"]

if not os.path.exists(MODEL_PATH) and os.path.exists(LEGACY_MODEL_PATH):
    MODEL_PATH = LEGACY_MODEL_PATH

# Palette de couleurs pour les classes
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


# ============================================================
# 1. Chargement des données
# ============================================================

print("=" * 60)
print("Évaluation — LSTM + Attention (NSL-KDD)")
print("=" * 60)

print("\n[1/6] Chargement des données...")

X_test      = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy"))
y_test      = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)
num_classes = len(class_names)

print(f"  X_test shape   : {X_test.shape}")
print(f"  y_test shape   : {y_test.shape}")
print(f"  Classes ({num_classes})   : {list(class_names)}")


# ============================================================
# 2. Chargement du modèle
# ============================================================

print("\n[2/6] Chargement du modèle...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TemporalAttention": TemporalAttention}
)

model.summary()


# ============================================================
# 3. Prédictions
# ============================================================

print("\n[3/6] Prédictions sur le test set...")

y_pred_proba = model.predict(X_test, batch_size=512, verbose=1)
y_pred       = np.argmax(y_pred_proba, axis=1)

print(f"  y_pred shape       : {y_pred.shape}")
print(f"  y_pred_proba shape : {y_pred_proba.shape}")


# ============================================================
# 4. Calcul des métriques
# ============================================================

print("\n[4/6] Calcul des métriques...")

accuracy = accuracy_score(y_test, y_pred)

precision_macro    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
recall_macro       = recall_score(y_test, y_pred,    average="macro",    zero_division=0)
f1_macro           = f1_score(y_test, y_pred,        average="macro",    zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1_weighted        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

kappa = cohen_kappa_score(y_test, y_pred)
mcc   = matthews_corrcoef(y_test, y_pred)

# ROC-AUC multiclasse One-vs-Rest
y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
try:
    roc_auc_ovr = roc_auc_score(
        y_test_bin, y_pred_proba,
        average="weighted", multi_class="ovr"
    )
except Exception as e:
    print(f"  ROC-AUC non calculable : {e}")
    roc_auc_ovr = None

try:
    pr_auc_weighted = average_precision_score(
        y_test_bin, y_pred_proba, average="weighted"
    )
except Exception as e:
    print(f"  PR-AUC non calculable : {e}")
    pr_auc_weighted = None

# Rapport de classification
report = classification_report(
    y_test, y_pred,
    target_names=class_names,
    zero_division=0
)

# Affichage
print(f"\n  Accuracy           : {accuracy:.4f}")
print(f"  Precision (macro)  : {precision_macro:.4f}")
print(f"  Recall (macro)     : {recall_macro:.4f}")
print(f"  F1 (macro)         : {f1_macro:.4f}")
print(f"  Precision (weighted): {precision_weighted:.4f}")
print(f"  Recall (weighted)  : {recall_weighted:.4f}")
print(f"  F1 (weighted)      : {f1_weighted:.4f}")
print(f"  Cohen's Kappa      : {kappa:.4f}")
print(f"  MCC                : {mcc:.4f}")
print(f"  ROC-AUC (OvR)      : {roc_auc_ovr:.4f}" if roc_auc_ovr else "  ROC-AUC : N/A")
print(f"  PR-AUC (weighted)  : {pr_auc_weighted:.4f}" if pr_auc_weighted else "  PR-AUC : N/A")

print("\nClassification Report :")
print(report)

pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
prediction_distribution = {
    str(class_names[int(cls)]): int(count)
    for cls, count in zip(pred_unique, pred_counts)
}

print("\nDistribution des prédictions :")
for cls_name, count in prediction_distribution.items():
    print(f"  {cls_name:10s} : {count:6d} ({100 * count / len(y_pred):5.2f}%)")

# Sauvegarde métriques JSON
metrics = {
    "accuracy":             round(accuracy, 4),
    "precision_macro":      round(precision_macro, 4),
    "recall_macro":         round(recall_macro, 4),
    "f1_macro":             round(f1_macro, 4),
    "precision_weighted":   round(precision_weighted, 4),
    "recall_weighted":      round(recall_weighted, 4),
    "f1_weighted":          round(f1_weighted, 4),
    "cohen_kappa":          round(kappa, 4),
    "matthews_corrcoef":    round(mcc, 4),
    "roc_auc_ovr_weighted": round(roc_auc_ovr, 4) if roc_auc_ovr else None,
    "pr_auc_weighted":      round(pr_auc_weighted, 4) if pr_auc_weighted else None,
    "prediction_distribution": prediction_distribution
}

with open(os.path.join(METRICS_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(METRICS_DIR, "lstm_attention_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

with open(os.path.join(REPORTS_DIR, "lstm_attention_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

print(f"\n  OK Metrics saved -> {METRICS_DIR}/evaluation_metrics.json")
print(f"  OK Report saved  -> {REPORTS_DIR}/classification_report.txt")


# ============================================================
# 5. Visualisations
# ============================================================

print("\n[5/6] Génération des visualisations...")

# ── 5a. Confusion Matrix absolue ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.title("Confusion Matrix — LSTM + Attention", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
path = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"  OK Confusion matrix -> {path}")

# ── 5b. Confusion Matrix normalisée ─────────────────────────────────────────
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    vmin=0, vmax=1
)
plt.title("Normalized Confusion Matrix — LSTM + Attention", fontsize=13, fontweight="bold")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
path = os.path.join(VISUALIZATION_DIR, "normalized_confusion_matrix.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"  OK Normalized confusion matrix -> {path}")

# ── 5c. Courbes ROC ──────────────────────────────────────────────────────────
# Une courbe par classe + courbe moyenne pondérée
plt.figure(figsize=(9, 7))

roc_auc_per_class = {}
for i, cls_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    auc_score    = auc(fpr, tpr)
    roc_auc_per_class[cls_name] = round(auc_score, 4)
    plt.plot(fpr, tpr, color=COLORS[i], lw=2,
             label=f"{cls_name} (AUC = {auc_score:.3f})")

# Ligne diagonale (aléatoire)
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves — LSTM + Attention (One-vs-Rest)", fontsize=13, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(VISUALIZATION_DIR, "roc_curves.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"  OK ROC curves -> {path}")

# Sauvegarde AUC par classe dans le JSON
metrics["roc_auc_per_class"] = roc_auc_per_class
with open(os.path.join(METRICS_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
with open(os.path.join(METRICS_DIR, "lstm_attention_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ── 5d. Courbes Precision-Recall ─────────────────────────────────────────────
plt.figure(figsize=(9, 7))

ap_per_class = {}
for i, cls_name in enumerate(class_names):
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test_bin[:, i], y_pred_proba[:, i]
    )
    ap = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
    ap_per_class[cls_name] = round(ap, 4)
    plt.plot(recall_curve, precision_curve, color=COLORS[i], lw=2,
             label=f"{cls_name} (AP = {ap:.3f})")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curves — LSTM + Attention", fontsize=13, fontweight="bold")
plt.legend(loc="upper right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(VISUALIZATION_DIR, "precision_recall_curves.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"  OK Precision-Recall curves -> {path}")

metrics["pr_auc_per_class"] = ap_per_class
with open(os.path.join(METRICS_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)
with open(os.path.join(METRICS_DIR, "lstm_attention_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)


# ============================================================
# 6. Sauvegarde des prédictions
# ============================================================

print("\n[6/6] Sauvegarde des prédictions...")

np.save(os.path.join(METRICS_DIR, "y_test.npy"),                    y_test)
np.save(os.path.join(METRICS_DIR, "y_pred_lstm_attention.npy"),     y_pred)
np.save(os.path.join(METRICS_DIR, "y_pred_proba_lstm_attention.npy"), y_pred_proba)

print(f"  OK y_test, y_pred, y_pred_proba -> {METRICS_DIR}/")

# ============================================================
# Résumé final
# ============================================================

print("\n" + "=" * 60)
print("Résumé de l'évaluation")
print("=" * 60)
print(f"  Accuracy    : {accuracy:.4f}")
print(f"  F1 (macro)  : {f1_macro:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")
print(f"  Kappa       : {kappa:.4f}")
print(f"  MCC         : {mcc:.4f}")
if roc_auc_ovr:
    print(f"  ROC-AUC     : {roc_auc_ovr:.4f}")
print("\nAUC par classe :")
for cls, score in roc_auc_per_class.items():
    print(f"  {cls:10s} : {score:.4f}")
print("\nEvaluation terminee.")
