"""
src/train.py
============
Script d'entraînement principal pour le modèle LSTM + Attention (IDS NSL-KDD).

Améliorations par rapport à la version précédente :
  - Validation stratifiée (train_test_split avec stratify) au lieu de validation_split=0.2
  - Class weights plafonnés (clip entre 0.3 et 15) pour stabiliser l'entraînement
  - Sauvegarde complète : best model, final model, history.pkl, history.csv
  - Affichage clair des distributions de classes
  - Compatible TF 2.10 + DirectML
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# Import du modèle local
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import LSTM_ATTENTION_NAME, PROCESSED_DIR, lstm_attention_dirs
from src.models.lstm_attention import build_lstm_model


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DIRS = lstm_attention_dirs()
MODELS_DIR    = DIRS["models"]
METRICS_DIR   = DIRS["metrics"]

BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 0.001
VAL_SIZE      = 0.15       # 15% du train pour la validation
RANDOM_SEED   = 42

# Plafond des class weights : évite que U2R (classe rare) explose l'entraînement.
# Sans pondération, le modèle prédit surtout la classe majoritaire "normal".
# Avec un plafond trop haut, l'entraînement devient instable. 5.0 est un
# compromis léger pour rééquilibrer R2L/U2R sans dominer la loss.
WEIGHT_MIN    = 0.3
WEIGHT_MAX    = 5.0


def set_global_seed(seed=42):
    """Fixe les seeds pour améliorer la reproductibilité des expériences."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ──────────────────────────────────────────────────────────────────────────────
# Chargement des données
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """Charge les fichiers .npy générés par feature_engineering.py."""
    print("=" * 60)
    print("Chargement des données")
    print("=" * 60)

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_seq.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train_seq.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)

    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_test  shape : {y_test.shape}")

    return X_train, y_train, X_test, y_test, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Affichage des distributions
# ──────────────────────────────────────────────────────────────────────────────

def print_distribution(y, class_names, split_name=""):
    """Affiche la distribution des classes dans un split."""
    print(f"\nDistribution des classes — {split_name} :")
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  {cls} ({name:8s}) : {cnt:6d}  ({100*cnt/total:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Calcul des class weights plafonnés
# ──────────────────────────────────────────────────────────────────────────────

def compute_capped_class_weights(y_train, class_names, w_min=0.3, w_max=15.0):
    """
    Calcule les class weights avec sklearn puis plafonne entre [w_min, w_max].

    Sans plafond, U2R peut atteindre ~484 ce qui rend l'entraînement instable.
    Avec plafond à 15, le modèle garde une sensibilité aux classes rares
    sans exploser les gradients.
    """
    classes = np.unique(y_train)
    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    print("\nClass weights (avant / après plafonnement) :")
    class_weight_dict = {}
    for cls, raw_w in zip(classes, raw_weights):
        capped_w = float(np.clip(raw_w, w_min, w_max))
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  {cls} ({name:8s}) : {raw_w:8.2f}  ->  {capped_w:.2f}")
        class_weight_dict[int(cls)] = capped_w

    return class_weight_dict


# ──────────────────────────────────────────────────────────────────────────────
# Script principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_global_seed(RANDOM_SEED)

    # 1. Chargement
    X_train_full, y_train_full, X_test, y_test, class_names = load_data()
    num_classes  = len(class_names)
    input_shape  = X_train_full.shape[1:]  # (10, 44)

    print(f"\nClasses ({num_classes}) : {list(enumerate(class_names))}")

    # 2. Split train / validation STRATIFIÉ
    # ─────────────────────────────────────────────────────────────────────────
    # On utilise stratify=y pour garantir que chaque classe est représentée
    # proportionnellement dans le set de validation.
    # C'est crucial avec U2R (52 exemples) pour ne pas se retrouver
    # avec 0 exemples U2R dans la validation.
    # ─────────────────────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full       # ← clé pour le déséquilibre
    )

    print(f"\nAprès split stratifié (val_size={VAL_SIZE}) :")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")

    print_distribution(y_train, class_names, "Train")
    print_distribution(y_val,   class_names, "Validation")
    print_distribution(y_test,  class_names, "Test")

    # 3. Class weights plafonnés
    class_weight_dict = compute_capped_class_weights(
        y_train, class_names, w_min=WEIGHT_MIN, w_max=WEIGHT_MAX
    )

    # 4. Construction du modèle
    print("\n" + "=" * 60)
    print("Construction du modèle")
    print("=" * 60)

    model = build_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=64,
        dense_units=64,
        dropout_rate_1=0.2,
        dropout_rate_2=0.1,
        l2_reg=1e-5,
        return_attention=False   # False pour l'entraînement (une seule sortie)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 5. Callbacks
    best_model_path  = os.path.join(MODELS_DIR, "best_model.keras")
    final_model_path = os.path.join(MODELS_DIR, "final_model.keras")

    callbacks = [
        # Sauvegarde du meilleur modèle selon val_loss
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Réduction du LR si val_loss stagne (patience=5)
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Arrêt si val_loss ne s'améliore plus (patience=10)
        # restore_best_weights=True pour récupérer le meilleur état
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # 6. Entraînement
    print("\n" + "=" * 60)
    print("Entraînement")
    print("=" * 60)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),   # validation explicite, pas de split aléatoire
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Sauvegarde
    print("\n" + "=" * 60)
    print("Sauvegarde")
    print("=" * 60)

    # Modèle final (dernier état, pas forcément le meilleur)
    model.save(final_model_path)
    print(f"  Final model  -> {final_model_path}")
    print(f"  Best model   -> {best_model_path}")

    # Historique en .pkl (compatible avec les notebooks)
    history_pkl = os.path.join(METRICS_DIR, "training_history.pkl")
    with open(history_pkl, "wb") as f:
        pickle.dump(history.history, f)
    print(f"  History .pkl -> {history_pkl}")

    # Historique en .csv (lisible directement)
    history_csv = os.path.join(METRICS_DIR, "training_history.csv")
    pd.DataFrame(history.history).to_csv(history_csv, index=False)
    print(f"  History .csv -> {history_csv}")

    # Sauvegarde des class_names (utile pour evaluate.py)
    class_names_path = os.path.join(METRICS_DIR, "class_names.npy")
    np.save(class_names_path, class_names)
    print(f"  class_names  -> {class_names_path}")

    # 8. Résumé final
    print("\n" + "=" * 60)
    print("Résumé de l'entraînement")
    print("=" * 60)
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = history.history["val_accuracy"][best_epoch - 1]
    print(f"  Meilleure époque : {best_epoch}")
    print(f"  Best val_loss    : {best_val_loss:.4f}")
    print(f"  Best val_acc     : {best_val_acc:.4f}")
    print(f"\nExpérience sauvegardée dans results/{LSTM_ATTENTION_NAME}/")
    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
