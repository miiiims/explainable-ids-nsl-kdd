"""
Lightweight adversarial evasion analysis for differentiable IDS models.

This script uses untargeted FGSM and PGD perturbations on normalized inputs. The
goal is not to build a full attacker, but to quantify whether small
gradient-based input changes can reduce IDS detection performance. A simple
confidence-threshold abstention analysis is also reported as a lightweight
security mitigation, not as a fully trained defense.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PROCESSED_DIR, RANDOM_SEED, lstm_attention_dirs, lstm_dirs, mlp_dirs
from src.models.lstm_attention import TemporalAttention


EPSILONS = [0.01, 0.03, 0.05]
BATCH_SIZE = 512
OUTPUT_DIR = os.path.join("results", "adversarial_evasion")
PGD_STEPS = 10
ABSTENTION_THRESHOLD = 0.80


def _loss_gradient(model, X_batch, y_batch):
    with tf.GradientTape() as tape:
        tape.watch(X_batch)
        predictions = model(X_batch, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        loss = tf.reduce_mean(loss)

    return tape.gradient(loss, X_batch)


def fgsm_attack(model, X, y, epsilon):
    """One-step untargeted FGSM under an L-infinity constraint."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)

    adv_batches = []
    for start in range(0, len(X), BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X_tensor[start:end]
        y_batch = y_tensor[start:end]

        gradient = _loss_gradient(model, X_batch, y_batch)
        X_adv = X_batch + epsilon * tf.sign(gradient)
        X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)
        adv_batches.append(X_adv.numpy())

    return np.concatenate(adv_batches, axis=0)


def pgd_attack(model, X, y, epsilon, steps=PGD_STEPS):
    """Iterative untargeted PGD constrained to an L-infinity epsilon ball."""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    alpha = epsilon / max(steps // 2, 1)

    adv_batches = []
    for start in range(0, len(X), BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X_tensor[start:end]
        y_batch = y_tensor[start:end]
        X_origin = tf.clip_by_value(X_batch, 0.0, 1.0)
        X_adv = tf.identity(X_origin)

        for _ in range(steps):
            gradient = _loss_gradient(model, X_adv, y_batch)
            X_adv = X_adv + alpha * tf.sign(gradient)
            X_adv = tf.minimum(tf.maximum(X_adv, X_origin - epsilon), X_origin + epsilon)
            X_adv = tf.clip_by_value(X_adv, 0.0, 1.0)

        adv_batches.append(X_adv.numpy())

    return np.concatenate(adv_batches, axis=0)


def confidence_abstention_metrics(y_true, proba, threshold, normal_index):
    confidence = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    accepted = confidence >= threshold
    attack_mask = y_true != normal_index
    attack_to_normal = attack_mask & accepted & (pred == normal_index)

    accepted_accuracy = None
    if np.any(accepted):
        accepted_accuracy = accuracy_score(y_true[accepted], pred[accepted])

    return {
        "abstention_threshold": threshold,
        "coverage": float(np.mean(accepted)),
        "abstention_rate": float(1.0 - np.mean(accepted)),
        "accepted_accuracy": accepted_accuracy,
        "attack_to_normal_rate": (
            float(np.mean(attack_to_normal[attack_mask])) if np.any(attack_mask) else None
        ),
    }


def evaluate_model(name, model, X_test, y_test, normal_index):
    clean_proba = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    clean_pred = np.argmax(clean_proba, axis=1)
    clean_correct = clean_pred == y_test

    rows = []
    defense_rows = [
        {
            "model": name,
            "attack": "clean",
            "epsilon": 0.0,
            **confidence_abstention_metrics(
                y_test, clean_proba, ABSTENTION_THRESHOLD, normal_index
            ),
        }
    ]

    attacks = {
        "FGSM": fgsm_attack,
        "PGD": pgd_attack,
    }

    for attack_name, attack_fn in attacks.items():
        for epsilon in EPSILONS:
            print(f"{name}: {attack_name} epsilon={epsilon}")
            X_adv = attack_fn(model, X_test, y_test, epsilon)
            adv_proba = model.predict(X_adv, batch_size=BATCH_SIZE, verbose=0)
            adv_pred = np.argmax(adv_proba, axis=1)

            attack_success_rate = float(np.mean(clean_correct & (adv_pred != y_test)))
            rows.append(
                {
                    "model": name,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    "clean_accuracy": accuracy_score(y_test, clean_pred),
                    "adversarial_accuracy": accuracy_score(y_test, adv_pred),
                    "clean_f1_weighted": f1_score(
                        y_test, clean_pred, average="weighted", zero_division=0
                    ),
                    "adversarial_f1_weighted": f1_score(
                        y_test, adv_pred, average="weighted", zero_division=0
                    ),
                    "prediction_stability": float(np.mean(clean_pred == adv_pred)),
                    "attack_success_rate_on_clean_correct": attack_success_rate,
                    "mean_linf_perturbation": epsilon,
                }
            )
            defense_rows.append(
                {
                    "model": name,
                    "attack": attack_name,
                    "epsilon": epsilon,
                    **confidence_abstention_metrics(
                        y_test, adv_proba, ABSTENTION_THRESHOLD, normal_index
                    ),
                }
            )

    return rows, defense_rows


def plot_results(df):
    plt.figure(figsize=(9, 5))
    for (model_name, attack_name), group in df.groupby(["model", "attack"]):
        plt.plot(
            group["epsilon"],
            group["adversarial_accuracy"],
            marker="o",
            label=f"{model_name} - {attack_name}",
        )
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial accuracy")
    plt.title("Adversarial evasion sensitivity")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "adversarial_accuracy_fgsm_pgd.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print("=" * 60)
    print("Adversarial evasion analysis - FGSM and PGD")
    print("=" * 60)

    class_names = np.load(os.path.join(PROCESSED_DIR, "class_names.npy"), allow_pickle=True)
    normal_index = int(np.where(class_names == "normal")[0][0])

    experiments = []

    mlp = tf.keras.models.load_model(os.path.join(mlp_dirs()["models"], "best_model.keras"))
    X_mlp = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy")).astype("float32")
    y_mlp = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    experiments.append(("MLP", mlp, X_mlp, y_mlp))

    lstm = tf.keras.models.load_model(os.path.join(lstm_dirs()["models"], "best_model.keras"))
    X_lstm = np.load(os.path.join(PROCESSED_DIR, "X_test_seq.npy")).astype("float32")
    y_lstm = np.load(os.path.join(PROCESSED_DIR, "y_test_seq.npy"))
    experiments.append(("LSTM", lstm, X_lstm, y_lstm))

    lstm_attention = tf.keras.models.load_model(
        os.path.join(lstm_attention_dirs()["models"], "best_model.keras"),
        custom_objects={"TemporalAttention": TemporalAttention},
    )
    experiments.append(("LSTM + Attention", lstm_attention, X_lstm, y_lstm))

    rows = []
    defense_rows = []
    for name, model, X_test, y_test in experiments:
        model_rows, model_defense_rows = evaluate_model(
            name, model, X_test, y_test, normal_index
        )
        rows.extend(model_rows)
        defense_rows.extend(model_defense_rows)

    df = pd.DataFrame(rows)
    defense_df = pd.DataFrame(defense_rows)
    csv_path = os.path.join(OUTPUT_DIR, "adversarial_evasion_metrics.csv")
    legacy_csv_path = os.path.join(OUTPUT_DIR, "fgsm_evasion_metrics.csv")
    defense_csv_path = os.path.join(OUTPUT_DIR, "confidence_abstention_metrics.csv")
    json_path = os.path.join(OUTPUT_DIR, "adversarial_evasion_metrics.json")
    legacy_json_path = os.path.join(OUTPUT_DIR, "fgsm_evasion_metrics.json")
    plot_path = plot_results(df)

    df.to_csv(csv_path, index=False)
    df.to_csv(legacy_csv_path, index=False)
    defense_df.to_csv(defense_csv_path, index=False)
    summary = {
        "method": "Untargeted FGSM and PGD on normalized inputs",
        "epsilons": EPSILONS,
        "pgd_steps": PGD_STEPS,
        "defense_probe": (
            f"Confidence-threshold abstention at threshold={ABSTENTION_THRESHOLD}. "
            "This is a lightweight mitigation analysis, not a trained defense."
        ),
        "models": sorted(df["model"].unique().tolist()),
        "results": df.round(6).to_dict(orient="records"),
        "confidence_abstention": defense_df.round(6).to_dict(orient="records"),
        "files": {
            "csv": csv_path,
            "legacy_csv": legacy_csv_path,
            "defense_csv": defense_csv_path,
            "plot": plot_path,
        },
        "interpretation_note": (
            "FGSM and PGD are used as lightweight white-box evasion stress tests "
            "for differentiable IDS models. They complement Gaussian stability "
            "analysis but do not replace a full adversarial security evaluation "
            "with domain-constrained network features."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    with open(legacy_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nAdversarial evasion results:")
    print(df.round(4).to_string(index=False))
    print("\nConfidence-threshold abstention probe:")
    print(defense_df.round(4).to_string(index=False))
    print(f"\nSaved outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
