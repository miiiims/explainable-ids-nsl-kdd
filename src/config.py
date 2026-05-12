"""
Shared project paths and experiment identifiers.
"""

import os


RANDOM_SEED = 42
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"

LSTM_ATTENTION_NAME = "lstm_attention"
MLP_NAME = "mlp"
RANDOM_FOREST_NAME = "random_forest"


def experiment_dir(experiment_name):
    return os.path.join(RESULTS_DIR, experiment_name)


def experiment_subdir(experiment_name, subdir):
    path = os.path.join(experiment_dir(experiment_name), subdir)
    os.makedirs(path, exist_ok=True)
    return path


def lstm_attention_dirs():
    return {
        "root": experiment_dir(LSTM_ATTENTION_NAME),
        "models": experiment_subdir(LSTM_ATTENTION_NAME, "models"),
        "metrics": experiment_subdir(LSTM_ATTENTION_NAME, "metrics"),
        "reports": experiment_subdir(LSTM_ATTENTION_NAME, "reports"),
        "visualizations": experiment_subdir(LSTM_ATTENTION_NAME, "visualizations"),
        "explainability": experiment_subdir(LSTM_ATTENTION_NAME, "explainability"),
        "explainability_visualizations": experiment_subdir(
            LSTM_ATTENTION_NAME, os.path.join("visualizations", "explainability")
        ),
    }


def mlp_dirs():
    return {
        "root": experiment_dir(MLP_NAME),
        "models": experiment_subdir(MLP_NAME, "models"),
        "metrics": experiment_subdir(MLP_NAME, "metrics"),
        "reports": experiment_subdir(MLP_NAME, "reports"),
        "visualizations": experiment_subdir(MLP_NAME, "visualizations"),
        "explainability": experiment_subdir(MLP_NAME, "explainability"),
        "explainability_visualizations": experiment_subdir(
            MLP_NAME, os.path.join("visualizations", "explainability")
        ),
    }
