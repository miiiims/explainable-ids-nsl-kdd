"""
Entry point for Experiment 3 evaluation: LSTM + Attention.

This wrapper keeps model-specific commands explicit:
  python src/evaluation/evaluate_lstm_attention.py
"""

import os
import runpy


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "evaluate.py")
    runpy.run_path(path, run_name="__main__")
