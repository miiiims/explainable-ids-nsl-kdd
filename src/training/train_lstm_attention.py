"""
Entry point for Experiment 3: LSTM + Attention.

This wrapper keeps the project architecture explicit:
  python src/training/train_lstm_attention.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.train import main


if __name__ == "__main__":
    main()
