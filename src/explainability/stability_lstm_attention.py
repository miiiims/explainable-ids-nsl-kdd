"""
Entry point for LSTM + Attention stability analysis.
"""

import os
import runpy


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "stability_analysis.py")
    runpy.run_path(path, run_name="__main__")
