"""
Entry point for LSTM + Attention SHAP explanations.
"""

import os
import runpy


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "shap_explainer.py")
    runpy.run_path(path, run_name="__main__")
