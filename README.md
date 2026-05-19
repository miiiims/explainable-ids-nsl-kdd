# IDS LSTM Project

## Overview
This project implements an LSTM-based Intrusion Detection System (IDS) using the NSL-KDD dataset. It compares deep learning (LSTM) with classical machine learning (Random Forest) approaches.

## Project Structure
```
IDS_LSTM_Project/
├── data/
│   ├── raw/                 # Original KDD dataset
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── processed/           # Normalized & encoded data
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── y_train.pkl
│       └── y_test.pkl
├── src/
│   ├── data_engineering.py  # Data preprocessing & normalization
│   ├── preprocessing/       # Preprocessing and sequence creation
│   │   └── feature_engineering.py
│   ├── training/            # Training entrypoints for experiments
│   │   ├── train_lstm_attention.py
│   │   ├── train_lstm.py
│   │   ├── train_mlp.py
│   │   ├── train_random_forest.py
│   │   └── train_hist_gradient_boosting.py
│   ├── evaluation/          # Evaluation entrypoints for experiments
│   │   ├── evaluate_lstm_attention.py
│   │   ├── evaluate_lstm.py
│   │   ├── evaluate_mlp.py
│   │   ├── evaluate_random_forest.py
│   │   └── evaluate_hist_gradient_boosting.py
│   ├── explainability/      # Explainability scripts
│   │   ├── attention_visualizer.py
│   │   ├── shap_lstm_attention.py
│   │   ├── shap_lstm.py
│   │   ├── shap_mlp.py
│   │   ├── shap_random_forest.py
│   │   └── stability_lstm_attention.py
├── results/
│   ├── lstm_attention/      # Mini-dossier LSTM + Attention
│   │   ├── models/
│   │   ├── metrics/
│   │   ├── reports/
│   │   ├── visualizations/
│   │   │   └── explainability/
│   │   └── explainability/
│   ├── lstm/                # Mini-dossier LSTM sans attention
│   ├── mlp/                 # Mini-dossier MLP
│   ├── random_forest/       # Mini-dossier Random Forest
│   └── hist_gradient_boosting/
├── notebooks/               # Jupyter notebooks
├── logs/                    # Training logs
└── requirements.txt         # Dependencies
```

## Quick Start

### 1. **First Time Setup** (Restore from Backup)

```bash
# Clone the repository (if using GitHub)
git clone https://github.com/your-username/IDS_LSTM_Project.git
cd IDS_LSTM_Project

# Create and activate virtual environment
python -m venv venv

# On Windows
.\venv\Scripts\Activate.ps1

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run the Project**

```bash
# Step 1: Preprocess data
python src/preprocessing/feature_engineering.py

# Step 2: Train the experiments
python src/training/train_lstm_attention.py      # LSTM + Attention
python src/training/train_lstm.py                # LSTM sans attention
python src/training/train_random_forest.py       # Random Forest baseline
python src/training/train_hist_gradient_boosting.py
python src/training/train_mlp.py                 # MLP tabular baseline

# Step 3: Evaluate each experiment
python src/evaluation/evaluate_lstm_attention.py
python src/evaluation/evaluate_lstm.py
python src/evaluation/evaluate_random_forest.py
python src/evaluation/evaluate_hist_gradient_boosting.py
python src/evaluation/evaluate_mlp.py
```

## Key Results

### Final Test Performance

All metrics below are computed on the held-out NSL-KDD test set. LSTM-based
models use sequence windows, so their test set contains 22,540 samples instead
of 22,544 tabular samples.

| Model | Accuracy | Macro F1 | Weighted F1 | Weighted ROC-AUC | Weighted PR-AUC |
|-------|---------:|---------:|------------:|-----------------:|----------------:|
| LSTM + Attention | 0.7652 | 0.5705 | 0.7288 | 0.9349 | 0.8482 |
| LSTM | 0.7536 | 0.5658 | 0.7195 | 0.9272 | 0.8368 |
| MLP | 0.7575 | 0.5764 | 0.7255 | 0.9340 | 0.8739 |
| Random Forest | 0.7471 | 0.4866 | 0.6983 | 0.9346 | 0.8848 |
| HistGradientBoosting | 0.7692 | 0.5644 | 0.7360 | 0.9517 | 0.8956 |

### Interpretation

- **Best overall test accuracy:** HistGradientBoosting, slightly ahead of
  LSTM + Attention.
- **Best macro F1:** MLP, slightly ahead of the LSTM variants.
- **Best weighted ROC-AUC and PR-AUC:** HistGradientBoosting.
- **Main limitation:** all models struggle with rare attacks, especially R2L
  and U2R. Boosting improves Random Forest on these classes, but R2L/U2R
  remain difficult.

The validation accuracy can be much higher than the test accuracy because the
NSL-KDD test set has a different class distribution and contains harder attack
patterns. For example, R2L is only 0.79% of the tabular training set but 12.81%
of the tabular test set. Therefore, the test metrics above are the numbers to
use for final conclusions.

### Stability Under Gaussian Perturbations

Prediction stability is the fraction of predictions that remain unchanged after
adding Gaussian noise to normalized inputs. Higher is better.

| Model | Stability sigma=0.01 | Stability sigma=0.05 | Stability sigma=0.10 | Accuracy sigma=0.10 |
|-------|---------------------:|---------------------:|---------------------:|--------------------:|
| LSTM + Attention | 0.9811 | 0.8594 | 0.7356 | 0.6446 |
| LSTM | 0.9807 | 0.8404 | 0.7436 | 0.6264 |
| MLP | 0.9870 | 0.8978 | 0.8215 | 0.6789 |
| Random Forest | 0.9248 | 0.8745 | 0.8244 | 0.5994 |
| HistGradientBoosting | 0.7738 | 0.7024 | 0.6323 | 0.5298 |

HistGradientBoosting gives the best clean performance, but it is the least
stable under synthetic input perturbations. MLP and Random Forest are more
stable in prediction consistency, while the LSTM variants degrade more smoothly
than HistGradientBoosting under stronger noise.

### Output Files
Each experiment writes to its own mini-folder:

| Experiment | Output folder |
|------------|---------------|
| LSTM + Attention | `results/lstm_attention/` |
| LSTM sans attention | `results/lstm/` |
| MLP | `results/mlp/` |
| Random Forest | `results/random_forest/` |
| HistGradientBoosting | `results/hist_gradient_boosting/` |

Inside each folder, use the same layout:

```text
models/                 trained model checkpoints
metrics/                JSON/CSV metrics, predictions, history
reports/                classification reports
visualizations/         confusion matrices, ROC/PR curves
visualizations/explainability/
explainability/         SHAP, attention, and stability raw outputs
```

## Important Files to Back Up

| File | Size | Importance | Include in Git |
|------|------|-----------|--------|
| `src/*.py` | ~50KB | **CRITICAL** | ✅ YES |
| `requirements.txt` | <1KB | **CRITICAL** | ✅ YES |
| `results/<experiment>/models/` | varies | HIGH | ⚠️ Optional |
| `data/raw/*.txt` | ~150MB | HIGH | ⚠️ Optional |
| `data/processed/` | ~200MB | HIGH | ⚠️ Optional |
| `pyrightconfig.json` | <1KB | MEDIUM | ✅ YES |
| `README.md` | <10KB | MEDIUM | ✅ YES |

## How to Save Everything

### Option 1: **Git + Local Storage** (Recommended for Personal Use)

```bash
# Initialize Git repo
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: IDS LSTM project with data preprocessing, training, and evaluation"

# View commit history
git log
```

### Option 2: **GitHub** (Best for Backup & Collaboration)

```bash
# Create new repo on GitHub (don't initialize with README)

# Add remote
git remote add origin https://github.com/your-username/IDS_LSTM_Project.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 3: **Cloud Storage** (Recommended for Large Data)

```bash
# For large files (model.h5, processed data), use:
# 1. Google Drive
# 2. OneDrive
# 3. Dropbox
# 4. AWS S3

# Example: Create a backup script
```

## Restoring After Restart

```bash
# After turning computer back on:

# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Reinstall dependencies (if needed)
pip install -r requirements.txt

# 3. Run any step (data is already processed)
python src/evaluate.py
```

## Tracked vs Untracked Files

### Files to Save (in Git)
```
✅ All Python source code (src/*.py)
✅ Configuration files (pyrightconfig.json, requirements.txt)
✅ Documentation (README.md, .gitignore)
✅ Results CSV files (results/*.csv)
❌ Large data files (add to .gitignore)
❌ Virtual environment (venv/)
```

### Files to Back Up Separately
```
📦 Raw data: data/raw/*.txt (150MB)
📦 Processed data: data/processed/*.npy (200MB)
📦 Model weights: results/<experiment>/models/
📦 Visualizations: results/<experiment>/visualizations/*.png
```

## Git Workflow Example

```bash
# After making changes
git add src/my_changes.py
git commit -m "Improve model architecture"

# Update requirements (after pip install new package)
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"

# Check status
git status

# View changes
git diff
```

## Total Size Guide

| Category | Size | Storage Type |
|----------|------|--------------|
| Git repo (code only) | ~100KB | Free Git |
| + Raw data | +150MB | GitHub LFS / Cloud |
| + Processed data | +200MB | Cloud only |
| + Model weights | +500KB | Cloud / USB |
| **Total** | **~350MB** | Multi-location |

## Recommendations

1. **USE GIT** - Always track code in version control (free & essential)
2. **BACKUP DATA** - Use cloud storage for large files (data, models)
3. **DOCUMENT RESULTS** - Save .csv reports and visualizations in Git
4. **GITIGNORE** - Use .gitignore to avoid committing unnecessary files
5. **REQUIREMENTS.TXT** - Always keep updated for easy reinstalls

## Troubleshooting

### Q: I deleted something by accident, how do I recover?
```bash
# Restore file from Git
git checkout HEAD -- path/to/file.py

# See all versions
git log path/to/file.py
```

### Q: How do I start fresh from the last saved state?
```bash
# Clone from backup
git clone <repo-url> IDS_LSTM_Project_v2
cd IDS_LSTM_Project_v2
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Q: Can I undo my last commit?
```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1
```

## Next Steps

1. ✅ **Initialize Git** - `git init`
2. ✅ **Create first commit** - `git commit -m "Initial"`
3. ✅ **Add to GitHub** - Push to remote
4. ✅ **Set up auto-backup** - Use GitHub Actions or cloud sync
5. ✅ **Document your changes** - Commit messages & README

---

**Last Updated**: April 15, 2026  
**Project Status**: Complete (Tasks 1-5)  
**Maintainer**: Meria  
