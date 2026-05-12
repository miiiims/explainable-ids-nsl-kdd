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
│   ├── model.py             # LSTM model architecture
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation & comparison
│   └── features.py          # Feature engineering utilities
├── models/
│   ├── baseline/            # Baseline models
│   └── lstm/                # LSTM model checkpoints
├── results/
│   ├── visualizations/      # Performance charts
│   └── model_comparison.csv # Evaluation metrics
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
python src/data_engineering.py

# Step 2: Train LSTM model
python src/train.py

# Step 3: Evaluate and compare models
python src/evaluate.py
```

## Key Results

### Model Performance (Task 5)
- **LSTM Accuracy**: 62.79% | **F1-Score**: 51.00% | **ROC-AUC**: 77.76%
- **Random Forest Accuracy**: 72.15% ⭐ | **F1-Score**: 61.95%
- **Optimal Sequence Length**: 5 frames (65.89% accuracy)

### Output Files
- Training history: `results/`
- Visualizations: `results/visualizations/`
- Model comparison: `results/model_comparison.csv`
- Evaluation report: `results/evaluation_report.txt`

## Important Files to Back Up

| File | Size | Importance | Include in Git |
|------|------|-----------|--------|
| `src/*.py` | ~50KB | **CRITICAL** | ✅ YES |
| `requirements.txt` | <1KB | **CRITICAL** | ✅ YES |
| `models/lstm_model.h5` | ~500KB | HIGH | ⚠️ Optional |
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
📦 Model weights: models/lstm_model.h5 (500KB)
📦 Visualizations: results/visualizations/*.png
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
