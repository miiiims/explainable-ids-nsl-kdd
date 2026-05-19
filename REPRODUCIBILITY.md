# Reproducibility Commands

Run all commands from the project root:

```powershell
cd c:\Users\meria\IDS_LSTM_Project
```

## Shared preprocessing

```powershell
.\venv_dml\Scripts\python.exe src\preprocessing\feature_engineering.py
```

## Experiment 3: LSTM + Attention

Training:

```powershell
.\venv_dml\Scripts\python.exe src\training\train_lstm_attention.py
```

Evaluation:

```powershell
.\venv_dml\Scripts\python.exe src\evaluation\evaluate_lstm_attention.py
```

Attention explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\attention_visualizer.py
```

SHAP explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\shap_lstm_attention.py
```

Stability analysis:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\stability_lstm_attention.py
```

## Result layout

```text
results/
  lstm_attention/
    models/
    metrics/
    reports/
    visualizations/
      explainability/
    explainability/
```

Future experiments should use the same layout:

```text
results/
  mlp/
  random_forest/
```

## Experiment 2: MLP

Training:

```powershell
.\venv_dml\Scripts\python.exe src\training\train_mlp.py
```

Evaluation:

```powershell
.\venv_dml\Scripts\python.exe src\evaluation\evaluate_mlp.py
```

SHAP explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\shap_mlp.py
```

Stability analysis:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\stability_mlp.py
```

## Lightweight adversarial evasion analysis

This white-box FGSM/PGD stress test is applied to the differentiable IDS models
MLP, LSTM, and LSTM + Attention. It also reports a simple confidence-threshold
abstention probe as a lightweight mitigation analysis.

```powershell
.\venv_dml\Scripts\python.exe src\explainability\adversarial_evasion_analysis.py
```
