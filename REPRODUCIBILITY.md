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

## Experiment 3b: LSTM sans attention

Training:

```powershell
.\venv_dml\Scripts\python.exe src\training\train_lstm.py
```

Evaluation:

```powershell
.\venv_dml\Scripts\python.exe src\evaluation\evaluate_lstm.py
```

SHAP explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\shap_lstm.py
```

Stability analysis:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\stability_lstm.py
```

## Experiment 1: Random Forest baseline

Training:

```powershell
.\venv_dml\Scripts\python.exe src\training\train_random_forest.py
```

Evaluation:

```powershell
.\venv_dml\Scripts\python.exe src\evaluation\evaluate_random_forest.py
```

SHAP explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\shap_random_forest.py
```

Stability analysis:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\stability_random_forest.py
```

## Experiment 1b: HistGradientBoosting baseline

Training:

```powershell
.\venv_dml\Scripts\python.exe src\training\train_hist_gradient_boosting.py
```

Evaluation:

```powershell
.\venv_dml\Scripts\python.exe src\evaluation\evaluate_hist_gradient_boosting.py
```

SHAP explanations:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\shap_hist_gradient_boosting.py
```

Stability analysis:

```powershell
.\venv_dml\Scripts\python.exe src\explainability\stability_hist_gradient_boosting.py
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
  lstm/
    models/
    metrics/
    reports/
    visualizations/
      explainability/
    explainability/
  mlp/
    models/
    metrics/
    reports/
    visualizations/
      explainability/
    explainability/
  random_forest/
    models/
    metrics/
    reports/
    visualizations/
      explainability/
    explainability/
  hist_gradient_boosting/
    models/
    metrics/
    reports/
    visualizations/
      explainability/
    explainability/
```

The legacy global folders such as `results/metrics/`, `results/models/`,
`results/reports/`, `results/visualizations/`, and `results/explainability/`
are kept only for backward compatibility. New runs should use the mini-folder
for the experiment name.

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
