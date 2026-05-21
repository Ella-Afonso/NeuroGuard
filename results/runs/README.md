# results/runs/

Per-experiment output folders. **Contents are gitignored.**

## Structure

Each run creates a folder named `{timestamp}_{experiment}_{config_hash[:8]}/`
containing:

```
20260521_120000_mvp_full_factorial_a1b2c3d4/
├── config_snapshot.yaml    # Frozen experiment config
├── metrics.json            # Evaluation metrics (ROC AUC, PR AUC, F1, ...)
├── predictions.parquet     # Model predictions on test set
├── plots/                  # Auto-generated figures
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── model_artefact.joblib   # Trained model (if saved)
```

## Manifest

A `manifest.json` file in this directory indexes all runs with config
hashes, timestamps, and headline metrics for programmatic lookup:

```json
[
  {
    "run_id": "20260521_120000_mvp_full_factorial_a1b2c3d4",
    "experiment": "mvp_full_factorial",
    "config_hash": "a1b2c3d4...",
    "timestamp": "2026-05-21T12:00:00Z",
    "best_model": "xgboost",
    "pr_auc": 0.74,
    "roc_auc": 0.82
  }
]
```
