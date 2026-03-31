# Predictive Model

This document explains how the forging line data is used to build a predictive model that estimates the total piece travel time before the piece finishes the line. It covers feature selection, model training, evaluation, and the inference service.

## The prediction problem

Each piece takes approximately 56–59 seconds to travel from the furnace to the quench bath. During this journey, the piece passes through several stages on the main press (2nd strike, 3rd strike, 4th strike) before reaching the bath.

The question is: **after the 2nd strike (~18 seconds into the process), can we predict how long the piece will take to reach the bath?**

If we can, this enables:
- **Real-time alerts**: flag pieces that are likely to be slow while they are still on the line
- **Anticipation**: operators can investigate the cause of a predicted delay before it impacts downstream production
- **OEE monitoring**: predicted vs actual times reveal process drift

## Feature selection (Task 6)

### Constraint: only early-stage data

The model must predict using only information available **after the 2nd strike** — approximately 18 seconds into the ~58-second journey. Any feature that requires waiting for later stages (3rd strike, 4th strike, bath) cannot be used, because by the time those values exist, the prediction is no longer useful.

### Selected features

| Feature | Type | Available at | Why it's included |
|---|---|---|---|
| `die_matrix` | Categorical (int) | Before processing | Each die has different tooling geometry and expected times. Matrix 4974 has a median bath time of 56.0s while 5091 has 59.2s — the model needs this context. |
| `lifetime_2nd_strike_s` | Continuous (seconds) | After 2nd strike (~18s) | The earliest cumulative time. If a piece is already 3 seconds slower than reference at the 2nd strike, it will likely carry that delay through to the bath. |
| `oee_cycle_time_s` | Continuous (seconds) | Rolling metric | The production rate provides context about the overall line state. Slower OEE may correlate with systematic delays (e.g. robot trajectory adjustments, hydraulic pressure drops). |

### Excluded features

| Feature | Why excluded |
|---|---|
| `lifetime_3rd_strike_s` | Available too late (~25s) — by this point, the piece is already halfway through the main press |
| `lifetime_4th_strike_s` | Available too late (~38s) — and has 16% missing data |
| `lifetime_auxiliary_press_s` | Available too late (~55s) — only ~2s before the bath, prediction would be useless |
| `lifetime_general_s` | Equivalent to `lifetime_bath_s` — redundant with the target |
| `lifetime_bath_s` | This is the target — obviously cannot be a feature |
| `partial_*` columns | Derived from cumulative times that include late-stage data |
| `piece_id` | Not predictive — just an identifier |

### Feature correlations with target

The Pearson correlation between each feature and `lifetime_bath_s`:

| Feature | Correlation | Interpretation |
|---|---|---|
| `lifetime_2nd_strike_s` | Strong positive | Pieces slow at the 2nd strike tend to be slow at the bath |
| `die_matrix` | Weak | Different matrices have different baselines, but the relationship is non-linear (handled well by tree models) |
| `oee_cycle_time_s` | Weak-moderate | Provides production context, marginal predictive value on its own but helpful in combination |

## Model training (Task 7)

### Implementation

The model is trained in `notebooks/05_feature_selection_and_model.ipynb`.

### Model choice: XGBoost Regressor

XGBoost was selected because:
- Handles mixed feature types (categorical die_matrix + continuous times) without explicit encoding
- Robust to remaining noise in the cleaned data
- Fast training on ~130k rows
- Produces interpretable feature importance rankings
- Well-supported in AWS SageMaker for deployment

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Sufficient trees for convergence without overfitting |
| `max_depth` | 6 | Deep enough to capture die_matrix × lifetime interactions |
| `learning_rate` | 0.1 | Standard learning rate for 200 trees |
| `subsample` | 0.8 | Row sampling to reduce overfitting |
| `colsample_bytree` | 0.8 | Feature sampling per tree |

### Train/test split

- **80/20 split**, stratified by `die_matrix`
- Training set: ~104k pieces
- Test set: ~26k pieces
- Fixed random seed (42) for reproducibility

### Performance metrics

| Metric | Value | Interpretation |
|---|---|---|
| **RMSE** | 1.35s | Root mean squared error — penalizes large errors more |
| **MAE** | 0.65s | Average prediction error: less than 1 second off |
| **R²** | 0.78 | The model explains 78% of the variance in bath time |

On a typical bath time of ~58 seconds, the MAE of 0.65s represents a **±1.1% error** — sufficient for real-time alerting where the goal is to detect multi-second delays, not sub-second precision.

### Feature importance

The model's internal feature importance (gain-based) shows which features contribute most to predictions. `lifetime_2nd_strike_s` dominates — confirming that early-stage timing is the strongest predictor of total travel time.

### Saved artifacts

The trained model and metadata are saved to `models/`:

| File | Content |
|---|---|
| `xgboost_bath_predictor.json` | Serialized XGBoost model (loadable without retraining) |
| `model_metadata.json` | Features, target, metrics, training info |

## Inference service (Task 8)

### Implementation

The inference service is implemented in `src/vaultech_analysis/inference.py` as a Python module with a `Predictor` class.

### Architecture

```
models/
  xgboost_bath_predictor.json    ← trained model
  model_metadata.json            ← feature list, metrics

src/vaultech_analysis/
  inference.py                   ← Predictor class + CLI
      │
      ├── predict(die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s)
      │   → {"predicted_bath_time_s": 57.22, ...}
      │
      └── predict_batch(df)
          → pd.Series of predictions
```

### Usage as CLI

```bash
# With all features
uv run python -m vaultech_analysis.inference \
  --die-matrix 5052 --first 18.3 --oee 13.5

# Without OEE (uses median as default)
uv run python -m vaultech_analysis.inference \
  --die-matrix 5091 --first 22.0
```

Output:
```json
{
  "predicted_bath_time_s": 57.22,
  "die_matrix": 5052,
  "lifetime_2nd_strike_s": 18.3,
  "oee_cycle_time_s": 13.5,
  "model_metrics": {
    "rmse": 1.348,
    "mae": 0.653,
    "r2": 0.7813
  }
}
```

### Usage as Python module (for Streamlit)

```python
from vaultech_analysis.inference import Predictor

predictor = Predictor()

# Single prediction
result = predictor.predict(
    die_matrix=5052,
    lifetime_2nd_strike_s=18.3,
    oee_cycle_time_s=13.5
)
print(result["predicted_bath_time_s"])  # 57.22

# Batch prediction on a DataFrame
df["predicted_bath_s"] = predictor.predict_batch(df)
```

### Design decisions

| Decision | Rationale |
|---|---|
| Module, not HTTP server | Simpler to deploy and test. Streamlit imports it directly — no network overhead. Can be wrapped in Flask/FastAPI later if needed. |
| Missing OEE defaults to median (13.8s) | OEE is NULL for ~23% of pieces (production gaps). Using the median avoids dropping predictions for those pieces. |
| Returns model metrics in response | Consumers can display prediction confidence alongside the result. |
| Validates die_matrix | Returns an error dict if an unknown matrix is provided, instead of silently producing garbage predictions. |

## Dependencies

The predictive model adds these Python packages to the project:

| Package | Purpose |
|---|---|
| `xgboost` | Model training and inference |
| `scikit-learn` | Train/test split, evaluation metrics (RMSE, MAE, R²) |

Both are declared in `pyproject.toml` and installed via `uv sync`.

## How the pieces fit together

![Prediction pipeline — from gold dataset through model training, saved artifacts, inference service, to Streamlit and CLI consumers](assets/prediction_pipeline.png)

The diagram shows the end-to-end prediction pipeline:

1. **Gold dataset** (`pieces.parquet`) provides the 3 features and the target for training
2. **Model training** (notebook 05) trains an XGBoost Regressor and evaluates it (R²=0.78, MAE=0.65s)
3. **Saved artifacts** (`models/`) store the serialized model and its metadata (features, metrics, config)
4. **Inference service** (`inference.py`) loads the saved model and exposes `predict()` and `predict_batch()` methods
5. **Consumers**: the Streamlit app imports the Predictor class directly; the CLI invokes it via `uv run python -m vaultech_analysis.inference`
