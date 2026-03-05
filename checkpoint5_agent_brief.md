# Energy Demand Forecast (OPSD Germany) — Agent Brief for Checkpoint 5

This file summarizes the current project state and what to implement in **Checkpoint 5**.
It’s written so the code agent (Antigravity) can implement CP5 without re-reading all notebooks.

---

## Project snapshot (what already exists)

### Repo
- Repo: `lburdman/energy-demand-forecast`
- Notebooks:
  - `notebooks/01_data_exploration.ipynb` (Checkpoint 1)
  - `notebooks/02_feature_engineering.ipynb` (Checkpoint 2)
  - `notebooks/03_modeling.ipynb` (Checkpoint 3)
  - `notebooks/04_interpretability_diagnostics.ipynb` (Checkpoint 4)

### Dataset (OPSD)
- File downloaded into Drive: `data/raw/time_series_60min_singleindex.csv`
- Download URL hardcoded in `src/data_loader.ensure_opsd_download()`:
  - `https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv`
- Loader selects Germany columns dynamically:
  - timestamp candidate: `utc_timestamp` (renamed to `timestamp`)
  - load: first `DE_*load*` non-forecast (renamed to `load`)
  - optional exogenous: `solar`, `wind`, `temperature` if present
- Cleaning:
  - parse timestamp → set index → sort
  - interpolate missing values with `method="time"`

### Modeling formulation
- **Forecast horizon:** `t+24` (next-day same hour), i.e. `y = load.shift(-24)`
- Split: chronological 80/20 via `models.train_test_split_time(df, split_ratio=0.8)`
- Feature table: built by `features.build_feature_table(df)`:
  - Calendar: `hour`, `day_of_week`, `month`, `is_weekend`
  - Lags: `lag_1`, `lag_24`, `lag_168`
  - Rolling (computed safely on `load.shift(1)`): `roll_mean_24`, `roll_std_24`, `roll_mean_168`, `roll_std_168`
  - Exogenous if present: typically `solar`, `wind` (+ `temperature` if available)
  - Drops NaNs created by shifts/horizon.

### Current model results (from `model_metrics.csv`)
| Model        |    RMSE |     MAE |      MAPE |
|:-------------|--------:|--------:|----------:|
| Naive        | 8993.08 | 6792.21 | 0.129343  |
| Ridge        | 5016.67 | 3948.16 | 0.0755203 |
| RandomForest | 2258.22 | 1475.35 | 0.0287505 |
| XGBoost      | 2239    | 1482.16 | 0.029087  |

**Current winner:** XGBoost and RandomForest are very close; XGBoost slightly better RMSE, RandomForest slightly better MAPE (tiny difference).

### Known issues / decisions so far
- SARIMA performed extremely poorly and slow; it is de-emphasized.
- We focus on **classical ML** (no deep learning) with strong time-series validation.

---

## Checkpoint 5 — Goal

Make the project **portfolio-grade** by adding:
1. **Statistical validation** (stationarity + residual diagnostics).
2. **Robust evaluation** beyond single holdout: **rolling-origin backtesting**.
3. **Uncertainty / prediction intervals** (simple, defensible approach).
4. **Insight narrative**: what drives demand + when models fail.

This checkpoint should produce figures + a short written section that reads like a report.

---

## CP5 — Deliverables (what to generate)

### A) Backtesting (rolling-origin evaluation)
Implement a function that runs multiple temporal folds.

**Suggested design**
- Use the engineered dataset `features_dataset.parquet` if present (or rebuild from raw if missing).
- Choose e.g. **5 folds** on the last ~20% of the timeline:
  - Fold i: train = [start .. cutoff_i], test = next `test_window` (e.g. 30 days * 24 = 720 points)
  - Step forward by `test_window`
- For each fold, train:
  - Naive (`lag_24`)
  - Ridge (optional baseline)
  - RandomForest
  - XGBoost
- Compute RMSE/MAE/MAPE per fold, then summarize:
  - mean ± std across folds
  - plot metrics by fold (line plot)
  - plot “XGB vs Naive improvement” distribution by fold.

**Output**
- `results/diagnostics/backtest_metrics.csv`
- `results/diagnostics/backtest_rmse_by_fold.png` (and optionally MAE/MAPE)
- A short markdown cell explaining why rolling-origin matters.

### B) Residual diagnostics (for the final selected model)
Using the **test split predictions** (or fold predictions):
- Residual series: `e_t = y_true - y_pred`
- Plots:
  1. Residual time series (with rolling mean/std)
  2. Histogram/KDE of residuals
  3. QQ plot (normality sanity check)
  4. Residual ACF (autocorrelation of errors)
  5. Residual vs predicted scatter (heteroscedasticity)
- Tests (lightweight, optional but strong):
  - Ljung–Box on residuals (autocorrelation)
  - ADF/KPSS on the target series (daily average or detrended series) to support discussion around stationarity

**Output**
- Figures saved under `results/diagnostics/`
- Small table with test p-values and interpretation (“evidence of autocorrelation remains”, etc.)

### C) Error segmentation (when the model fails)
Leverage existing utilities in `src/diagnostics.py`:
- Segment by:
  - hour
  - day_of_week
  - month
  - weekend vs weekday
- Produce bar plots of mean absolute error and mean MAPE by segment.
- Add “worst 5 days” table (already supported by `get_worst_days`).

**Output**
- `results/diagnostics/error_by_hour.png`, `error_by_dow.png`, `error_by_month.png`, `worst_days.csv`

### D) Prediction intervals (uncertainty)
Keep it simple and defensible for a portfolio:

**Option 1 (recommended): Conformal intervals**
- Use residuals on a calibration slice (e.g. last 10% of train).
- Compute q = quantile(|residual|, 0.9) for 80% PI or 0.95 for 90% PI.
- Interval: `[y_hat - q, y_hat + q]`
- Evaluate empirical coverage on test.

**Output**
- Coverage % (should be close to nominal, discuss deviations)
- Plot: actual vs prediction with shaded interval for 7–14 days.

---

## Implementation notes (important)
- Avoid leakage:
  - Any scaling must be fit on train only.
  - Rolling windows already use `shift(1)` in `features.py`.
  - When backtesting, create splits BEFORE fitting models.
- Speed:
  - Use modest RF params (`n_estimators=200`, `max_depth=10`).
  - XGB uses `tree_method='hist'`; prefer CPU unless GPU is available.
- Keep all outputs reproducible and saved to Drive paths.

---

## “Done” definition for CP5
Checkpoint 5 is complete when:
- Backtest table + plots exist and are referenced in the notebook narrative.
- Residual diagnostics plots + at least one formal test (Ljung–Box) are included.
- Error segmentation plots exist (hour/day/week/month + worst days).
- A simple prediction interval is implemented + coverage is reported.

