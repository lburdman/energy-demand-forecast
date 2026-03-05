# ⚡ Energy Demand Forecasting with Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-demand-forecast.streamlit.app)

**A portfolio-grade machine learning project implementing a robust 24-hour ahead electricity demand forecasting pipeline using the Open Power System Data (OPSD).**

## 📖 Project Overview
This project tackles the **electricity demand forecasting problem**, targeting stable power delivery via accurate predictive scaling. The objective is to construct a **24-hour ahead** load forecast minimizing standard error limits against extremely volatile, non-linear sequences dynamically responding to cyclic shifts securely.

## 📊 Dataset
We natively utilize the high-quality **Open Power System Data (Germany)** framework. The dataset provides multiple years of historical hourly electricity demand alongside renewable generation bounds forming ideal conditions for complex autoregressive time-series structures.
- [View the OPSD Dataset](https://data.open-power-system-data.org/)

## ⚙️ Feature Engineering
We natively avoid data leakage dynamically predicting the following boundaries without future overlaps:
- **Calendar Features**: `hour`, `day_of_week`, `month`, and boolean `is_weekend` mapping basic human cycles natively.
- **Lag Features**: Auto-regressive values from $t-1$, $t-24$, and $t-168$ past boundaries. `lag-24` inherently predicts daily repeating patterns intuitively.
- **Rolling Features**: Calculating moving momentum `mean` and `std` boundaries for 24h & 168h windows securely shifted by `t-1`.

## 🤖 Models Evaluated
- **Naive baseline (lag 24)**
- **Ridge Regression**
- **Random Forest**
- **XGBoost (Selected Model)**

## 📈 Results Preview & Key Findings

### 1. Daily Seasonality Dominates
![Prediction vs Actual](results/figures/prediction_vs_actual.png)
The models inherently learn that strong daily cycles map the majority of the predictive weight dynamically scaling with peak loads accurately.

### 2. Nonlinearity Outperforms Linear Mapping
Tree models (XGBoost/RF) drastically outperform the Naive baseline and the Ridge regression mapping complex, non-linear interactions accurately across changing conditions safely without over-fitting limitations.

### 3. Stability Across Rolling-Origin Backtesting
![Backtest RMSE by Fold](results/figures/backtest_rmse_by_fold.png)
Using robust rolling-origin validation, we mapped sequential fold tests securely verifying structural accuracy remains completely tight and completely stable without unmapped degradation randomly gracefully.

### 4. Ramp & Peak Error Concentrations
![Error by Hour](results/figures/error_by_hour.png)
Predictive limits mathematically suffer inherently around rapidly shifting peak cycles intuitively demonstrating inherent tracking latency across steep ramp periods logically.

### 5. Uncertainty Modeling via Conformal Prediction
![Prediction Interval Plot](results/figures/prediction_interval_plot.png)
Residual matrices inherently map heavy-tailed non-Gaussian logic cleanly justifying safe Conformal Prediction interval tracking dynamically outputting an implicit 95% uncertainty scaling logically surrounding limits effectively.

## 💻 Run the Dashboard
A fast native Python web application charting the predictions interactively!
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 📁 Repository Structure
```text
├── README.md
├── requirements.txt
├── report/
│   └── REPORT.md               <-- Concise project paper
├── dashboard/
│   └── app.py                  <-- Streamlit interactive dashboard
├── notebooks/                  <-- Core Colab pipeline (01-05)
└── src/                        
    ├── data_loader.py          <-- Data ingestion 
    ├── diagnostics.py          <-- Error plotting
    ├── features.py             <-- Feature engineering mappings
    ├── models.py               <-- Baseline architectures
    └── validation.py           <-- Backtest/Conformal logic
```

## 🚀 Future Work
- Implementing structured **Holiday Calendars** bounding deviations.
- Integrating highly complex **Temperature/Exogenous features**.
- Experimenting natively with **Quantile Regression** evaluating direct probabilistic outputs securely natively.
