# ⚡ Energy Demand Forecast

**A portfolio-grade machine learning project implementing a robust 24-hour ahead electricity demand forecasting pipeline using the Open Power System Data (OPSD) for Germany.**

## 📖 What's Inside

The project follows a rigorous, production-like data science workflow documented across 5 standalone Google Colab Notebooks:
- `notebooks/01_data_exploration.ipynb`: Data Extraction and Exploratory Data Analysis (EDA).
- `notebooks/02_feature_engineering.ipynb`: Time-series feature engineering (calendar variables, auto-regressive lags, rolling windows).
- `notebooks/03_modeling.ipynb`: Baseline and advanced model benchmarks (Naive, Ridge Regression, Random Forest, XGBoost).
- `notebooks/04_interpretability_diagnostics.ipynb`: Global feature importance, cyclical error segmentation, and worst-case isolation.
- `notebooks/05_statistical_validation.ipynb`: Rigorous 5-fold rolling-origin backtesting, residual analysis (Ljung-Box test), and conformal prediction bounds.

Also included is a standalone Streamlit dashboard for interactive evaluation:
- `dashboard/app.py`: A native Python web app charting the predictions natively against actual bounds with dynamic zooming.

## 🚀 Quickstart

This project is optimized to run seamlessly in **Google Colab** with storage persisting securely on **Google Drive**:
1. Open [`notebooks/05_statistical_validation.ipynb`](notebooks/05_statistical_validation.ipynb) in Google Colab.
2. The notebook will automatically guide you to mount your Google Drive.
3. Run the notebook top-to-bottom. Data is fetched automatically, and the XGBoost model outputs are stored directly in your Drive.
4. *(Optional)* Run the Streamlit Dashboard locally:
   ```bash
   pip install -r requirements.txt
   streamlit run dashboard/app.py
   ```

## 📊 Key Results Preview

Tree-based architectures significantly outperformed baseline and linear models effectively. XGBoost was selected structurally for optimal accuracy and cyclic mapping performance across stable 5-fold rolling-origin backtesting metrics.

| Model        | Avg RMSE (MW) | Avg MAE (MW) | Avg MAPE (%) |
|--------------|---------------|--------------|--------------|
| Naive      | High          | High         | High         |
| Ridge        | Medium        | Medium       | Medium       |
| RF           | Low           | Low          | Low          |
| XGBoost      | Lowest        | Lowest       | Lowest       |
*(Actual precision limits logged explicitly inside `results/diagnostics/backtest_metrics.csv`)*

### Residual Analysis
Residual tests (Ljung-Box p-values) validated the core structural stability natively, capturing broad seasonality efficiently while tracking remaining cyclic volatility intuitively mapped across:
![Prediction Interval Plot](https://github.com/lburdman/energy-demand-forecast/raw/main/results/diagnostics/prediction_interval_plot.png)

## 📁 Folder Structure

```text
├── README.md
├── requirements.txt
├── report/
│   └── REPORT.md               <-- Concise project paper
├── dashboard/
│   └── app.py                  <-- Streamlit exploration tool
├── notebooks/                  <-- Core Colab steps (01-05)
└── src/                        
    ├── data_loader.py          <-- Data ingestion methods
    ├── diagnostics.py          <-- Core plotting behaviors
    ├── features.py             <-- Modeling targets
    ├── models.py               <-- Baseline architectures
    └── validation.py           <-- Core conformal methods
```

## ⚖️ License & Attribution

This codebase is provided under the standard MIT License.
**Data Source:** The dataset natively utilizes the `time_series_60min_singleindex.csv` provided proudly under an open data standard by the [Open Power System Data (OPSD)](https://data.open-power-system-data.org/) consortium.
