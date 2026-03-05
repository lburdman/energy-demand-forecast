import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide", page_title="Energy Demand Forecast")

# Header
st.title("⚡ Energy Demand Forecast Dashboard")
st.markdown("This dashboard presents the results of a 24-hour ahead electricity demand forecasting model using Open Power System Data (OPSD) for Germany. It compares advanced tree-based architectures against naive and linear baselines.")

# Path Setup (Relative for Streamlit Cloud and Root Execution)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DIAG_DIR = RESULTS_DIR / "diagnostics"

# 1. Forecast Explorer
st.header("Forecast Explorer (28-Day Sample)")
preds_path = RESULTS_DIR / "predictions_sample.csv"

if preds_path.exists():
    df_preds = pd.read_csv(preds_path)
    df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])
    
    min_date = df_preds['timestamp'].min().date()
    max_date = df_preds['timestamp'].max().date()
    
    date_range = st.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_preds['timestamp'].dt.date >= start_date) & (df_preds['timestamp'].dt.date <= end_date)
        df_filtered = df_preds.loc[mask]
        
        fig = go.Figure()
        
        # Actual Load
        fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered['actual'], 
                                 mode='lines', name='Actual Load', line=dict(color='black', width=2)))
        
        # XGBoost Prediction
        fig.add_trace(go.Scatter(x=df_filtered['timestamp'], y=df_filtered['xgb_prediction'], 
                                 mode='lines', name='XGBoost Prediction', line=dict(color='blue', dash='dash')))
        
        # Conformal Prediction Intervals
        if not df_filtered['lower_PI'].isna().all() and not df_filtered['upper_PI'].isna().all():
            fig.add_trace(go.Scatter(
                x=pd.concat([df_filtered['timestamp'], df_filtered['timestamp'][::-1]]),
                y=pd.concat([df_filtered['upper_PI'], df_filtered['lower_PI'][::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='95% Prediction Interval'
            ))
            
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Load (MW)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(f"Prediction sample file not found at '{preds_path}'. Please ensure Checkpoint 5 has been fully executed.")

st.markdown("---")

# 2. Metrics
st.header("Model Performance")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Holdout Test Metrics")
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        st.dataframe(df_metrics.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color='lightgreen', axis=0), use_container_width=True)
    else:
        st.info("Holdout metrics not found.")

with col_m2:
    st.subheader("Rolling-Origin Backtest Metrics (Aggregated)")
    backtest_path = DIAG_DIR / "backtest_metrics.csv"
    if backtest_path.exists():
        df_bt = pd.read_csv(backtest_path)
        df_bt_agg = df_bt.groupby("model")[["RMSE", "MAE", "MAPE"]].mean().reset_index()
        st.dataframe(df_bt_agg.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color='lightgreen', axis=0), use_container_width=True)
    else:
        st.info("Backtest metrics not found.")

st.markdown("---")

# 3. Key Figures
st.header("Key Analytical Figures")
col_f1, col_f2 = st.columns(2)

def safe_image(path, caption):
    if path.exists():
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found: {path.name}")
        
with col_f1:
    safe_image(FIG_DIR / "prediction_vs_actual.png", "Prediction vs Actual")
    safe_image(FIG_DIR / "prediction_interval_plot.png", "Conformal Prediction Intervals")

with col_f2:
    safe_image(FIG_DIR / "backtest_rmse_by_fold.png", "RMSE Stability across Folds")
    safe_image(FIG_DIR / "error_by_hour.png", "Absolute Error mapped by Hour")

# Expander for Diagnostics
with st.expander("View Extended Residual Diagnostics"):
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        safe_image(FIG_DIR / "residual_acf.png", "Auto-Correlation Function")
        safe_image(FIG_DIR / "residual_hist.png", "Residual Histogram")
        safe_image(FIG_DIR / "model_comparison.png", "Model Comparison Barplots")
    with col_e2:
        safe_image(FIG_DIR / "residual_qq.png", "Quantile-Quantile (QQ) Plot")
        safe_image(FIG_DIR / "residuals_plot.png", "Residual Time Series")

st.markdown("---")

# 4. Statistical Tests & Worst Days
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.subheader("Statistical Tests")
    stat_path = DIAG_DIR / "stat_tests.csv"
    if stat_path.exists():
        st.dataframe(pd.read_csv(stat_path), use_container_width=True)
    else:
        st.info("Statistical test results not found.")

with col_s2:
    st.subheader("Top Worst Prediction Days")
    worst_path = DIAG_DIR / "worst_days.csv"
    if worst_path.exists():
        st.dataframe(pd.read_csv(worst_path), use_container_width=True)
    else:
        st.info("Worst days log not found.")
