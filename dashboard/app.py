import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

st.set_page_config(layout="wide", page_title="Energy Demand Forecast")
st.title("⚡ Energy Demand Forecast Dashboard")

# Paths
DRIVE_ROOT = "/content/drive/MyDrive/energy-demand-forecast"
if os.path.exists(DRIVE_ROOT):
    RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
else:
    RESULTS_DIR = "results"

DIAG_DIR = os.path.join(RESULTS_DIR, "diagnostics")

st.markdown("This dashboard presents the results of the 24h-ahead energy demand forecasting model using Open Power System Data (OPSD) for Germany.")

# Model Metrics
st.subheader("Model Performance (Rolling-Origin Backtest)")
backtest_path = os.path.join(DIAG_DIR, "backtest_metrics.csv")
if os.path.exists(backtest_path):
    df_metrics = pd.read_csv(backtest_path)
    avg_metrics = df_metrics.groupby("model")[["RMSE", "MAE", "MAPE"]].mean().reset_index()
    st.dataframe(avg_metrics.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color='lightgreen', axis=0))
else:
    st.info(f"Backtest metrics not found at {backtest_path}. Please run Checkpoint 5 notebook.")

# Interactive Plot
st.subheader("Actual vs. Predicted Load")
preds_path = os.path.join(RESULTS_DIR, "predictions.parquet")
if os.path.exists(preds_path):
    preds_df = pd.read_parquet(preds_path)
    
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(preds_df.index):
        if 'timestamp' in preds_df.columns:
            preds_df['timestamp'] = pd.to_datetime(preds_df['timestamp'])
            preds_df.set_index('timestamp', inplace=True)
        else:
            preds_df.index = pd.to_datetime(preds_df.index)
            
    min_date = preds_df.index.min().date()
    max_date = preds_df.index.max().date()
    
    date_range = st.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (preds_df.index.date >= start_date) & (preds_df.index.date <= end_date)
        filtered_df = preds_df.loc[mask]
        
        st.line_chart(filtered_df)
else:
    st.info(f"Predictions file not found at {preds_path}.")

st.markdown("---")

# Layout for plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Error by Hour")
    hour_img_path = os.path.join(DIAG_DIR, "error_by_hour.png")
    if os.path.exists(hour_img_path):
        st.image(Image.open(hour_img_path), use_container_width=True)
    else:
        st.info("Error by hour plot not found.")

with col2:
    st.subheader("Prediction Intervals (95%)")
    pi_img_path = os.path.join(DIAG_DIR, "prediction_interval_plot.png")
    if os.path.exists(pi_img_path):
        st.image(Image.open(pi_img_path), use_container_width=True)
    else:
        st.info("Prediction interval plot not found.")
