import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import os

def segment_errors(predictions_df: pd.DataFrame, true_col: str, pred_col: str, naive_col: str = None) -> pd.DataFrame:
    """
    Computes absolute errors natively and joins them back to the original timestamps
    to allow for grouping by hour, day_of_week, month, and is_weekend.
    
    Args:
        predictions_df (pd.DataFrame): The test predictions dataframe (with a datetime index)
        true_col: name of the ground truth column
        pred_col: name of the model prediction column
        naive_col: (Optional) name of the naive benchmark column for comparison.
        
    Returns:
        pd.DataFrame: Contains abs_error, naive_abs_error, error_improvement, and calendar components.
    """
    df = predictions_df.copy()
    
    # Calculate Absolute Errors
    df['abs_error'] = (df[true_col] - df[pred_col]).abs()
    df['mape_error'] = (df['abs_error'] / df[true_col].replace(0, np.nan)).abs()
    
    if naive_col and naive_col in df.columns:
        df['naive_abs_error'] = (df[true_col] - df[naive_col]).abs()
        df['error_improvement'] = df['naive_abs_error'] - df['abs_error']
        
    # Inject Calendar Features back directly for segmentation mapping from the index
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    else:
        raise ValueError("predictions_df must have a DatetimeIndex or a 'timestamp' column.")
        
    return df

def aggregate_error_segments(segmented_df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    """
    Groups errors generically calculating the mean MAE logically across subgroups.
    """
    agg_funcs = {'abs_error': 'mean'}
    
    if 'mape_error' in segmented_df.columns:
        agg_funcs['mape_error'] = 'mean'
    if 'error_improvement' in segmented_df.columns:
        agg_funcs['error_improvement'] = 'mean'
    
    return segmented_df.groupby(groupby_col).agg(agg_funcs).reset_index()

def get_worst_days(segmented_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Returns the top N worst days mathematically by computing the daily mean abs error.
    """
    df_daily = segmented_df.copy()
    
    if isinstance(df_daily.index, pd.DatetimeIndex):
        df_daily['date'] = df_daily.index.date
    else:
        df_daily['date'] = df_daily['timestamp'].dt.date
        
    daily_error = df_daily.groupby('date')['abs_error'].mean().reset_index()
    daily_error = daily_error.sort_values(by='abs_error', ascending=False)
    
    return daily_error.head(n)

def plot_feature_importance(model, feature_names: List[str], max_num_features: int = 15, title: str = "XGBoost Internal Feature Importance", save_path: str = None):
    """
    Graphs native regressor importance using the regressor object dynamically.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:max_num_features]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=[feature_names[i] for i in indices], palette="viridis")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
