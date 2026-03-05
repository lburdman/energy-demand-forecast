import numpy as pd
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculates standard regression metrics comparing predicted values mathematically 
    against known targets explicitly evaluating RMSE, MAE, and MAPE.
    
    Args:
        y_true (pd.Series): Actual observed load targets.
        y_pred (pd.Series): Predicted computational parameters uniformly mapped identically.
        
    Returns:
        Dict[str, float]: Mathematical output limits containing precise mapped outputs.
    """
    # Drop NAs from both arrays together uniformly to securely handle any stray padding logic correctly
    valid_idx = y_true.notna() & y_pred.notna()
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    
    if len(y_true_clean) == 0:
        return {"RMSE": float('nan'), "MAE": float('nan'), "MAPE": float('nan')}
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

def train_test_split_time(df: pd.DataFrame, target_col: str = 'y', split_ratio: float = 0.8) -> tuple:
    """
    Chronologically splits data into training and testing sets cleanly retaining structure.
    """
    X = df.drop(columns=[target_col])
    if 'timestamp' in X.columns:
        X = X.drop(columns=['timestamp'])
        
    y = df[target_col]
    
    split_idx = int(len(df) * split_ratio)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def predict_naive_lag24(X_test: pd.DataFrame) -> pd.Series:
    """Returns the naïve lag-24 estimation."""
    return X_test['lag_24']

def train_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Pipeline:
    """Trains a Ridge linear baseline explicitly scaling inputs internally."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 200, max_depth: int = 10) -> RandomForestRegressor:
    """Trains the structural Random Forest representation natively."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        n_jobs=-1, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 500, learning_rate: float = 0.05, max_depth: int = 6) -> XGBRegressor:
    """Trains parameter-optimized XGBoost explicitly built for CPU/GPU environments flexibly."""
    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    xgb.fit(X_train, y_train)
    return xgb
