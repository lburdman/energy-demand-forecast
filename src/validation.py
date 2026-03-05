import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Tuple
from src.models import calculate_metrics

def rolling_origin_backtest(df: pd.DataFrame, 
                            target_col: str, 
                            train_funcs: Dict[str, Callable],
                            predict_funcs: Dict[str, Callable],
                            test_window: int = 720, 
                            n_folds: int = 5) -> pd.DataFrame:
    """
    Performs rolling-origin (walk-forward) backtesting on time series data.
    
    Args:
        df: The feature-engineered dataframe, sorted by time.
        target_col: Name of the target column ('y').
        train_funcs: Dictionary of model names to training functions e.g. {'Ridge': train_ridge}
        predict_funcs: Dictionary of model names to prediction functions e.g. {'Ridge': lambda m, x: m.predict(x)}
        test_window: Number of samples in each test fold.
        n_folds: Number of walk-forward folds.
        
    Returns:
        A DataFrame containing fold, model, and metrics.
    """
    results = []
    
    total_len = len(df)
    
    # Calculate cutoff indices safely
    cutoffs = []
    for i in range(n_folds, 0, -1):
        cutoff = total_len - (i * test_window)
        if cutoff <= 0:
            raise ValueError(f"Data too small for {n_folds} folds of size {test_window}.")
        cutoffs.append(cutoff)
        
    for fold_idx, cutoff_i in enumerate(cutoffs, 1):
        train_df = df.iloc[:cutoff_i]
        test_df = df.iloc[cutoff_i:cutoff_i + test_window]
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        if 'timestamp' in X_train.columns:
            X_train = X_train.drop(columns=['timestamp'])
            
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        if 'timestamp' in X_test.columns:
            X_test = X_test.drop(columns=['timestamp'])
        
        # Train and predict for each model
        for model_name, train_func in train_funcs.items():
            if model_name == "Naive":
                # Naive model uses a special prediction function that does not need training
                y_pred = predict_funcs[model_name](None, X_test)
            else:
                model = train_func(X_train, y_train)
                y_pred = predict_funcs[model_name](model, X_test)
            
            y_pred = pd.Series(y_pred, index=y_test.index)
            metrics = calculate_metrics(y_test, y_pred)
            
            results.append({
                "fold": fold_idx,
                "model": model_name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "MAPE": metrics["MAPE"]
            })
            
    return pd.DataFrame(results)

def conformal_prediction_interval(y_true_calib: pd.Series, y_pred_calib: pd.Series, y_pred_test: pd.Series, alpha: float = 0.05) -> Tuple[pd.Series, pd.Series, float]:
    """
    Computes simple conformal prediction intervals using calibration residuals.
    
    Args:
        y_true_calib: Actual values from the calibration set.
        y_pred_calib: Predicted values from the calibration set.
        y_pred_test: Predicted values for the test set.
        alpha: Error rate (e.g., 0.05 for 95% coverage, 0.10 for 90% coverage).
        
    Returns:
        A tuple of (lower_bound, upper_bound, q_value).
    """
    # Compute absolute residuals on the calibration set
    residuals = np.abs(y_true_calib - y_pred_calib)
    
    # Calculate the quantile q
    # We want the (1 - alpha) quantile. We adjust for finite sample size: (n+1)(1-alpha)/n
    n = len(residuals)
    quantile_level = min(1.0, (1 - alpha) * (n + 1) / n)
    q = np.quantile(residuals, quantile_level)
    
    lower_bound = y_pred_test - q
    upper_bound = y_pred_test + q
    
    return lower_bound, upper_bound, q

