import numpy as pd
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

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
    
    import numpy as np
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }
