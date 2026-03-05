import pandas as pd
import numpy as np
from typing import Tuple, List

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-related features including hour, day of the week, 
    month, and a boolean flag for weekends.
    """
    df = df.copy()
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    return df

def create_lag_features(df: pd.DataFrame, target_col: str = "load", lags: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Creates lag targets by shifting backward iteratively.
    """
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
        
    return df

def create_rolling_features(df: pd.DataFrame, target_col: str = "load", windows: List[int] = [24, 168]) -> pd.DataFrame:
    """
    Derives heavily stabilized rolling means and rolling stds.
    Crucially uses `.shift(1)` BEFORE applying the rolling logic to avoid 
    leaking present data into historical moving averages.
    """
    df = df.copy()
    
    # Safe historical series
    s = df[target_col].shift(1)
    
    for w in windows:
        df[f'roll_mean_{w}'] = s.rolling(window=w).mean()
        df[f'roll_std_{w}'] = s.rolling(window=w).std()
        
    return df

def create_target(df: pd.DataFrame, target_col: str = "load", horizon: int = 24) -> pd.DataFrame:
    """
    Produces the model target value 'y' mapped across the specific look-forward horizon.
    """
    df = df.copy()
    
    df['y'] = df[target_col].shift(-horizon)
    
    return df

def build_feature_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Master pipeline constructing the explicit features required for typical tree-based 
    TS learning algorithms alongside SARIMA matrices, mapping targets accurately while
    avoiding all form of leakage.

    Drops NaNs inherently formed by the shift distributions.
    """
    # Exogenous variables natively collected (could be missing/empty initially safely)
    base_cols = list(df.columns)
    
    # Check if load actually exists natively in frame mappings
    if 'load' not in base_cols:
        raise ValueError("DataFrame must contain a target metric column 'load' implicitly.")
        
    # Execute chronological derivations
    df_feat = create_calendar_features(df)
    df_feat = create_lag_features(df_feat, target_col='load', lags=[1, 24, 168])
    df_feat = create_rolling_features(df_feat, target_col='load', windows=[24, 168])
    df_feat = create_target(df_feat, target_col='load', horizon=24)
    
    # Truncate edges affected by shifts/horizons natively across all indexes cleanly
    df_feat = df_feat.dropna()
    
    # Extract structural feature signatures alone natively for downstream regression outputs
    ignore_cols = ['y', 'load'] # Usually 'load' itself shouldn't be a feature to strictly avoid leakage alongside 'y' (but 'lag_X' handles this specifically)
    
    # Exogenous columns inherently included strictly if discovered earlier smoothly ('temperature', 'solar', etc.)
    all_current_cols = df_feat.columns.tolist()
    feature_columns = [col for col in all_current_cols if col not in ignore_cols]
    
    return df_feat, feature_columns
