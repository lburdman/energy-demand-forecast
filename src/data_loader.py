import os
import urllib.request
import pandas as pd

def get_drive_paths(drive_root: str) -> dict:
    """
    Returns a dictionary of paths required for the project, creating directories if they do not exist.
    """
    paths = {
        "raw_data": os.path.join(drive_root, "data", "raw"),
        "processed_data": os.path.join(drive_root, "data", "processed"),
        "figures": os.path.join(drive_root, "results", "figures")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

def ensure_opsd_download(raw_path: str) -> str:
    """
    Downloads the OPSD time series dataset if it does not already exist in the raw directory.
    Returns the full path to the downloaded CSV.
    """
    file_name = "time_series_60min_singleindex.csv"
    file_path = os.path.join(raw_path, file_name)
    
    # We use a stable URL for testing. 
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    
    if not os.path.exists(file_path):
        print(f"Downloading OPSD dataset to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"File {file_name} already exists. Skipping download.")
        
    return file_path

def load_opsd_germany(raw_file_path: str) -> pd.DataFrame:
    """
    Loads the OPSD CSV, filtering dynamically for Germany columns:
    - Load
    - Optional: Solar, Wind, Temperature
    """
    print(f"Loading data from {raw_file_path}")
    
    # Read headers to identify columns safely
    all_cols = pd.read_csv(raw_file_path, nrows=0).columns.tolist()
    
    # 1. Identify Timestamp Column
    ts_candidates = ['utc_timestamp', 'timestamp', 'datetime']
    ts_col = next((c for c in ts_candidates if c in all_cols), None)
    if not ts_col:
        raise ValueError(f"No timestamp column found among candidates {ts_candidates} in: {all_cols[:10]}")
    
    # 2. Identify Germany DE_ Columns
    de_cols = [c for c in all_cols if c.startswith('DE_')]
    
    # 3. Identify specific Load, Temp, Solar, Wind dynamically
    try:
        load_col = next(c for c in de_cols if 'load' in c.lower() and 'forecast' not in c.lower())
    except StopIteration:
        raise ValueError(f"Could not find a 'load' column belonging to DE_ in: {de_cols}")
        
    temp_col = next((c for c in de_cols if 'temp' in c.lower()), None)
    solar_col = next((c for c in de_cols if 'solar' in c.lower() and 'profile' not in c.lower() and 'capacity' not in c.lower()), None)
    wind_col = next((c for c in de_cols if 'wind' in c.lower() and 'profile' not in c.lower() and 'capacity' not in c.lower() and 'offshore' not in c.lower() and 'onshore' not in c.lower()), None)
    
    cols_to_use = [ts_col, load_col]
    rename_mapping = {ts_col: "timestamp", load_col: "load"}
    
    print(f"-> Selected Timestamp col: '{ts_col}'")
    print(f"-> Selected Load col: '{load_col}'")
    
    if temp_col:
        cols_to_use.append(temp_col)
        rename_mapping[temp_col] = "temperature"
        print(f"-> Selected Temp col: '{temp_col}'")
    if solar_col:
        cols_to_use.append(solar_col)
        rename_mapping[solar_col] = "solar"
        print(f"-> Selected Solar col: '{solar_col}'")
    if wind_col:
        cols_to_use.append(wind_col)
        rename_mapping[wind_col] = "wind"
        print(f"-> Selected Wind col: '{wind_col}'")
        
    df = pd.read_csv(raw_file_path, usecols=cols_to_use)
    df = df.rename(columns=rename_mapping)
    
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses datetime, sets index, sorts, and handles missing values using time interpolation.
    """
    # Parse timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    # Report missing values before interpolation
    missing_before = df.isnull().sum()
    print("Missing values before cleaning:")
    print(missing_before)
    
    # Interpolate using time method (suitable for continuous time series)
    df = df.interpolate(method='time')
    
    print("\nMissing values after time interpolation:")
    print(df.isnull().sum())
    
    return df
