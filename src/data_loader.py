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
    - Load (DE_load_actual_entsoe_transparency)
    - Solar generation (DE_solar_generation_actual)
    - Wind generation (DE_wind_generation_actual)
    - Temperature (DE_temperature)
    """
    print(f"Loading data from {raw_file_path}")
    
    # We will only load the necessary columns to save memory
    cols_to_use = [
        "utc_timestamp",
        "DE_load_actual_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual",
        "DE_temperature"
    ]
    
    df = pd.read_csv(raw_file_path, usecols=cols_to_use)
    
    df = df.rename(columns={
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load",
        "DE_solar_generation_actual": "solar",
        "DE_wind_generation_actual": "wind",
        "DE_temperature": "temperature"
    })
    
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
