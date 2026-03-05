# Energy Demand Forecast

A portfolio-grade time series project forecasting energy demand, utilizing the Open Power System Data (OPSD) dataset for Germany.

## Google Colab Setup Instructions

This project is built to run entirely inside Google Colab, leveraging Google Drive for data and figure persistent storage.

To reproduce this project, please follow these steps:

1. **Open Google Colab:** Navigate to [Google Colab](https://colab.research.google.com/) and start a new Notebook or upload the notebooks from this repository.
2. **Mount Google Drive:** We use Google Drive to save the downloaded dataset and the generated plots/figures. This is handled automatically by the notebooks, but requires you to authenticate.
3. **Clone the Repository in Colab:**
   Inside the notebook you will see a cell that clones this repository directly into your Colab environment:
   ```python
   !git clone https://github.com/<USERNAME>/energy-demand-forecast.git
   %cd energy-demand-forecast
   ```
   *Note: Replace `<USERNAME>` with your GitHub username if you fork the repository.*
4. **Run Notebooks:** 
   - Open `notebooks/01_data_exploration.ipynb` and run it top-to-bottom for Data Extraction and basic EDA.
   - Open `notebooks/02_feature_engineering.ipynb` and run it top-to-bottom to build structural models, outputting processed target structures natively matching predictive pipelines natively towards Google Drive parquets.
   - Open `notebooks/03_modeling.ipynb` to evaluate and compare multiple predictive TS architectures natively producing final residual figures and metric benchmarks systematically back into Google Drive.
   - Open `notebooks/04_interpretability_diagnostics.ipynb` to natively dive into XGBoost feature drop-off maps natively logging precise bounding logic generating comprehensive worst-case analytics automatically sent back towards Google Drive.
   - Open `notebooks/05_statistical_validation.ipynb` to execute rolling-origin backtesting, analyze residual distributions, compute sequence stationarity statistically, map error segments, and calculate conformal prediction limits bounding uncertainty efficiently inside Drive properties.

## Data and Figures Storage

- Data and results will be stored dynamically in your Google Drive at the path `/content/drive/MyDrive/energy-demand-forecast`.
- Raw data will be downloaded to `data/raw/` inside the Drive root if it doesn't already exist.
- Generated figures are automatically saved as PNGs under `results/figures/` in the same Drive root.
