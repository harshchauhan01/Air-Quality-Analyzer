# Air Quality Analyzer – CA2 Project

## Overview
- End-to-end pipeline for Indian air-quality monitoring and forecasting.
- Pulls live AQI data from data.gov.in, stores consolidated CSVs, and powers an interactive Streamlit dashboard.
- Includes notebooks for exploration/modeling and a packaged `model_artifacts.pkl` used by the app for predictions.

## Project Structure
- `fetch_weather_data.py` – hourly/adhoc ingest from the Government AQI API into `weather_data_v2.csv` (appends, keeps header on first run).
- `weather_data*.csv` – historical datasets consumed by the dashboard (and example model inputs).
- `ML_Model/app.py` – Streamlit dashboard with prediction, analytics, city/state comparisons, and download options (loads `model_artifacts.pkl`).
- `Air Quality Forecasting and Alert System for Indian Cities/` – notebook(s) for exploratory analysis/feature engineering.
- `ML_Model/main_v2.ipynb` – model training and evaluation workflow.
- `requirements.txt` – minimal deps for data pull; dashboard needs extra libs noted below.

## Quickstart (Windows PowerShell)
```powershell
# create and activate virtual env
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# base deps for ingest + dashboard
pip install -r requirements.txt streamlit plotly numpy

# run data ingest (optional; updates weather_data_v2.csv)
python fetch_weather_data.py

# launch dashboard (from repo root)
cd ML_Model; streamlit run app.py
```

## Data Ingestion
- Uses API `https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69` with provided key in the script.
- Appends new pulls to `weather_data_v2.csv`; does not overwrite existing data.
- Add scheduling (e.g., Task Scheduler/cron) for continuous updates.

## Dashboard Features (Streamlit)
- Real-time data load: historical CSVs + live API pull (with caching and refresh control).
- AQI computation per Indian standards; category badges and health advisories.
- Filters by date range, pollutant, data source; export filtered CSV.
- Pages: Home overview, AQI prediction, analytics, city comparison, state trends, About.

## Notebooks
- Use the notebooks to inspect data quality, experiment with features, and retrain models. Export updated artifacts to `ML_Model/model_artifacts.pkl` for the app.

## Notes & Next Steps
- Keep `model_artifacts.pkl` in `ML_Model/` alongside `app.py` when deploying/running.
- If you extend features, add any new Python deps to `requirements.txt` and re-install.
