# ChurnSense:360

Enterprise-style churn intelligence dashboard built with Streamlit.

### 1. Open PowerShell in the project folder
Use the extracted folder where `app.py` is located.

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard
```powershell
streamlit run app.py
```

### 5. Login
## Overview
ChurnSense:360 is a customer retention intelligence platform that combines:

- churn risk prediction,
- explainability and driver analysis,
- revenue exposure monitoring,
- intervention simulation,
- and model diagnostics for back-end engineering visibility.

The application is designed to run directly from pre-generated artifacts, with an optional rebuild path when retraining is required.

## Project Structure
- Dashboard: `app.py`
- Training pipeline: `train.py`
- Feature/model engine: `retention_engine.py`
- Model and serving artifacts: CSV/JSON/PKL files in the project root

## Pre-Generated Artifacts
The repository includes ready-to-run artifacts such as:

- `enriched_data.csv`
- `churn_model.pkl`
- `model_diagnostics.json`
- `training_baseline.json`
- `customer_monthly_history.csv`
- `customer_monthly_history.parquet`
- `feature_importance.json`
- `shap_summary.json`
- `clv_diagnostics.json`
- `model_comparison.csv`
- `marketing_strategy.csv`

## Authentication
Default login:

- Username: `admin`
- Password: `123`

Optional environment variable override:

- `CHURNSENSE_USERNAME`
- `CHURNSENSE_PASSWORD`

## Run Instructions

### Windows (PowerShell)

1. Open PowerShell in the project folder.
2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the dashboard:

```powershell
streamlit run app.py
```

### Linux / macOS

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run app.py
```

## Optional Rebuild
If you want to regenerate model artifacts before running the app:

```bash
python train.py
streamlit run app.py
```
