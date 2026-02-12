import os
import pickle
import json

import pandas as pd

from retention_engine import FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS



def load_serving_artifacts(data_path='enriched_data.csv', model_path='churn_model.pkl'):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data artifact not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    data = pd.read_csv(data_path)

    diagnostics_path = 'model_diagnostics.json'
    feature_columns = list(DEFAULT_FEATURE_COLUMNS)
    if os.path.exists(diagnostics_path):
        try:
            with open(diagnostics_path, 'r') as f:
                diagnostics = json.load(f)
            feature_columns = diagnostics.get('feature_columns', feature_columns)
        except Exception:
            feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    missing = [col for col in feature_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = data[feature_columns].copy()
    for col in feature_columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
    features = features.fillna(0)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return data, features, model