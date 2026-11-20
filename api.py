"""Scoring API for ChurnSense:360.

Run locally:   uvicorn api:app --reload --port 8000
Interactive docs:  http://localhost:8000/docs

Endpoints
    GET  /health                 service and model status
    GET  /predict/{customer_id}  score a customer already in the portfolio
    POST /predict                score a known customer ID, or raw feature values

Risk bands (shown as ``risk_band``):
    Low     risk below 20%
    High    risk at or above the model's decision threshold
    Medium  everything in between
"""
from __future__ import annotations

import json
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from reasons import explain_frame, make_explainer

ARTIFACT_DIR = Path(os.getenv('CHURNSENSE_ARTIFACT_DIR', Path(__file__).resolve().parent))
LOW_RISK_CUTOFF = 0.20
DEFAULT_FEATURES = [
    'Total_Spend_6M', 'Total_Orders_6M', 'Average_Order_Value_6M', 'Active_Months_6M',
    'Spend_Momentum', 'Orders_Momentum', 'Recency_Months', 'Spend_Volatility_6M',
]


class Features(BaseModel):
    """Behaviour over the last 6 months, exactly as the model was trained."""

    model_config = ConfigDict(json_schema_extra={'example': {
        'Total_Spend_6M': 420.0, 'Total_Orders_6M': 3, 'Average_Order_Value_6M': 140.0,
        'Active_Months_6M': 2, 'Spend_Momentum': 0.4, 'Orders_Momentum': 0.5,
        'Recency_Months': 4, 'Spend_Volatility_6M': 150.0,
    }})

    Total_Spend_6M: float = Field(ge=0, description='Total spend over the last 6 months (GBP)')
    Total_Orders_6M: float = Field(ge=0, description='Number of orders over the last 6 months')
    Average_Order_Value_6M: float = Field(ge=0, description='Average order value over the last 6 months (GBP)')
    Active_Months_6M: float = Field(ge=0, le=6, description='Months with a purchase, out of the last 6')
    Spend_Momentum: float = Field(ge=0, le=5, description='Recent 6-month spend divided by the previous 6 months (capped at 5)')
    Orders_Momentum: float = Field(ge=0, le=5, description='Recent 6-month orders divided by the previous 6 months (capped at 5)')
    Recency_Months: float = Field(ge=0, description='Months since the last purchase')
    Spend_Volatility_6M: float = Field(ge=0, description='Standard deviation of monthly spend over the last 6 months')


class PredictRequest(BaseModel):
    customer_id: Optional[str] = Field(default=None, description='A customer already in the portfolio')
    features: Optional[Features] = Field(default=None, description='Or raw feature values for any customer')

    @model_validator(mode='after')
    def _exactly_one(self):
        if (self.customer_id is None) == (self.features is None):
            raise ValueError('Send either customer_id or features, not both and not neither.')
        return self


class Reason(BaseModel):
    feature: str
    value: float
    effect: str
    text: str


class PredictResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_flag: bool
    risk_band: str
    decision_threshold: float
    reasons: List[Reason]
    predicted_clv_90d: Optional[float] = Field(default=None, description='Forecast spend over the next 90 days (GBP)')
    expected_revenue_at_risk: Optional[float] = None
    recommended_action: Optional[str] = None
    model: str


class State:
    model = None
    explainer = None
    portfolio: Optional[pd.DataFrame] = None
    features: List[str] = DEFAULT_FEATURES
    threshold: float = 0.5
    model_name: str = 'unknown'
    reference: Dict[str, float] = {}


state = State()


def load_artifacts(directory: Path = ARTIFACT_DIR) -> None:
    diagnostics_path = directory / 'model_diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    state.features = diagnostics.get('feature_columns', DEFAULT_FEATURES)
    state.threshold = float(diagnostics.get('selected_threshold', 0.5))
    state.model_name = str(diagnostics.get('selected_model', 'unknown'))

    with open(directory / 'churn_model.pkl', 'rb') as handle:
        state.model = pickle.load(handle)
    state.explainer = make_explainer(state.model)

    portfolio = pd.read_csv(directory / 'enriched_data.csv')
    portfolio['Customer_ID'] = portfolio['Customer_ID'].astype(str)
    state.portfolio = portfolio.set_index('Customer_ID', drop=False)
    # Portfolio medians let the reason text avoid calling a high value "low".
    state.reference = {c: float(portfolio[c].median()) for c in state.features if c in portfolio.columns}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title='ChurnSense:360 Scoring API',
    description='Churn probability, plain-English reasons and 90-day value for a customer.',
    version='1.0.0',
    lifespan=lifespan,
)


def _risk_band(probability: float) -> str:
    if probability < LOW_RISK_CUTOFF:
        return 'Low'
    if probability >= state.threshold:
        return 'High'
    return 'Medium'


def _score(frame: pd.DataFrame) -> tuple[float, List[Dict[str, object]]]:
    probability = float(state.model.predict_proba(frame)[:, 1][0])
    reasons = explain_frame(state.explainer, frame, n=3, reference=state.reference)[0]
    return probability, reasons


@app.get('/health')
def health() -> Dict[str, object]:
    return {
        'status': 'ok' if state.model is not None else 'loading',
        'model': state.model_name,
        'decision_threshold': state.threshold,
        'customers_loaded': 0 if state.portfolio is None else int(len(state.portfolio)),
        'features': state.features,
    }


def _predict_known_customer(customer_id: str) -> PredictResponse:
    key = str(customer_id).strip()
    if state.portfolio is None or key not in state.portfolio.index:
        raise HTTPException(status_code=404, detail=f'Customer {key!r} not found in the portfolio.')
    row = state.portfolio.loc[key]
    frame = pd.DataFrame([[float(row[column]) for column in state.features]], columns=state.features)
    probability, reasons = _score(frame)
    clv = float(row['Predicted_CLV']) if 'Predicted_CLV' in row else None
    return PredictResponse(
        customer_id=key,
        churn_probability=probability,
        churn_flag=probability >= state.threshold,
        risk_band=_risk_band(probability),
        decision_threshold=state.threshold,
        reasons=[Reason(**{k: r[k] for k in ('feature', 'value', 'effect', 'text')}) for r in reasons],
        predicted_clv_90d=clv,
        expected_revenue_at_risk=None if clv is None else probability * clv,
        recommended_action=str(row['Recommended_Action']) if 'Recommended_Action' in row else None,
        model=state.model_name,
    )


@app.get('/predict/{customer_id}', response_model=PredictResponse)
def predict_by_id(customer_id: str) -> PredictResponse:
    return _predict_known_customer(customer_id)


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if request.customer_id is not None:
        return _predict_known_customer(request.customer_id)

    values = request.features.model_dump()
    frame = pd.DataFrame([[values[column] for column in state.features]], columns=state.features)
    probability, reasons = _score(frame)
    return PredictResponse(
        churn_probability=probability,
        churn_flag=probability >= state.threshold,
        risk_band=_risk_band(probability),
        decision_threshold=state.threshold,
        reasons=[Reason(**{k: r[k] for k in ('feature', 'value', 'effect', 'text')}) for r in reasons],
        model=state.model_name,
    )
