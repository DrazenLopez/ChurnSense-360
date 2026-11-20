"""Plain-English churn reasons, shared by the scoring API and the dashboard.

SHAP values say how much each feature pushed one customer's risk score up or down.
This module turns the biggest pushes into sentences a retention team can read.
A positive SHAP value means the feature *raised* the predicted churn risk.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


def _plural(value: float, singular: str) -> str:
    return f"{value:.0f} {singular}" if round(value) == 1 else f"{value:.0f} {singular}s"


LABELS = {
    'Recency_Months': 'Time since last purchase',
    'Active_Months_6M': 'Active months in the last 6',
    'Total_Spend_6M': 'Spend over the last 6 months',
    'Total_Orders_6M': 'Orders over the last 6 months',
    'Average_Order_Value_6M': 'Average order value',
    'Spend_Momentum': 'Spend trend',
    'Orders_Momentum': 'Order trend',
    'Spend_Volatility_6M': 'Month-to-month spend swings',
}
# For these a HIGHER value is healthier; for Recency and Volatility a higher value is worse.
HIGHER_IS_HEALTHIER = {
    'Total_Spend_6M', 'Total_Orders_6M', 'Average_Order_Value_6M',
    'Active_Months_6M', 'Spend_Momentum', 'Orders_Momentum',
}


def _value_text(feature: str, v: float) -> str:
    if feature in ('Total_Spend_6M', 'Average_Order_Value_6M', 'Spend_Volatility_6M'):
        return f"£{v:,.0f}"
    if feature in ('Spend_Momentum', 'Orders_Momentum'):
        return f"{v:.0%} of the previous 6 months"
    if feature == 'Recency_Months':
        return _plural(v, 'month') + ' ago'
    return f"{v:.0f}"


def _is_aligned(feature: str, value: float, raises: bool, reference: float) -> bool:
    """Does the raw value point the same way as the SHAP effect? (e.g. low spend raising risk)"""
    higher_is_healthier = feature in HIGHER_IS_HEALTHIER
    # A value equal to the median counts as "risky-looking" so ties keep the direct wording.
    looks_risky = value <= reference if higher_is_healthier else value >= reference
    return looks_risky if raises else (not looks_risky)


def describe_feature(feature: str, value: float, shap_value: float, reference: Optional[float] = None) -> str:
    """One readable sentence for one feature of one customer.

    ``reference`` is the portfolio median for the feature. When the value does not point the same way
    as the model's push (e.g. high spend that still raises risk because of other signals), the sentence
    says so instead of calling a high number "low".
    """
    raises = shap_value > 0
    v = float(value)

    if reference is not None and feature in LABELS and not _is_aligned(feature, v, raises, float(reference)):
        tail = ('is adding to the risk in combination with other signals' if raises
                else 'is a point in this customer\'s favour')
        return f"{LABELS[feature]} ({_value_text(feature, v)}) {tail}"

    if feature == 'Recency_Months':
        if raises:
            return f"Last purchase was {_plural(v, 'month')} ago"
        return "Bought this month" if v < 1 else f"Bought recently ({_plural(v, 'month')} ago)"
    if feature == 'Active_Months_6M':
        if raises:
            return f"Bought in only {v:.0f} of the last 6 months"
        return f"Bought in {v:.0f} of the last 6 months"
    if feature == 'Total_Spend_6M':
        return (f"Low spend over the last 6 months (£{v:,.0f})" if raises
                else f"Strong spend over the last 6 months (£{v:,.0f})")
    if feature == 'Total_Orders_6M':
        return (f"Few orders in the last 6 months ({v:.0f})" if raises
                else f"Frequent orders in the last 6 months ({v:.0f})")
    if feature == 'Average_Order_Value_6M':
        return (f"Small average order (£{v:,.0f})" if raises
                else f"Healthy average order (£{v:,.0f})")
    if feature == 'Spend_Momentum':
        return (f"Spend trend is weak ({v:.0%} of the previous 6 months)" if raises
                else f"Spend is holding up or growing ({v:.0%} of the previous 6 months)")
    if feature == 'Orders_Momentum':
        return (f"Order trend is weak ({v:.0%} of the previous 6 months)" if raises
                else f"Orders are holding up or growing ({v:.0%} of the previous 6 months)")
    if feature == 'Spend_Volatility_6M':
        return "Erratic month-to-month spending" if raises else "Steady month-to-month spending"

    label = feature.replace('_', ' ')
    return f"{label} = {v:,.2f} ({'raises' if raises else 'lowers'} risk)"


def top_reasons(
    feature_values: Sequence[float],
    shap_values: Sequence[float],
    feature_names: Sequence[str],
    n: int = 3,
    reference: Optional[Dict[str, float]] = None,
) -> List[Dict[str, object]]:
    """The n features that moved this customer's score the most, with readable text."""
    shap_array = np.asarray(shap_values, dtype=float)
    order = np.argsort(-np.abs(shap_array), kind='mergesort')[:n]
    reasons = []
    for index in order:
        feature = feature_names[int(index)]
        reasons.append({
            'feature': feature,
            'value': float(feature_values[int(index)]),
            'effect': 'raises risk' if shap_array[int(index)] > 0 else 'lowers risk',
            'shap_value': float(shap_array[int(index)]),
            'text': describe_feature(
                feature, feature_values[int(index)], shap_array[int(index)],
                reference=None if reference is None else reference.get(feature),
            ),
        })
    return reasons


def make_explainer(model):
    """SHAP explainer for the tree model (imported lazily so tests of the text logic stay light)."""
    import shap

    return shap.TreeExplainer(model)


def shap_matrix(explainer, features: pd.DataFrame) -> np.ndarray:
    """SHAP values for the churn class as a 2-D array (rows x features)."""
    raw = explainer.shap_values(features)
    if isinstance(raw, list):
        raw = raw[-1]
    values = np.asarray(raw)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def explain_frame(
    explainer,
    features: pd.DataFrame,
    n: int = 3,
    reference: Optional[Dict[str, float]] = None,
) -> List[List[Dict[str, object]]]:
    """Top-n readable reasons for every row of ``features``. ``reference`` = portfolio medians."""
    values = shap_matrix(explainer, features)
    names = list(features.columns)
    matrix = features.to_numpy(dtype=float)
    return [top_reasons(matrix[i], values[i], names, n=n, reference=reference) for i in range(len(features))]


def join_reasons(reasons: Iterable[Dict[str, object]]) -> str:
    return '; '.join(str(item['text']) for item in reasons)
