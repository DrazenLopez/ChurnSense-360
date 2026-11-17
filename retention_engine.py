from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    'Total_Spend_6M',
    'Total_Orders_6M',
    'Average_Order_Value_6M',
    'Active_Months_6M',
    'Spend_Momentum',
    'Orders_Momentum',
    'Recency_Months',
    'Spend_Volatility_6M',
]

MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()


def load_transactions(file_path: str = 'online_retail_II.xlsx') -> pd.DataFrame:
    csv_path = 'fast_retail_data.csv'

    if pd.io.common.file_exists(csv_path):
        raw = pd.read_csv(csv_path)
        raw = raw.dropna(subset=['Customer ID']).copy()
        raw['Customer ID'] = pd.to_numeric(raw['Customer ID'], errors='coerce').astype('Int64').astype(str)
        raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'])
        raw['Quantity'] = pd.to_numeric(raw['Quantity'], errors='coerce').fillna(0)
        raw['Price'] = pd.to_numeric(raw['Price'], errors='coerce').fillna(0)
        raw['Invoice'] = pd.to_numeric(raw['Invoice'], errors='coerce').fillna(0).astype(int)
        raw['Total_Value'] = raw['Quantity'] * raw['Price']
        raw = raw[(raw['Quantity'] > 0) & (raw['Price'] > 0) & (raw['Total_Value'] > 0)].copy()
        return raw

    raw = pd.concat(
        [
            pd.read_excel(file_path, sheet_name='Year 2009-2010'),
            pd.read_excel(file_path, sheet_name='Year 2010-2011')
        ],
        ignore_index=True,
    )
    raw = raw.dropna(subset=['Customer ID']).copy()
    raw['Customer ID'] = raw['Customer ID'].astype(float).astype(int).astype(str)
    raw['InvoiceDate'] = pd.to_datetime(raw['InvoiceDate'])
    raw['Total_Value'] = pd.to_numeric(raw['Quantity'], errors='coerce').fillna(0) * pd.to_numeric(raw['Price'], errors='coerce').fillna(0)
    raw['Quantity'] = pd.to_numeric(raw['Quantity'], errors='coerce').fillna(0)
    raw['Price'] = pd.to_numeric(raw['Price'], errors='coerce').fillna(0)
    raw['Invoice'] = raw.get('Invoice', np.arange(len(raw)))
    raw = raw[(raw['Quantity'] > 0) & (raw['Price'] > 0) & (raw['Total_Value'] > 0)].copy()
    return raw


def _vectorized_recency(purchase_flags: np.ndarray) -> np.ndarray:
    recency = np.full(len(purchase_flags), np.nan, dtype=float)
    last_seen = -1
    for index, flag in enumerate(purchase_flags):
        if flag > 0:
            last_seen = index
            recency[index] = 0.0
        elif last_seen >= 0:
            recency[index] = float(index - last_seen)
    return recency


def _future_window_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    if len(values) == 0:
        return values.astype(float)
    cumulative = np.concatenate(([0.0], np.cumsum(values.astype(float))))
    indexes = np.arange(len(values))
    end_indexes = np.minimum(indexes + 1 + horizon, len(values))
    return cumulative[end_indexes] - cumulative[indexes + 1]


def build_customer_month_panel(
    transactions: pd.DataFrame,
    lookback_months: int = 6,
    outcome_months: int = 3,
) -> pd.DataFrame:
    monthly = (
        transactions.assign(Month=transactions['InvoiceDate'].dt.to_period('M').dt.to_timestamp())
        .groupby(['Customer ID', 'Month'], as_index=False)
        .agg(
            Total_Spend=('Total_Value', 'sum'),
            Total_Orders=('Invoice', 'nunique'),
            Total_Units=('Quantity', 'sum'),
            Active_Days=('InvoiceDate', lambda series: series.dt.date.nunique()),
            Last_Purchase_Date=('InvoiceDate', 'max'),
        )
        .sort_values(['Customer ID', 'Month'])
    )

    all_months = pd.period_range(
        start=monthly['Month'].min().to_period('M'),
        end=monthly['Month'].max().to_period('M'),
        freq='M',
    ).to_timestamp()
    all_customers = monthly['Customer ID'].drop_duplicates().sort_values().tolist()

    full_index = pd.MultiIndex.from_product(
        [all_customers, all_months],
        names=['Customer ID', 'Month'],
    )

    panel = (
        monthly.set_index(['Customer ID', 'Month'])
        .reindex(full_index)
        .reset_index()
        .sort_values(['Customer ID', 'Month'])
    )
    panel['Total_Spend'] = panel['Total_Spend'].fillna(0.0)
    panel['Total_Orders'] = panel['Total_Orders'].fillna(0.0)
    panel['Total_Units'] = panel['Total_Units'].fillna(0.0)
    panel['Active_Days'] = panel['Active_Days'].fillna(0.0)
    panel['Active_Flag'] = (panel['Total_Spend'] > 0).astype(int)

    feature_rows: List[pd.DataFrame] = []
    for customer_id, group in panel.groupby('Customer ID', sort=False):
        group = group.sort_values('Month').copy()
        spend = group['Total_Spend'].to_numpy(dtype=float)
        orders = group['Total_Orders'].to_numpy(dtype=float)
        active_flag = group['Active_Flag'].to_numpy(dtype=int)

        recent_spend = pd.Series(spend).rolling(lookback_months, min_periods=1).sum().to_numpy()
        prior_spend = pd.Series(spend).shift(lookback_months).rolling(lookback_months, min_periods=1).sum().fillna(0).to_numpy()
        recent_orders = pd.Series(orders).rolling(lookback_months, min_periods=1).sum().to_numpy()
        prior_orders = pd.Series(orders).shift(lookback_months).rolling(lookback_months, min_periods=1).sum().fillna(0).to_numpy()
        rolling_spend_std = pd.Series(spend).rolling(lookback_months, min_periods=2).std().fillna(0).to_numpy()
        active_months = pd.Series(active_flag).rolling(lookback_months, min_periods=1).sum().to_numpy()
        recency = _vectorized_recency(active_flag)
        future_spend = _future_window_sum(spend, outcome_months)
        future_available = np.arange(len(group)) <= (len(group) - outcome_months - 1)

        group['Total_Spend_6M'] = recent_spend
        group['Total_Orders_6M'] = recent_orders
        group['Average_Order_Value_6M'] = np.divide(recent_spend, np.maximum(recent_orders, 1.0))
        group['Active_Months_6M'] = active_months

        # Stabilize momentum ratios: when prior windows are near zero,
        # use bounded defaults to prevent extreme values that can distort model learning.
        prior_spend_safe = np.where(prior_spend >= 1.0, prior_spend, np.nan)
        prior_orders_safe = np.where(prior_orders >= 1.0, prior_orders, np.nan)

        spend_momentum = np.divide(recent_spend, prior_spend_safe)
        orders_momentum = np.divide(recent_orders, prior_orders_safe)

        spend_momentum = np.where(np.isnan(spend_momentum), np.where(recent_spend > 0, 1.0, 0.0), spend_momentum)
        orders_momentum = np.where(np.isnan(orders_momentum), np.where(recent_orders > 0, 1.0, 0.0), orders_momentum)

        group['Spend_Momentum'] = np.clip(spend_momentum, 0.0, 5.0)
        group['Orders_Momentum'] = np.clip(orders_momentum, 0.0, 5.0)
        group['Recency_Months'] = recency
        group['Spend_Volatility_6M'] = rolling_spend_std
        group['Future_Spend_3M'] = future_spend
        group['Is_Leaving'] = ((future_spend <= 0) & future_available).astype(int)
        group['Has_Label'] = future_available.astype(int)

        feature_rows.append(group)

    panel_features = pd.concat(feature_rows, ignore_index=True)
    panel_features['Recency_Months'] = panel_features['Recency_Months'].fillna(panel_features['Recency_Months'].max())
    panel_features['Spend_Volatility_6M'] = panel_features['Spend_Volatility_6M'].fillna(0.0)
    panel_features['Average_Order_Value_6M'] = panel_features['Average_Order_Value_6M'].fillna(0.0)
    panel_features['Spend_Momentum'] = panel_features['Spend_Momentum'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    panel_features['Orders_Momentum'] = panel_features['Orders_Momentum'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return panel_features


def split_temporally(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    labeled = panel[panel['Has_Label'] == 1].copy()
    labeled = labeled.dropna(subset=FEATURE_COLUMNS + ['Is_Leaving'])
    labeled['Month'] = pd.to_datetime(labeled['Month'])
    months = sorted(labeled['Month'].drop_duplicates())
    if len(months) < 6:
        raise ValueError('Not enough monthly snapshots to perform temporal splitting.')

    train_end = max(1, int(len(months) * 0.6))
    val_end = max(train_end + 1, int(len(months) * 0.8))
    if val_end >= len(months):
        val_end = len(months) - 1
        train_end = max(1, val_end - 1)

    train_months = set(months[:train_end])
    val_months = set(months[train_end:val_end])
    test_months = set(months[val_end:])

    train_df = labeled[labeled['Month'].isin(train_months)].copy()
    val_df = labeled[labeled['Month'].isin(val_months)].copy()
    test_df = labeled[labeled['Month'].isin(test_months)].copy()

    split_info = {
        'train_start': str(min(train_months).date()),
        'train_end': str(max(train_months).date()),
        'val_start': str(min(val_months).date()),
        'val_end': str(max(val_months).date()),
        'test_start': str(min(test_months).date()),
        'test_end': str(max(test_months).date()),
    }

    return train_df, val_df, test_df, split_info


def _limit_frame_size(frame: pd.DataFrame, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()

    if 'Is_Leaving' in frame.columns and frame['Is_Leaving'].nunique() > 1:
        sampled, _ = train_test_split(
            frame,
            train_size=max_rows,
            stratify=frame['Is_Leaving'],
            random_state=random_state,
        )
        return sampled.copy()

    return frame.sample(n=max_rows, random_state=random_state).copy()


def build_model_candidates(random_state: int = 42) -> Dict[str, object]:
    return {
        'logistic_regression': make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state),
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1,
        ),
        'xgboost': XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_alpha=0.0,
            reg_lambda=1.0,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def _get_estimator_from_pipeline(model):
    if isinstance(model, Pipeline):
        return model.named_steps[model.steps[-1][0]]
    return model


def evaluate_classifier(model, X, y) -> Dict[str, object]:
    proba = model.predict_proba(X)[:, 1]
    threshold = 0.5
    predictions = (proba >= threshold).astype(int)
    brier = float(brier_score_loss(y, proba))
    metrics = {
        'roc_auc': float(roc_auc_score(y, proba)),
        'pr_auc': float(average_precision_score(y, proba)),
        'brier_score': brier,
        'log_loss': float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, predictions).tolist(),
        'probabilities': proba.tolist(),
        'calibration_curve': calibration_curve_data(y, proba),
    }
    best_threshold, threshold_metrics = optimize_threshold(y, proba)
    metrics['best_threshold'] = best_threshold
    metrics['threshold_metrics'] = threshold_metrics
    return metrics


def calibration_curve_data(y_true, proba):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy='quantile')
    return {
        'mean_predicted_value': [float(x) for x in mean_pred],
        'fraction_of_positives': [float(x) for x in frac_pos],
    }


def optimize_threshold(y_true, proba):
    thresholds = np.unique(np.round(proba, 5))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = {}
    for threshold in thresholds:
        predictions = (proba >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
            best_metrics = {
                'precision': float(precision_score(y_true, predictions, zero_division=0)),
                'recall': float(recall_score(y_true, predictions, zero_division=0)),
                'f1': float(f1),
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist(),
            }
    return best_threshold, best_metrics


def select_best_model(model_reports: Dict[str, Dict[str, object]]) -> str:
    def sort_key(item):
        report = item[1]
        return (report['validation']['pr_auc'], -report['validation']['brier_score'])

    return max(model_reports.items(), key=sort_key)[0]


def fit_clv_model(transactions: pd.DataFrame, cutoff_date: pd.Timestamp, prediction_days: int = 90) -> Tuple[pd.DataFrame, Dict[str, object]]:
    history = transactions[transactions['InvoiceDate'] <= cutoff_date].copy()
    # Returns/credit notes carry negative values and break the Gamma-Gamma spend model; fit on purchases only.
    history = history[history['Total_Value'] > 0]
    summary = summary_data_from_transaction_data(
        history,
        customer_id_col='Customer ID',
        datetime_col='InvoiceDate',
        monetary_value_col='Total_Value',
        observation_period_end=cutoff_date,
        freq='D',
        include_first_transaction=False,
    ).reset_index()

    rename_map = {}
    for candidate in ['customer_id', 'Customer ID', 'Customer_ID', 'index']:
        if candidate in summary.columns:
            rename_map[candidate] = 'Customer_ID'
            break
    summary = summary.rename(columns=rename_map)

    if summary.empty:
        return pd.DataFrame(columns=['Customer_ID', 'Predicted_CLV', 'Expected_Purchases', 'CLV_Segment']), {
            'status': 'unavailable'
        }

    positive_monetary = summary.loc[summary['frequency'] > 0, 'monetary_value']
    fallback_monetary = float(positive_monetary.median()) if not positive_monetary.empty else float(summary['monetary_value'].replace(0, np.nan).median())
    if np.isnan(fallback_monetary) or fallback_monetary <= 0:
        fallback_monetary = float(summary['monetary_value'].replace(0, np.nan).mean()) if summary['monetary_value'].replace(0, np.nan).notna().any() else 1.0

    summary['monetary_value'] = summary['monetary_value'].fillna(fallback_monetary).replace(0, fallback_monetary)

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

    repeat_customers = summary[summary['frequency'] > 0].copy()
    if repeat_customers.empty:
        repeat_customers = summary.copy()
        repeat_customers['monetary_value'] = fallback_monetary

    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(repeat_customers['frequency'], repeat_customers['monetary_value'])

    summary['Expected_Purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        prediction_days,
        summary['frequency'],
        summary['recency'],
        summary['T'],
    )
    summary['Expected_AOV'] = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])

    horizon_discount = 1.0 / ((1.0 + 0.01) ** (prediction_days / 365.25))
    summary['Predicted_CLV'] = summary['Expected_Purchases'] * summary['Expected_AOV'] * horizon_discount

    valid_clv = summary['Predicted_CLV'].replace([np.inf, -np.inf], np.nan)
    summary['Predicted_CLV'] = valid_clv.fillna(valid_clv.median()).clip(lower=0.0)
    summary['CLV_Segment'] = pd.qcut(summary['Predicted_CLV'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])

    diagnostics = {
        'status': 'fit',
        'customer_count': int(len(summary)),
        'repeat_customer_count': int(len(repeat_customers)),
        'prediction_days': prediction_days,
        'method': 'expected_purchases_times_expected_aov_discounted',
        'predicted_clv_min': float(summary['Predicted_CLV'].min()),
        'predicted_clv_median': float(summary['Predicted_CLV'].median()),
        'predicted_clv_max': float(summary['Predicted_CLV'].max()),
    }

    return summary[['Customer_ID', 'frequency', 'recency', 'T', 'monetary_value', 'Expected_Purchases', 'Expected_AOV', 'Predicted_CLV', 'CLV_Segment']], diagnostics


def build_strategy_table(
    merged: pd.DataFrame,
    risk_quantiles: Tuple[float, float] = (0.33, 0.66),
) -> pd.DataFrame:
    frame = merged.copy()
    frame['Risk_Norm'] = frame['Risk_Score'].rank(pct=True)
    frame['CLV_Norm'] = frame['Predicted_CLV'].rank(pct=True)
    frame['Momentum_Norm'] = frame['Spend_Momentum'].replace([np.inf, -np.inf], np.nan).fillna(0).rank(pct=True)
    frame['Recency_Norm'] = frame['Recency_Months'].rank(pct=True)
    frame['Order_Norm'] = frame['Total_Orders_6M'].rank(pct=True)

    action_scores = []
    for _, row in frame.iterrows():
        risk = float(row['Risk_Norm'])
        clv = float(row['CLV_Norm'])
        momentum = float(row['Momentum_Norm'])
        recency = float(row['Recency_Norm'])
        orders = float(row['Order_Norm'])

        scores = {
            'High-Touch Recovery': 0.45 * risk + 0.35 * clv + 0.20 * recency,
            'Targeted Incentive': 0.42 * risk + 0.28 * clv + 0.30 * (1 - momentum),
            'Loyalty Growth': 0.48 * clv + 0.27 * (1 - risk) + 0.25 * orders,
            'Automated Nurture': 0.35 * (1 - risk) + 0.30 * momentum + 0.35 * (1 - recency),
        }
        best_action = max(scores, key=scores.get)
        best_score = float(scores[best_action])
        action_scores.append((best_action, best_score, scores))

    frame['Recommended_Action'] = [item[0] for item in action_scores]
    frame['Action_Score'] = [item[1] for item in action_scores]
    frame['Action_Score_Details'] = [item[2] for item in action_scores]
    frame['Expected_Revenue_at_Risk'] = frame['Risk_Score'] * frame['Predicted_CLV']
    frame['Priority_Score'] = frame['Expected_Revenue_at_Risk'].rank(pct=True)

    def explain(row):
        if row['Recommended_Action'] == 'High-Touch Recovery':
            driver = 'recent disengagement and high financial exposure'
        elif row['Recommended_Action'] == 'Targeted Incentive':
            driver = 'risk escalation with slowing momentum'
        elif row['Recommended_Action'] == 'Loyalty Growth':
            driver = 'high value with manageable churn exposure'
        else:
            driver = 'moderate risk with room for low-cost reactivation'
        return f"Recommended because the account shows {driver}."

    frame['Recommendation_Reason'] = frame.apply(explain, axis=1)
    return frame
