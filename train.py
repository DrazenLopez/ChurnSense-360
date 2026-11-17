from __future__ import annotations

import json
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

from retention_engine import (
    FEATURE_COLUMNS,
    build_customer_month_panel,
    build_model_candidates,
    _limit_frame_size,
    build_strategy_table,
    evaluate_classifier,
    fit_clv_model,
    load_transactions,
    select_best_model,
    split_temporally,
)


OUTPUT_FILES = [
    'enriched_data.csv',
    'customer_monthly_history.csv',
    'customer_monthly_history.parquet',
    'training_baseline.json',
    'model_diagnostics.json',
    'feature_importance.json',
    'shap_summary.json',
    'clv_diagnostics.json',
    'marketing_strategy.csv',
    'churn_model.pkl',
    'model_comparison.csv',
]


def _serialise_model_report(report: dict) -> dict:
    serialised = deepcopy(report)
    for split_name in ['train', 'validation', 'test']:
        if split_name in serialised:
            serialised[split_name].pop('probabilities', None)
    return serialised


def _extract_feature_importance(model, model_name: str, feature_names: list[str]) -> dict:
    estimator = model
    if hasattr(model, 'named_steps'):
        estimator = model.steps[-1][1]

    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        return {
            feature: float(value)
            for feature, value in sorted(
                zip(feature_names, importances.tolist()),
                key=lambda pair: pair[1],
                reverse=True,
            )
        }

    if hasattr(estimator, 'coef_'):
        coefficients = np.abs(np.asarray(estimator.coef_).ravel())
        return {
            feature: float(value)
            for feature, value in sorted(
                zip(feature_names, coefficients.tolist()),
                key=lambda pair: pair[1],
                reverse=True,
            )
        }

    return {feature: 0.0 for feature in feature_names}


def _compute_shap_reasons(model, feature_matrix: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(feature_matrix)
    shap_values = np.array(shap_values_raw)

    if shap_values.ndim == 3:
        shap_values = shap_values[0]

    reason_texts: list[str] = []
    for index in range(len(feature_matrix)):
        top_feature_index = int(np.argmax(np.abs(shap_values[index])))
        top_feature = feature_matrix.columns[top_feature_index].replace('_', ' ')
        reason_texts.append(f"Primary driver: {top_feature}")

    return shap_values, reason_texts


def _compute_global_shap_summary(feature_matrix: pd.DataFrame, shap_values: np.ndarray) -> list[dict]:
    if shap_values.ndim == 3:
        shap_values = shap_values[0]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    summary_rows = []
    for feature_name, feature_values in zip(feature_matrix.columns, shap_values.T):
        mean_abs_shap = float(np.mean(np.abs(feature_values)))
        mean_signed_shap = float(np.mean(feature_values))
        direction = 'positive' if mean_signed_shap > 0 else 'negative' if mean_signed_shap < 0 else 'neutral'
        summary_rows.append(
            {
                'feature': feature_name,
                'mean_abs_shap': mean_abs_shap,
                'mean_signed_shap': mean_signed_shap,
                'direction': direction,
            }
        )

    summary_rows.sort(key=lambda item: item['mean_abs_shap'], reverse=True)
    return summary_rows


def _rebuild_customer_history(transactions: pd.DataFrame) -> pd.DataFrame:
    monthly_history = (
        transactions.assign(Month=transactions['InvoiceDate'].dt.to_period('M').dt.to_timestamp())
        .groupby(['Customer ID', 'Month'], as_index=False)['Total_Value']
        .sum()
        .rename(columns={'Customer ID': 'Customer_ID', 'Total_Value': 'Monthly_Spend'})
        .sort_values(['Customer_ID', 'Month'])
    )
    return monthly_history


def run_engine():
    started_at = time.perf_counter()

    file_path = 'online_retail_II.xlsx'
    if not Path(file_path).exists():
        print('Error: Target Excel file not found.')
        return

    print('Status: Ingesting retail transactions...', flush=True)
    stage_at = time.perf_counter()
    transactions = load_transactions(file_path)
    print(f'Status: Ingested transactions in {time.perf_counter() - stage_at:.1f}s', flush=True)

    print('Status: Building temporal customer panel...', flush=True)
    stage_at = time.perf_counter()
    panel = build_customer_month_panel(transactions, lookback_months=6, outcome_months=3)
    print(f'Status: Built temporal customer panel in {time.perf_counter() - stage_at:.1f}s', flush=True)

    stage_at = time.perf_counter()
    train_df, val_df, test_df, split_info = split_temporally(panel)
    print(f'Status: Temporal split ready in {time.perf_counter() - stage_at:.1f}s', flush=True)

    stage_at = time.perf_counter()
    train_df = _limit_frame_size(train_df, max_rows=20000, random_state=42)
    val_df = _limit_frame_size(val_df, max_rows=10000, random_state=42)
    test_df = _limit_frame_size(test_df, max_rows=10000, random_state=42)
    print(
        'Status: Capped split sizes to '
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} in {time.perf_counter() - stage_at:.1f}s",
        flush=True,
    )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df['Is_Leaving'].astype(int)
    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df['Is_Leaving'].astype(int)
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df['Is_Leaving'].astype(int)

    print('Status: Training and comparing churn models...', flush=True)
    stage_at = time.perf_counter()
    candidates = build_model_candidates(random_state=42)
    model_reports = {}

    for model_name, model in candidates.items():
        model.fit(X_train, y_train)
        train_report = evaluate_classifier(model, X_train, y_train)
        validation_report = evaluate_classifier(model, X_val, y_val)
        test_report = evaluate_classifier(model, X_test, y_test)

        model_reports[model_name] = {
            'estimator': model,
            'train': train_report,
            'validation': validation_report,
            'test': test_report,
        }

        print(f'Status: Completed {model_name} in {time.perf_counter() - stage_at:.1f}s', flush=True)
        stage_at = time.perf_counter()

    selected_model_name = select_best_model(model_reports)
    print(f'Status: Selected model => {selected_model_name}', flush=True)

    selected_model = candidates[selected_model_name]
    combined_train = pd.concat([X_train, X_val], ignore_index=True)
    combined_target = pd.concat([y_train, y_val], ignore_index=True)
    selected_model.fit(combined_train, combined_target)
    selected_feature_importance = _extract_feature_importance(selected_model, selected_model_name, FEATURE_COLUMNS)

    validation_for_threshold = evaluate_classifier(candidates[selected_model_name].fit(X_train, y_train), X_val, y_val)
    selected_threshold = float(validation_for_threshold['best_threshold'])
    selected_test_report = evaluate_classifier(selected_model, X_test, y_test)

    print('Status: Fitting CLV model...', flush=True)
    stage_at = time.perf_counter()
    clv_cutoff = transactions['InvoiceDate'].max().normalize()
    clv_scores, clv_diagnostics = fit_clv_model(transactions, pd.Timestamp(clv_cutoff), prediction_days=90)
    print(f'Status: Fitted CLV model in {time.perf_counter() - stage_at:.1f}s', flush=True)
    clv_scores = clv_scores.rename(columns={'Customer_ID': 'Customer_ID'})
    clv_scores['Value_Segment'] = clv_scores['CLV_Segment'].astype(str)

    latest_month = panel['Month'].max()
    serving_snapshot = panel[panel['Month'] == latest_month].copy()
    serving_snapshot = serving_snapshot.rename(columns={'Customer ID': 'Customer_ID'})
    serving_snapshot['Snapshot_Month'] = pd.to_datetime(serving_snapshot['Month'])

    merged_serving = serving_snapshot.merge(clv_scores, on='Customer_ID', how='left')
    merged_serving['Predicted_CLV'] = merged_serving['Predicted_CLV'].fillna(merged_serving['Total_Spend_6M'])
    merged_serving['Predicted_CLV'] = merged_serving['Predicted_CLV'].clip(lower=0.0)
    merged_serving['CLV_Segment'] = merged_serving['CLV_Segment'].fillna('Low')

    print('Status: Building churn predictions for serving snapshot...', flush=True)
    stage_at = time.perf_counter()
    merged_serving['Risk_Score'] = selected_model.predict_proba(merged_serving[FEATURE_COLUMNS])[:, 1]
    merged_serving['Predicted_Churn_Flag'] = (merged_serving['Risk_Score'] >= selected_threshold).astype(int)

    if selected_model_name == 'xgboost':
        shap_values, reasons = _compute_shap_reasons(selected_model, merged_serving[FEATURE_COLUMNS])
        shap_summary = _compute_global_shap_summary(merged_serving[FEATURE_COLUMNS], shap_values)
        top_reason_series = pd.Series(reasons, index=merged_serving.index)
    else:
        # Fallback reason text for non-tree models.
        shap_values = np.zeros((len(merged_serving), len(FEATURE_COLUMNS)))
        # Weight each feature's standardised deviation from the median by the model's importance,
        # so the "primary driver" is not simply whichever feature has the biggest raw number.
        feature_frame = merged_serving[FEATURE_COLUMNS]
        spread = feature_frame.std().replace(0, 1.0)
        weights = pd.Series(selected_feature_importance).reindex(FEATURE_COLUMNS).fillna(0.0)
        driver_scores = ((feature_frame - feature_frame.median()) / spread).abs().mul(weights, axis=1)
        top_reason_series = pd.Series([
            f'Primary driver: {feature}' for feature in driver_scores.idxmax(axis=1)
        ], index=merged_serving.index)
        shap_summary = [
            {
                'feature': feature,
                'mean_abs_shap': float(value),
                'mean_signed_shap': 0.0,
                'direction': 'importance',
            }
            for feature, value in selected_feature_importance.items()
        ]

    merged_serving['Reason'] = top_reason_series.values

    print('Status: Generating data-driven strategy recommendations...', flush=True)
    stage_at = time.perf_counter()
    strategy_table = build_strategy_table(merged_serving)
    merged_serving = merged_serving.merge(
        strategy_table[
            ['Customer_ID', 'Recommended_Action', 'Action_Score', 'Expected_Revenue_at_Risk', 'Priority_Score', 'Recommendation_Reason']
        ],
        on='Customer_ID',
        how='left',
    )
    merged_serving['Action_Plan'] = merged_serving['Recommended_Action'] + ' | ' + merged_serving['Recommendation_Reason']
    merged_serving['Customer_Class'] = np.where(
        merged_serving['Predicted_CLV'].rank(pct=True) >= 0.9,
        'VIP Regular',
        'Standard Regular',
    )
    merged_serving['Growth_Momentum'] = merged_serving['Spend_Momentum']

    print(f'Status: Built serving snapshot in {time.perf_counter() - stage_at:.1f}s', flush=True)

    print('Status: Saving artifacts...', flush=True)
    stage_at = time.perf_counter()
    monthly_history = _rebuild_customer_history(transactions)
    monthly_history.to_csv('customer_monthly_history.csv', index=False)
    try:
        monthly_history.to_parquet('customer_monthly_history.parquet', index=False)
    except Exception:
        print('Status: Parquet export unavailable, CSV history artifact saved.')

    enriched_export = merged_serving.copy()
    export_columns = [
        'Customer_ID',
        'Snapshot_Month',
        'Total_Spend',
        'Total_Orders',
        'Total_Spend_6M',
        'Total_Orders_6M',
        'Average_Order_Value_6M',
        'Active_Months_6M',
        'Spend_Momentum',
        'Growth_Momentum',
        'Orders_Momentum',
        'Recency_Months',
        'Spend_Volatility_6M',
        'Is_Leaving',
        'Predicted_CLV',
        'CLV_Segment',
        'Value_Segment',
        'Risk_Score',
        'Predicted_Churn_Flag',
        'Reason',
        'Recommended_Action',
        'Action_Score',
        'Expected_Revenue_at_Risk',
        'Priority_Score',
        'Recommendation_Reason',
        'Action_Plan',
        'Customer_Class',
    ]
    available_columns = [column for column in export_columns if column in enriched_export.columns]
    enriched_export[available_columns].to_csv('enriched_data.csv', index=False)

    model_comparison_rows = []
    for model_name, report in model_reports.items():
        model_comparison_rows.append(
            {
                'model_name': model_name,
                'validation_roc_auc': report['validation']['roc_auc'],
                'validation_pr_auc': report['validation']['pr_auc'],
                'validation_brier_score': report['validation']['brier_score'],
                'test_roc_auc': report['test']['roc_auc'],
                'test_pr_auc': report['test']['pr_auc'],
                'test_brier_score': report['test']['brier_score'],
                'test_f1': report['test']['f1'],
                'test_precision': report['test']['precision'],
                'test_recall': report['test']['recall'],
                'best_threshold': report['validation']['best_threshold'],
            }
        )

    model_comparison_df = pd.DataFrame(model_comparison_rows).sort_values('validation_pr_auc', ascending=False)
    model_comparison_df.to_csv('model_comparison.csv', index=False)

    diagnostics_payload = {
        'selected_model': selected_model_name,
        'selected_threshold': selected_threshold,
        'split_info': split_info,
        'latest_snapshot_month': str(pd.to_datetime(latest_month).date()),
        'model_comparison': {
            row['model_name']: {
                'validation': {
                    'roc_auc': float(row['validation_roc_auc']),
                    'pr_auc': float(row['validation_pr_auc']),
                    'brier_score': float(row['validation_brier_score']),
                },
                'test': {
                    'roc_auc': float(row['test_roc_auc']),
                    'pr_auc': float(row['test_pr_auc']),
                    'brier_score': float(row['test_brier_score']),
                    'f1': float(row['test_f1']),
                    'precision': float(row['test_precision']),
                    'recall': float(row['test_recall']),
                },
                'best_threshold': float(row['best_threshold']),
            }
            for row in model_comparison_rows
        },
        'selected_model_metrics': {
            'roc_auc': float(selected_test_report['roc_auc']),
            'pr_auc': float(selected_test_report['pr_auc']),
            'brier_score': float(selected_test_report['brier_score']),
            'log_loss': float(selected_test_report['log_loss']),
            'accuracy': float(selected_test_report['accuracy']),
            'precision': float(selected_test_report['precision']),
            'recall': float(selected_test_report['recall']),
            'f1': float(selected_test_report['f1']),
            'confusion_matrix': selected_test_report['confusion_matrix'],
            'best_threshold_validation': selected_threshold,
            'threshold_metrics_test_at_0p5': selected_test_report['threshold_metrics'],
            'calibration_curve': selected_test_report['calibration_curve'],
        },
        'feature_columns': FEATURE_COLUMNS,
        'feature_importance': selected_feature_importance,
        'shap_summary': shap_summary,
        'clv_diagnostics': clv_diagnostics,
        'portfolio_statistics': {
            'serving_customers': int(len(merged_serving)),
            'mean_risk_score': float(merged_serving['Risk_Score'].mean()),
            'mean_predicted_clv': float(merged_serving['Predicted_CLV'].mean()),
            'mean_expected_revenue_at_risk': float(merged_serving['Expected_Revenue_at_Risk'].mean()),
            'high_risk_customers': int((merged_serving['Risk_Score'] >= 0.5).sum()),
            'high_value_customers': int((merged_serving['CLV_Segment'] == 'High').sum()),
        },
    }

    with open('model_diagnostics.json', 'w') as f:
        json.dump(diagnostics_payload, f, indent=2)

    with open('feature_importance.json', 'w') as f:
        json.dump(selected_feature_importance, f, indent=2)

    with open('shap_summary.json', 'w') as f:
        json.dump(shap_summary, f, indent=2)

    with open('clv_diagnostics.json', 'w') as f:
        json.dump(clv_diagnostics, f, indent=2)

    with open('training_baseline.json', 'w') as f:
        json.dump(
            {
                'Total_Spend_6M': float(panel['Total_Spend_6M'].mean()),
                'Spend_Momentum': float(panel['Spend_Momentum'].mean()),
            },
            f,
            indent=2,
        )

    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(selected_model, f)

    print(f"Status: Selected Model ROC-AUC: {selected_test_report['roc_auc']:.4f}", flush=True)
    print(f"Status: Selected Model PR-AUC: {selected_test_report['pr_auc']:.4f}", flush=True)
    print(f"Status: Selected Model Brier Score: {selected_test_report['brier_score']:.4f}", flush=True)
    print(f'Status: Saved artifacts in {time.perf_counter() - stage_at:.1f}s', flush=True)
    print(f'Success: Execution complete in {time.perf_counter() - started_at:.1f}s', flush=True)


if __name__ == '__main__':
    run_engine()