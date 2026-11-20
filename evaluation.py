"""Extra model checks that answer "is the model actually useful?" rather than "what is its AUC?".

Everything here is reported alongside the main metrics in ``model_evaluation.json``:

* simple rule-based baselines (can the ML model beat a one-line rule?)
* precision / lift at the top k% of customers (what a limited outreach budget would actually get)
* an unseen-customer test (customers the model has never seen, in later months)
* a cost-based threshold (replaces "maximise F1" with an explicit cost trade-off)
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# PLACEHOLDER COSTS - set these from real business numbers before relying on the
# cost-based threshold. Units are arbitrary; only the ratio matters.
#   COST_MISSED_LEAVER : value lost when a customer who will leave is not contacted
#   COST_WASTED_OFFER  : cost of an offer sent to a customer who would have stayed
# ---------------------------------------------------------------------------
COST_MISSED_LEAVER = float(os.getenv('CHURNSENSE_COST_MISSED_LEAVER', '5'))
COST_WASTED_OFFER = float(os.getenv('CHURNSENSE_COST_WASTED_OFFER', '1'))


def _safe_auc(y_true, score) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {'roc_auc': float('nan'), 'pr_auc': float('nan')}
    return {
        'roc_auc': float(roc_auc_score(y_true, score)),
        'pr_auc': float(average_precision_score(y_true, score)),
    }


def baseline_report(test_df: pd.DataFrame, model_proba: np.ndarray) -> Dict[str, object]:
    """Compare the model with trivial one-feature rules on the same test rows."""
    y = test_df['Is_Leaving'].astype(int).to_numpy()
    rules = {
        'model': model_proba,
        # The longer since the last purchase, the likelier the customer is gone.
        'rule_recency_months': test_df['Recency_Months'].to_numpy(dtype=float),
        # Fewer active months in the last 6 means likelier to leave.
        'rule_few_active_months': -test_df['Active_Months_6M'].to_numpy(dtype=float),
        # Lower recent spend means likelier to leave.
        'rule_low_recent_spend': -test_df['Total_Spend_6M'].to_numpy(dtype=float),
    }
    out = {name: _safe_auc(y, score) for name, score in rules.items()}
    out['base_rate_leaving'] = float(y.mean())
    return out


def lift_table(y_true, score, fractions: Iterable[float] = (0.05, 0.10, 0.20, 0.30, 0.50)) -> list:
    """Precision and lift when only the top k% highest-risk customers are contacted."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(score), kind='mergesort')
    base_rate = float(y.mean())
    rows = []
    for fraction in fractions:
        k = max(1, int(round(len(y) * fraction)))
        precision = float(y[order[:k]].mean())
        rows.append({
            'top_fraction': float(fraction),
            'customers_contacted': int(k),
            'precision': precision,
            'lift_vs_random': precision / base_rate if base_rate > 0 else float('nan'),
            'share_of_leavers_caught': float(y[order[:k]].sum() / max(y.sum(), 1)),
        })
    return rows


def cost_based_threshold(
    y_true,
    proba,
    cost_missed: float = COST_MISSED_LEAVER,
    cost_wasted: float = COST_WASTED_OFFER,
) -> Dict[str, float]:
    """Pick the threshold with the lowest expected cost, and report the textbook value too.

    For a calibrated model the optimal rule is: contact when p >= wasted / (wasted + missed).
    The empirical value is searched on the supplied (validation) data.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], np.round(p, 4), [1.0])))
    best_t, best_cost = 0.5, float('inf')
    for t in candidates:
        flag = p >= t
        cost = cost_missed * np.sum((~flag) & (y == 1)) + cost_wasted * np.sum(flag & (y == 0))
        if cost < best_cost:
            best_cost, best_t = float(cost), float(t)
    return {
        'cost_missed_leaver': float(cost_missed),
        'cost_wasted_offer': float(cost_wasted),
        'theoretical_threshold': float(cost_wasted / (cost_wasted + cost_missed)),
        'empirical_threshold': best_t,
        'expected_cost_at_empirical': best_cost,
        'share_flagged_at_empirical': float(np.mean(p >= best_t)),
    }


def unseen_customer_report(
    labeled: pd.DataFrame,
    feature_columns: list,
    model_factory: Callable[[], object],
    test_start: pd.Timestamp,
    holdout_fraction: float = 0.2,
    max_train_rows: int = 20000,
    max_test_rows: int = 10000,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train on some customers in early months, test on *different* customers in later months.

    The standard temporal split still lets the same customers appear in both train and test.
    This stricter check removes that overlap.
    """
    frame = labeled.copy()
    frame['Month'] = pd.to_datetime(frame['Month'])
    customers = frame['Customer ID'].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    held_out = set(rng.choice(customers, size=max(1, int(len(customers) * holdout_fraction)), replace=False))

    train_part = frame[(frame['Month'] < test_start) & (~frame['Customer ID'].isin(held_out))]
    test_part = frame[(frame['Month'] >= test_start) & (frame['Customer ID'].isin(held_out))]
    if len(train_part) > max_train_rows:
        train_part = train_part.sample(n=max_train_rows, random_state=random_state)
    if len(test_part) > max_test_rows:
        test_part = test_part.sample(n=max_test_rows, random_state=random_state)

    model = model_factory()
    model.fit(train_part[feature_columns], train_part['Is_Leaving'].astype(int))
    proba = model.predict_proba(test_part[feature_columns])[:, 1]
    y = test_part['Is_Leaving'].astype(int).to_numpy()
    report = _safe_auc(y, proba)
    report.update({
        'held_out_customers': int(len(held_out)),
        'train_rows': int(len(train_part)),
        'test_rows': int(len(test_part)),
        'base_rate_leaving': float(y.mean()),
        'lift': lift_table(y, proba),
    })
    return report


def integrity_report(
    model_factory: Callable[[], object],
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    n_repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, object]:
    """Sanity checks that would expose leakage or a model resting on one or two fragile features.

    * permutation importance: how much the test score drops when one feature is shuffled
    * label-copy test: a feature that copies the answer MUST drive the score to ~1.0
      (proves the evaluation would notice leakage)
    * top-2 removed: retrain without the two most important features to see how much they carry
    """
    rng = np.random.default_rng(random_state)
    y_test_arr = np.asarray(y_test).astype(int)

    model = model_factory()
    model.fit(X_train, y_train)
    base = _safe_auc(y_test_arr, model.predict_proba(X_test)[:, 1])

    importance = []
    for feature in X_train.columns:
        roc_drops, pr_drops = [], []
        for _ in range(n_repeats):
            shuffled = X_test.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            scores = _safe_auc(y_test_arr, model.predict_proba(shuffled)[:, 1])
            roc_drops.append(base['roc_auc'] - scores['roc_auc'])
            pr_drops.append(base['pr_auc'] - scores['pr_auc'])
        importance.append({
            'feature': feature,
            'delta_roc_auc': float(np.mean(roc_drops)),
            'delta_pr_auc': float(np.mean(pr_drops)),
        })
    importance.sort(key=lambda row: row['delta_roc_auc'], reverse=True)

    leaky_train, leaky_test = X_train.copy(), X_test.copy()
    leaky_train['LABEL_COPY'] = np.asarray(y_train).astype(int)
    leaky_test['LABEL_COPY'] = y_test_arr
    leaky_model = model_factory()
    leaky_model.fit(leaky_train, y_train)
    leak_scores = _safe_auc(y_test_arr, leaky_model.predict_proba(leaky_test)[:, 1])

    top_two = [row['feature'] for row in importance[:2]]
    reduced_model = model_factory()
    reduced_model.fit(X_train.drop(columns=top_two), y_train)
    reduced_scores = _safe_auc(y_test_arr, reduced_model.predict_proba(X_test.drop(columns=top_two))[:, 1])

    return {
        'baseline_temporal_test': base,
        'permutation_importance': importance,
        'leak_stress_test': {
            'with_explicit_label_copy_roc_auc': leak_scores['roc_auc'],
            'with_explicit_label_copy_pr_auc': leak_scores['pr_auc'],
            'without_top2_features_removed_roc_auc': reduced_scores['roc_auc'],
            'without_top2_features_removed_pr_auc': reduced_scores['pr_auc'],
            'removed_features': top_two,
        },
        'note': 'Customer-level holdout results are in model_evaluation.json under unseen_customers.',
    }
