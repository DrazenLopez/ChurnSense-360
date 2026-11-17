import numpy as np
import pandas as pd
import pytest

from evaluation import cost_based_threshold, lift_table
from retention_engine import FEATURE_COLUMNS, build_customer_month_panel


def _transactions(rows):
    frame = pd.DataFrame(rows, columns=['Customer ID', 'InvoiceDate', 'Invoice', 'Quantity', 'Price'])
    frame['InvoiceDate'] = pd.to_datetime(frame['InvoiceDate'])
    frame['Total_Value'] = frame['Quantity'] * frame['Price']
    return frame


def _monthly_purchases(customer, months, start='2010-01-15'):
    start = pd.Timestamp(start)
    return [
        (customer, (start + pd.DateOffset(months=m)).strftime('%Y-%m-%d'), f'{customer}-{m}', 1, 10.0)
        for m in months
    ]


def test_label_marks_customer_who_stops_buying_as_leaving():
    # Customer A buys in months 0-3 then stops; customer B buys every month for 12 months.
    rows = _monthly_purchases('A', range(0, 4)) + _monthly_purchases('B', range(0, 12))
    panel = build_customer_month_panel(_transactions(rows))
    a = panel[panel['Customer ID'] == 'A'].set_index('Month')

    # Month 3 is A's last purchase month; the following 3 months have zero spend.
    assert a.loc[pd.Timestamp('2010-04-01'), 'Is_Leaving'] == 1
    b = panel[panel['Customer ID'] == 'B'].set_index('Month')
    assert b.loc[pd.Timestamp('2010-04-01'), 'Is_Leaving'] == 0


def test_rows_without_a_full_future_window_are_not_labelled():
    rows = _monthly_purchases('A', range(0, 12))
    panel = build_customer_month_panel(_transactions(rows))
    last_month = panel['Month'].max()
    assert panel.loc[panel['Month'] == last_month, 'Has_Label'].eq(0).all()


def test_features_do_not_use_future_transactions():
    base = _monthly_purchases('A', range(0, 12)) + _monthly_purchases('B', range(0, 12))
    changed = [r for r in base if not (r[0] == 'A' and pd.Timestamp(r[1]) >= pd.Timestamp('2010-09-01'))]
    changed += [('A', '2010-11-20', 'A-late', 50, 100.0)]

    original = build_customer_month_panel(_transactions(base))
    modified = build_customer_month_panel(_transactions(changed))

    cutoff = pd.Timestamp('2010-07-01')
    cols = ['Customer ID', 'Month'] + FEATURE_COLUMNS
    left = original[original['Month'] <= cutoff][cols].sort_values(['Customer ID', 'Month']).reset_index(drop=True)
    right = modified[modified['Month'] <= cutoff][cols].sort_values(['Customer ID', 'Month']).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_lift_table_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    score = np.concatenate([np.linspace(0.9, 0.8, 10), np.linspace(0.1, 0.0, 90)])
    top10 = lift_table(y, score, fractions=(0.10,))[0]
    assert top10['precision'] == 1.0
    assert top10['share_of_leavers_caught'] == 1.0
    assert top10['lift_vs_random'] == pytest.approx(10.0)


def test_cost_threshold_matches_textbook_value_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    result = cost_based_threshold(y, p, cost_missed=4.0, cost_wasted=1.0)
    assert result['theoretical_threshold'] == pytest.approx(0.2)
    assert result['empirical_threshold'] == pytest.approx(0.2, abs=0.05)
