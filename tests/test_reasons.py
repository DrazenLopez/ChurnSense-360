import numpy as np

from reasons import describe_feature, top_reasons

NAMES = ['Recency_Months', 'Active_Months_6M', 'Total_Spend_6M']


def test_top_reasons_are_ordered_by_strength_and_limited_to_n():
    reasons = top_reasons([5.0, 2.0, 100.0], [0.1, -0.9, 0.5], NAMES, n=2)
    assert [r['feature'] for r in reasons] == ['Active_Months_6M', 'Total_Spend_6M']
    assert reasons[0]['effect'] == 'lowers risk'
    assert reasons[1]['effect'] == 'raises risk'


def test_text_is_plain_english_not_raw_numbers_only():
    text = describe_feature('Recency_Months', 4, shap_value=0.3)
    assert text == 'Last purchase was 4 months ago'
    assert describe_feature('Recency_Months', 1, shap_value=0.3) == 'Last purchase was 1 month ago'
    assert describe_feature('Recency_Months', 0, shap_value=-0.3) == 'Bought this month'
    assert 'shap' not in describe_feature('Spend_Momentum', 0.4, 0.2).lower()


def test_direction_changes_the_wording():
    assert 'Low spend' in describe_feature('Total_Spend_6M', 50, 0.2)
    assert 'Strong spend' in describe_feature('Total_Spend_6M', 5000, -0.2)


def test_unknown_feature_falls_back_gracefully():
    assert 'raises risk' in describe_feature('Some_New_Feature', 1.5, 0.1)


def test_high_value_that_still_raises_risk_is_not_called_low():
    # £12,602 spend is far above the median (£500), so "low spend" would be false.
    text = describe_feature('Total_Spend_6M', 12602, shap_value=0.4, reference=500)
    assert 'Low spend' not in text
    assert '£12,602' in text
    assert 'combination with other signals' in text


def test_aligned_values_keep_the_direct_wording_when_reference_given():
    assert 'Low spend' in describe_feature('Total_Spend_6M', 50, 0.4, reference=500)
    assert 'Strong spend' in describe_feature('Total_Spend_6M', 5000, -0.4, reference=500)
    # Recency: a high value raising risk is aligned; a high value lowering risk is not.
    assert describe_feature('Recency_Months', 9, 0.4, reference=2) == 'Last purchase was 9 months ago'
    assert 'point in this customer' in describe_feature('Recency_Months', 9, -0.4, reference=2)
