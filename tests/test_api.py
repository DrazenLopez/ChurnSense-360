import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as test_client:
        yield test_client


GOOD_FEATURES = {
    'Total_Spend_6M': 420.0, 'Total_Orders_6M': 3, 'Average_Order_Value_6M': 140.0,
    'Active_Months_6M': 2, 'Spend_Momentum': 0.4, 'Orders_Momentum': 0.5,
    'Recency_Months': 4, 'Spend_Volatility_6M': 150.0,
}


def test_health(client):
    body = client.get('/health').json()
    assert body['status'] == 'ok'
    assert body['customers_loaded'] > 1000
    assert 0 < body['decision_threshold'] < 1


def test_known_customer_returns_full_scorecard(client):
    body = client.get('/predict/12346').json()
    assert 0 <= body['churn_probability'] <= 1
    assert body['risk_band'] in {'Low', 'Medium', 'High'}
    assert len(body['reasons']) == 3
    assert all(r['effect'] in {'raises risk', 'lowers risk'} for r in body['reasons'])
    assert body['predicted_clv_90d'] >= 0
    assert body['recommended_action']


def test_post_with_customer_id_matches_get(client):
    via_get = client.get('/predict/12346').json()
    via_post = client.post('/predict', json={'customer_id': '12346'}).json()
    assert via_post['churn_probability'] == pytest.approx(via_get['churn_probability'])


def test_unknown_customer_is_404(client):
    assert client.get('/predict/does-not-exist').status_code == 404


def test_raw_features_are_scored_without_clv(client):
    body = client.post('/predict', json={'features': GOOD_FEATURES}).json()
    assert 0 <= body['churn_probability'] <= 1
    assert body['predicted_clv_90d'] is None
    assert len(body['reasons']) == 3


def test_worse_behaviour_scores_higher_risk(client):
    healthy = dict(GOOD_FEATURES, Recency_Months=0, Active_Months_6M=6, Total_Orders_6M=12, Total_Spend_6M=3000)
    lapsed = dict(GOOD_FEATURES, Recency_Months=9, Active_Months_6M=1, Total_Orders_6M=1, Total_Spend_6M=50)
    p_healthy = client.post('/predict', json={'features': healthy}).json()['churn_probability']
    p_lapsed = client.post('/predict', json={'features': lapsed}).json()['churn_probability']
    assert p_lapsed > p_healthy


def test_invalid_requests_are_rejected(client):
    assert client.post('/predict', json={}).status_code == 422
    assert client.post('/predict', json={'customer_id': '1', 'features': GOOD_FEATURES}).status_code == 422
    assert client.post('/predict', json={'features': dict(GOOD_FEATURES, Active_Months_6M=9)}).status_code == 422
    assert client.post('/predict', json={'features': {'Total_Spend_6M': 1}}).status_code == 422
