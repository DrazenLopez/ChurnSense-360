# ChurnSense:360

Customer churn prediction and retention dashboard for a UK online retailer, built with Python, XGBoost, SHAP and Streamlit.

It answers four questions: **who is likely to stop buying, why, how much revenue that puts at risk, and what to do about it.**

## What it does

| Module | What you get |
|---|---|
| Churn model | A monthly risk score for each customer (XGBoost, compared with Logistic Regression and Random Forest) |
| Explainability | The main reason behind each customer's score (SHAP) |
| Customer value | A 90-day expected-spend forecast per customer (BG/NBD + Gamma-Gamma) |
| Strategy | A recommended action per customer: High-Touch Recovery, Targeted Incentive, Loyalty Growth, Automated Nurture |
| Simulator | A what-if tool: change spend or order inputs and see how the predicted risk moves |
| Back-end tab | Model comparison, calibration, threshold and drift monitoring |

## How it is built

- **Data:** UCI *Online Retail II* (UK retailer, Dec 2009 to Dec 2011, about 5,900 customers).
- **Target:** a customer is "leaving" in a month if they spend nothing in the following 3 months.
- **Features (8):** spend, orders, average order value, active months and recency over the last 6 months, plus spend and order momentum and spend volatility.
- **Split:** by time (60% / 20% / 20% of months), so the model is always tested on later months than it trained on. The final model is refitted on train + validation before scoring.
- **Model choice:** the model with the best validation PR-AUC (ties broken by Brier score). XGBoost won.

## Results

Test period: May to Sep 2011. In this data **59% of customer-months are "leaving"**, so a model has to beat that base rate to be useful.

| | ROC-AUC | PR-AUC |
|---|---|---|
| **XGBoost (selected)** | **0.77** | **0.81** |
| Best one-line rule (few active months or low recent spend) | 0.69 | 0.71 |
| Rule: months since last purchase | 0.64 | 0.64 |

**Unseen customers.** Re-training on 80% of customers and testing on the other 20% (in later months) gives ROC-AUC 0.77 and PR-AUC 0.81, so the score does not depend on the model having seen the same customers before.

**Contacting only the highest-risk customers.** Precision is the share of contacted customers who really left:

| Contact the top… | Precision | vs. random (59%) |
|---|---|---|
| 5% | 91% | 1.5x |
| 10% | 91% | 1.5x |
| 20% | 88% | 1.5x |
| 50% | 79% | 1.3x |

Full numbers are in `model_evaluation.json` and `model_diagnostics.json`.

## Limitations (read before relying on the numbers)

- **Retail purchases are irregular.** "Zero spend for 3 months" is a proxy for churn, not a cancellation. Many "leavers" will buy again.
- **Because most customers are labelled as leaving, the model flags most of them** (about 85% at the chosen threshold). It is more useful for *ranking* customers than for a yes/no decision. Use the top-k table above.
- **The decision threshold is a trade-off, not a fact.** With a 5:1 cost for missing a leaver versus wasting an offer, almost every customer should be contacted. The cost values in `evaluation.py` are placeholders; set `CHURNSENSE_COST_MISSED_LEAVER` and `CHURNSENSE_COST_WASTED_OFFER` from real numbers.
- **The simulator is not causal.** It shows how the model's score reacts to changed inputs. It does not prove that a coupon would keep a customer.
- **90-day customer value is a forecast with wide uncertainty**, and about 29% of customers have a forecast of zero.
- **One dataset, one retailer, 2009 to 2011.** Nothing here has been validated on current or other data.
- **The "2-week" action plans in the dashboard are suggested cadences**, not model outputs. The model itself is monthly.

## Run it

```bash
python -m venv venv
# Windows: .\venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker:

```bash
docker build -t churnsense360 .
docker run -p 8501:8501 churnsense360
```

Then open http://localhost:8501.

The repository includes pre-generated model and data artifacts, so the dashboard starts immediately.

**Login.** No login is required by default. To require one, set `CHURNSENSE_PASSWORD` (and optionally `CHURNSENSE_USERNAME`, default `admin`) before starting.

**Retrain** (about 3 to 4 minutes; needs `fast_retail_data.csv` or `online_retail_II.xlsx`):

```bash
python train.py
```

**Tests:**

```bash
pip install pytest
python -m pytest tests
```

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `train.py` | Runs the whole pipeline and writes the artifacts |
| `retention_engine.py` | Customer-month features, labels, splits, models, customer value, strategy table |
| `evaluation.py` | Baselines, top-k lift, unseen-customer test, cost-based threshold |
| `pipeline_io.py` | Loads the saved model and data for scoring |
| `explain.py`, `recommendations.py` | Command-line SHAP reasons and strategy export |
| `etl.py` | Small synthetic star-schema demo, separate from the main pipeline |
| `tests/` | Unit tests (labels, no future leakage, lift and cost functions) |
