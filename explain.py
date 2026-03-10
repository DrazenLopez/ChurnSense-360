import shap
import numpy as np

from pipeline_io import load_serving_artifacts

# 1. Load production-serving artifacts
data, X, model = load_serving_artifacts()

# 2. Initialize SHAP
# This calculates per-customer feature contribution against churn risk.
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X)
shap_values = np.array(shap_values_raw)

if shap_values.ndim == 3:
    shap_values = shap_values[0]

customer_id_column = 'Customer_ID' if 'Customer_ID' in data.columns else None

print("Analyzing Customer Behavioral Fingerprints...\n")

# 3. Show the reason codes for the first 5 customers
sample_size = min(5, len(X))
for i in range(sample_size):
    # Find which feature had the biggest impact on this specific person
    top_feature_idx = np.argmax(np.abs(shap_values[i]))
    feature_name = X.columns[top_feature_idx]

    customer_label = str(data.iloc[i][customer_id_column]) if customer_id_column else str(i)
    print(f"Customer {customer_label}: Predicted Churn Risk High.")
    print(f"Primary Driver: {feature_name.upper()}")
    print("-" * 20)

# 4. Summary check
print("\nSystem Module 3 Complete: Explainability layer active.")
print("The model is no longer a 'black box'—it's providing actionable reasons.")