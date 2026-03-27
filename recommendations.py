import pandas as pd

from pipeline_io import load_serving_artifacts
from retention_engine import build_strategy_table


def run_engine():
    data, _, _ = load_serving_artifacts()
    strategy_table = build_strategy_table(data)

    output = strategy_table[
        [
            'Customer_ID',
            'Risk_Score',
            'Predicted_CLV',
            'CLV_Segment',
            'Recommended_Action',
            'Action_Score',
            'Expected_Revenue_at_Risk',
            'Priority_Score',
            'Recommendation_Reason',
        ]
    ].copy()

    output['Action_Plan'] = output['Recommended_Action'] + ' | ' + output['Recommendation_Reason']

    output.to_csv('marketing_strategy.csv', index=False)

    print(output.head(10))
    print('\nStrategy Generated: marketing_strategy.csv is ready for executive review.')


if __name__ == '__main__':
    run_engine()