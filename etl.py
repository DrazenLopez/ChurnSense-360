import pandas as pd
import numpy as np

def save_star_schema(raw_df):
    fact_transactions = raw_df.copy()

    dim_customers = (
        raw_df.groupby('customer_id', as_index=False)
        .agg(
            first_purchase_date=('date', 'min'),
            last_purchase_date=('date', 'max'),
            total_spend=('amount', 'sum')
        )
        .sort_values('customer_id')
    )

    dim_products = (
        raw_df[['product_id', 'product_name', 'category']]
        .drop_duplicates()
        .sort_values('product_id')
        .reset_index(drop=True)
    )

    fact_transactions.to_csv('fact_transactions.csv', index=False)
    dim_customers.to_csv('dim_customers.csv', index=False)
    dim_products.to_csv('dim_products.csv', index=False)

    return fact_transactions, dim_customers, dim_products


# 1. Generate Raw Transaction Data
np.random.seed(42)
days = pd.date_range(start='2025-01-01', periods=365, freq='D')
product_catalog = {
    1000: ('Starter Kit', 'Core'),
    1001: ('Pro Bundle', 'Core'),
    1002: ('Refill Pack', 'Consumable'),
    1003: ('Premium Add-on', 'Add-on'),
    1004: ('Enterprise Module', 'Add-on')
}

data = []
transaction_id = 1
for customer_id in range(100):
    for day in days:
        if np.random.rand() > 0.8:  # Randomly shopping
            product_id = int(np.random.choice(list(product_catalog.keys())))
            product_name, category = product_catalog[product_id]
            quantity = int(np.random.randint(1, 5))
            unit_price = float(np.random.uniform(10, 120))
            amount = float(quantity * unit_price)

            data.append([
                transaction_id,
                customer_id,
                product_id,
                product_name,
                category,
                day,
                quantity,
                unit_price,
                amount
            ])
            transaction_id += 1

df = pd.DataFrame(
    data,
    columns=[
        'transaction_id',
        'customer_id',
        'product_id',
        'product_name',
        'category',
        'date',
        'quantity',
        'unit_price',
        'amount'
    ]
)

fact_transactions, dim_customers, dim_products = save_star_schema(df)

# 2. System Module 1: High-Velocity Data Pipeline
# We pivot to see spend per week
weekly = fact_transactions.groupby(['customer_id', pd.Grouper(key='date', freq='W')])['amount'].sum().unstack().fillna(0)

# Velocity = This week's spend / average of last 3 weeks
current_week = weekly.iloc[:, -1]
prev_3_avg = weekly.iloc[:, -4:-1].mean(axis=1)
velocity = current_week / (prev_3_avg + 1e-9)

# 3. Create Feature Table
features = pd.DataFrame({
    'velocity': velocity,
    'total_spend': weekly.sum(axis=1),
    'is_churn': (current_week == 0).astype(int) # Target: 1 if they didn't shop this week
}).reset_index()

features.to_csv('processed_data.csv', index=False)
print("System Module 1 Complete: Star schema saved (fact_transactions, dim_customers, dim_products).")
print("System Module 1 Complete: 'processed_data.csv' is ready for model processing.")