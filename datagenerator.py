import numpy as np
import pandas as pd

np.random.seed(42)
N = 50000

CATEGORIES = {
    'Electronics':   {'base_price': 150, 'cost_pct': 0.60, 'elasticity': -2.1, 'base_demand': 80},
    'Groceries':     {'base_price': 10,  'cost_pct': 0.55, 'elasticity': -0.8, 'base_demand': 400},
    'Clothing':      {'base_price': 45,  'cost_pct': 0.40, 'elasticity': -1.5, 'base_demand': 150},
    'Home & Garden': {'base_price': 35,  'cost_pct': 0.50, 'elasticity': -1.2, 'base_demand': 120},
    'Toys':          {'base_price': 28,  'cost_pct': 0.45, 'elasticity': -1.8, 'base_demand': 200},
    'Sports':        {'base_price': 60,  'cost_pct': 0.50, 'elasticity': -1.6, 'base_demand': 100},
    'Beauty':        {'base_price': 22,  'cost_pct': 0.35, 'elasticity': -1.3, 'base_demand': 180},
    'Books':         {'base_price': 18,  'cost_pct': 0.45, 'elasticity': -0.9, 'base_demand': 250},
}

rows = []
category_names = list(CATEGORIES.keys())
product_ids = {cat: [f"{cat[:3].upper()}{i:03d}" for i in range(1, 21)]
               for cat in category_names}

for _ in range(N):
    cat = np.random.choice(category_names)
    cfg = CATEGORIES[cat]
    product_id = np.random.choice(product_ids[cat])

    # Price variation ±30% around base
    price_factor = np.random.uniform(0.90, 1.10)
    base_price = cfg['base_price'] * price_factor

    # Discount between 0% and 50%
    discount_pct = np.random.choice(
        [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        p=[0.25, 0.10, 0.15, 0.15, 0.12, 0.08, 0.07, 0.05, 0.03]
    )
    effective_price = base_price * (1 - discount_pct)

    # Seasonality — week of year
    week = np.random.randint(1, 53)
    seasonality = 1 + 0.3 * np.sin(2 * np.pi * week / 52)  # peaks mid-year
    if cat == 'Toys' and week >= 48:
        seasonality += 0.6  # holiday spike for toys
    if cat == 'Sports' and 10 <= week <= 20:
        seasonality += 0.3  # spring fitness boost

    # Store features
    store_id = np.random.randint(1, 51)
    store_size = np.random.choice(['Small', 'Medium', 'Large'], p=[0.3, 0.5, 0.2])
    store_size_factor = {'Small': 0.8, 'Medium': 1.0, 'Large': 1.4}[store_size]
    is_holiday = 1 if week in [1, 26, 35, 48, 49, 50, 51, 52] else 0
    competitor_discount = np.random.uniform(0, 0.40)

    # Demand model: power law elasticity + competitor effect + noise
    price_ratio = effective_price / cfg['base_price']
    demand = (cfg['base_demand']
              * (price_ratio ** cfg['elasticity'])
              * seasonality
              * store_size_factor
              * (1 + 0.15 * is_holiday)
              * (1 + 0.10 * (competitor_discount - discount_pct))
              * np.random.lognormal(0, 0.12))  # realistic noise
    demand = max(1, round(demand))

    cost = base_price * cfg['cost_pct']
    revenue = effective_price * demand
    profit = (effective_price - cost) * demand

    rows.append({
        'product_id': product_id,
        'category': cat,
        'week': week,
        'store_id': store_id,
        'store_size': store_size,
        'base_price': round(base_price, 2),
        'discount_pct': discount_pct,
        'effective_price': round(effective_price, 2),
        'cost': round(cost, 2),
        'competitor_discount': round(competitor_discount, 3),
        'is_holiday': is_holiday,
        'seasonality_index': round(seasonality, 4),
        'units_sold': demand,
        'revenue': round(revenue, 2),
        'profit': round(profit, 2),
    })

df = pd.DataFrame(rows)
df.to_csv('retail_data.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(df.head())
print("\nCategory distribution:")
print(df['category'].value_counts())
print("\nUnits sold stats:")
print(df['units_sold'].describe())