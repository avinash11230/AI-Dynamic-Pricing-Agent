import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize_scalar

model     = joblib.load('demand_model.pkl')
le_cat    = joblib.load('le_category.pkl')
le_store  = joblib.load('le_store.pkl')
df        = pd.read_csv('retail_data_features.csv')

FEATURES = [
    'effective_price', 'discount_pct', 'base_price',
    'log_effective_price', 'price_discount_interaction',
    'competitor_discount', 'competitor_gap',
    'is_holiday', 'seasonality_index',
    'week_sin', 'week_cos',
    'category_enc', 'store_size_enc',
    'price_vs_category_mean'
]

cat_mean_prices = df.groupby('category')['effective_price'].mean().to_dict()

def build_feature_vector(base_price, discount_pct, category,
                         store_size, week, competitor_discount,
                         is_holiday, seasonality_index):
    effective_price = base_price * (1 - discount_pct)
    cat_enc   = le_cat.transform([category])[0]
    store_enc = le_store.transform([store_size])[0]
    cat_mean  = cat_mean_prices.get(category, base_price)

    return {
        'effective_price':          effective_price,
        'discount_pct':             discount_pct,
        'base_price':               base_price,
        'log_effective_price':      np.log(max(effective_price, 0.01)),
        'price_discount_interaction': effective_price * discount_pct,
        'competitor_discount':      competitor_discount,
        'competitor_gap':           discount_pct - competitor_discount,
        'is_holiday':               is_holiday,
        'seasonality_index':        seasonality_index,
        'week_sin':                 np.sin(2 * np.pi * week / 52),
        'week_cos':                 np.cos(2 * np.pi * week / 52),
        'category_enc':             cat_enc,
        'store_size_enc':           store_enc,
        'price_vs_category_mean':   effective_price / cat_mean,
    }

def predict_demand(discount_pct, base_price, category, store_size,
                   week, competitor_discount, is_holiday, seasonality_index):
    fv = build_feature_vector(base_price, discount_pct, category, store_size,
                              week, competitor_discount, is_holiday, seasonality_index)
    X  = pd.DataFrame([fv])[FEATURES]
    return max(0, model.predict(X)[0])

def objective(discount_pct, target, base_price, cost, category, store_size,
              week, competitor_discount, is_holiday, seasonality_index):
    demand = predict_demand(discount_pct, base_price, category, store_size,
                            week, competitor_discount, is_holiday, seasonality_index)
    effective_price = base_price * (1 - discount_pct)
    if target == 'revenue':
        return -(effective_price * demand)
    elif target == 'profit':
        return -((effective_price - cost) * demand)
    elif target == 'volume':
        return -demand

def optimize(base_price, cost, category, store_size='Medium', week=26,
             competitor_discount=0.15, is_holiday=0, seasonality_index=1.0,
             target='revenue'):
    result = minimize_scalar(
        objective,
        bounds=(0.0, 0.50),
        method='bounded',
        args=(target, base_price, cost, category, store_size,
              week, competitor_discount, is_holiday, seasonality_index)
    )
    return round(result.x, 4)

def simulate_scenarios(base_price, cost, category, store_size='Medium',
                       week=26, competitor_discount=0.15,
                       is_holiday=0, seasonality_index=1.0, target='revenue'):
    optimal_disc = optimize(base_price, cost, category, store_size,
                            week, competitor_discount, is_holiday,
                            seasonality_index, target)
    discount_points = sorted(set(
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, round(optimal_disc, 2)]
    ))
    scenarios = []
    for d in discount_points:
        eff_price = base_price * (1 - d)
        demand    = predict_demand(d, base_price, category, store_size,
                                   week, competitor_discount, is_holiday,
                                   seasonality_index)
        revenue   = eff_price * demand
        profit    = (eff_price - cost) * demand
        scenarios.append({
            'discount_pct':     f"{d*100:.0f}%",
            'effective_price':  round(eff_price, 2),
            'predicted_demand': round(demand, 1),
            'revenue':          round(revenue, 2),
            'profit':           round(profit, 2),
            'is_optimal':       abs(d - optimal_disc) < 0.01
        })
    return pd.DataFrame(scenarios), optimal_disc

# Quick sanity test
if __name__ == '__main__':
    print("Optimizer sanity check — Electronics, base_price=$150, cost=$90\n")
    scenarios, opt = simulate_scenarios(
        base_price=150, cost=90, category='Electronics',
        target='profit', competitor_discount=0.20
    )
    print(scenarios.to_string(index=False))
    print(f"\nOptimal discount: {opt*100:.1f}%")