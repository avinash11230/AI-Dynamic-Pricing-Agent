import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('retail_data.csv')

# ── Feature Engineering ──────────────────────────────────────────────
df['log_effective_price'] = np.log(df['effective_price'])
df['log_units_sold'] = np.log(df['units_sold'])
df['price_discount_interaction'] = df['effective_price'] * df['discount_pct']
df['competitor_gap'] = df['discount_pct'] - df['competitor_discount']
df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
df['price_vs_category_mean'] = df.groupby('category')['effective_price'].transform(
    lambda x: x / x.mean())

le_cat = LabelEncoder()
le_store = LabelEncoder()
df['category_enc'] = le_cat.fit_transform(df['category'])
df['store_size_enc'] = le_store.fit_transform(df['store_size'])

FEATURES = [
    'effective_price', 'discount_pct', 'base_price',
    'log_effective_price', 'price_discount_interaction',
    'competitor_discount', 'competitor_gap',
    'is_holiday', 'seasonality_index',
    'week_sin', 'week_cos',
    'category_enc', 'store_size_enc',
    'price_vs_category_mean'
]
TARGET = 'units_sold'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Model Training ───────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=12,
                                               n_jobs=-1, random_state=42),
    'XGBoost':           xgb.XGBRegressor(n_estimators=300, max_depth=6,
                                           learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, random_state=42,
                                           verbosity=0),
}

results = {}
print(f"{'Model':<22} {'R² Train':>10} {'R² Test':>10} {'RMSE Test':>12} {'CV R² Mean':>12}")
print("─" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test  = r2_score(y_test, y_pred)
    rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)

    results[name] = {'model': model, 'r2_test': r2_test, 'rmse': rmse}
    print(f"{name:<22} {r2_train:>10.4f} {r2_test:>10.4f} {rmse:>12.2f} {cv_scores.mean():>12.4f}")

# Save best model + metadata
best_name = max(results, key=lambda k: results[k]['r2_test'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name} (R²={results[best_name]['r2_test']:.4f})")

joblib.dump(best_model, 'demand_model.pkl')
joblib.dump(le_cat, 'le_category.pkl')
joblib.dump(le_store, 'le_store.pkl')

# Feature importance (XGBoost)
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({'feature': FEATURES,
                       'importance': best_model.feature_importances_})
    fi.sort_values('importance', ascending=False, inplace=True)
    fi.to_csv('feature_importance.csv', index=False)
    print("\nTop 5 features:")
    print(fi.head())

df.to_csv('retail_data_features.csv', index=False)
print("\nModel training complete.")