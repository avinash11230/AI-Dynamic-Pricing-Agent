import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib

df = pd.read_csv('retail_data_features.csv')

# ── 1. Compute Price Elasticity Per Category ─────────────────────────
# Using log-log regression: log(Q) = α + ε * log(P) → ε is elasticity
from sklearn.linear_model import LinearRegression

elasticity = {}
for cat in df['category'].unique():
    sub = df[df['category'] == cat][['log_effective_price', 'log_units_sold']].dropna()
    lr = LinearRegression().fit(sub[['log_effective_price']], sub['log_units_sold'])
    elasticity[cat] = round(lr.coef_[0], 3)

elasticity_df = pd.DataFrame(list(elasticity.items()),
                              columns=['Category', 'Elasticity'])
elasticity_df.sort_values('Elasticity', inplace=True)
elasticity_df.to_csv('elasticity_by_category.csv', index=False)
print("Price Elasticity by Category:")
print(elasticity_df.to_string(index=False))

# ── 2. Revenue Curve — Average per Discount Bucket ───────────────────
df['discount_bucket'] = pd.cut(df['discount_pct'], bins=10)
rev_curve = df.groupby(['category', 'discount_bucket'])['revenue'].mean().reset_index()

# ── MASTER VISUALIZATION ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Smart Retail Pricing Optimizer — Analytical Dashboard',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

COLORS = ['#2196F3','#4CAF50','#FF5722','#9C27B0',
          '#FF9800','#00BCD4','#E91E63','#795548']

# Plot 1: Elasticity Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
colors_bar = ['#E53935' if e < -1.5 else '#FB8C00' if e < -1.0 else '#43A047'
              for e in elasticity_df['Elasticity']]
bars = ax1.barh(elasticity_df['Category'], elasticity_df['Elasticity'],
                color=colors_bar, edgecolor='white', linewidth=0.5)
ax1.axvline(x=-1, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
            label='Elasticity = -1 (unit elastic)')
ax1.set_xlabel('Price Elasticity of Demand')
ax1.set_title('Price Elasticity by Category', fontweight='bold')
ax1.legend(fontsize=8)
for bar, val in zip(bars, elasticity_df['Elasticity']):
    ax1.text(val - 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', ha='right', fontsize=9, color='white', fontweight='bold')
ax1.text(0.98, 0.02, '← Elastic (discount helps)\n→ Inelastic (discount less effective)',
         transform=ax1.transAxes, fontsize=7.5, va='bottom', ha='right',
         style='italic', color='gray')

# Plot 2: Demand Curve per Category (price vs units sold)
ax2 = fig.add_subplot(gs[0, 1])
for i, cat in enumerate(df['category'].unique()):
    sub = df[df['category'] == cat].copy()
    sub['price_bin'] = pd.qcut(sub['effective_price'], q=12, duplicates='drop')
    demand_curve = sub.groupby('price_bin')['units_sold'].mean()
    mid_prices = [interval.mid for interval in demand_curve.index]
    ax2.plot(mid_prices, demand_curve.values, 'o-', label=cat,
             color=COLORS[i], linewidth=2, markersize=4, alpha=0.85)
ax2.set_xlabel('Effective Price ($)')
ax2.set_ylabel('Avg Units Sold')
ax2.set_title('Demand Curves by Category', fontweight='bold')
ax2.legend(fontsize=7.5, ncol=2)
ax2.grid(True, alpha=0.3)

# Plot 3: Revenue vs Discount % per Category
ax3 = fig.add_subplot(gs[1, 0])
for i, cat in enumerate(df['category'].unique()):
    sub = df[df['category'] == cat].copy()
    sub['disc_bin'] = pd.cut(sub['discount_pct'], bins=np.arange(0, 0.55, 0.05))
    rev = sub.groupby('disc_bin')['revenue'].mean()
    midpoints = [b.mid for b in rev.index]
    ax3.plot([m * 100 for m in midpoints], rev.values, 'o-',
             label=cat, color=COLORS[i], linewidth=2, markersize=4, alpha=0.85)
ax3.set_xlabel('Discount % Applied')
ax3.set_ylabel('Avg Revenue ($)')
ax3.set_title('Revenue vs Discount % by Category', fontweight='bold')
ax3.legend(fontsize=7.5, ncol=2)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Plot 4: Feature Importance
ax4 = fig.add_subplot(gs[1, 1])
try:
    fi = pd.read_csv('feature_importance.csv').head(10)
    colors_fi = ['#1565C0'] * len(fi)
    colors_fi[0] = '#B71C1C'
    ax4.barh(fi['feature'][::-1], fi['importance'][::-1],
             color=colors_fi[::-1], edgecolor='white')
    ax4.set_xlabel('Feature Importance (XGBoost)')
    ax4.set_title('Top 10 Most Influential Features', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
except FileNotFoundError:
    ax4.text(0.5, 0.5, 'Feature importance\nnot available\n(no tree model)',
             ha='center', va='center', transform=ax4.transAxes)

plt.savefig('pricing_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
print("\nDashboard saved → pricing_dashboard.png")
print("\nElasticity insights:")
for _, row in elasticity_df.iterrows():
    e = row['Elasticity']
    label = "HIGHLY elastic — discounts strongly increase demand" if e < -1.5 \
        else "Elastic — discounts moderately boost demand" if e < -1.0 \
        else "Inelastic — discounts have limited demand impact"
    print(f"  {row['Category']:<18}: E={e:>6.3f}  → {label}")