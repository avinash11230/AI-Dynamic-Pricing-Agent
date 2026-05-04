# Smart Retail Pricing Optimizer 

> An AI-powered dynamic pricing agent that recommends revenue and profit-maximizing discounts across retail product categories using demand elasticity modeling, XGBoost regression, and SciPy optimization.

---

## What This Project Does

Retailers lose millions annually by applying flat discounts blindly — discounting Electronics during peak holiday season when customers are already buying, or discounting Groceries where demand barely responds to price changes.

This agent solves that. Given a product's price, cost, category, store context, seasonality, and competitor discounts — it finds the mathematically optimal discount percentage and explains **why**, with full scenario analysis across 9 discount levels.

Inspired by real-world pricing systems used at companies like **Procter & Gamble**, where ML-driven pricing agents balance demand elasticity, competitor behavior, and margin constraints simultaneously.

---

## Demo

| Sidebar Configuration | Results Dashboard |
|---|---|
| Select category, price, cost, week, competitor discount | KPI cards + scenario table + revenue curve |

**Three contrasting scenarios the agent handles differently:**

| Product | Situation | Agent Recommendation | Why |
|---|---|---|---|
| Samsung 4K TV | Holiday week, high seasonality | ~0% discount | Demand already elevated — protect margin |
| Generic Laptop | Normal week, competitor at 30% off | ~22% discount | Elastic demand, aggressive competition |
| Organic Milk | Any week | 0% discount | Inelastic (E=-0.75) — discounts waste margin |

---

## Project Architecture

```
smart-pricing-agent/
│
├── data_generator.py          # Synthetic dataset generation (50,000 transactions)
├── model.py                   # Feature engineering + model training + evaluation
├── analysis.py                # Elasticity analysis + visualization dashboard
├── optimizer.py               # SciPy bounded optimization engine + scenario simulator
├── agent.py                   # Autonomous pricing agent with reasoning engine
├── app.py                     # Streamlit interactive dashboard
│
├── retail_data.csv            # Raw generated dataset (50k rows, 15 columns)
├── retail_data_features.csv   # Feature-engineered dataset (21 columns)
├── demand_model.pkl           # Trained XGBoost model (frozen)
├── le_category.pkl            # LabelEncoder for product categories
├── le_store.pkl               # LabelEncoder for store sizes
├── feature_importance.csv     # XGBoost feature importance scores
├── elasticity_by_category.csv # Computed price elasticity per category
└── pricing_dashboard.png      # Generated analytical dashboard
```

---

## How It Works

### 1. Data Generation (`data_generator.py`)
Generates 50,000 realistic retail transactions across 8 product categories. Each row encodes:
- **Price elasticity** as a power-law demand function: `demand ∝ price^elasticity`
- **Seasonality** via sine waves calibrated per category (Toy holiday spikes, Sports spring peaks)
- **Competitor effects**, store size factors, holiday demand boosts, and lognormal noise

This produces a statistically rigorous dataset where ground-truth elasticity values are known — enabling validation of the model's learned elasticity estimates.

### 2. Demand Model (`model.py`)
Trains and compares three models on the task of predicting `units_sold`:

| Model | R² Train | R² Test | Adj R² | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.8073 | 0.8051 | 0.8050 | 71.05 |
| Random Forest | 0.9711 | 0.9438 | 0.9437 | 38.16 |
| **XGBoost** | **0.9621** | **0.9468** | **0.9467** | **37.12** |

**Feature engineering highlights:**
- `log_effective_price` — linearises the curved price-demand relationship for tree models
- `week_sin / week_cos` — cyclical encoding so Week 1 and Week 52 are mathematically adjacent
- `price_vs_category_mean` — relative price signal (cheap vs expensive within its category)
- `price_discount_interaction` — captures the joint effect of base price and discount magnitude

### 3. Elasticity Analysis (`analysis.py`)
Computes price elasticity per category using **log-log regression**:

```
log(demand) = α + ε × log(price)   →   ε = price elasticity
```

| Category | Elasticity | Classification |
|---|---|---|
| Electronics | -2.03 | Highly elastic |
| Toys | -1.89 | Highly elastic |
| Sports | -1.59 | Elastic |
| Clothing | -1.45 | Elastic |
| Beauty | -1.28 | Elastic |
| Home & Garden | -1.18 | Moderately inelastic |
| Books | -0.89 | Inelastic |
| Groceries | -0.75 | Inelastic |

### 4. Optimization Engine (`optimizer.py`)
Uses **SciPy's `minimize_scalar`** with Brent's method to find the exact discount `d ∈ [0%, 50%]` that maximises revenue, profit, or volume:

```python
result = minimize_scalar(
    objective,           # −revenue or −profit (minimize negative = maximize)
    bounds=(0.0, 0.50),
    method='bounded'     # Brent's method — converges in ~10 iterations
)
```

Supports three optimization targets:
- **Revenue** — maximise `effective_price × predicted_demand`
- **Profit** — maximise `(effective_price − cost) × predicted_demand`
- **Volume** — maximise `predicted_demand` (market share / clearance strategy)

### 5. Pricing Agent (`agent.py`)
Wraps the optimizer with a rule-based reasoning engine that generates structured natural language reports:

```
PERCEIVE  → ingest product context (price, cost, category, week, competitor...)
RETRIEVE  → look up category elasticity from computed elasticity table
REASON    → classify elasticity, assess competitor gap, check contextual flags
ACT       → call SciPy optimizer, run scenario simulation (9 discount levels)
EXPLAIN   → generate structured recommendation report with confidence level
RETURN    → optimal discount + lift metrics + scenario table + full reasoning
```

**Contextual flags the agent checks:**
- Holiday period detected → recommend holding margin
- Elevated seasonality → moderate discounts sufficient
- High cost structure (>65% of price) → warn against deep discounts

---

## Visualizations

The analytical dashboard (`pricing_dashboard.png`) contains four plots:

**Top Left — Price Elasticity by Category**
Horizontal bar chart showing elasticity values with a critical threshold line at E=-1.0. Red = highly elastic, orange = elastic, green = inelastic.

**Top Right — Demand Curves by Category**
Price vs units sold curves for all 8 categories. Steep curves = elastic categories. Validates that learned elasticity values match the visual demand curve shapes.

**Bottom Left — Revenue vs Discount %**
Shows how average revenue responds to increasing discounts per category. Electronics revenue climbs with deeper discounts; Grocery revenue is flat — directly actionable insight.

**Bottom Right — Feature Importance**
XGBoost feature importance. `log_effective_price` dominates at ~40% — confirming price is the primary demand driver, as economic theory predicts.

---

## Setup and Installation

**Requirements:**
```
Python 3.8+
```

**Install dependencies:**
```bash
pip install numpy pandas scikit-learn xgboost scipy matplotlib seaborn streamlit joblib
```

**Run in order:**
```bash
# Step 1 — Generate dataset
python data_generator.py

# Step 2 — Train demand model
python model.py

# Step 3 — Compute elasticity + generate dashboard
python analysis.py

# Step 4 — Test the optimizer (sanity check)
python optimizer.py

# Step 5 — Test the agent (3 demo scenarios)
python agent.py

# Step 6 — Launch Streamlit app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Key Results

- **XGBoost demand model** achieves R²=0.9468 on holdout test set (10,000 unseen transactions)
- **Elasticity estimates** from log-log regression closely match ground-truth values embedded in data generator (validates end-to-end pipeline)
- **Optimizer** correctly recommends ~0% discount during holiday/high-seasonality periods — protecting margin when organic demand is already elevated
- **Profit advantage** of optimal vs naive 25% flat discount: up to **+275%** on holiday Electronics scenarios

---

## Tech Stack

| Component | Technology |
|---|---|
| Data generation | NumPy, Pandas |
| Demand modeling | scikit-learn, XGBoost |
| Optimization | SciPy (`minimize_scalar`, Brent's method) |
| Elasticity analysis | Log-log regression (scikit-learn LinearRegression) |
| Visualization | Matplotlib, Seaborn |
| Web application | Streamlit |
| Model persistence | Joblib |

---

## Concepts Demonstrated

- Price elasticity of demand (log-log regression)
- Power-law demand modeling
- Cyclical feature encoding (sin/cos transformation for week of year)
- Feature interaction engineering
- XGBoost regression with cross-validation
- Bounded mathematical optimization (Brent's method)
- Rule-based reasoning engine design
- Adjusted R² vs R² tradeoffs with large datasets

---

## Motivation

This project was built to understand the kind of ML systems deployed in large consumer goods companies — where pricing decisions across thousands of SKUs and hundreds of stores cannot be made manually. The agent architecture mirrors real pricing intelligence systems: a demand model trained on historical data, an optimizer finding the mathematical optimum, and a reasoning layer that makes the recommendation interpretable to business stakeholders.

---

## Author

**Pothanaku Avinash Babu**
Engineering Physics, IIT Madras

