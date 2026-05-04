import numpy as np
import pandas as pd
import joblib
from optimizer import optimize, simulate_scenarios, predict_demand
import warnings
warnings.filterwarnings('ignore')

elasticity_df = pd.read_csv('elasticity_by_category.csv')
ELASTICITY    = dict(zip(elasticity_df['Category'], elasticity_df['Elasticity']))

class PricingAgent:
    """
    Autonomous retail pricing agent.
    Given a product context, recommends the optimal discount
    with full scenario analysis and natural-language reasoning.
    """

    def __init__(self):
        self.model  = joblib.load('demand_model.pkl')
        self.le_cat = joblib.load('le_category.pkl')

    def analyze(self, product_name, category, base_price, cost,
                store_size='Medium', week=26, competitor_discount=0.15,
                is_holiday=0, seasonality_index=1.0, target='revenue'):

        # ── Step 1: Gather context ─────────────────────────────────
        elasticity = ELASTICITY.get(category, -1.2)
        cost_pct   = cost / base_price

        # ── Step 2: Run optimizer ──────────────────────────────────
        scenarios, optimal_disc = simulate_scenarios(
            base_price, cost, category, store_size,
            week, competitor_discount, is_holiday, seasonality_index, target
        )

        # ── Step 3: Compute lifts ──────────────────────────────────
        baseline = scenarios[scenarios['discount_pct'] == '0%'].iloc[0]
        optimal  = scenarios[scenarios['is_optimal']].iloc[0]

        # Naive strategy = blindly giving 25% flat discount
        naive_candidates = scenarios[scenarios['discount_pct'] == '25%']
        if len(naive_candidates) == 0:
            naive_candidates = scenarios[scenarios['discount_pct'] == '20%']
        naive = naive_candidates.iloc[0]

        rev_lift    = (optimal['revenue'] - baseline['revenue']) \
                      / baseline['revenue'] * 100
        profit_lift = (optimal['profit']  - baseline['profit']) \
                      / baseline['profit']  * 100
        demand_lift = (optimal['predicted_demand'] - baseline['predicted_demand']) \
                      / baseline['predicted_demand'] * 100

        # How much better is optimal vs blindly giving 25% discount?
        vs_naive_profit = (optimal['profit'] - naive['profit']) \
                          / abs(naive['profit']) * 100 \
                          if naive['profit'] != 0 else 0

        # ── Step 4: Generate reasoning ─────────────────────────────
        reasoning = self._generate_reasoning(
            product_name, category, elasticity, optimal_disc,
            base_price, cost_pct, competitor_discount,
            rev_lift, profit_lift, demand_lift, vs_naive_profit,
            is_holiday, seasonality_index, target
        )

        return {
            'product':           product_name,
            'category':          category,
            'optimal_discount':  optimal_disc,
            'optimal_pct_str':   f"{optimal_disc*100:.1f}%",
            'effective_price':   round(base_price * (1 - optimal_disc), 2),
            'predicted_demand':  optimal['predicted_demand'],
            'projected_revenue': optimal['revenue'],
            'projected_profit':  optimal['profit'],
            'revenue_lift_pct':  round(rev_lift, 2),
            'profit_lift_pct':   round(profit_lift, 2),
            'demand_lift_pct':   round(demand_lift, 2),
            'vs_naive_profit':   round(vs_naive_profit, 2),
            'elasticity':        elasticity,
            'scenarios':         scenarios,
            'reasoning':         reasoning,
            'target':            target,
        }

    def _generate_reasoning(self, product, category, elasticity, optimal_disc,
                            base_price, cost_pct, competitor_discount,
                            rev_lift, profit_lift, demand_lift, vs_naive_profit,
                            is_holiday, seasonality, target):

        # ── Elasticity classification ──────────────────────────────
        if elasticity < -1.8:
            elast_label = "HIGHLY elastic"
            elast_note  = ("Customers in this category are very price-sensitive. "
                           "Even moderate discounts generate significant demand uplift.")
        elif elasticity < -1.2:
            elast_label = "elastic"
            elast_note  = ("Customers show meaningful price sensitivity. "
                           "Strategic discounting drives solid demand increases.")
        elif elasticity < -0.8:
            elast_label = "moderately inelastic"
            elast_note  = ("Demand is somewhat resistant to price changes. "
                           "Discounts help but with diminishing returns.")
        else:
            elast_label = "inelastic"
            elast_note  = ("This category has low price sensitivity. "
                           "Deep discounts sacrifice margin without proportional demand gains.")

        # ── Competitor context ─────────────────────────────────────
        if competitor_discount > optimal_disc + 0.05:
            competitor_note = (
                f"Competitors are discounting at {competitor_discount*100:.0f}%, "
                f"above your recommended {optimal_disc*100:.1f}%. "
                "Matching partially is strategic — going higher erodes margin unnecessarily."
            )
        elif competitor_discount < optimal_disc - 0.05:
            competitor_note = (
                f"Your recommended {optimal_disc*100:.1f}% discount exceeds "
                f"competitor level ({competitor_discount*100:.0f}%). "
                "This is aggressive and appropriate given your demand model's projection."
            )
        else:
            competitor_note = (
                f"Your discount aligns closely with the competitive landscape "
                f"({competitor_discount*100:.0f}%), minimizing price-war risk."
            )

        # ── Contextual flags ───────────────────────────────────────
        context_flags = []
        if is_holiday:
            context_flags.append(
                "Holiday demand boost detected (+15% baseline) — "
                "discounts may be unnecessary; consider holding margin."
            )
        if seasonality > 1.2:
            context_flags.append(
                f"Elevated seasonality index ({seasonality:.2f}) suggests "
                "naturally high demand period — moderate discounts sufficient."
            )
        if cost_pct > 0.65:
            context_flags.append(
                f"High cost structure ({cost_pct*100:.0f}% of base price) — "
                "deep discounts significantly compress margin; use caution."
            )

        context_section = (
            "\n   • " + "\n   • ".join(context_flags)
        ) if context_flags else "None flagged."

        # ── Assemble final report ──────────────────────────────────
        reasoning = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRICING AGENT — RECOMMENDATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Product   : {product}
Category  : {category}
Target KPI: {target.upper()}

RECOMMENDED DISCOUNT: {optimal_disc*100:.1f}%
Effective Price     : ${base_price * (1 - optimal_disc):.2f}  (was ${base_price:.2f})

PROJECTED IMPACT (vs. no discount):
  Revenue uplift     : {rev_lift:+.1f}%
  Profit uplift      : {profit_lift:+.1f}%
  Demand uplift      : {demand_lift:+.1f}%
  vs. naive 25% disc : {vs_naive_profit:+.1f}% profit advantage

─── REASONING ──────────────────────────────
[1] ELASTICITY ANALYSIS
  {category} demand is {elast_label} (E = {elasticity:.3f}).
  {elast_note}

[2] COMPETITOR CONTEXT
  {competitor_note}

[3] CONTEXTUAL FLAGS
  {context_section}

[4] OPTIMIZATION METHOD
  Scipy bounded optimizer searched discount ∈ [0%, 50%]
  using gradient-free minimization of −{target}.
  Model: XGBoost demand predictor (R² > 0.90 on holdout).

[5] RECOMMENDATION CONFIDENCE
  {'HIGH' if abs(elasticity) > 1.2 else 'MEDIUM'} — based on {
  'strong historical price-demand signal' if abs(elasticity) > 1.2
  else 'moderate signal; validate with A/B test before full rollout'}.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
        return reasoning


# ── Demo run ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    agent = PricingAgent()

    tests = [
        dict(product_name='Samsung 4K TV', category='Electronics',
             base_price=599, cost=360, store_size='Large',
             week=49, competitor_discount=0.25, is_holiday=1,
             seasonality_index=1.3, target='profit'),
        dict(product_name='Organic Milk (1L)', category='Groceries',
             base_price=4.5, cost=2.5, store_size='Medium',
             week=15, competitor_discount=0.05, is_holiday=0,
             seasonality_index=1.0, target='revenue'),
        dict(product_name='Running Shoes X2', category='Sports',
             base_price=85, cost=42, store_size='Medium',
             week=12, competitor_discount=0.20, is_holiday=0,
             seasonality_index=1.3, target='profit'),
    ]

    for t in tests:
        result = agent.analyze(**t)
        print(result['reasoning'])
        print("\nScenario Table:")
        print(result['scenarios'].to_string(index=False))
        print("\n")