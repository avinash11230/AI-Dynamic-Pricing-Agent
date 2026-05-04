import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from agent import PricingAgent
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Smart Retail Pricing Optimizer",
                   page_icon="💹", layout="wide")

@st.cache_resource
def load_agent():
    return PricingAgent()

agent = load_agent()

st.title("💹 Smart Retail Pricing Optimizer")
st.markdown("*AI-powered discount recommendation engine using demand elasticity modeling + SciPy optimization*")

# ── Sidebar Inputs ────────────────────────────────────────────────────
st.sidebar.header("Product Configuration")
product_name       = st.sidebar.text_input("Product Name", "Samsung 4K TV")
category           = st.sidebar.selectbox("Category",
    ['Electronics','Groceries','Clothing','Home & Garden','Toys','Sports','Beauty','Books'])
base_price         = st.sidebar.number_input("Base Price ($)", 1.0, 5000.0, 150.0, step=1.0)
cost               = st.sidebar.number_input("Cost / Unit ($)", 0.5, 4000.0,
                                              round(base_price * 0.55, 2), step=1.0)
store_size         = st.sidebar.selectbox("Store Size", ['Small','Medium','Large'], index=1)
week               = st.sidebar.slider("Week of Year", 1, 52, 26)
competitor_discount= st.sidebar.slider("Competitor Discount (%)", 0, 50, 15) / 100
is_holiday         = int(st.sidebar.checkbox("Holiday Period?"))
seasonality_index  = st.sidebar.slider("Seasonality Index", 0.5, 2.0, 1.0, 0.05)
target             = st.sidebar.radio("Optimization Target",
                                       ['revenue','profit','volume'], index=0)

run = st.sidebar.button("🚀 Run Pricing Agent", type="primary", use_container_width=True)

if run:
    with st.spinner("Running optimization engine..."):
        result = agent.analyze(
            product_name=product_name, category=category,
            base_price=base_price, cost=cost, store_size=store_size,
            week=week, competitor_discount=competitor_discount,
            is_holiday=is_holiday, seasonality_index=seasonality_index,
            target=target
        )

    # ── KPI Cards ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Optimal Discount",     result['optimal_pct_str'])
    c2.metric("Effective Price",      f"${result['effective_price']}")
    c3.metric("Predicted Demand",     f"{result['predicted_demand']:.0f} units")
    c4.metric("Revenue Lift",         f"{result['revenue_lift_pct']:+.1f}%")
    c5.metric("Profit Lift",          f"{result['profit_lift_pct']:+.1f}%")

    st.divider()
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.subheader("Scenario Analysis")
        df_sc = result['scenarios'].copy()
        df_sc[''] = df_sc['is_optimal'].apply(lambda x: '⭐ OPTIMAL' if x else '')
        df_sc = df_sc.drop(columns=['is_optimal'])
        st.dataframe(df_sc.style.apply(
            lambda row: ['background-color: #e8f5e9; font-weight: bold'
                         if '⭐' in str(row['']) else '' for _ in row], axis=1
        ), use_container_width=True)

        # Revenue curve plot
        fig, ax = plt.subplots(figsize=(7, 3.5))
        discounts = [float(d.replace('%', '')) / 100
                     for d in result['scenarios']['discount_pct']]
        revenues  = result['scenarios']['revenue'].tolist()
        profits   = result['scenarios']['profit'].tolist()
        ax.plot([d*100 for d in discounts], revenues, 'o-',
                color='#1565C0', linewidth=2.5, label='Revenue', markersize=5)
        ax.plot([d*100 for d in discounts], profits, 's--',
                color='#2E7D32', linewidth=2, label='Profit', markersize=5, alpha=0.8)
        opt_d = result['optimal_discount'] * 100
        ax.axvline(opt_d, color='#D32F2F', linestyle=':', linewidth=2,
                   label=f'Optimal: {opt_d:.1f}%')
        ax.set_xlabel('Discount (%)')
        ax.set_ylabel('Value ($)')
        ax.set_title(f'Revenue & Profit vs Discount — {product_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        st.subheader("Agent Reasoning")
        st.code(result['reasoning'], language=None)

        # Elasticity gauge
        st.subheader(f"Price Elasticity: {result['elasticity']:.3f}")
        elast_normalized = min(max((abs(result['elasticity']) - 0.5) / 1.5, 0), 1)
        st.progress(elast_normalized,
                    text=f"{'High' if elast_normalized > 0.6 else 'Moderate' if elast_normalized > 0.3 else 'Low'} price sensitivity")

    # ── Elasticity Heatmap (across all categories) ────────────────────
    st.divider()
    st.subheader("Elasticity Dashboard — All Categories")
    try:
        elast_df = pd.read_csv('elasticity_by_category.csv')
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        colors = ['#C62828' if e < -1.5 else '#EF6C00' if e < -1.0 else '#2E7D32'
                  for e in elast_df['Elasticity']]
        bars = ax2.bar(elast_df['Category'], elast_df['Elasticity'],
                       color=colors, edgecolor='white', linewidth=0.5)
        ax2.axhline(-1, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax2.set_ylabel('Elasticity')
        ax2.set_title('Price Elasticity of Demand by Product Category')
        legend_items = [
            mpatches.Patch(color='#C62828', label='Highly elastic (< -1.5)'),
            mpatches.Patch(color='#EF6C00', label='Elastic (-1.5 to -1.0)'),
            mpatches.Patch(color='#2E7D32', label='Inelastic (> -1.0)'),
        ]
        ax2.legend(handles=legend_items, loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig2)
    except FileNotFoundError:
        st.info("Run analysis.py first to generate elasticity data.")

else:
    st.info("Configure your product in the sidebar and click **Run Pricing Agent** to get a recommendation.")
    st.image('pricing_dashboard.png', caption='Analysis Dashboard (from analysis.py)')