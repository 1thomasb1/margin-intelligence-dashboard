"""MarginIQ | Finance and pricing intelligence in Streamlit.

Run from this project root: python -m streamlit run app.py
All built-in data are fictional. Optional AI transmits aggregates only.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import RAW_COLUMNS, filter_data, generate_synthetic_data, load_demo_data, parse_uploaded_csv
from src.insights import build_ai_payload, generate_ai_commentary, make_rule_based_insights
from src.metrics import (
    budget_vs_actual,
    forecast_revenue,
    group_performance,
    monthly_performance,
    pricing_scenario,
    summarize,
)

st.set_page_config(
    page_title="MarginIQ | Finance Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#13243a"
TEAL = "#00a59a"
BLUE = "#5377bd"

st.markdown("""
<style>
.block-container { padding-top: 1.6rem; max-width: 1450px; }
h1, h2, h3 { letter-spacing: -.035em; }
[data-testid="stMetric"] { border: 1px solid #dfe6ec; border-radius: 12px; padding: .85rem 1rem; }
[data-testid="stMetricValue"] { font-weight: 700; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def fmt_signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.0f}"


def month_string(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def line_chart(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, col, color in [
        ("Net revenue", "revenue", BLUE),
        ("Gross profit", "gross_profit", TEAL),
    ]:
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly[col], name=name,
            mode="lines+markers", line={"color": color, "width": 3},
            hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        template="plotly_white", height=350, margin={"t": 10, "l": 10, "r": 10, "b": 10},
        hovermode="x unified", yaxis_title="USD", xaxis_title="",
        legend={"orientation": "h", "y": 1.13, "x": 0},
    )
    return fig


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    try:
        key = st.secrets.get("OPENAI_API_KEY", key)
    except (FileNotFoundError, KeyError):
        pass
    return key


@st.cache_data(show_spinner=False)
def demo_data():
    return load_demo_data()


st.title("MarginIQ")
st.caption("Revenue & Profitability Intelligence  ·  FP&A + SQL + Pricing Analytics + Optional AI")
st.info("**Portfolio demonstration:** All built-in brands, customer segments, transactions, and budgets are simulated. No real employer data is included.")

with st.sidebar:
    st.header("Data & filters")
    mode = st.radio("Data source", ("Fictional demo", "Upload your CSV"), horizontal=False)
    budgets = None
    if mode == "Fictional demo":
        transactions, budgets = demo_data()
        st.caption(f"{len(transactions):,} fictional invoices in SQLite. Seed: 42.")
        if st.button("Reload demo database", help="Refresh cached results after regenerating SQLite"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.warning("Only upload data you have permission to process. Publicly deployed apps are not appropriate for confidential company records.")
        example_csv = generate_synthetic_data(n=12)[RAW_COLUMNS].to_csv(index=False).encode()
        st.download_button("Download CSV template", example_csv, "marginiq_template.csv", "text/csv")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is None:
            st.caption("Select a file to activate the analysis. Monetary fields are USD; discount rates are decimals (e.g. 0.15 = 15%).")
            st.stop()
        try:
            transactions = parse_uploaded_csv(uploaded)
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            st.error(f"Could not process CSV: {exc}")
            st.stop()
        st.caption(f"{len(transactions):,} uploaded invoices. Budget comparisons disabled.")

    brands = st.multiselect(
        "Brand", sorted(transactions["brand"].unique().tolist()),
        default=sorted(transactions["brand"].unique().tolist()),
    )
    segments = st.multiselect(
        "Customer segment", sorted(transactions["customer_segment"].unique().tolist()),
        default=sorted(transactions["customer_segment"].unique().tolist()),
    )
    regions = st.multiselect(
        "Region", sorted(transactions["region"].unique().tolist()),
        default=sorted(transactions["region"].unique().tolist()),
    )
    all_months = sorted(transactions["order_date"].dt.to_period("M").astype(str).unique().tolist())
    if len(all_months) > 1:
        start_month, end_month = st.select_slider(
            "Reporting months (whole months)", options=all_months,
            value=(all_months[0], all_months[-1]),
        )
    else:
        start_month = end_month = all_months[0]
        st.caption(f"Reporting month: {start_month}")
    st.caption("Filters update all analytics together.")

start_date = pd.Timestamp(start_month + "-01")
end_date = pd.Timestamp(end_month + "-01") + pd.offsets.MonthEnd(0)
filtered = filter_data(transactions, brands, segments, regions, start_date, end_date)
if filtered.empty:
    st.warning("Nothing matches these filters. Expand the date range or choose more groups.")
    st.stop()

monthly = monthly_performance(filtered)
score = summarize(filtered)
insights = make_rule_based_insights(filtered)

if budgets is not None:
    budget_selected = budgets.loc[
        budgets["brand"].isin(brands)
        & budgets["customer_segment"].isin(segments)
        & budgets["region"].isin(regions)
        & budgets["month"].between(start_date, end_date)
    ].copy()
    monthly_bva = budget_vs_actual(monthly, budget_selected)
    budget_rev = float(budget_selected["budget_revenue"].sum())
    budget_gp = float(budget_selected["budget_gross_profit"].sum())
    budget_sum = {
        "budget_revenue": round(budget_rev, 2),
        "actual_minus_budget_revenue": round(score["revenue"] - budget_rev, 2),
        "budget_gross_profit": round(budget_gp, 2),
        "actual_minus_budget_gp": round(score["gross_profit"] - budget_gp, 2),
    }
else:
    monthly_bva = None
    budget_sum = None


tab_overview, tab_pricing, tab_plan, tab_audit, tab_ai, tab_method = st.tabs([
    "📊 Overview", "💹 Pricing simulator", "📈 Budget & forecast",
    "🧾 Transaction audit", "🤖 AI analyst", "📘 Methodology",
])

with tab_overview:
    st.subheader("Executive scorecard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net revenue", fmt_money(score["revenue"]))
    c2.metric("Gross profit", fmt_money(score["gross_profit"]))
    c3.metric("Gross margin", f"{score['gross_margin']:.1%}")
    c4.metric("Weighted discount", f"{score['weighted_discount']:.1%}")
    c5.metric("Over-policy price gap", fmt_money(score["policy_gap"]),
              help="Illustrative price difference assuming unchanged volumes and contract terms; NOT recoverable cash.")
    st.caption(f"{score['transactions']:,} transactions · {score['units']:,} units · {score['policy_exceptions']:,} orders above fictional policy")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("#### Revenue and gross profit over time")
        st.plotly_chart(line_chart(monthly), use_container_width=True)
    with right:
        st.markdown("#### Over-policy gap by brand")
        brands_chart = group_performance(filtered, "brand").nlargest(10, "policy_gap")
        fig = px.bar(
            brands_chart.sort_values("policy_gap"), x="policy_gap", y="brand", orientation="h",
            labels={"policy_gap": "Illustrative gap (USD)", "brand": ""},
            color_discrete_sequence=[TEAL],
        )
        fig.update_layout(template="plotly_white", height=350,
                          margin={"t": 10, "l": 10, "r": 10, "b": 10}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Gross margin by customer segment")
        segment = group_performance(filtered, "customer_segment").sort_values("gross_margin")
        fig = px.bar(segment, x="gross_margin", y="customer_segment", orientation="h",
                     labels={"gross_margin": "Gross margin", "customer_segment": ""},
                     color_discrete_sequence=[BLUE])
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(template="plotly_white", height=300, showlegend=False,
                          margin={"t": 10, "l": 10, "r": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Net revenue by region")
        region = group_performance(filtered, "region")
        fig = px.pie(region, values="revenue", names="region", hole=0.55,
                     color_discrete_sequence=[NAVY, TEAL, BLUE, "#9db9ce"])
        fig.update_layout(height=300, margin={"t": 10, "l": 10, "r": 10, "b": 10}, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Automated financial observations")
    for bullet in insights:
        st.markdown("- " + bullet)

with tab_pricing:
    st.subheader("Pricing-policy scenario simulator")
    st.write("Stress-test an illustrative policy correction and a uniform change in unit volume. This is a **sensitivity model**, not a demand forecast.")
    left, right = st.columns(2)
    with left:
        correction_pct = st.slider("Close this share of above-policy discounts", 0, 100, 50, 5, format="%d%%")
    with right:
        volume_pct = st.slider("Assumed volume change on all orders", -25, 25, 0, 1, format="%d%%")
    scenario = pricing_scenario(filtered, correction_pct / 100, volume_pct / 100)
    c1, c2, c3 = st.columns(3)
    c1.metric("Modeled revenue", fmt_money(scenario["scenario_revenue"]),
              delta=fmt_signed_money(scenario["delta_revenue"]))
    c2.metric("Modeled gross profit", fmt_money(scenario["scenario_gp"]),
              delta=fmt_signed_money(scenario["delta_gp"]))
    c3.metric("Modeled gross margin", f"{scenario['scenario_margin']:.1%}",
              delta=f"{(scenario['scenario_margin'] - scenario['baseline_margin'])*100:+.2f} pp")

    fig = go.Figure(data=[
        go.Bar(name="Baseline", x=["Revenue", "Gross profit"],
               y=[scenario["baseline_revenue"], scenario["baseline_gp"]], marker_color=BLUE),
        go.Bar(name="Scenario", x=["Revenue", "Gross profit"],
               y=[scenario["scenario_revenue"], scenario["scenario_gp"]], marker_color=TEAL),
    ])
    fig.update_layout(barmode="group", template="plotly_white", height=330,
                      margin={"t": 10, "l": 10, "r": 10, "b": 10}, yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)
    if scenario["break_even_volume_loss"] is not None and correction_pct > 0:
        st.info(
            "With the selected discount correction **before** the volume adjustment, "
            f"the modeled gross-profit break-even volume decline is approximately "
            f"**{scenario['break_even_volume_loss']:.1%}**. This is a hypothetical threshold, "
            "not an elasticity estimate."
        )
    st.caption("Assumptions: each above-policy discount is reduced by the chosen fraction; all transaction units change proportionally; list prices and unit costs remain fixed; no change in customer/product mix, taxes, overhead, or contracts.")

with tab_plan:
    st.subheader("Budget variance and forward revenue trend")
    if monthly_bva is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Synthetic revenue budget", fmt_money(budget_sum["budget_revenue"]))
        col2.metric("Revenue variance", fmt_signed_money(budget_sum["actual_minus_budget_revenue"]))
        col3.metric("Gross profit variance", fmt_signed_money(budget_sum["actual_minus_budget_gp"]))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_bva["month"], y=monthly_bva["budget_revenue"],
                             name="Simulated revenue budget", marker_color="#c6d9e7"))
        fig.add_trace(go.Scatter(x=monthly_bva["month"], y=monthly_bva["revenue"],
                                 name="Revenue", mode="lines+markers",
                                 line={"color": NAVY, "width": 3}))
        fig.update_layout(template="plotly_white", height=330, yaxis_title="USD",
                          margin={"t": 10, "l": 10, "r": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Demo budgets are constructed from simulated realized sales plus random variations. They illustrate FP&A variance reporting, not a genuinely independent budget process.")
    else:
        st.info("A matching budget is available only for the built-in fictional demo data.")

    st.markdown("#### Three-month baseline revenue trend")
    forecast = forecast_revenue(monthly, months_ahead=3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["revenue"], name="Historical revenue",
                             mode="lines+markers", line={"color": BLUE, "width": 3}))
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=[monthly.iloc[-1]["month"], *forecast["month"].tolist()],
            y=[monthly.iloc[-1]["revenue"], *forecast["forecast_revenue"].tolist()],
            name="Simple trend (illustrative)", mode="lines+markers",
            line={"color": TEAL, "width": 3, "dash": "dash"},
        ))
    fig.update_layout(template="plotly_white", height=350, yaxis_title="USD",
                      margin={"t": 10, "l": 10, "r": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Uses a linear trend fit on up to the last six filtered months. No seasonality, confidence interval, scenario weighting, or forecast validation. Months with few transactions can make this unreliable.")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(monthly[["month", "revenue", "gross_profit", "gross_margin", "policy_gap"]],
                     hide_index=True, use_container_width=True)
    with c2:
        st.dataframe(forecast, hide_index=True, use_container_width=True)
        st.download_button("Download monthly performance CSV", monthly.to_csv(index=False),
                           file_name="marginiq_monthly_performance.csv", mime="text/csv")

with tab_audit:
    st.subheader("Order-level review")
    st.write("Verify where the pricing-policy arithmetic comes from, then export the filtered results.")
    audit = filtered.sort_values("policy_gap", ascending=False)
    selected_cols = [
        "order_id", "order_date", "brand", "customer_segment", "region", "product_group",
        "units", "list_price", "discount_rate", "allowed_discount_rate", "unit_cost",
        "revenue", "cogs", "gross_profit", "policy_gap",
    ]
    st.dataframe(audit[selected_cols], hide_index=True, use_container_width=True, height=460,
                 column_config={
                     "discount_rate": st.column_config.NumberColumn("Actual discount", format="%.2f"),
                     "allowed_discount_rate": st.column_config.NumberColumn("Policy discount", format="%.2f"),
                     "revenue": st.column_config.NumberColumn("Net revenue ($)", format="%.2f"),
                     "policy_gap": st.column_config.NumberColumn("Over-policy gap ($)", format="%.2f"),
                 })
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download selected transactions (CSV)", audit[selected_cols].to_csv(index=False),
                           file_name="marginiq_transaction_audit.csv", mime="text/csv")
    with col2:
        summary_export = group_performance(filtered, "brand")
        st.download_button("Download brand scorecard (CSV)", summary_export.to_csv(index=False),
                           file_name="marginiq_brand_scorecard.csv", mime="text/csv")
    st.markdown("#### Brand scorecard")
    st.dataframe(summary_export, hide_index=True, use_container_width=True)

with tab_ai:
    st.subheader("AI-assisted FP&A commentary")
    st.write("The **rule-based analyst** below works without any API key and uses verified calculations.")
    for observation in insights:
        st.markdown("- " + observation)
    st.divider()
    st.markdown("#### Optional OpenAI executive brief")
    st.caption("Only aggregated metrics, a modeled scenario and the statements above are sent to the API. No raw rows, order IDs, or customer data are sent by this feature.")
    key = get_api_key()
    if not key:
        st.info("To activate AI generation, set `OPENAI_API_KEY` in your environment or `.streamlit/secrets.toml` (see README). The rest of the app works without it.")
    else:
        st.caption("Uses your configured API key. API usage may incur charges.")
    # Match AI narrative to the scenario controls without requiring the visitor to open that tab.
    scenario_for_ai = pricing_scenario(filtered, correction_pct / 100, volume_pct / 100)
    payload = build_ai_payload(score, insights, scenario_for_ai, budget_sum)
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if st.button("Generate executive brief with AI", type="primary", disabled=not bool(key)):
        try:
            with st.spinner("Reviewing aggregate finance metrics..."):
                reply = generate_ai_commentary(payload, key, model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
            st.session_state["ai_result"] = {"hash": payload_hash, "text": reply}
        except Exception as exc:
            st.error(f"AI generation failed: {type(exc).__name__}. Verify the key, model, network, and API billing.")
    saved = st.session_state.get("ai_result")
    if saved and saved["hash"] == payload_hash:
        st.markdown(saved["text"])
        st.download_button("Download AI brief (TXT)", saved["text"],
                           file_name="marginiq_ai_brief.txt", mime="text/plain")
    elif saved:
        st.caption("Your filters or scenario changed; regenerate the brief for the current numbers.")
    st.warning("AI text can contain interpretation errors. Cross-check claims against the scorecard before using it in a presentation or decision.")

with tab_method:
    st.subheader("Financial definitions and reproducibility")
    st.markdown("""
| Metric | Formula |
|---|---|
| List revenue | Units × List price |
| Net revenue | List revenue × (1 − Actual discount) |
| Cost of goods sold | Units × Unit cost |
| Gross profit | Net revenue − COGS |
| Gross margin | Sum(Gross profit) ÷ Sum(Net revenue) |
| Weighted discount | 1 − Sum(Net revenue) ÷ Sum(List revenue) |
| Over-policy price gap | List revenue × max(Actual discount − Allowed discount, 0) |
| Revenue variance | Actual revenue − Fictional budget revenue |
| Simulated revenue | Adjusted list-price revenue × (1 + Volume change) |

The application uses SQLite to store demo invoices and budget observations. Pandas computes
invoice calculations and group-level finance metrics. Graphs are built with Plotly.

**Interpretation limits:** The over-policy gap is not proven margin leakage, collectable
revenue, or increased cash flow. It omits demand elasticity, contracts, competitor responses,
volume/mix changes, selling costs, overhead, tax, and working capital. The optional AI
analyst receives only supplied aggregates and must be manually reviewed.

**Data safety:** The bundled data are synthetic. Custom CSV inputs remain in the running
app session and are not written into the sample SQLite database. Avoid uploading
confidential data to any public deployment.
    """)
    st.markdown("#### Example SQL query")
    st.code("""SELECT brand,
       ROUND(SUM(revenue), 2) AS revenue,
       ROUND(100.0 * SUM(gross_profit) / NULLIF(SUM(revenue), 0), 2) AS gross_margin_pct,
       ROUND(SUM(policy_gap), 2) AS over_policy_price_gap
FROM transactions
GROUP BY brand
ORDER BY over_policy_price_gap DESC;""", language="sql")
    st.caption("MarginIQ v1.0 · Demonstration software, not a financial recommendation or pricing decision tool.")
