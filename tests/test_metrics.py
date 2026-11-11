"""Financial logic must remain independently testable without Streamlit."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    add_financial_metrics, budget_vs_actual, forecast_revenue,
    group_performance, monthly_performance, pricing_scenario, summarize,
)


def fixtures():
    raw = pd.DataFrame({
        "order_id": ["A", "B"],
        "order_date": pd.to_datetime(["2026-01-10", "2026-02-11"]),
        "brand": ["Demo A", "Demo B"],
        "customer_segment": ["Enterprise", "OEM"],
        "region": ["North", "South"],
        "product_group": ["Pumps", "Valves"],
        "channel": ["Direct", "Online"],
        "units": [2, 4],
        "list_price": [100.0, 50.0],
        "discount_rate": [0.20, 0.10],
        "allowed_discount_rate": [0.10, 0.10],
        "unit_cost": [50.0, 25.0],
    })
    return add_financial_metrics(raw)


def test_invoice_math():
    rows = fixtures()
    assert rows["revenue"].tolist() == pytest.approx([160.0, 180.0])
    assert rows["cogs"].tolist() == pytest.approx([100.0, 100.0])
    assert rows["gross_profit"].tolist() == pytest.approx([60.0, 80.0])
    assert rows["policy_gap"].tolist() == pytest.approx([20.0, 0.0])


def test_metric_function_does_not_modify_input():
    rows = fixtures()
    copy = rows.copy(deep=True)
    add_financial_metrics(rows)
    pd.testing.assert_frame_equal(rows, copy)


def test_summary_calculates_weighted_ratios():
    stats = summarize(fixtures())
    assert stats["revenue"] == pytest.approx(340)
    assert stats["gross_profit"] == pytest.approx(140)
    assert stats["gross_margin"] == pytest.approx(140 / 340)
    assert stats["weighted_discount"] == pytest.approx(1 - 340 / 400)
    assert stats["policy_exceptions"] == 1
    assert stats["transactions"] == 2


def test_full_discount_policy_close_at_flat_volume_equals_gap():
    df = fixtures()
    result = pricing_scenario(df, recovery_fraction=1, volume_change=0)
    assert result["delta_revenue"] == pytest.approx(20)
    assert result["delta_gp"] == pytest.approx(20)
    assert result["scenario_gp"] == pytest.approx(160)
    assert result["break_even_volume_loss"] == pytest.approx(1 - 140 / 160)


def test_zero_discount_change_zero_delta():
    r = pricing_scenario(fixtures(), 0, 0)
    assert r["delta_revenue"] == pytest.approx(0)
    assert r["delta_gp"] == pytest.approx(0)


def test_uniform_volume_response_applies_to_both_revenue_and_cogs():
    r = pricing_scenario(fixtures(), 1, -0.10)
    assert r["scenario_revenue"] == pytest.approx(360 * 0.9)
    assert r["scenario_gp"] == pytest.approx((360 - 200) * 0.9)


@pytest.mark.parametrize("discount", [-0.1, 1.1])
def test_invalid_discounts_rejected(discount):
    df = fixtures()
    df["discount_rate"] = discount
    with pytest.raises(ValueError):
        add_financial_metrics(df)


def test_invalid_units_rejected():
    df = fixtures()
    df.loc[0, "units"] = 0
    with pytest.raises(ValueError):
        add_financial_metrics(df)


def test_invalid_scenario_inputs_rejected():
    with pytest.raises(ValueError):
        pricing_scenario(fixtures(), 1.1, 0)
    with pytest.raises(ValueError):
        pricing_scenario(fixtures(), 0.5, -1.1)


def test_monthly_grouping():
    monthly = monthly_performance(fixtures())
    assert len(monthly) == 2
    assert monthly["revenue"].tolist() == pytest.approx([160, 180])


def test_grouped_brand_margin_is_correct():
    grouped = group_performance(fixtures(), "brand")
    assert len(grouped) == 2
    demo_a = grouped.set_index("brand").loc["Demo A"]
    assert demo_a["gross_margin"] == pytest.approx(60 / 160)


def test_groupby_rejects_arbitrary_names():
    with pytest.raises(ValueError):
        group_performance(fixtures(), "order_id")


def test_budget_variance():
    monthly = monthly_performance(fixtures())
    budgets = pd.DataFrame({
        "month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "budget_revenue": [150.0, 200.0],
        "budget_gross_profit": [55.0, 70.0],
    })
    result = budget_vs_actual(monthly, budgets)
    assert result["revenue_variance"].tolist() == pytest.approx([10, -20])
    assert result["gp_variance"].tolist() == pytest.approx([5, 10])


def test_forecast_has_three_future_months():
    monthly = pd.DataFrame({
        "month": pd.date_range("2026-01-01", periods=6, freq="MS"),
        "revenue": [100, 110, 120, 130, 140, 150],
    })
    forecast = forecast_revenue(monthly, months_ahead=3)
    assert forecast["month"].tolist() == list(pd.date_range("2026-07-01", periods=3, freq="MS"))
    assert forecast["forecast_revenue"].tolist() == pytest.approx([160, 170, 180])


def test_single_month_forecast_flat():
    monthly = pd.DataFrame({"month": [pd.Timestamp("2026-05-01")], "revenue": [200.0]})
    result = forecast_revenue(monthly)
    assert result["forecast_revenue"].tolist() == pytest.approx([200, 200, 200])


def test_empty_forecast():
    out = forecast_revenue(pd.DataFrame(columns=["month", "revenue"]))
    assert out.empty
