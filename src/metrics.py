"""Auditable financial calculations for MarginIQ.

Money is represented in USD floats for a portfolio demonstration. For a
production accounting system, use decimal arithmetic and approved rounding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_FINANCIAL_COLUMNS = (
    "units", "list_price", "discount_rate", "allowed_discount_rate", "unit_cost"
)


def add_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute invoice-level financial metrics without changing the input.

    The policy gap is an *arithmetic price difference*, not cash that can
    necessarily be recovered. It ignores price elasticity and contracts.
    """
    missing = set(REQUIRED_FINANCIAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    for key in REQUIRED_FINANCIAL_COLUMNS:
        out[key] = pd.to_numeric(out[key], errors="raise")
        if not np.isfinite(out[key].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {key} includes missing or infinite values")

    if (out["units"] <= 0).any() or (out["units"] % 1 != 0).any():
        raise ValueError("units must be positive whole numbers")
    if (out[["list_price", "unit_cost"]] < 0).any().any():
        raise ValueError("Prices and costs cannot be negative")
    for key in ("discount_rate", "allowed_discount_rate"):
        if ((out[key] < 0) | (out[key] > 1)).any():
            raise ValueError(f"{key} must be a decimal between 0 and 1")

    out["list_revenue"] = out["units"] * out["list_price"]
    out["revenue"] = out["list_revenue"] * (1 - out["discount_rate"])
    out["cogs"] = out["units"] * out["unit_cost"]
    out["gross_profit"] = out["revenue"] - out["cogs"]
    out["policy_gap"] = out["list_revenue"] * (
        out["discount_rate"] - out["allowed_discount_rate"]
    ).clip(lower=0)
    out["above_policy"] = out["discount_rate"] > out["allowed_discount_rate"] + 1e-9
    return out


def summarize(df: pd.DataFrame) -> dict[str, float]:
    """Summarize a set of invoices. Ratios are calculated from totals."""
    revenue = float(df["revenue"].sum())
    gp = float(df["gross_profit"].sum())
    list_revenue = float(df["list_revenue"].sum())
    return {
        "revenue": revenue,
        "gross_profit": gp,
        "cogs": float(df["cogs"].sum()),
        "gross_margin": gp / revenue if revenue else 0.0,
        "policy_gap": float(df["policy_gap"].sum()),
        "weighted_discount": 1 - revenue / list_revenue if list_revenue else 0.0,
        "transactions": int(len(df)),
        "units": int(df["units"].sum()),
        "policy_exceptions": int(df["above_policy"].sum()),
    }


def group_performance(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate by any approved dimension."""
    approved = {"brand", "customer_segment", "region", "product_group", "channel"}
    if by not in approved:
        raise ValueError(f"Unsupported group: {by}")
    result = df.groupby(by, as_index=False).agg(
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        policy_gap=("policy_gap", "sum"),
        orders=("order_id", "count"),
    )
    result["gross_margin"] = np.where(
        result["revenue"] != 0,
        result["gross_profit"] / result["revenue"], 0.0
    )
    return result.sort_values("revenue", ascending=False).reset_index(drop=True)


def monthly_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly realized performance; assumes order_date is datetime-like."""
    working = df.copy()
    working["month"] = pd.to_datetime(working["order_date"]).dt.to_period("M").dt.to_timestamp()
    out = working.groupby("month", as_index=False).agg(
        revenue=("revenue", "sum"),
        gross_profit=("gross_profit", "sum"),
        cogs=("cogs", "sum"),
        policy_gap=("policy_gap", "sum"),
        transactions=("order_id", "count"),
    )
    out["gross_margin"] = np.where(
        out["revenue"] != 0, out["gross_profit"] / out["revenue"], 0.0
    )
    return out.sort_values("month").reset_index(drop=True)


def pricing_scenario(
    df: pd.DataFrame, recovery_fraction: float, volume_change: float
) -> dict[str, float]:
    """Model a discount-correction scenario with a uniform volume response.

    recovery_fraction=1 enforces policy on every above-policy invoice.
    volume_change=-0.05 assumes 5% fewer units on *all* modeled invoices.
    Unit cost and mix stay unchanged; no contract or demand modeling is implied.
    """
    if not 0 <= recovery_fraction <= 1:
        raise ValueError("recovery_fraction must be between 0 and 1")
    if not -1 <= volume_change <= 1:
        raise ValueError("volume_change must be between -1 and 1")

    actual = summarize(df)
    excessive = (df["discount_rate"] - df["allowed_discount_rate"]).clip(lower=0)
    proposed_discounts = df["discount_rate"] - recovery_fraction * excessive
    before_volume_revenue = float((df["list_revenue"] * (1 - proposed_discounts)).sum())
    before_volume_gp = before_volume_revenue - actual["cogs"]
    factor = 1 + volume_change
    scenario_revenue = before_volume_revenue * factor
    scenario_cogs = actual["cogs"] * factor
    scenario_gp = scenario_revenue - scenario_cogs
    # At this relative loss, baseline gross profit equals scenario gross profit.
    max_volume_loss = (
        max(0.0, min(1.0, 1 - actual["gross_profit"] / before_volume_gp))
        if before_volume_gp > 0 and actual["gross_profit"] >= 0 else None
    )
    return {
        "baseline_revenue": actual["revenue"],
        "baseline_gp": actual["gross_profit"],
        "baseline_margin": actual["gross_margin"],
        "scenario_revenue": scenario_revenue,
        "scenario_gp": scenario_gp,
        "scenario_margin": scenario_gp / scenario_revenue if scenario_revenue else 0.0,
        "delta_revenue": scenario_revenue - actual["revenue"],
        "delta_gp": scenario_gp - actual["gross_profit"],
        "break_even_volume_loss": max_volume_loss,
        "full_policy_gap": actual["policy_gap"],
    }


def budget_vs_actual(
    monthly: pd.DataFrame, budgets: pd.DataFrame
) -> pd.DataFrame:
    """Compare *matching* monthly series (already filtered to same dimensions)."""
    b = budgets.groupby("month", as_index=False)[
        ["budget_revenue", "budget_gross_profit"]
    ].sum()
    out = monthly.merge(b, on="month", how="left", validate="one_to_one")
    out["revenue_variance"] = out["revenue"] - out["budget_revenue"]
    out["gp_variance"] = out["gross_profit"] - out["budget_gross_profit"]
    out["revenue_variance_pct"] = np.where(
        out["budget_revenue"] != 0,
        out["revenue_variance"] / out["budget_revenue"], np.nan
    )
    return out


def forecast_revenue(monthly: pd.DataFrame, months_ahead: int = 3, lookback: int = 6) -> pd.DataFrame:
    """Simple trend baseline using up to 6 recent full months; no seasonality.

    Forecasts are illustrative, not model-validated predictions. A single
    observation uses a flat run rate. Results are clipped at zero.
    """
    if not 1 <= months_ahead <= 12:
        raise ValueError("months_ahead must be 1..12")
    if not 2 <= lookback <= 24:
        raise ValueError("lookback must be 2..24")
    if monthly.empty:
        return pd.DataFrame(columns=["month", "forecast_revenue"])
    recent = monthly.sort_values("month").tail(lookback)
    y = recent["revenue"].to_numpy(dtype=float)
    if len(y) >= 2:
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        predicted = [max(0.0, slope * (len(y) + h) + intercept) for h in range(months_ahead)]
    else:
        predicted = [max(0.0, y[-1])] * months_ahead
    next_month = pd.Timestamp(recent["month"].iloc[-1]) + pd.offsets.MonthBegin(1)
    months = pd.date_range(next_month, periods=months_ahead, freq="MS")
    return pd.DataFrame({"month": months, "forecast_revenue": predicted})
