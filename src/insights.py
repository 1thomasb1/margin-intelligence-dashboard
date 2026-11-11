"""Deterministic analyst observations and optional LLM commentary."""
from __future__ import annotations

import json

import pandas as pd

from src.metrics import group_performance, monthly_performance, summarize


def make_rule_based_insights(df: pd.DataFrame) -> list[str]:
    """Reproducible comments based solely on actual filtered data."""
    if df.empty:
        return ["No transactions match the selection."]
    stats = summarize(df)
    points = [
        f"The selected {stats['transactions']:,} orders produced ${stats['revenue']:,.0f} "
        f"in revenue and a {stats['gross_margin']:.1%} weighted gross margin."
    ]
    by_brand = group_performance(df, "brand").sort_values("policy_gap", ascending=False)
    if stats["policy_gap"] > 0 and not by_brand.empty:
        top = by_brand.iloc[0]
        pct = float(top["policy_gap"] / stats["policy_gap"])
        points.append(
            f"{top['brand']} accounts for ${top['policy_gap']:,.0f} ({pct:.1%}) of the "
            "illustrative over-policy discount gap. Review exceptions before any pricing action."
        )
    months = monthly_performance(df)
    if len(months) >= 2:
        earlier, latest = months.iloc[-2], months.iloc[-1]
        if earlier["revenue"] > 0:
            growth = latest["revenue"] / earlier["revenue"] - 1
            direction = "increased" if growth >= 0 else "decreased"
            points.append(
                f"Revenue {direction} {abs(growth):.1%} from "
                f"{pd.Timestamp(earlier['month']):%b %Y} to {pd.Timestamp(latest['month']):%b %Y}."
            )
    points.append(
        "Policy gaps are price differences under fixed-volume assumptions, "
        "not verified recoverable profit or cash."
    )
    return points


def build_ai_payload(
    stats: dict, insights: list[str], scenario: dict, budget_summary: dict | None
) -> dict:
    """Only aggregates go to an external LLM; no customer/order records."""
    return {
        "metrics_usd": {k: round(float(stats[k]), 2) for k in (
            "revenue", "gross_profit", "policy_gap", "gross_margin", "weighted_discount"
        )},
        "transactions": stats["transactions"],
        "validated_observations": insights,
        "scenario_usd": {
            "delta_revenue": round(float(scenario["delta_revenue"]), 2),
            "delta_gross_profit": round(float(scenario["delta_gp"]), 2),
        },
        "budget_comparison_usd": budget_summary,
    }


def generate_ai_commentary(payload: dict, api_key: str, model: str = "gpt-4.1-mini") -> str:
    """Generate OPTIONAL commentary from aggregated finance facts.

    Called only after the end-user clicks the app's AI button. A paid API
    key is required; no data are transmitted when the feature isn't used.
    """
    if not api_key:
        raise ValueError("Missing OpenAI API key")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=25.0)
    result = client.responses.create(
        model=model,
        instructions=(
            "You are an FP&A analyst reviewing a fully synthetic demonstration. "
            "Using only the supplied numeric metrics and observations, write a "
            "concise executive summary with exactly 3 numbered findings and 2 "
            "specific due-diligence questions. Do not fabricate data, causes, "
            "forecast confidence, or claims that policy gaps are collectable. "
            "If inputs are insufficient, explicitly say so. Use USD amounts "
            "and explain scenario assumptions. No markdown tables."
        ),
        input=json.dumps(payload, default=str),
        max_output_tokens=500,
    )
    return result.output_text.strip()
