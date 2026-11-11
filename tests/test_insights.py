from src.data import generate_synthetic_data
from src.insights import build_ai_payload, make_rule_based_insights
from src.metrics import pricing_scenario, summarize


def test_insights_reference_real_computed_figures():
    df = generate_synthetic_data(n=100)
    words = " ".join(make_rule_based_insights(df))
    assert "gross margin" in words
    assert "not verified recoverable" in words


def test_ai_payload_contains_aggregates_not_customer_rows():
    df = generate_synthetic_data(n=20)
    payload = build_ai_payload(
        summarize(df), make_rule_based_insights(df), pricing_scenario(df, 0.5, 0), None
    )
    assert "transactions" in payload
    assert "order_id" not in str(payload)
    assert "DEMO-" not in str(payload)
