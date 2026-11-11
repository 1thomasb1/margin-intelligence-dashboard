import io
import sqlite3

import pandas as pd
import pytest

from src.data import (
    RAW_COLUMNS, filter_data, generate_synthetic_data, load_demo_data,
    make_database, parse_uploaded_csv,
)
from src.metrics import summarize


def test_generate_reproducible_and_fictional():
    left = generate_synthetic_data(n=100, seed=7)
    right = generate_synthetic_data(n=100, seed=7)
    pd.testing.assert_frame_equal(left, right)
    assert all(str(x).startswith("Demo Brand") for x in left["brand"])
    assert left["order_id"].nunique() == 100


def test_database_round_trip(tmp_path):
    db = tmp_path / "test_demo.db"
    make_database(db, n=250, seed=12)
    invoices, budgets = load_demo_data(db)
    assert len(invoices) == 250
    assert len(budgets) > 0
    assert invoices["order_date"].dtype.kind == "M"
    assert budgets["month"].dtype.kind == "M"
    assert set(budgets["region"]) <= set(invoices["region"])
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 250
        assert con.execute("SELECT schema_version FROM metadata").fetchone()[0] == 2


def test_filter_can_isolate_one_brand():
    df = generate_synthetic_data(n=200, seed=12)
    brand = df["brand"].iloc[0]
    selected = filter_data(
        df, [brand], df["customer_segment"].unique().tolist(),
        df["region"].unique().tolist(),
        pd.Timestamp("2025-04-01"), pd.Timestamp("2026-09-30"),
    )
    assert len(selected) > 0
    assert set(selected["brand"]) == {brand}


def test_csv_upload_success():
    example = generate_synthetic_data(n=8)[RAW_COLUMNS]
    csv = io.StringIO(example.to_csv(index=False))
    parsed = parse_uploaded_csv(csv)
    assert len(parsed) == 8
    assert summarize(parsed)["revenue"] > 0


def test_csv_upload_missing_column():
    example = generate_synthetic_data(n=8)[RAW_COLUMNS].drop(columns=["unit_cost"])
    with pytest.raises(ValueError, match="unit_cost"):
        parse_uploaded_csv(io.StringIO(example.to_csv(index=False)))


def test_csv_upload_rejects_bad_discount():
    example = generate_synthetic_data(n=8)[RAW_COLUMNS]
    example.loc[0, "discount_rate"] = 1.2
    with pytest.raises(ValueError, match="discount_rate"):
        parse_uploaded_csv(io.StringIO(example.to_csv(index=False)))


def test_csv_upload_adds_optional_fields():
    example = generate_synthetic_data(n=8)[RAW_COLUMNS]
    example = example.drop(columns=["order_id", "region", "channel", "product_group"])
    parsed = parse_uploaded_csv(io.StringIO(example.to_csv(index=False)))
    assert parsed["region"].iloc[0] == "Unspecified"
    assert parsed["order_id"].iloc[0].startswith("UPLOAD-")
