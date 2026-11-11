"""Synthetic transactional data and the MarginIQ SQLite demo store."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import add_financial_metrics

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "marginiq_demo.db"
BRANDS = [f"Demo Brand {i:02d}" for i in range(1, 18)]
PRODUCT_GROUPS = ["Pumps", "Valves", "Control Systems", "Flow Components"]
SEGMENT_POLICIES = {
    "Enterprise": 0.16,
    "Distributor": 0.19,
    "OEM": 0.13,
    "Municipal": 0.10,
    "Small Business": 0.07,
}
REGIONS = ["Northeast", "South", "Midwest", "West"]
CHANNELS = ["Direct", "Partner", "Online"]
RAW_COLUMNS = [
    "order_id", "order_date", "brand", "customer_segment", "region",
    "product_group", "channel", "units", "list_price", "discount_rate",
    "allowed_discount_rate", "unit_cost",
]


def generate_synthetic_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Deterministic *fictional* B2B invoices, with no real employer data."""
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    brands = rng.choice(BRANDS, n)
    segments = rng.choice(
        list(SEGMENT_POLICIES), n, p=[0.25, 0.26, 0.20, 0.13, 0.16]
    )
    days = pd.date_range("2025-04-01", "2026-09-30", freq="D")
    # Mild seasonal variation gives a non-flat reporting series.
    weights = np.asarray([1 + 0.22 * np.sin((d.month - 1) / 12 * 2 * np.pi) for d in days])
    weights = weights / weights.sum()
    date_indices = rng.choice(len(days), n, p=weights)
    dates = days[date_indices]
    brand_index = np.array([BRANDS.index(b) for b in brands])
    groups = np.array([PRODUCT_GROUPS[i % len(PRODUCT_GROUPS)] for i in brand_index])
    regions = rng.choice(REGIONS, n, p=[0.20, 0.33, 0.28, 0.19])
    channels = rng.choice(CHANNELS, n, p=[0.52, 0.34, 0.14])
    units = rng.integers(1, 38, n)
    list_price = np.round(rng.uniform(175, 1425, n) * (0.87 + 0.055 * (brand_index % 6)), 2)
    brand_adjust = (brand_index % 5 - 2) * 0.006
    allowed = np.clip(np.array([SEGMENT_POLICIES[s] for s in segments]) + brand_adjust, 0, 0.40)
    exceptions = rng.random(n) < 0.20
    actual = np.clip(allowed + rng.normal(-0.009, 0.028, n) + exceptions * 0.064, 0, 0.45)
    unit_cost = np.round(list_price * rng.uniform(0.52, 0.72, n), 2)

    raw = pd.DataFrame({
        "order_id": [f"DEMO-{i:06d}" for i in range(1, n + 1)],
        "order_date": dates,
        "brand": brands,
        "customer_segment": segments,
        "region": regions,
        "product_group": groups,
        "channel": channels,
        "units": units,
        "list_price": list_price,
        "discount_rate": np.round(actual, 5),
        "allowed_discount_rate": np.round(allowed, 5),
        "unit_cost": unit_cost,
    })
    return add_financial_metrics(raw)


def generate_demo_budgets(df: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    """Create *illustrative* plans for each brand/segment/month.

    These demo budgets are generated from synthetic realized sales plus noise;
    they are NOT independently prepared business forecasts or actual budgets.
    """
    rng = np.random.default_rng(seed)
    work = df.assign(month=df["order_date"].dt.to_period("M").dt.to_timestamp())
    budgets = work.groupby(
        ["month", "brand", "customer_segment", "region"], as_index=False
    ).agg(realized_revenue=("revenue", "sum"), realized_gp=("gross_profit", "sum"))
    rev_factor = np.clip(rng.normal(1.025, 0.075, len(budgets)), 0.75, 1.30)
    gp_factor = np.clip(rng.normal(1.045, 0.095, len(budgets)), 0.70, 1.35)
    budgets["budget_revenue"] = budgets["realized_revenue"] * rev_factor
    budgets["budget_gross_profit"] = budgets["realized_gp"] * gp_factor
    return budgets[[
        "month", "brand", "customer_segment", "region", "budget_revenue", "budget_gross_profit"
    ]]


def make_database(path: Path = DB_PATH, n: int = 6000, seed: int = 42) -> Path:
    """Create a local reproducible SQLite demo database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transactions = generate_synthetic_data(n=n, seed=seed)
    budgets = generate_demo_budgets(transactions, seed=seed + 1)
    with sqlite3.connect(path) as con:
        transactions.to_sql("transactions", con, if_exists="replace", index=False)
        budgets.to_sql("budgets", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_brand ON transactions(brand)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_orderdate ON transactions(order_date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_budget_period ON budgets(month, brand, customer_segment, region)")
        pd.DataFrame([{"schema_version": 2, "seed": seed, "transaction_count": n}]).to_sql(
            "metadata", con, if_exists="replace", index=False
        )
    return path


def load_demo_data(path: Path = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch demo data using SQL; regenerate if schema is absent or outdated."""
    path = Path(path)
    if not path.exists():
        make_database(path)
    with sqlite3.connect(path) as con:
        try:
            version = con.execute("SELECT schema_version FROM metadata LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            version = None
    if version is None or version[0] != 2:
        make_database(path)
    with sqlite3.connect(path) as con:
        transactions = pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY order_date, order_id", con,
            parse_dates=["order_date"],
        )
        budgets = pd.read_sql_query(
            "SELECT * FROM budgets ORDER BY month, brand, customer_segment, region", con,
            parse_dates=["month"],
        )
    transactions["above_policy"] = transactions["above_policy"].astype(bool)
    return transactions, budgets


def parse_uploaded_csv(upload) -> pd.DataFrame:
    """Validate raw invoice CSVs; calculate metrics rather than trusting them."""
    raw = pd.read_csv(upload)
    essential = {
        "order_date", "brand", "customer_segment", "units", "list_price",
        "discount_rate", "allowed_discount_rate", "unit_cost",
    }
    missing = essential.difference(raw.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    if raw.empty:
        raise ValueError("CSV contains no transactions")
    if len(raw) > 100_000:
        raise ValueError("CSV limit is 100,000 rows")
    raw["order_date"] = pd.to_datetime(raw["order_date"], errors="raise")
    if raw["order_date"].isna().any():
        raise ValueError("order_date cannot be empty")
    if "order_id" not in raw:
        raw["order_id"] = [f"UPLOAD-{i:06d}" for i in range(1, len(raw) + 1)]
    for optional in ("region", "product_group", "channel"):
        if optional not in raw:
            raw[optional] = "Unspecified"
    for name in ("order_id", "brand", "customer_segment", "region", "product_group", "channel"):
        if raw[name].isna().any() or (raw[name].astype(str).str.strip() == "").any():
            raise ValueError(f"Column {name} cannot have blank values")
    return add_financial_metrics(raw[RAW_COLUMNS])


def filter_data(
    transactions: pd.DataFrame, brands: list[str], segments: list[str],
    regions: list[str], period_start: pd.Timestamp, period_end: pd.Timestamp,
) -> pd.DataFrame:
    return transactions.loc[
        transactions["brand"].isin(brands)
        & transactions["customer_segment"].isin(segments)
        & transactions["region"].isin(regions)
        & transactions["order_date"].between(period_start, period_end)
    ].copy()
