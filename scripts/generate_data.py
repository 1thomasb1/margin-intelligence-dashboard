"""Regenerate the fictional local SQLite demo database.

Usage: python -m scripts.generate_data --rows 6000 --seed 42
"""
import argparse
from src.data import DB_PATH, make_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MarginIQ demo SQLite data")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = make_database(path=DB_PATH, n=args.rows, seed=args.seed)
    print(f"Generated {args.rows:,} fictional transactions: {output}")


if __name__ == "__main__":
    main()
