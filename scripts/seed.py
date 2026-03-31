"""
Seed the database with CSV data using PostgreSQL COPY.

Supports both plain .csv and gzip-compressed .csv.gz files.
Works with both local PostgreSQL (Docker) and AWS RDS — just change the
connection parameters via environment variables or .env file.

Usage:
    uv run python scripts/seed.py
    uv run python scripts/seed.py --env infra/.env
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import psycopg2


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEEDS = [
    {
        "table": "bronze.raw_lifetime",
        "file": DATA_DIR / "forging_line_lifetime.csv.gz",
        "columns": ("timestamp", "signal", "value"),
    },
    {
        "table": "bronze.raw_piece_info",
        "file": DATA_DIR / "die_matrix_ids.csv.gz",
        "columns": ("timestamp", "signal", "value"),
    },
]


def load_env(path: str):
    """Load key=value pairs from a .env file into environment."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "vaultech"),
        user=os.environ.get("POSTGRES_USER", "vaultech"),
        password=os.environ.get("POSTGRES_PASSWORD", "vaultech_dev"),
    )


def open_file(filepath: Path):
    """Open a file, transparently decompressing .gz files."""
    if filepath.suffix == ".gz":
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def seed_table(cur, table: str, filepath: Path, columns: tuple[str, ...]):
    """Load a CSV file into a table using COPY (fast bulk insert)."""
    col_list = ", ".join(columns)
    copy_sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, HEADER true)"

    cur.execute(f"SELECT COUNT(*) FROM {table}")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"  {table}: already has {existing:,} rows, skipping")
        return

    print(f"  {table}: loading {filepath.name}...", end=" ", flush=True)
    with open_file(filepath) as f:
        cur.copy_expert(copy_sql, f)
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"{count:,} rows loaded")


def main():
    parser = argparse.ArgumentParser(description="Seed database with CSV data")
    parser.add_argument("--env", default="infra/.env", help="Path to .env file")
    args = parser.parse_args()

    if os.path.exists(args.env):
        load_env(args.env)

    print("Connecting to PostgreSQL...")
    try:
        conn = get_connection()
    except psycopg2.OperationalError as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    conn.autocommit = False
    cur = conn.cursor()

    print("Seeding tables:")
    for seed in SEEDS:
        if not seed["file"].exists():
            print(f"  {seed['table']}: {seed['file']} not found, skipping")
            continue
        seed_table(cur, seed["table"], seed["file"], seed["columns"])

    conn.commit()
    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
