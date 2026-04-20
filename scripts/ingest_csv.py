"""Ingest structured IPL CSV data into a SQLite database with indexes."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
	if "batsman" not in df.columns and "batter" in df.columns:
		df = df.rename(columns={"batter": "batsman"})
	return df


def _sqlite_type(series: pd.Series) -> str:
	if pd.api.types.is_integer_dtype(series.dtype):
		return "INTEGER"
	if pd.api.types.is_float_dtype(series.dtype):
		return "REAL"
	return "TEXT"


def _create_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
	cols = []
	for col in df.columns:
		cols.append(f'"{col}" {_sqlite_type(df[col])}')
	conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
	conn.execute(f'CREATE TABLE "{table_name}" ({", ".join(cols)})')


def _insert_df(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
	if df.empty:
		return
	placeholders = ",".join(["?"] * len(df.columns))
	cols = ",".join([f'"{c}"' for c in df.columns])
	sql = f'INSERT INTO "{table_name}" ({cols}) VALUES ({placeholders})'
	conn.executemany(sql, df.itertuples(index=False, name=None))


def main() -> None:
	"""Read CSV files, normalize schema, and load into SQLite with indexes."""
	load_dotenv()

	base_dir = Path(__file__).resolve().parents[1]
	matches_path = base_dir / "data" / "structured" / "matches.csv"
	deliveries_path = base_dir / "data" / "structured" / "deliveries.csv"

	db_path = Path(os.getenv("SQLITE_DB_PATH", "data/ipl.db"))
	if not db_path.is_absolute():
		db_path = base_dir / db_path
	db_path.parent.mkdir(parents=True, exist_ok=True)

	matches_df = _clean_columns(pd.read_csv(matches_path))
	deliveries_df = _clean_columns(pd.read_csv(deliveries_path))

	with sqlite3.connect(db_path) as conn:
		_create_table(conn, "matches", matches_df)
		_insert_df(conn, "matches", matches_df)

		_create_table(conn, "deliveries", deliveries_df)
		_insert_df(conn, "deliveries", deliveries_df)

		conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season)")
		conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_team1 ON matches(team1)")
		conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_team2 ON matches(team2)")
		conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_match_id ON deliveries(match_id)")
		conn.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_batsman ON deliveries(batsman)")
		conn.commit()

		matches_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
		deliveries_count = conn.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]

	print(f"Loaded SQLite database: {db_path}")
	print(f"matches rows: {matches_count}")
	print(f"deliveries rows: {deliveries_count}")


if __name__ == "__main__":
	main()
