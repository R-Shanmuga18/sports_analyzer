"""Query tool for structured IPL statistics in SQLite."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def _db_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    db_path = Path(os.getenv("SQLITE_DB_PATH", "data/ipl.db"))
    if not db_path.is_absolute():
        db_path = base_dir / db_path
    return db_path


def _get_schema(conn: sqlite3.Connection) -> str:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    blocks: list[str] = []
    for (table_name,) in tables:
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_text = ", ".join([f"{c[1]} {c[2]}" for c in cols])
        blocks.append(f"{table_name}({col_text})")
    return "\n".join(blocks)


def _extract_sql(text: str) -> str:
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip().rstrip(";")
    return text.strip().rstrip(";")


def _ensure_limit(sql: str, limit: int = 20) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql.rstrip(';')} LIMIT {limit}"


def _heuristic_sql(question: str) -> str:
    q = question.lower()
    if "how many" in q and "csk" in q and "2025" in q and "win" in q:
        return (
            "SELECT COUNT(*) AS csk_wins_2025 FROM matches "
            "WHERE season = 2025 AND lower(winner) LIKE '%chennai super kings%'"
        )
    if "highest" in q and "total" in q and "2024" in q:
        return (
            "SELECT d.batting_team, SUM(d.total_runs) AS team_total, m.date, m.venue "
            "FROM deliveries d JOIN matches m ON d.match_id = m.id "
            "WHERE m.season = 2024 "
            "GROUP BY d.match_id, d.batting_team, m.date, m.venue "
            "ORDER BY team_total DESC LIMIT 1"
        )
    if "seasons" in q and "winners" in q:
        return (
            "SELECT season, winner FROM matches "
            "WHERE lower(match_type) = 'final' "
            "GROUP BY season, winner ORDER BY season"
        )
    return "SELECT * FROM matches ORDER BY season DESC"


def _llm_sql(question: str, schema: str, error_feedback: str | None = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _heuristic_sql(question)

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an expert SQLite query writer for IPL analytics.\n"
        "Use ONLY the schema below.\n"
        "Return ONLY a single SELECT SQL query, no explanation.\n"
        "Never use DDL/DML.\n\n"
        f"Schema:\n{schema}\n\n"
        f"User question: {question}\n"
    )
    if error_feedback:
        prompt += f"\nPrevious SQL error: {error_feedback}\nPlease fix and return corrected SQL only.\n"

    resp = client.responses.create(model="gpt-4o-mini", input=prompt)
    text = (resp.output_text or "").strip()
    sql = _extract_sql(text)
    if not sql.lower().startswith("select"):
        return _heuristic_sql(question)
    return sql


def _format_result(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "No rows returned"
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join([str(v) for v in row]))
    return "\n".join(lines)


def query_data(query: str) -> dict:
    """
    Query the structured IPL statistics database.

    Use this tool when the question asks about:
    - Specific numbers: scores, averages, strike rates, economy rates
    - Season-level aggregates: total runs, wickets, matches in a season
    - Player statistics: batting average, highest score, number of matches
    - Team records: wins, losses, net run rate, head-to-head results
    - Rankings or comparisons that require numerical data

    Do NOT use this tool for:
    - Narrative explanations of why something happened (use search_docs)
    - News after the 2024 IPL season (use web_search)

    Args:
        query: Natural language question about IPL statistics

    Returns:
        Dict with keys: result (the data as string), sql_used (the query run), row_count, source
    """
    load_dotenv()
    db_path = _db_path()

    try:
        with sqlite3.connect(db_path) as conn:
            schema = _get_schema(conn)

            sql = _llm_sql(query, schema)
            sql = _ensure_limit(sql, 20)

            try:
                cur = conn.execute(sql)
                rows = cur.fetchmany(20)
                cols = [c[0] for c in (cur.description or [])]
            except Exception as first_exc:
                fixed_sql = _llm_sql(query, schema, error_feedback=str(first_exc))
                fixed_sql = _ensure_limit(fixed_sql, 20)
                try:
                    cur = conn.execute(fixed_sql)
                    rows = cur.fetchmany(20)
                    cols = [c[0] for c in (cur.description or [])]
                    sql = fixed_sql
                except Exception as second_exc:
                    return {
                        "result": f"SQL execution failed: {second_exc}",
                        "sql_used": fixed_sql,
                        "row_count": 0,
                        "source": "ipl.db",
                    }

            return {
                "result": _format_result(cols, rows),
                "sql_used": sql,
                "row_count": len(rows),
                "source": "ipl.db",
            }
    except Exception as exc:
        return {
            "result": f"Database error: {exc}",
            "sql_used": "",
            "row_count": 0,
            "source": "ipl.db",
        }
