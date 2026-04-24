"""Query tool for structured IPL statistics in SQLite."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TEAM_ALIASES: dict[str, str] = {
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "rcb": "Royal Challengers Bangalore",
    "rcbengaluru": "Royal Challengers Bengaluru",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "dd": "Delhi Daredevils",
    "rr": "Rajasthan Royals",
    "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans",
    "pbks": "Punjab Kings",
    "kxip": "Kings XI Punjab",
    "pw": "Pune Warriors",
    "ris": "Rising Pune Supergiant",
}

_SAMPLE_VALUES: dict[str, list] = {
    "match_type": ["League", "Qualifier 1", "Qualifier 2", "Eliminator", "Final"],
    "result": ["runs", "wickets", "tie", "no result"],
    "toss_decision": ["bat", "field"],
    "dl_applied": [0, 1],
}


def _db_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    db_path = Path(os.getenv("SQLITE_DB_PATH", "data/ipl.db"))
    if not db_path.is_absolute():
        db_path = base_dir / db_path
    return db_path


def _get_schema(conn: sqlite3.Connection) -> str:
    """
    Return a rich schema string including column types AND sample categorical
    values so the LLM writes correct WHERE clauses on the first attempt.
    """
    tables = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()

    blocks: list[str] = []
    for (table_name,) in tables:
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        col_lines: list[str] = []
        for c in cols:
            col_name, col_type = c[1], c[2]
            hint = ""
            if col_name in _SAMPLE_VALUES:
                samples = ", ".join(repr(v) for v in _SAMPLE_VALUES[col_name])
                hint = f"  -- sample values: {samples}"
            elif col_type.upper() in ("TEXT", "VARCHAR") and col_name not in (
                "venue", "city", "date", "umpire1", "umpire2", "player_of_match",
            ):
                try:
                    rows = conn.execute(
                        f"SELECT DISTINCT {col_name} FROM {table_name} "
                        f"WHERE {col_name} IS NOT NULL LIMIT 4"
                    ).fetchall()
                    if rows:
                        samples = ", ".join(repr(r[0]) for r in rows)
                        hint = f"  -- e.g. {samples}"
                except Exception:
                    pass
            col_lines.append(f"  {col_name} {col_type}{hint}")
        blocks.append(f"TABLE {table_name}(\n" + "\n".join(col_lines) + "\n)")

    for (table_name,) in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            for i, block in enumerate(blocks):
                if block.startswith(f"TABLE {table_name}"):
                    blocks[i] = block.rstrip(")") + f"\n)  -- {count} rows"
        except Exception:
            pass

    return "\n\n".join(blocks)


@lru_cache(maxsize=1)
def _cached_schema(db_path_str: str) -> str:
    with sqlite3.connect(db_path_str) as conn:
        return _get_schema(conn)


def _extract_sql(text: str) -> str:
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidate = fence.group(1).strip().rstrip(";")
    else:
        candidate = text.strip().rstrip(";")
    start_idx = candidate.upper().find("SELECT")
    if start_idx != -1:
        return candidate[start_idx:].rstrip(";")
    return candidate


def _ensure_limit(sql: str, limit: int = 20) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql.rstrip('; ')} LIMIT {limit}"


def _normalize_question(question: str) -> str:
    """Replace team abbreviations with full names directly in the question text."""
    normalized = question
    for short, full in TEAM_ALIASES.items():
        normalized = re.sub(
            rf"\b{re.escape(short)}\b", full, normalized, flags=re.IGNORECASE
        )
    return normalized


def _llm_sql(question: str, schema: str, error_feedback: str | None = None) -> str:
    """
    Use Groq LLaMA to convert a natural language question into SQLite SQL.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.warning("GROQ_API_KEY not set")
        return "SELECT 'GROQ_API_KEY not configured' AS error"

    normalized_question = _normalize_question(question)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

        system_msg = (
            "You are an expert SQLite query writer for IPL cricket analytics.\n\n"
            "HARD RULES — violating any of these produces completely wrong answers:\n\n"

            "1. Return ONLY a single SELECT SQL statement. "
            "No explanation, no markdown fences, no comments, no semicolons.\n\n"

            "2. DB SCHEMA CONTEXT:\n"
            "   - `matches` table has one row per match (`id` is the match ID).\n"
            "   - `deliveries` table has one row per ball bowled (`match_id` links to `matches.id`).\n"
            "   - `batsman_runs` is runs off the bat for ONE BALL (max 6).\n"
            "   - `total_runs` is batsman_runs + extra_runs for ONE BALL.\n\n"

            "3. HIGHEST INDIVIDUAL BATTING SCORE:\n"
            "   NEVER use MAX(batsman_runs), that only returns 6. You MUST sum the runs grouped by match.\n"
            "   CORRECT:   SELECT batsman, SUM(batsman_runs) AS match_score FROM deliveries GROUP BY match_id, batsman ORDER BY match_score DESC LIMIT 1\n\n"

            "4. COUNTING WICKETS FOR A BOWLER:\n"
            "   `is_wicket` = 1 includes run outs, which do NOT count towards a bowler's stats.\n"
            "   CORRECT:   SUM(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 ELSE 0 END)\n\n"

            "5. MAIDEN OVERS:\n"
            "   A maiden over is an over where 0 runs are scored off the bowler. You must group by match, inning, and over.\n"
            "   CORRECT:   SELECT bowler, COUNT(*) as maidens FROM (SELECT bowler, match_id, inning, over FROM deliveries GROUP BY match_id, inning, over, bowler HAVING SUM(total_runs) = 0 AND COUNT(ball) >= 6) GROUP BY bowler ORDER BY maidens DESC LIMIT 1\n\n"

            "6. COUNTING WINS OR MATCHES:\n"
            "   Always count on the `matches` table alone. NEVER join with `deliveries` just to count wins, it multiplies the rows by 300.\n"
            "   CORRECT:   SELECT COUNT(*) FROM matches WHERE winner = 'Team Name'\n\n"

            "7. IPL SEASON TITLE WINNER:\n"
            "   CORRECT:   SELECT winner FROM matches WHERE season = '2024' AND lower(match_type) = 'final' LIMIT 1\n"
            "   WRONG:     SELECT winner FROM matches WHERE season = '2024'  ← returns all match winners, not just the final\n\n"

            "8. BATTING AVERAGE:\n"
            "   average = SUM(batsman_runs) / NULLIF(COUNT(CASE WHEN is_wicket=1 THEN 1 END), 0)\n"
            "   Group by batsman. is_wicket=1 means the batsman was dismissed.\n\n"

            "9. ECONOMY RATE AND BEST BOWLERS:\n"
            "   economy = (total runs conceded) / (total legal balls) * 6\n"
            "   CRITICAL CRICKET RULE: The 'best' or 'top' economy rate is the LOWEST number. If asked for best economy, you MUST use ORDER BY economy_rate ASC.\n"
            "   If asked for 'top wicket-taker', use ORDER BY wickets DESC.\n"
            "   Always filter out part-timers by adding: HAVING COUNT(ball) > 60\n"
            "   CORRECT WICKET-TAKER QUERY: SELECT bowler, COUNT(CASE WHEN is_wicket = 1 AND dismissal_kind NOT IN ('run out', 'retired hurt') THEN 1 END) AS wickets, (SUM(total_runs) * 6.0) / NULLIF(COUNT(CASE WHEN extras_type NOT IN ('wides', 'noballs') OR extras_type IS NULL THEN 1 END), 0) AS economy_rate FROM deliveries JOIN matches ON deliveries.match_id = matches.id WHERE matches.season = '2024' GROUP BY bowler HAVING COUNT(ball) > 60 ORDER BY wickets DESC LIMIT 1\n\n"
            
            "10. TEAM INNINGS TOTAL:\n"
            "    To get a team's total score in a match, use: SUM(total_runs) grouped by match_id AND inning.\n\n"

            "11. FILTERING BY SEASON:\n"
            "    The `deliveries` table has NO season column. To filter a ball-by-ball stat by year, you MUST JOIN:\n"
            "    FROM deliveries JOIN matches ON deliveries.match_id = matches.id WHERE matches.season = '2023'\n\n"

            "12. TEAM NAMES:\n"
            "    `team1`, `team2`, `winner`, `batting_team`, and `bowling_team` store FULL names (e.g., 'Chennai Super Kings'). Never use abbreviations in SQL values.\n\n"

            "13. Always append `LIMIT 20` unless it's a scalar aggregate (like a single COUNT/SUM/MAX).\n"

            "14. TEAM WITH MOST WINS IN A SEASON:\n"
            "    To find who won the most matches, you MUST group by the winner column and count.\n"
            "    CORRECT:   SELECT winner, COUNT(*) as wins FROM matches WHERE season = '2024' GROUP BY winner ORDER BY wins DESC LIMIT 1\n\n"
        )

        user_msg = f"Schema:\n{schema}\n\nQuestion: {normalized_question}\n"
        if error_feedback:
            user_msg += (
                f"\nPrevious SQL failed with error: {error_feedback}\n"
                "Fix the SQL. Return only the corrected SELECT statement, nothing else.\n"
            )

        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        text = ((resp.choices[0].message.content if resp.choices else "") or "").strip()
        sql = _extract_sql(text)

        if not sql.upper().lstrip().startswith("SELECT"):
            logger.warning("LLM returned non-SELECT SQL: %s", sql[:120])
            return "SELECT 'Could not generate valid SQL for this question' AS error"

        return sql

    except Exception as exc:
        logger.error("LLM SQL generation failed: %s", exc)
        return "SELECT 'SQL generation failed — please rephrase the question' AS error"


def _format_result(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "No matching records found in the database."
    header = " | ".join(str(c) for c in columns)
    separator = "-" * len(header)
    lines = [header, separator]
    for row in rows:
        lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
    lines.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''} returned)")
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
    - If the question contains any number, statistic, count, total, average,
      highest, lowest, or ranking — this is the right tool

    Do NOT use this tool for:
    - Narrative explanations of why something happened (use search_docs)
    - News after the 2024 IPL season (use web_search)

    Args:
        query: Natural language question about IPL statistics

    Returns:
        Dict with keys: result, sql_used, row_count, source
    """
    if not query or not query.strip():
        return {
            "result": "Empty query received — please provide a specific statistics question.",
            "sql_used": "",
            "row_count": 0,
            "source": "ipl.db",
        }

    db_path = _db_path()
    if not db_path.exists():
        return {
            "result": "Database not found. Run scripts/ingest_csv.py first.",
            "sql_used": "",
            "row_count": 0,
            "source": "ipl.db",
        }

    try:
        with sqlite3.connect(str(db_path)) as conn:
            schema = _cached_schema(str(db_path))
            sql = _llm_sql(query, schema)
            sql = _ensure_limit(sql, 20)
            logger.debug("Generated SQL: %s", sql)

            try:
                cur = conn.execute(sql)
                rows = cur.fetchmany(20)
                cols = [c[0] for c in (cur.description or [])]
            except Exception as first_exc:
                logger.warning("SQL first attempt failed: %s — retrying", first_exc)
                sql = _llm_sql(query, schema, error_feedback=str(first_exc))
                sql = _ensure_limit(sql, 20)
                try:
                    cur = conn.execute(sql)
                    rows = cur.fetchmany(20)
                    cols = [c[0] for c in (cur.description or [])]
                except Exception as second_exc:
                    logger.error("SQL second attempt also failed: %s", second_exc)
                    return {
                        "result": (
                            f"Could not execute a valid SQL query for this question. "
                            f"Error: {second_exc}"
                        ),
                        "sql_used": sql,
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
        logger.error("Database error in query_data: %s", exc)
        return {
            "result": f"Database error: {exc}",
            "sql_used": "",
            "row_count": 0,
            "source": "ipl.db",
        }