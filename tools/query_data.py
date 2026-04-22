"""Query tool for structured IPL statistics in SQLite."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


TEAM_ALIASES = {
    "csk": "Chennai Super Kings",
    "mi": "Mumbai Indians",
    "rcb": "Royal Challengers Bangalore",
    "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad",
    "dc": "Delhi Capitals",
    "rr": "Rajasthan Royals",
    "lsg": "Lucknow Super Giants",
    "gt": "Gujarat Titans",
    "pbks": "Punjab Kings",
}


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


def _extract_years(question: str) -> list[int]:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
    return sorted(list(dict.fromkeys(years)))


def _extract_top_n(question: str, default: int = 3) -> int:
    m = re.search(r"\btop\s+(\d+)\b", question.lower())
    if m:
        return max(1, int(m.group(1)))
    return default


def _extract_team_names(question: str) -> list[str]:
    q = question.lower()
    teams: list[str] = []
    for short, full in TEAM_ALIASES.items():
        if re.search(rf"\b{re.escape(short)}\b", q) and full not in teams:
            teams.append(full)
    return teams


def _heuristic_sql(question: str) -> str:
    q = question.lower()
    years = _extract_years(question)
    team_names = _extract_team_names(question)

    if "final" in q and "score" in q:
        year_filter = f"m.season = {years[0]} AND " if years else ""
        return (
            "SELECT m.team1, m.team2, m.winner, m.result, m.result_margin, "
            "MAX(CASE WHEN d.inning = 1 THEN team_score END) AS team1_score, "
            "MAX(CASE WHEN d.inning = 2 THEN team_score END) AS team2_score "
            "FROM matches m "
            "JOIN ("
            "SELECT match_id, inning, SUM(total_runs) AS team_score "
            "FROM deliveries GROUP BY match_id, inning"
            ") d ON d.match_id = m.id "
            f"WHERE {year_filter}lower(m.match_type) = 'final' "
            "GROUP BY m.id, m.team1, m.team2, m.winner, m.result, m.result_margin"
        )

    if "most titles" in q or ("titles" in q and "overall" in q):
        return (
            "SELECT winner AS team, COUNT(*) AS titles "
            "FROM matches WHERE lower(match_type) = 'final' AND winner IS NOT NULL "
            "GROUP BY winner ORDER BY titles DESC"
        )

    if "run" in q and "top" in q and ("scorer" in q or "run scorers" in q):
        year_filter = f"WHERE m.season = {years[0]} " if years else ""
        top_n = _extract_top_n(question, default=3)
        return (
            "SELECT d.batsman, SUM(d.batsman_runs) AS total_runs "
            "FROM deliveries d JOIN matches m ON d.match_id = m.id "
            f"{year_filter}"
            f"GROUP BY d.batsman ORDER BY total_runs DESC LIMIT {top_n}"
        )

    if "highest" in q and "individual score" in q:
        return (
            "SELECT batsman, innings_runs FROM ("
            "SELECT batsman, match_id, SUM(batsman_runs) AS innings_runs "
            "FROM deliveries GROUP BY batsman, match_id"
            ") t ORDER BY innings_runs DESC LIMIT 1"
        )

    if "wicket" in q and ("top" in q or "highest" in q):
        year_filter = f"m.season = {years[0]} AND " if years else ""
        top_n = _extract_top_n(question, default=1)
        return (
            "SELECT d.bowler, COUNT(*) AS wickets "
            "FROM deliveries d JOIN matches m ON d.match_id = m.id "
            f"WHERE {year_filter}d.is_wicket = 1 "
            "AND lower(COALESCE(d.dismissal_kind, '')) NOT IN ('run out', 'retired hurt', 'obstructing the field') "
            f"GROUP BY d.bowler ORDER BY wickets DESC LIMIT {top_n}"
        )

    if "win rate" in q and team_names:
        season_filter = ", ".join([str(y) for y in years]) if years else "2023, 2024"
        union_teams = " UNION ALL ".join([f"SELECT '{t}' AS team" for t in team_names])
        return (
            "SELECT season, team, wins, matches_played, "
            "ROUND((wins * 100.0) / NULLIF(matches_played, 0), 2) AS win_rate_pct "
            "FROM ("
            "SELECT m.season AS season, t.team AS team, "
            "SUM(CASE WHEN lower(m.winner) = lower(t.team) THEN 1 ELSE 0 END) AS wins, "
            "SUM(CASE WHEN lower(m.team1) = lower(t.team) OR lower(m.team2) = lower(t.team) THEN 1 ELSE 0 END) AS matches_played "
            "FROM matches m "
            f"JOIN ({union_teams}) t "
            f"WHERE m.season IN ({season_filter}) "
            "GROUP BY m.season, t.team"
            ") s ORDER BY season, team"
        )

    if ("compared" in q or "vs" in q) and "win" in q and team_names and len(years) >= 2:
        team_like = team_names[0].lower().replace("'", "''")
        y1, y2 = years[0], years[1]
        return (
            "SELECT "
            f"SUM(CASE WHEN season = {y1} AND lower(winner) LIKE '%{team_like.split()[0]}%' THEN 1 ELSE 0 END) AS wins_{y1}, "
            f"SUM(CASE WHEN season = {y2} AND lower(winner) LIKE '%{team_like.split()[0]}%' THEN 1 ELSE 0 END) AS wins_{y2} "
            "FROM matches"
        )

    if "how many" in q and "win" in q and team_names and years:
        team = team_names[0].lower().replace("'", "''")
        year = years[0]
        return (
            f"SELECT COUNT(*) AS team_wins FROM matches WHERE season = {year} "
            f"AND lower(winner) LIKE '%{team.split()[0]}%'"
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


def _normalize_question(question: str) -> str:
    normalized = f" {question.lower()} "
    for short, full in TEAM_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(short)}\b", full.lower(), normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _should_use_heuristic(question: str) -> bool:
    q = question.lower()
    has_team_alias = bool(re.search(r"\b(csk|mi|rcb|kkr|srh|dc|rr|lsg|gt|pbks)\b", q))
    return any(
        [
            "most titles" in q,
            ("top" in q and "run" in q and "scorer" in q),
            ("highest" in q and "individual score" in q),
            ("win rate" in q and has_team_alias),
            ("win" in q and ("compared" in q or "vs" in q) and has_team_alias),
            ("wicket" in q and ("top" in q or "highest" in q)),
            ("final" in q and "score" in q),
        ]
    )


def _llm_sql(question: str, schema: str, error_feedback: str | None = None) -> str:
    if _should_use_heuristic(question):
        return _heuristic_sql(question)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _heuristic_sql(question)

    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    normalized_question = _normalize_question(question)
    prompt = (
        "You are an expert SQLite query writer for IPL analytics.\n"
        "Use ONLY the schema below.\n"
        "Return ONLY a single SELECT SQL query, no explanation.\n"
        "Never use DDL/DML.\n\n"
        "Important query-writing constraints for this schema:\n"
        "- The deliveries table does NOT have a season column. If season is needed, JOIN deliveries.match_id = matches.id and filter on matches.season.\n"
        "- Team short names may appear in user questions (CSK, MI, RCB, KKR). Map them to full team names in matches.team1/team2/winner.\n"
        "- For highest individual batting score, aggregate SUM(batsman_runs) by batsman and match_id, then take MAX over innings total.\n"
        "- Return portable SQLite syntax only.\n\n"
        f"Schema:\n{schema}\n\n"
        f"User question: {question}\n"
        f"Normalized question: {normalized_question}\n"
    )
    if error_feedback:
        prompt += f"\nPrevious SQL error: {error_feedback}\nPlease fix and return corrected SQL only.\n"

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )
    text = ((resp.choices[0].message.content if resp.choices else "") or "").strip()
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

            if not rows:
                heuristic_sql = _ensure_limit(_heuristic_sql(query), 20)
                if heuristic_sql != sql:
                    try:
                        cur = conn.execute(heuristic_sql)
                        heuristic_rows = cur.fetchmany(20)
                        heuristic_cols = [c[0] for c in (cur.description or [])]
                        if heuristic_rows:
                            rows = heuristic_rows
                            cols = heuristic_cols
                            sql = heuristic_sql
                    except Exception:
                        pass

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
