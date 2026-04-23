"""Live web search tool for recent IPL information using Tavily."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_tavily_client():
    """
    Create and return a TavilyClient, raising a clear RuntimeError if the
    API key is missing. This prevents silent failures where an empty key
    causes an auth error that gets swallowed and returned as a "no results"
    result — which would make the LLM answer from its training data instead.
    """
    from tavily import TavilyClient  # lazy import

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Add it to your .env file to enable web search."
        )
    return TavilyClient(api_key=api_key)


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the live web for recent IPL news and current information.

    Use this tool when the question asks about:
    - Events from 2025 onwards (transfers, auctions, injuries, coaching changes)
    - Current player status, current squads, or current team updates
    - Live or recent match results and announcements
    - Questions containing words like: latest, current, recent, now, today

    Do NOT use this tool for:
    - Historical statistics from 2023 or 2024 seasons (use query_data)
    - Narrative explanations from season reviews (use search_docs)
    - Anything already available in local data (use query_data or search_docs first)

    ONLY call this for information from 2025 onwards or explicitly
    current/latest/recent status questions.

    Args:
        query: Short search phrase, ideally under 10 words.
               Good: "IPL 2025 auction top buys"
               Bad: "Tell me everything about the latest IPL season"
        max_results: Number of results to return (default 3, max 5)

    Returns:
        List of dicts with keys: title, snippet, url, published_date
        On failure, returns a single-item list with a clear error message
        so the agent knows the tool failed rather than returning empty results.
    """
    if not query or not query.strip():
        return [
            {
                "title": "Empty search query",
                "snippet": "Please provide a specific search phrase.",
                "url": "",
                "published_date": "",
            }
        ]

    # Cap max_results to a sensible range
    max_results = max(1, min(max_results, 5))

    try:
        client = _get_tavily_client()
        response = client.search(query, max_results=max_results)

        raw_results = (
            response.get("results", []) if isinstance(response, dict) else []
        )

        if not raw_results:
            return [
                {
                    "title": "No results found",
                    "snippet": (
                        f"Tavily returned no results for query: '{query}'. "
                        "Try rephrasing or use query_data for historical stats."
                    ),
                    "url": "",
                    "published_date": "",
                }
            ]

        results: list[dict] = []
        for item in raw_results[:max_results]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "snippet": str(item.get("content", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "published_date": str(item.get("published_date", "")).strip(),
                }
            )

        if not results:
            return [
                {
                    "title": "No usable results",
                    "snippet": "Results were returned but could not be parsed.",
                    "url": "",
                    "published_date": "",
                }
            ]

        return results

    except RuntimeError as exc:
        # API key missing — this is a configuration error, not a search failure.
        # Return a message that clearly tells the LLM this tool is unavailable.
        logger.error("Web search configuration error: %s", exc)
        return [
            {
                "title": "Web search unavailable",
                "snippet": str(exc),
                "url": "",
                "published_date": "",
            }
        ]

    except Exception as exc:
        logger.error("Web search failed for query '%s': %s", query, exc)
        return [
            {
                "title": "Web search failed",
                "snippet": (
                    f"Search encountered an error: {exc}. "
                    "Try rephrasing or use a different tool."
                ),
                "url": "",
                "published_date": "",
            }
        ]