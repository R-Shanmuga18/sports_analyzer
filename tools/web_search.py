"""Live web search tool for recent IPL information using Tavily."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from tavily import TavilyClient


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the live web for recent IPL news and current information.

    Use this tool when the question asks about:
    - Events after the 2024 IPL season (transfers, auctions, injuries)
    - Current player status, team rosters, or coaching changes
    - Live or recent match results not in the database
    - Any question containing words like: latest, current, recent, now, today

    Do NOT use this tool for:
    - Historical statistics from 2023 or 2024 seasons (use query_data)
    - Narrative explanations from season reviews (use search_docs)

    Args:
        query: Short search query, ideally under 10 words
        max_results: Number of results to return (default 3)

    Returns:
        List of dicts, each with keys: title, snippet, url, published_date
    """
    try:
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        if not raw_results:
            return [
                {
                    "title": "No results found",
                    "snippet": "",
                    "url": "",
                    "published_date": "",
                }
            ]

        results: list[dict] = []
        for item in raw_results[: max_results if max_results > 0 else 1]:
            results.append(
                {
                    "title": item.get("title", "") if isinstance(item, dict) else "",
                    "snippet": item.get("content", "") if isinstance(item, dict) else "",
                    "url": item.get("url", "") if isinstance(item, dict) else "",
                    "published_date": item.get("published_date", "") if isinstance(item, dict) else "",
                }
            )

        if not results:
            return [
                {
                    "title": "No results found",
                    "snippet": "",
                    "url": "",
                    "published_date": "",
                }
            ]
        return results
    except Exception as error:
        return [
            {
                "title": "Web search failed",
                "snippet": str(error),
                "url": "",
                "published_date": "",
            }
        ]
