"""Function-calling schemas and dispatcher for the three agent tools."""

from __future__ import annotations

import json
import logging

from tools.query_data import query_data
from tools.search_docs import search_docs
from tools.web_search import web_search

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Use this tool to search IPL season review documents for narrative information.\n\n"
                "CALL THIS TOOL when the question asks about:\n"
                "- How a match was won or lost (game narrative, key moments)\n"
                "- Player performances described in text form "
                "- Team strategy, tournament storylines, or notable events\n"
                "- Reasons or explanations behind outcomes\n"
                "- Any answer that would appear in a written article or match report\n\n"
                "DO NOT call this tool when:\n"
                "- The question asks for a specific number (runs, average, economy, "
                "wins, titles) — use query_data instead\n"
                "- The question contains words like 'latest', 'current', 'recent', "
                "'now' — use web_search instead\n"
                "- You already retrieved a narrative chunk and still need numbers "
                "— use query_data for those\n\n"
                "Input must be a specific search phrase, not the full user question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific natural language phrase to search for in the documents. "
                            "Be precise — e.g. 'KKR winning strategy 2024 final' "
                            "not 'tell me about KKR'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Use this tool to query structured IPL statistics from the database.\n\n"
                "CALL THIS TOOL when the question asks about:\n"
                "- Exact numbers: runs, wickets, strike rate, economy, averages, "
                "wins, losses, totals\n"
                "- Season-level or player-level statistical rankings or comparisons\n"
                "- Historical records: highest score, most titles, best bowling figures\n"
                "- Head-to-head results between teams\n"
                "- Any question that can be answered with data from a match statistics table\n"
                "- If the question contains any number, statistic, count, total, average, "
                "highest, lowest, or ranking — this is the right tool\n\n"
                "DO NOT call this tool when:\n"
                "- The user asks for narrative explanation, strategy, or storyline "
                "— use search_docs instead\n"
                "- The question is about latest/current/recent/post-2024 live updates "
                "— use web_search instead\n\n"
                "Input must be one clear natural language statistics question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language question about IPL statistics. "
                            "Include the entity name and season where relevant, "
                            "e.g. 'How many matches did Mumbai Indians win in 2023?'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Use this tool to search the live web for IPL information not in local data.\n\n"
                "CALL THIS TOOL when the question asks about:\n"
                "- Events from 2025 onwards (transfers, auctions, injuries, "
                "coaching changes)\n"
                "- Current player status, current squads, or current team updates\n"
                "- Live or recent results and announcements\n"
                "- Questions containing words like 'latest', 'current', 'recent', "
                "'now', or 'today'\n\n"
                "DO NOT call this tool when:\n"
                "- The question is about historical 2023 or 2024 statistics "
                "— use query_data instead\n"
                "- The question asks for narrative explanations from season reviews "
                "— use search_docs instead\n"
                "- The answer already exists in local data "
                "— always try query_data or search_docs first\n\n"
                "ONLY call this for information from 2025 onwards or "
                "explicitly current/latest/recent status questions.\n"
                "Input must be a short focused phrase under 10 words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Short web search phrase under 10 words, "
                            "e.g. 'IPL 2025 auction retained players'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    Route a tool call from the agent loop to the correct tool function.

    Validates tool name and arguments before dispatching.
    Returns JSON string of the result for appending to the messages list.

    Args:
        tool_name: One of "search_docs", "query_data", "web_search"
        tool_args: Arguments dict parsed from the LLM tool call

    Returns:
        JSON-encoded string of the tool result

    Raises:
        ValueError: If tool_name is not recognised or query is empty
    """
    _KNOWN_TOOLS = {"search_docs", "query_data", "web_search"}

    if tool_name not in _KNOWN_TOOLS:
        raise ValueError(
            f"Unknown tool '{tool_name}'. "
            f"Valid tools are: {sorted(_KNOWN_TOOLS)}"
        )

    # Validate that query arg is present and non-empty
    query = tool_args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            f"Tool '{tool_name}' requires a non-empty 'query' argument. "
            f"Received: {repr(query)}"
        )

    logger.debug("Dispatching tool '%s' with args: %s", tool_name, tool_args)

    if tool_name == "search_docs":
        result = search_docs(**tool_args)
    elif tool_name == "query_data":
        result = query_data(**tool_args)
    else:  # web_search
        result = web_search(**tool_args)

    return json.dumps(result, ensure_ascii=False, indent=2)