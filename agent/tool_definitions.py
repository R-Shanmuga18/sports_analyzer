"""Function-calling schemas and dispatcher for the three agent tools."""

from __future__ import annotations


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": """Use this tool to search IPL season review documents for narrative information.

CALL THIS TOOL when the question asks about:
- How a match was won or lost (game narrative, key moments)
- Player performances described in text form (for example, "Kohli played an anchor role")
- Team strategy, tournament storylines, or notable events
- Reasons or explanations behind outcomes
- Any answer that would appear in a written article or match report

DO NOT call this tool when:
- The question asks for a specific number (runs, average, economy) - use query_data instead
- The question contains words like "latest", "current", "recent", "now" - use web_search instead
- You already retrieved a relevant chunk and need more numbers - use query_data instead

Input must be a specific search phrase describing what information you need, not the full user question.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific natural language phrase to search for in the documents. Be precise - for example 'KKR winning strategy 2024 final' not 'tell me about KKR'.",
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
            "description": """Use this tool to query structured IPL statistics from the database.

CALL THIS TOOL when the question asks about:
- Exact numbers such as runs, wickets, strike rate, economy, averages, wins, losses
- Aggregates such as totals, counts, maximums, minimums, rankings, comparisons
- Season-level or player-level statistical analysis
- Any question that can be answered with SQL over tabular match data
- If the question contains any number, statistic, count, total, average, highest, lowest, or ranking - this is your tool

DO NOT call this tool when:
- The user asks for narrative explanation, strategy, or match storyline - use search_docs instead
- The question is about latest/current/recent/post-2024 live updates - use web_search instead
- The answer depends on article-style commentary rather than measurable statistics - use search_docs instead

Input must be one clear natural language statistics question that can be translated to SQL.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question about IPL statistics. Include entity and season when possible, for example 'How many matches did MI win in 2023?'.",
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
            "description": """Use this tool to search the live web for IPL information that is not covered by local data.

CALL THIS TOOL when the question asks about:
- Events from 2025 onwards (transfers, auctions, injuries, coaching changes)
- Current player status, current squads, or current team updates
- Live or recent results and announcements
- Questions containing words like "latest", "current", "recent", "now", or "today"

DO NOT call this tool when:
- The question is about historical 2023 or 2024 statistics - use query_data instead
- The question asks for narrative explanations from season reviews - use search_docs instead
- The answer already exists in local structured or document data - use query_data or search_docs first

ONLY call this for information from 2025 onwards or explicitly current/latest/recent status questions.

Input must be a short, focused search phrase under 10 words, not a long paragraph.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short web search phrase under 10 words, for example 'IPL 2025 auction retained players'.",
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

    Args:
        tool_name: Name of the tool as returned by groq-llama 3.1 8b model (search_docs, query_data, web_search)
        tool_args: Arguments dict as returned by groq-llama 3.1 8b model (already parsed from JSON)

    Returns:
        String representation of the tool result (to be added to messages)

    Raises:
        ValueError: If tool_name is not one of the three known tools
    """
    import json

    from tools.query_data import query_data
    from tools.search_docs import search_docs
    from tools.web_search import web_search

    if tool_name == "search_docs":
        result = search_docs(**tool_args)
    elif tool_name == "query_data":
        result = query_data(**tool_args)
    elif tool_name == "web_search":
        result = web_search(**tool_args)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    return json.dumps(result, ensure_ascii=False, indent=2)
