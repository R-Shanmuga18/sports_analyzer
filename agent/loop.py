"""Core custom while-loop agent for IPL RAG tool orchestration."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agent.cache import cached_llm_call
from agent.tool_definitions import TOOL_DEFINITIONS, dispatch_tool
from agent.tracer import Tracer


SYSTEM_PROMPT = """You are an IPL cricket analyst assistant with access to three tools.

Your job is to answer questions about IPL cricket by retrieving information from the right sources.

TOOLS AVAILABLE:
- search_docs: Search IPL season review documents for narrative information
- query_data: Query the structured IPL statistics database for numbers and facts
- web_search: Search the live web for recent/current information

RULES YOU MUST FOLLOW:
1. Always use a tool to retrieve information before answering - do not answer from memory alone
2. If a question needs both narrative explanation AND statistics, call both search_docs AND query_data
3. After getting tool results, decide if you have enough information. If not, call another tool
    - After each tool result, explicitly check: does the question have multiple parts and have all parts been answered?
    - If any part is missing, call another tool before finalizing
4. When writing your final answer, cite your sources explicitly:
   - For search_docs results: mention the document name (e.g. "according to ipl_2024_season.txt")
   - For query_data results: mention the database (e.g. "according to the IPL database")
   - For web_search results: include the URL
    - Vague citations like "a document" or "the web" are not acceptable
5. If you cannot find the answer after using tools, say so clearly - do not guess or hallucinate
6. If asked for investment advice, predictions, or anything outside cricket analysis, refuse politely
7. Keep answers concise but complete. Use bullet points for lists of statistics.
8. If the question is about 2025 or later, or asks for latest/current/recent info, call web_search first.
9. If a question combines historical records (titles/history/stats) and current status (current squad/latest roster), call query_data for historical part and web_search for current part before finalizing.
10. If the question is purely narrative (how/why/strategy/storyline) and does not ask for specific numbers, use search_docs only.
"""

REFUSAL_PATTERNS = [
    "invest",
    "buy",
    "sell",
    "bet",
    "bet on",
    "stock price prediction",
    "which team will win",
    "who will win",
    "guarantee",
    "sure bet",
]


def extract_citations(answer: str, tool_results: list[dict]) -> list[str]:
    """
    Build a citation list from the tool calls that were actually made.
    Each tool call that produced a result contributes one citation.
    """
    del answer
    citations: list[str] = []
    seen: set[str] = set()
    for item in tool_results:
        tool_name = str(item.get("tool_name", ""))
        raw_output = str(item.get("output", ""))
        source_name = "unknown"
        try:
            parsed = json.loads(raw_output)
            if tool_name == "search_docs" and isinstance(parsed, list) and parsed:
                source_name = str(parsed[0].get("source", "unknown"))
            elif tool_name == "query_data" and isinstance(parsed, dict):
                source_name = str(parsed.get("source", "ipl.db"))
            elif tool_name == "web_search" and isinstance(parsed, list) and parsed:
                source_name = str(parsed[0].get("url", "web"))
        except Exception:
            source_name = "unknown"
        citation = f"{tool_name}({source_name})"
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)
    return citations


def _needs_web_search_first(question: str) -> bool:
    q = question.lower()
    if any(k in q for k in ["latest", "current", "recent", "now", "today"]):
        return True
    year_match = re.findall(r"\b(20\d{2})\b", q)
    return any(int(y) >= 2025 for y in year_match)


def _needs_mixed_history_current(question: str) -> bool:
    q = question.lower()
    historical = any(k in q for k in ["most titles", "overall", "history", "historical", "record"])
    current = any(k in q for k in ["current squad", "current", "latest", "recent", "roster"])
    return historical and current


class AgentLoop:
    def __init__(self, model: str = "llama-3.1-8b-instant", max_steps: int = 8):
        self.model = model
        self.max_steps = max_steps
        self.tracer = Tracer()

    def _latest_trace_file(self) -> str | None:
        files = list(Path(self.tracer.traces_dir).glob("trace_*.json"))
        if not files:
            return None
        return str(max(files, key=lambda p: p.stat().st_mtime))

    def run(self, question: str) -> dict:
        """
        Run the agent on a question and return a result dict.

        Returns dict with keys:
        - answer: str (final answer or refusal message)
        - citations: list[str]
        - steps_used: int
        - status: str (success | max_steps_exceeded | refusal)
        - trace_file: str (path to saved trace)
        """
        lower_q = question.lower()
        if any(pattern in lower_q for pattern in REFUSAL_PATTERNS):
            return {
                "answer": "I'm sorry, I can't help with predictions or investment advice. I can answer factual questions about IPL history and statistics.",
                "citations": [],
                "steps_used": 0,
                "status": "refusal",
                "trace_file": None,
            }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        if _needs_web_search_first(question):
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "This query is post-2024 or current. You must call web_search first before giving the final answer.",
                },
            )
        if _needs_mixed_history_current(question):
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": "This question has historical and current parts. Call query_data for the historical part and web_search for the current part before writing the final answer.",
                },
            )
        step_count = 0
        tool_results: list[dict] = []

        self.tracer.start(question)
        if self.tracer.current_trace is not None:
            self.tracer.current_trace.max_steps = self.max_steps

        while step_count < self.max_steps:
            try:
                response = cached_llm_call(model=self.model, messages=messages, tools=TOOL_DEFINITIONS, temperature=0)
            except Exception as exc:
                if "tool_use_failed" not in str(exc):
                    raise
                response = cached_llm_call(model=self.model, messages=messages, tools=None, temperature=0)
            response_msg = response.choices[0].message
            tool_calls = getattr(response_msg, "tool_calls", None) or []

            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                assistant_content = getattr(response_msg, "content", "") or ""

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments or "{}",
                                },
                            }
                        ],
                    }
                )

                start = time.perf_counter()
                tool_output = dispatch_tool(tool_name, tool_args)
                latency_ms = (time.perf_counter() - start) * 1000

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                )

                step_count += 1
                self.tracer.log_tool_call(step_count, tool_name, tool_args, tool_output, latency_ms)
                tool_results.append({"tool_name": tool_name, "output": tool_output})
                continue

            final_answer = (getattr(response_msg, "content", "") or "").strip()
            citations = extract_citations(final_answer, tool_results)
            trace = self.tracer.finish(final_answer, citations, "success")
            self.tracer.print_trace(trace)
            return {
                "answer": final_answer,
                "citations": citations,
                "steps_used": step_count,
                "status": "success",
                "trace_file": self._latest_trace_file(),
            }

        cap_message = (
            "I reached the maximum number of reasoning steps for this question. "
            "Please rephrase or narrow the request so I can answer reliably."
        )
        citations = extract_citations(cap_message, tool_results)
        trace = self.tracer.finish(cap_message, citations, "max_steps_exceeded")
        self.tracer.print_trace(trace)
        return {
            "answer": cap_message,
            "citations": citations,
            "steps_used": step_count,
            "status": "max_steps_exceeded",
            "trace_file": self._latest_trace_file(),
        }
