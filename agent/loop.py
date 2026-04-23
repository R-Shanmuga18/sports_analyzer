"""Core custom while-loop agent for IPL RAG tool orchestration."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from agent.cache import cached_llm_call
from agent.tool_definitions import TOOL_DEFINITIONS, dispatch_tool
from agent.tracer import Tracer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_BASE = """You are an IPL cricket analyst assistant with access to three tools.

Your job is to answer questions about IPL cricket by retrieving information from the right sources.

TOOLS AVAILABLE:
- search_docs: Search IPL season review documents for narrative information
- query_data: Query the structured IPL statistics database for numbers and facts
- web_search: Search the live web for events from 2025 onwards or "current/latest" questions

RULES YOU MUST FOLLOW:
1. Always use a tool to retrieve information before answering — do not answer from memory alone.
2. If a question needs both narrative explanation AND statistics, call both search_docs AND query_data.
3. After each tool result, explicitly check: does the question have multiple parts?
   Have all parts been addressed? If not, call another tool before writing the final answer.
4. When writing your final answer, cite sources explicitly:
   - For search_docs results: name the exact file from the result's 'source' field
     (e.g. "according to ipl_2024_season.txt")
   - For query_data results: say "according to the IPL database (ipl.db)"
   - For web_search results: include the URL from the result
   - Vague citations like "a document" or "the web" are not acceptable.
5. If tool results contain no relevant information, say so honestly.
   Do not guess or fabricate an answer.
6. If asked for investment advice, future predictions, or anything outside
   cricket analysis, refuse politely without calling any tool.
7. Keep answers concise but complete. Use bullet points for lists of statistics.
8. For ambiguous questions (e.g. "the finals" without specifying a year),
   state your assumption clearly before answering.
9. For multi-part questions where only part can be answered from sources,
   answer the supported part and explicitly state what could not be found.
"""

_SYSTEM_PROMPT_WEB_FIRST = (
    "\nADDITIONAL INSTRUCTION: This question is about 2025 or later, or asks for "
    "current/latest information. You MUST call web_search before giving the final answer."
)

_SYSTEM_PROMPT_MIXED = (
    "\nADDITIONAL INSTRUCTION: This question has both a historical part and a current part. "
    "Call query_data for historical statistics and web_search for the current status "
    "before writing the final answer."
)

# ---------------------------------------------------------------------------
# Refusal patterns — checked before any tool call is made
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    "invest",
    "buy stock",
    "sell stock",
    " bet ",
    "bet on",
    "sure bet",
    "stock price prediction",
    "guarantee",
]

_NON_CRICKET_PATTERNS = [
    "write me a poem",
    "write a poem",
    "python homework",
    " homework",
    "recipe",
    "weather",
]

_CRICKET_SCOPE_TERMS = [
    "ipl",
    "cricket",
    "bcci",
    "indian premier league",
    "wicket",
    "batting",
    "bowling",
    "over",
    "innings",
    "run chase",
    # Full and short team names
    "chennai super kings", "csk",
    "mumbai indians", " mi ",
    "royal challengers", "rcb",
    "kolkata knight riders", "kkr",
    "sunrisers hyderabad", "srh",
    "delhi capitals", " dc ",
    "rajasthan royals", " rr ",
    "lucknow super giants", "lsg",
    "gujarat titans", " gt ",
    "punjab kings", "pbks",
    # Notable players often asked about
    "dhoni", "kohli", "rohit", "bumrah", "jadeja",
    # Tournament terms
    "auction", "drs", "powerplay", "death overs"
]


def _build_system_prompt(question: str) -> str:
    """
    Build the complete system prompt as a single string.
    Conditional sections are appended based on question analysis.
    This avoids injecting multiple system-role messages into the
    messages list, which is not officially supported and may be ignored.
    """
    prompt = _SYSTEM_PROMPT_BASE
    if _needs_web_search_first(question):
        prompt += _SYSTEM_PROMPT_WEB_FIRST
    if _needs_mixed_history_current(question):
        prompt += _SYSTEM_PROMPT_MIXED
    return prompt


def _is_prediction_or_investment(question: str) -> bool:
    """Return True if the question is asking for a bet, prediction, or investment advice."""
    q = f" {question.lower()} "
    if any(p in q for p in _REFUSAL_PATTERNS):
        return True
    # "Will [team] win the IPL?" — forward-looking prediction
    if re.search(r"\bwill\b.{0,40}\b(win|champion|title)\b", q) and "ipl" in q:
        return True
    # "chances/odds/predict/likely" + IPL context
    if re.search(r"\b(chances|odds|predict|prediction|likely to win)\b", q) and "ipl" in q:
        return True
    # "should I" / "best team to [action]"
    if re.search(r"\b(should i|who should|best team to)\b", q) and "ipl" in q:
        return True
    return False


def _is_non_cricket(question: str) -> bool:
    """Return True if the question is clearly not about cricket at all."""
    q = question.lower()
    return any(p in q for p in _NON_CRICKET_PATTERNS)


def _in_cricket_scope(question: str) -> bool:
    """
    Return True only if the question contains at least one IPL-specific term.
    Generic terms like "team", "player", "match" are NOT sufficient
    """
    q = f" {question.lower()} "
    return any(t in q for t in _CRICKET_SCOPE_TERMS)


def _needs_web_search_first(question: str) -> bool:
    """Return True if the question clearly needs live/current information."""
    q = question.lower()
    if any(k in q for k in ["latest", "current", "recent", "now", "today"]):
        return True
    years = re.findall(r"\b(20\d{2})\b", q)
    return any(int(y) >= 2025 for y in years)


def _needs_mixed_history_current(question: str) -> bool:
    """Return True if the question combines historical records with current status."""
    q = question.lower()
    historical = any(k in q for k in ["most titles", "overall", "history", "historical", "all time"])
    current = any(k in q for k in ["current squad", "current roster", "latest squad", "recent transfer"])
    return historical and current


def _needs_future_uncertainty_guard(question: str) -> bool:
    """Return True if the question asks about events in future calendar years."""
    q = question.lower()
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
    current_year = datetime.now().year
    # Only future years (strictly greater) trigger the guard
    asks_future = any(y > current_year for y in years)
    future_language = bool(re.search(r"\b(what will|will they|future|next year)\b", q))
    return asks_future and future_language


def extract_citations(answer: str, tool_results: list[dict]) -> list[str]:
    """
    Build a deduplicated citation list from tool calls that actually ran.

    Args:
        answer: Final answer text (unused but kept for interface compatibility)
        tool_results: List of {"tool_name": str, "output": str} dicts

    Returns:
        List of citation strings like ["search_docs(ipl_2024_season.txt)", "query_data(ipl.db)"]
    """
    del answer  # not used — citations come from tool call metadata
    citations: list[str] = []
    seen: set[str] = set()

    for item in tool_results:
        tool_name = str(item.get("tool_name", ""))
        raw_output = str(item.get("output", ""))
        source_name = "unknown"

        try:
            parsed = json.loads(raw_output)
            if tool_name == "search_docs" and isinstance(parsed, list) and parsed:
                # Collect all unique source files mentioned, not just the first
                sources = list(
                    dict.fromkeys(
                        r.get("source", "unknown")
                        for r in parsed
                        if isinstance(r, dict) and r.get("source") not in (None, "none", "unknown")
                    )
                )
                source_name = ", ".join(sources) if sources else "unknown"
            elif tool_name == "query_data" and isinstance(parsed, dict):
                source_name = str(parsed.get("source", "ipl.db"))
            elif tool_name == "web_search" and isinstance(parsed, list) and parsed:
                # Use the URL of the first real result
                urls = [
                    r.get("url", "")
                    for r in parsed
                    if isinstance(r, dict) and r.get("url", "").startswith("http")
                ]
                source_name = urls[0] if urls else "web"
        except (json.JSONDecodeError, TypeError):
            source_name = "unknown"

        citation = f"{tool_name}({source_name})"
        if citation not in seen:
            citations.append(citation)
            seen.add(citation)

    return citations


def _llm_call_with_retry(messages: list[dict], model: str):
    """
    Call the LLM with one retry on transient API failure.
    If the model explicitly cannot use tools (tool_use_failed error),
    retries without tools so we at least get a text response.
    """
    try:
        return cached_llm_call(
            model=model, messages=messages, tools=TOOL_DEFINITIONS, temperature=0
        )
    except Exception as first_exc:
        err_str = str(first_exc)
        if "tool_use_failed" in err_str or "tool_calls" in err_str:
            logger.warning("Tool use failed — retrying without tools: %s", first_exc)
            return cached_llm_call(
                model=model, messages=messages, tools=None, temperature=0
            )
        logger.warning("LLM call failed, retrying in 2s: %s", first_exc)
        time.sleep(2)
        try:
            return cached_llm_call(
                model=model, messages=messages, tools=TOOL_DEFINITIONS, temperature=0
            )
        except Exception as second_exc:
            if "tool_use_failed" in str(second_exc):
                return cached_llm_call(
                    model=model, messages=messages, tools=None, temperature=0
                )
            raise RuntimeError(
                f"LLM API failed after retry: {second_exc}"
            ) from second_exc


class AgentLoop:
    def __init__(self, model: str = "llama-3.1-8b-instant", max_steps: int = 8):
        self.model = model
        self.max_steps = max_steps
        self.tracer = Tracer()

    def _latest_trace_file(self) -> str | None:
        """Return path to the most recently written trace file."""
        files = list(Path(self.tracer.traces_dir).glob("trace_*.json"))
        if not files:
            return None
        return str(max(files, key=lambda p: p.stat().st_mtime))

    def run(self, question: str, step_callback=None) -> dict:
        """
        Run the agent on a question and return a result dict.

        Args:
            question: The user's natural language question
            step_callback: Optional callable(step_num, tool_name, tool_input,
                           tool_output, latency_ms) — called after each tool
                           execution so a UI can update live (e.g. Gradio streaming).

        Returns:
            Dict with keys:
            - answer: str — final answer or refusal/error message
            - citations: list[str]
            - steps_used: int
            - status: str — "success" | "max_steps_exceeded" | "refusal" |
                            "api_error" | "parse_error"
            - trace_file: str | None — path to saved trace JSON
        """
        # ------------------------------------------------------------------ #
        # Pre-flight refusal checks — no tool calls made for these            #
        # ------------------------------------------------------------------ #
        if _is_prediction_or_investment(question):
            return {
                "answer": (
                    "I'm sorry, I can't help with predictions or investment advice. "
                    "I can answer factual questions about IPL history and statistics."
                ),
                "citations": [],
                "steps_used": 0,
                "status": "refusal",
                "trace_file": None,
            }

        if _is_non_cricket(question):
            return {
                "answer": (
                    "I can only help with factual IPL cricket analysis. "
                    "This question is outside my scope."
                ),
                "citations": [],
                "steps_used": 0,
                "status": "refusal",
                "trace_file": None,
            }

        if not _in_cricket_scope(question):
            return {
                "answer": (
                    "I could not find relevant information about this topic in my sources. "
                    "I specialise in IPL cricket — please ask about IPL teams, players, "
                    "matches, or seasons."
                ),
                "citations": [],
                "steps_used": 0,
                "status": "refusal",
                "trace_file": None,
            }

        # ------------------------------------------------------------------ #
        # Initialise conversation — single system message at position 0       #
        # ------------------------------------------------------------------ #
        messages: list[dict] = [
            {"role": "system", "content": _build_system_prompt(question)},
            {"role": "user", "content": question},
        ]
        step_count = 0
        tool_results: list[dict] = []

        self.tracer.start(question)
        if self.tracer.current_trace is not None:
            self.tracer.current_trace.max_steps = self.max_steps

        # ------------------------------------------------------------------ #
        # Main agent loop                                                      #
        # ------------------------------------------------------------------ #
        while step_count < self.max_steps:
            # --- LLM call ------------------------------------------------- #
            try:
                response = _llm_call_with_retry(messages=messages, model=self.model)
            except Exception as exc:
                logger.error("LLM API error: %s", exc)
                trace = self.tracer.finish(
                    f"I could not complete this request due to an API error: {exc}",
                    extract_citations("", tool_results),
                    "api_error",
                )
                self.tracer.print_trace(trace)
                return {
                    "answer": f"I could not complete this request due to an API error: {exc}",
                    "citations": extract_citations("", tool_results),
                    "steps_used": step_count,
                    "status": "api_error",
                    "trace_file": self._latest_trace_file(),
                }

            response_msg = response.choices[0].message
            tool_calls = getattr(response_msg, "tool_calls", None) or []

            # --- Tool call(s) branch -------------------------------------- #
            if tool_calls:
                # Process ALL tool calls returned in this response.
                # OpenAI may return multiple parallel tool calls in one message.
                # The original code only processed tool_calls[0], silently
                # dropping the rest.
                assistant_content = getattr(response_msg, "content", "") or ""
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tool_call in tool_calls:
                    if step_count >= self.max_steps:
                        break  # hard cap — stop processing further calls

                    tool_name = tool_call.function.name
                    raw_args = tool_call.function.arguments or "{}"

                    # Parse arguments
                    try:
                        tool_args = json.loads(raw_args)
                    except json.JSONDecodeError as parse_exc:
                        logger.error(
                            "Failed to parse tool arguments for '%s': %s",
                            tool_name, parse_exc,
                        )
                        self.tracer.log_tool_call(
                            step_count + 1,
                            tool_name,
                            {"raw_arguments": raw_args},
                            f"Argument parse error: {parse_exc}",
                            0.0,
                        )
                        # Append a tool result so OpenAI doesn't error on the
                        # unmatched tool_call_id in the assistant message
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": json.dumps(
                                    {"error": f"Could not parse arguments: {parse_exc}"}
                                ),
                            }
                        )
                        step_count += 1
                        continue

                    # Execute tool
                    start = time.perf_counter()
                    try:
                        tool_output = dispatch_tool(tool_name, tool_args)
                    except Exception as tool_exc:
                        logger.warning("Tool '%s' raised: %s", tool_name, tool_exc)
                        tool_output = json.dumps(
                            {"error": f"Tool '{tool_name}' failed: {tool_exc}"}
                        )
                    latency_ms = (time.perf_counter() - start) * 1000

                    # Append tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_output,
                        }
                    )

                    step_count += 1
                    self.tracer.log_tool_call(
                        step_count, tool_name, tool_args, tool_output, latency_ms
                    )
                    tool_results.append({"tool_name": tool_name, "output": tool_output})
                    logger.debug(
                        "Step %d: %s completed in %.0fms", step_count, tool_name, latency_ms
                    )

                    if step_callback is not None:
                        try:
                            step_callback(
                                step_count, tool_name, tool_args, tool_output, latency_ms
                            )
                        except Exception as cb_exc:
                            logger.warning("step_callback raised: %s", cb_exc)

                continue  # go back to top of while loop for next LLM call

            # --- Final answer branch -------------------------------------- #
            final_answer = (getattr(response_msg, "content", "") or "").strip()

            # Guard: add explicit uncertainty note for future-year questions
            if _needs_future_uncertainty_guard(question):
                lower_answer = final_answer.lower()
                if not any(
                    k in lower_answer
                    for k in ["could not", "unable", "unknown", "not available", "cannot"]
                ):
                    final_answer += (
                        "\n\n*Note: I could not find source-backed information "
                        "to confirm future actions or outcomes with certainty.*"
                    )

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

        # ------------------------------------------------------------------ #
        # Hard cap reached                                                     #
        # ------------------------------------------------------------------ #
        cap_message = (
            "I could not find enough reliable information within the maximum "
            f"allowed reasoning steps ({self.max_steps}). "
            "Please rephrase or narrow your question so I can answer it reliably."
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