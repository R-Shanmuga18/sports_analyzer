"""UI helper utilities for formatting trace and evaluation data."""

from __future__ import annotations

import json
from pathlib import Path


def _truncate(text: str, limit: int = 80) -> str:
    """Return text truncated to the given character limit."""
    cleaned = str(text).replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def format_trace_for_display(trace_steps: list) -> list:
    """
    Convert raw trace step dicts into rows for the Gradio Dataframe.

    Each row: [step_num, tool_name, input_preview, output_preview, latency_ms].
    Input and output previews are truncated to 80 characters.
    """
    rows: list[list] = []
    for i, step in enumerate(trace_steps, start=1):
        step_num = step.get("step") or step.get("step_num") or i
        tool_name = step.get("tool_name", "")
        tool_input = step.get("input") if "input" in step else step.get("tool_input", "")
        tool_output = step.get("output") if "output" in step else step.get("tool_output", "")
        latency_ms = int(round(float(step.get("latency_ms", 0))))
        rows.append([
            step_num,
            tool_name,
            _truncate(json.dumps(tool_input, ensure_ascii=False) if isinstance(tool_input, (dict, list)) else str(tool_input)),
            _truncate(json.dumps(tool_output, ensure_ascii=False) if isinstance(tool_output, (dict, list)) else str(tool_output)),
            latency_ms,
        ])
    return rows


def format_answer_with_citations(answer: str, citations: list) -> str:
    """
    Format the final answer with citations appended as a clean footnote section.

    Output format:
    <answer text>

    ---
    Sources: search_docs(ipl_2024_season.txt) · query_data(ipl.db)
    """
    base = (answer or "").strip()
    if not citations:
        return base
    return f"{base}\n\n---\nSources: {' · '.join(str(c) for c in citations)}"


def load_evaluation_results() -> tuple[list, dict]:
    """
    Load eval/evaluation_results.json if it exists.

    Returns:
        (rows_for_dataframe, summary_metrics_dict)
    """
    path = Path(__file__).resolve().parents[1] / "eval" / "evaluation_results.json"
    if not path.exists():
        return [], {
            "overall_pass_rate": "0.0%",
            "single_tool_accuracy": "0.0%",
            "multi_tool_accuracy": "0.0%",
            "refusal_accuracy": "0.0%",
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[list] = []
    for row in data:
        tools = row.get("actual_tools_called", [])
        tools_called = ", ".join(tools) if isinstance(tools, list) else str(tools)
        rows.append([
            row.get("id", ""),
            row.get("category", ""),
            row.get("question", ""),
            tools_called,
            row.get("status", ""),
            row.get("steps_used", 0),
        ])

    def _acc(rows_subset: list) -> float:
        if not rows_subset:
            return 0.0
        good = sum(1 for r in rows_subset if r.get("answer_quality") is True)
        return (good / len(rows_subset)) * 100.0

    total = len(data)
    passed = sum(
        1
        for r in data
        if r.get("tool_choice_correct") is True and r.get("answer_quality") is True
    )
    single_rows = [r for r in data if r.get("category") == "single_tool"]
    multi_rows = [r for r in data if r.get("category") == "multi_tool"]
    refusal_rows = [r for r in data if r.get("category") == "refusal"]

    metrics = {
        "overall_pass_rate": f"{((passed / total) * 100.0 if total else 0.0):.1f}%",
        "single_tool_accuracy": f"{_acc(single_rows):.1f}%",
        "multi_tool_accuracy": f"{_acc(multi_rows):.1f}%",
        "refusal_accuracy": f"{_acc(refusal_rows):.1f}%",
    }
    return rows, metrics


def get_status_color(status: str) -> str:
    """Return a color string for status badges: green=success, red=error, amber=refusal."""
    value = (status or "").lower().strip()
    if value == "success":
        return "#16a34a"
    if value == "refusal":
        return "#d97706"
    return "#dc2626"
