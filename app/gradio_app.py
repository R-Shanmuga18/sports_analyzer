"""Full Gradio interface for the IPL Agentic RAG system."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import gradio as gr

from agent.loop import AgentLoop
from app.components import (
    format_answer_with_citations,
    format_trace_for_display,
    get_status_color,
    load_evaluation_results,
)
from eval.questions import EVALUATION_QUESTIONS

logger = logging.getLogger(__name__)

EXAMPLES = [
    ["Who won the 2024 IPL title?"],
    ["How did KKR win the 2024 IPL final and what was their season win record?"],
    ["Compare CSK and MI head-to-head results in 2023 and 2024."],
    ["Who were the top 3 run scorers in 2023 IPL?"],
    ["Which team should I bet on for IPL 2025?"],
    ["Who is the current IPL points table leader?"],
]


def _metric_card(label: str, value: str) -> str:
    """Render a compact metric card as markdown-compatible HTML."""
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:12px;padding:12px;'>"
        f"<div style='font-size:12px;color:#6b7280;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:700;margin-top:4px;'>{value}</div>"
        "</div>"
    )


def _load_eval_details() -> list[dict]:
    """Load full evaluation JSON rows for row-click detail view."""
    path = Path(__file__).resolve().parents[1] / "eval" / "evaluation_results.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse evaluation_results.json")
        return []


def _status_line(steps_used: int, status: str, total_ms: int, max_steps: int = 8) -> str:
    """Build a consistent status line for the trace section."""
    return f"Steps used: {steps_used} / {max_steps} | Status: {status} | Total time: {total_ms}ms"


def _style_answer(answer: str, status: str) -> str:
    """Apply required UI messaging and style by status."""
    if status == "max_steps_exceeded":
        return "I couldn't find a complete answer within the search limit. Here's what I found:\n\n" + answer
    if status == "refusal":
        return (
            "<div style='background:#fff7ed;border-left:4px solid #d97706;padding:10px;border-radius:8px;'>"
            + answer
            + "</div>"
        )
    return answer


def run_agent_streaming(question: str):
    """
    Generator that yields (answer, trace_data, status_text) tuples as the agent progresses.

    Yield 1: ("Thinking...", [], "Running...")
    Yield 2 onwards: ("Thinking...", [rows_so_far], "Step X/8...")
    Final yield: (final_answer, all_rows, "Done - X steps | Status: success")
    """
    question = (question or "").strip()
    if not question:
        yield ("Please enter a question.", [], "Idle")
        return

    yield ("Thinking...", [], "Running...")
    start = time.perf_counter()
    queue: Queue = Queue()
    raw_steps: list[dict] = []
    holder: dict = {}

    def step_callback(step_num, tool_name, tool_input, tool_output, latency_ms):
        queue.put(
            {
                "step_num": step_num,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
                "latency_ms": latency_ms,
            }
        )

    def worker() -> None:
        try:
            holder["result"] = AgentLoop().run(question, step_callback=step_callback)
        except Exception as exc:
            holder["error"] = exc
            logger.exception("Agent run failed")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive() or not queue.empty():
        try:
            step = queue.get(timeout=0.1)
            raw_steps.append(step)
            rows = format_trace_for_display(raw_steps)
            yield ("Thinking...", rows, f"Step {len(raw_steps)}/8...")
        except Empty:
            continue

    if "error" in holder:
        yield ("Something went wrong. Please try again.", format_trace_for_display(raw_steps), "Status: error")
        return

    result = holder.get("result", {})
    status = result.get("status", "error")
    answer = _style_answer(result.get("answer", ""), status)
    answer = format_answer_with_citations(answer, result.get("citations", []))
    rows = format_trace_for_display(raw_steps)
    total_ms = int((time.perf_counter() - start) * 1000)
    final_status = _status_line(int(result.get("steps_used", len(raw_steps))), status, total_ms)
    yield (answer, rows, final_status)


def _on_eval_row_select(eval_details: list[dict], evt: gr.SelectData):
    """Show full answer and trace details when a result row is selected."""
    if not eval_details:
        return "No evaluation data loaded.", [], ""

    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if idx < 0 or idx >= len(eval_details):
        return "Invalid row selection.", [], ""

    row = eval_details[idx]
    answer = row.get("actual_answer", "")
    status = row.get("status", "")
    detail_md = (
        f"### {row.get('id', '')} - {row.get('question', '')}\n"
        f"<span style='color:{get_status_color(status)};'>Status: {status}</span>"
    )

    trace_rows: list[list] = []
    trace_path = row.get("trace_file")
    if trace_path:
        path = Path(trace_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / trace_path
        try:
            if path.exists():
                trace_data = json.loads(path.read_text(encoding="utf-8"))
                trace_rows = format_trace_for_display(trace_data.get("steps", []))
        except Exception:
            logger.exception("Failed loading trace file: %s", path)

    return f"{detail_md}\n\n{answer}", trace_rows, _status_line(row.get("steps_used", 0), status, 0)


def _rerun_evaluation(progress=gr.Progress(track_tqdm=False)):
    """Re-run all 20 evaluation questions and refresh table plus summary metrics."""
    loop = AgentLoop()
    results: list[dict] = []
    total = len(EVALUATION_QUESTIONS)

    for i, q in enumerate(EVALUATION_QUESTIONS, start=1):
        progress(i / total, desc=f"Running {q['id']}")
        result = loop.run(q["question"])
        results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_tools": q["expected_tools"],
                "actual_tools_called": [],
                "expected_behavior": q["expected_behavior"],
                "actual_answer": result.get("answer", ""),
                "steps_used": result.get("steps_used", 0),
                "status": result.get("status", ""),
                "trace_file": result.get("trace_file"),
                "tool_choice_correct": None,
                "answer_quality": None,
                "notes": "",
            }
        )

    out = Path(__file__).resolve().parents[1] / "eval" / "evaluation_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    rows, metrics = load_evaluation_results()
    details = _load_eval_details()
    return (
        rows,
        _metric_card("Overall Pass Rate", metrics["overall_pass_rate"]),
        _metric_card("Single-Tool Accuracy", metrics["single_tool_accuracy"]),
        _metric_card("Multi-Tool Accuracy", metrics["multi_tool_accuracy"]),
        _metric_card("Refusal Accuracy", metrics["refusal_accuracy"]),
        details,
        gr.update(visible=False),
    )


def build_app() -> gr.Blocks:
    """Build and return the complete two-tab Gradio application."""
    rows, metrics = load_evaluation_results()
    eval_details = _load_eval_details()

    with gr.Blocks(title="IPL Agentic RAG") as app:
        gr.Markdown("# IPL Agentic RAG")

        with gr.Tabs():
            with gr.Tab("Ask the Agent"):
                with gr.Row():
                    with gr.Column(scale=5):
                        question_box = gr.Textbox(
                            label="Ask about IPL cricket",
                            placeholder="e.g. How did KKR win the 2024 IPL final?",
                            lines=6,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Submit", variant="primary")
                            clear_btn = gr.Button("Clear")

                    with gr.Column(scale=7):
                        gr.Markdown("### Answer")
                        answer_box = gr.Markdown(value="", elem_id="answer-box")
                        with gr.Accordion("Agent Trace", open=False):
                            trace_df = gr.Dataframe(
                                headers=["Step", "Tool Called", "Input", "Output Preview", "Time (ms)"],
                                value=[],
                                interactive=False,
                                wrap=True,
                            )
                        status_line = gr.Markdown(value="Steps used: 0 / 8 | Status: idle | Total time: 0ms")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[question_box],
                    outputs=[answer_box, trace_df, status_line],
                    fn=run_agent_streaming,
                    cache_examples=False,
                    run_on_click=True,
                )

                submit_btn.click(
                    fn=run_agent_streaming,
                    inputs=[question_box],
                    outputs=[answer_box, trace_df, status_line],
                )
                question_box.submit(
                    fn=run_agent_streaming,
                    inputs=[question_box],
                    outputs=[answer_box, trace_df, status_line],
                )
                clear_btn.click(
                    fn=lambda: ("", "", [], "Steps used: 0 / 8 | Status: idle | Total time: 0ms"),
                    outputs=[question_box, answer_box, trace_df, status_line],
                )

            with gr.Tab("Evaluation Results"):
                details_state = gr.State(value=eval_details)
                with gr.Row():
                    overall_md = gr.Markdown(_metric_card("Overall Pass Rate", metrics["overall_pass_rate"]))
                    single_md = gr.Markdown(_metric_card("Single-Tool Accuracy", metrics["single_tool_accuracy"]))
                    multi_md = gr.Markdown(_metric_card("Multi-Tool Accuracy", metrics["multi_tool_accuracy"]))
                    refusal_md = gr.Markdown(_metric_card("Refusal Accuracy", metrics["refusal_accuracy"]))

                eval_table = gr.Dataframe(
                    headers=["ID", "Category", "Question", "Tools Called", "Status", "Steps Used"],
                    value=rows,
                    interactive=False,
                    wrap=True,
                )

                rerun_btn = gr.Button("Re-run Evaluation", variant="primary")
                confirm_row = gr.Row(visible=False)
                with confirm_row:
                    gr.Markdown("Re-run all 20 evaluation questions now?")
                    confirm_btn = gr.Button("Confirm")
                    cancel_btn = gr.Button("Cancel")

                detail_answer = gr.Markdown("Select a row to view full answer and trace.")
                detail_trace = gr.Dataframe(
                    headers=["Step", "Tool Called", "Input", "Output Preview", "Time (ms)"],
                    value=[],
                    interactive=False,
                    wrap=True,
                )
                detail_status = gr.Markdown("")

                rerun_btn.click(lambda: gr.update(visible=True), outputs=[confirm_row])
                cancel_btn.click(lambda: gr.update(visible=False), outputs=[confirm_row])
                confirm_btn.click(
                    fn=_rerun_evaluation,
                    outputs=[eval_table, overall_md, single_md, multi_md, refusal_md, details_state, confirm_row],
                )
                eval_table.select(
                    fn=_on_eval_row_select,
                    inputs=[details_state],
                    outputs=[detail_answer, detail_trace, detail_status],
                )

    return app
