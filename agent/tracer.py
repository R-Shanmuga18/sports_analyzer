"""Structured trace logger for agent runs and tool-call observability."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import re
import time
from pathlib import Path


@dataclass
class ToolCall:
    step: int
    tool_name: str
    input: dict
    output: str
    latency_ms: float


@dataclass
class AgentTrace:
    question: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    steps: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    citations: list[str] = field(default_factory=list)
    steps_used: int = 0
    max_steps: int = 8
    status: str = "in_progress"
    total_time_ms: float = 0.0


class Tracer:
    def __init__(self, traces_dir: str = "traces"):
        try:
            self.traces_dir = Path(traces_dir)
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback keeps logger operational even if directory creation fails initially.
            self.traces_dir = Path("traces")
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace: AgentTrace | None = None
        self._start_time: float = 0.0

    def start(self, question: str) -> None:
        """Begin a new trace for a question."""
        try:
            self.current_trace = AgentTrace(question=question)
            self._start_time = time.perf_counter()
        except Exception:
            self.current_trace = AgentTrace(question=str(question))
            self._start_time = time.perf_counter()

    def log_tool_call(
        self,
        step: int,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        latency_ms: float,
    ) -> None:
        """Record a single tool call in the current trace."""
        try:
            if self.current_trace is None:
                self.start("(trace started implicitly)")

            call = ToolCall(
                step=int(step),
                tool_name=str(tool_name),
                input=tool_input if isinstance(tool_input, dict) else {"value": str(tool_input)},
                output=str(tool_output),
                latency_ms=float(latency_ms),
            )
            self.current_trace.steps.append(call)
            self.current_trace.steps_used = len(self.current_trace.steps)
        except Exception:
            # Never crash tracing logic.
            return

    def finish(self, final_answer: str, citations: list[str], status: str) -> AgentTrace:
        """Complete the trace and save it to disk. Returns the completed trace."""
        try:
            if self.current_trace is None:
                self.start("(trace finished without explicit start)")

            self.current_trace.final_answer = str(final_answer)
            self.current_trace.citations = [str(c) for c in (citations or [])]
            self.current_trace.status = str(status)
            self.current_trace.steps_used = len(self.current_trace.steps)
            elapsed_ms = (time.perf_counter() - self._start_time) * 1000 if self._start_time else 0.0
            self.current_trace.total_time_ms = max(0.0, float(elapsed_ms))
            self.save(self.current_trace)
            return self.current_trace
        except Exception:
            fallback = AgentTrace(
                question="(trace finish fallback)",
                final_answer=str(final_answer),
                citations=[str(c) for c in (citations or [])],
                status=str(status),
                total_time_ms=0.0,
            )
            self.save(fallback)
            return fallback

    def _safe_filename(self, trace: AgentTrace) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        words = trace.question.strip().split()[:5] if trace.question else ["empty"]
        slug_raw = "_".join(words).lower()
        slug = re.sub(r"[^a-z0-9_]+", "", slug_raw)
        slug = re.sub(r"_+", "_", slug).strip("_") or "empty"
        return f"trace_{timestamp}_{slug}.json"

    def save(self, trace: AgentTrace) -> str:
        """Save trace as JSON to traces/ directory. Returns filepath."""
        try:
            self.traces_dir.mkdir(parents=True, exist_ok=True)
            file_name = self._safe_filename(trace)
            file_path = self.traces_dir / file_name
            payload = asdict(trace)
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(file_path)
        except Exception:
            return ""

    def _preview_text(self, text: str, limit: int = 200) -> str:
        normalized = (text or "").replace("\n", " ").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."

    def print_trace(self, trace: AgentTrace) -> None:
        """Print a human-readable trace to stdout in the required format."""
        try:
            print("═" * 42)
            print(f"QUESTION: {trace.question}")
            print("═" * 42)

            for call in trace.steps:
                input_preview = self._preview_text(json.dumps(call.input, ensure_ascii=False))
                output_preview = self._preview_text(call.output)
                print(f"Step {call.step}: tool={call.tool_name}")
                print(f"  input : '{input_preview}'")
                print(f"  output: {output_preview}")
                print(f"  time  : {int(round(call.latency_ms))}ms")
                print("")

            print("─" * 42)
            print("FINAL ANSWER:")
            print(trace.final_answer)
            print("")
            citations_text = " | ".join(trace.citations) if trace.citations else "None"
            print(f"CITATIONS: {citations_text}")
            print(
                f"STATUS: {trace.status} | STEPS: {trace.steps_used}/{trace.max_steps} | "
                f"TIME: {int(round(trace.total_time_ms))}ms"
            )
            print("═" * 42)
        except Exception:
            # Failsafe to keep caller flow alive even if formatting fails.
            print("Trace unavailable")
