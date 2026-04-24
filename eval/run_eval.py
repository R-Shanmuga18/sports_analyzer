"""Prompt 9 evaluation runner for 20-question IPL Agentic RAG assessment."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from agent.loop import AgentLoop
from eval.questions import EVALUATION_QUESTIONS


def extract_tools_from_trace(trace_file: str | None) -> list[str]:
	"""Read trace JSON and return ordered unique tools called."""
	if not trace_file:
		return []
	try:
		path = Path(trace_file)
		if not path.exists():
			return []
		data = json.loads(path.read_text(encoding="utf-8"))
		steps = data.get("steps", [])
		tools: list[str] = []
		seen: set[str] = set()
		for s in steps:
			t = s.get("tool_name")
			if isinstance(t, str) and t and t not in seen:
				tools.append(t)
				seen.add(t)
		return tools
	except Exception:
		return []


def save_results(results: list[dict]) -> Path:
	"""Save evaluation rows to eval/evaluation_results.json."""
	out_path = Path(__file__).resolve().parent / "evaluation_results.json"
	out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
	return out_path


def print_summary(results: list[dict]) -> None:
	"""Print category summary using currently filled fields."""
	grouped: dict[str, list[dict]] = defaultdict(list)
	for r in results:
		grouped[r.get("category", "unknown")].append(r)

	print("\n===== Evaluation Summary =====")
	for category in ["single_tool", "multi_tool", "refusal", "edge_case"]:
		rows = grouped.get(category, [])
		total = len(rows)
		tool_ok = sum(1 for r in rows if r.get("tool_choice_correct") is True)
		ans_ok = sum(1 for r in rows if r.get("answer_quality") is True)
		print(f"{category:12} | total={total:2d} | tool_ok={tool_ok:2d} | answer_ok={ans_ok:2d}")


def run_evaluation() -> list[dict]:
	loop = AgentLoop()
	results: list[dict] = []

	for q in EVALUATION_QUESTIONS:
		print(f"\nRunning {q['id']}: {q['question'][:60]}...")
		result = loop.run(q["question"])

		evaluation = {
			"id": q["id"],
			"category": q["category"],
			"question": q["question"],
			"expected_tools": q["expected_tools"],
			"actual_tools_called": extract_tools_from_trace(result.get("trace_file")),
			"expected_behavior": q["expected_behavior"],
			"actual_answer": result.get("answer", ""),
			"steps_used": result.get("steps_used", 0),
			"status": result.get("status", ""),
			"trace_file": result.get("trace_file"),
			"tool_choice_correct": None,
			"answer_quality": None,
			"notes": "",
		}
		results.append(evaluation)

	out_path = save_results(results)
	print(f"\nSaved results: {out_path}")
	print_summary(results)
	return results


if __name__ == "__main__":
	run_evaluation()
