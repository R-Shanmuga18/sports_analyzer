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


def generate_markdown_report(results: list[dict]) -> Path:
    """Generate a clean EVALUATION.md file with Markdown tables."""
    out_path = Path(__file__).resolve().parents[1] / "EVALUATION.md"
    
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        grouped[r.get("category", "unknown")].append(r)

    md = ["# Evaluation Report & Reflection\n"]
    
    # Write Summary Table
    md.append("## 📊 Summary\n")
    md.append("| Category | Total Questions | Tools Correct | Status OK |")
    md.append("|---|---|---|---|")
    
    total_q = 0
    total_tools_ok = 0
    total_ans_ok = 0
    
    for category in ["single_tool", "multi_tool", "refusal", "edge_case"]:
        rows = grouped.get(category, [])
        total = len(rows)
        tool_ok = sum(1 for r in rows if r.get("tool_choice_correct") is True)
        
        # Heuristic for Answer OK (Success for normal, Refusal for refusals)
        if category == "refusal":
            ans_ok = sum(1 for r in rows if r.get("status") == "refusal")
        else:
            ans_ok = sum(1 for r in rows if r.get("status") == "success")
            
        total_q += total
        total_tools_ok += tool_ok
        total_ans_ok += ans_ok
        
        md.append(f"| **{category}** | {total} | {tool_ok} | {ans_ok} |")
        
    md.append(f"| **TOTAL** | **{total_q}** | **{total_tools_ok}** | **{total_ans_ok}** |\n")

    # Write Detailed Results Table
    md.append("## 📝 Detailed Results\n")
    md.append("| ID | Category | Question | Expected Tools | Actual Tools | Status | Answer Preview |")
    md.append("|---|---|---|---|---|---|---|")
    
    for r in results:
        id_str = r['id']
        cat = r['category']
        q_str = r['question'].replace("|", "\\|")
        exp_tools = ", ".join(r['expected_tools']) if r['expected_tools'] else "None"
        act_tools = ", ".join(r['actual_tools_called']) if r['actual_tools_called'] else "None"
        status = r['status']
        
        # Clean answer for markdown table (truncate to 100 chars, remove newlines)
        ans = str(r['actual_answer']).replace('\n', ' ').replace('|', '\\|')
        if len(ans) > 100:
            ans = ans[:97] + "..."
            
        # Add emojis for visual clarity
        tool_match = "✅" if r['tool_choice_correct'] else "❌"
        status_match = "✅" if (cat == "refusal" and status == "refusal") or (cat != "refusal" and status == "success") else "❌"
        
        md.append(f"| {id_str} | {cat} | {q_str} | {exp_tools} | {act_tools} {tool_match} | {status} {status_match} | {ans} |")

    # Add Reflection Section Placeholder
    md.append("\n## 🧠 Reflection on Failure Modes")
    md.append("\n*(Write your analysis of any failures here based on the DESIGN.md and our debugging sessions!)*")

    # Save to disk
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def run_evaluation() -> list[dict]:
    loop = AgentLoop()
    results: list[dict] = []

    for q in EVALUATION_QUESTIONS:
        print(f"\nRunning {q['id']}: {q['question'][:60]}...")
        result = loop.run(q["question"])

        actual_tools = extract_tools_from_trace(result.get("trace_file"))
        
        # Auto-grade the tools (Order doesn't matter, just checking if the sets match)
        tools_match = set(actual_tools) == set(q["expected_tools"])

        evaluation = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected_tools": q["expected_tools"],
            "actual_tools_called": actual_tools,
            "expected_behavior": q["expected_behavior"],
            "actual_answer": result.get("answer", ""),
            "steps_used": result.get("steps_used", 0),
            "status": result.get("status", ""),
            "trace_file": result.get("trace_file"),
            "tool_choice_correct": tools_match,
            "answer_quality": None, # Still requires human review, but summary handles it via status
            "notes": "",
        }
        results.append(evaluation)

    save_results(results)
    md_path = generate_markdown_report(results)
    
    print(f"\n✅ Evaluation complete!")
    print(f"✅ Markdown report generated at: {md_path}")
    print("Open EVALUATION.md in your editor to see the formatted tables!")
    
    return results


if __name__ == "__main__":
    run_evaluation()