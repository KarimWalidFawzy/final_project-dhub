"""Command-line entry point for the D-Hub research agent."""

import argparse
import json
from pathlib import Path
from typing import List

from actions import send_email, write_dashboard
from agent import MultiAgentSystem
from llm import summarize_with_llm
from retriever import retrieve
from summarizer import build_report, summarize


def decompose_task(task: str) -> List[str]:
    """Create useful research sub-tasks from a natural-language request."""
    return [
        f"Identify the key facts needed to answer: {task}",
        f"Compare evidence and note limitations related to: {task}",
        f"Form a concise answer to: {task}",
    ]


def run_research(task: str, sources: List[str]) -> dict:
    system = MultiAgentSystem()
    system.register("planner", lambda state: {"subtasks": decompose_task(state["task"])})
    system.register("researcher", lambda state: {"documents": retrieve(state["sources"], state["task"])})
    system.register("analyst", lambda state: {
        "findings": [
            {"subtask": subtask, "summary": summarize_with_llm(
                " ".join(document.text for document in state["documents"]), subtask, 2
            ) or "No evidence matched this subtask."}
            for subtask in state["subtasks"]
        ]
    })
    system.register("synthesizer", lambda state: {
        "report": build_report(state["task"], state["documents"], subtasks=state["subtasks"]),
        "llm_summary": summarize_with_llm(
            " ".join(document.text for document in state["documents"]), state["task"]
        ),
    })
    result = system.run({"task": task, "sources": sources})
    result["sources"] = [document.source for document in result.pop("documents")]
    result["history"] = [{"agent": item.name, "error": item.error} for item in system.history]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Research and summarize local files or web pages.")
    parser.add_argument("task", help="Question or research task")
    parser.add_argument("sources", nargs="+", help="Text files or HTTP(S) URLs to inspect")
    parser.add_argument("--output", type=Path, help="Write the report to this file")
    parser.add_argument("--dashboard", type=Path, help="Write machine-readable dashboard metrics")
    parser.add_argument("--email-to", help="Send the report through configured SMTP")
    parser.add_argument("--json", action="store_true", help="Print structured JSON instead of the report")
    args = parser.parse_args()
    result = run_research(args.task, args.sources)
    rendered = json.dumps(result, indent=2) if args.json else result["report"]
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.dashboard:
        write_dashboard(result, args.dashboard)
    if args.email_to:
        send_email(result["report"], args.email_to)


if __name__ == "__main__":
    main()
