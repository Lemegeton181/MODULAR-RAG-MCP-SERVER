#!/usr/bin/env python
"""CLI demo for the Phase K legal tool layer.

Drives :func:`src.legal.tool_runner.run_legal_tool` and prints a single
JSON object so Agent / MCP integrators can copy-paste the contract.

Examples::

    .venv\\Scripts\\python.exe scripts\\legal_tool_demo.py \\
        --tool answer --query "为什么申请人认为裁决应撤销？" --db data\\db\\legal.db

    .venv\\Scripts\\python.exe scripts\\legal_tool_demo.py \\
        --tool search --query "惩罚性赔偿" --db data\\db\\legal.db

    .venv\\Scripts\\python.exe scripts\\legal_tool_demo.py \\
        --tool self_check --db data\\db\\legal.db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.tool_runner import list_tools, run_legal_tool  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a legal agent tool and print its JSON result."
    )
    parser.add_argument(
        "--tool",
        required=True,
        help=(
            "Tool name. Full: legal_search_tool / legal_answer_tool / "
            "legal_reindex_fields_tool / legal_eval_tool / "
            "legal_self_check_tool. Short aliases: search / answer / "
            "reindex_fields / eval / self_check."
        ),
    )
    parser.add_argument("--query", default=None)
    parser.add_argument("--db", dest="db_path", default="data/db/legal.db")
    parser.add_argument("--mode", default="hybrid", choices=["fts", "semantic", "hybrid"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--llm", default="none", choices=["none", "local"])
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--case-id", dest="case_id", default=None)
    parser.add_argument("--use-context", dest="use_context", action="store_true")
    parser.add_argument("--save-context", dest="save_context", action="store_true")
    parser.add_argument(
        "--golden-path",
        dest="golden_path",
        default="evaluation/legal_golden_queries.jsonl",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print available tool names and exit.",
    )
    args = parser.parse_args()

    if args.list_tools:
        print(json.dumps({"tools": list_tools()}, ensure_ascii=False, indent=2))
        return 0

    tool_args = {
        "query": args.query,
        "db_path": args.db_path,
        "mode": args.mode,
        "limit": args.limit,
        "llm": args.llm,
        "model_path": args.model_path,
        "case_id": args.case_id,
        "use_context": args.use_context,
        "save_context": args.save_context,
        "golden_path": args.golden_path,
    }
    # Drop Nones so each tool sees only what's relevant.
    tool_args = {k: v for k, v in tool_args.items() if v is not None}

    result = run_legal_tool(args.tool, tool_args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
