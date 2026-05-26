"""Central dispatch for the legal agent tools (Phase K).

A tiny indirection so callers (CLI demo, MCP wrapper, function-calling
loop) never import the tool functions directly — they go through one
name-keyed entry point. Unknown names and argument errors become
``{"ok": False, "error": ...}`` instead of raising.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from src.legal.tools import (
    legal_answer_tool,
    legal_eval_tool,
    legal_reindex_fields_tool,
    legal_search_tool,
    legal_self_check_tool,
)


_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "legal_search_tool": legal_search_tool,
    "legal_answer_tool": legal_answer_tool,
    "legal_reindex_fields_tool": legal_reindex_fields_tool,
    "legal_eval_tool": legal_eval_tool,
    "legal_self_check_tool": legal_self_check_tool,
    # Friendly short aliases mirrored from the CLI demo.
    "search": legal_search_tool,
    "answer": legal_answer_tool,
    "reindex_fields": legal_reindex_fields_tool,
    "eval": legal_eval_tool,
    "self_check": legal_self_check_tool,
}


def run_legal_tool(tool_name: str, args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not tool_name or tool_name not in _REGISTRY:
        return {
            "ok": False,
            "error": f"unknown_tool: {tool_name!r}",
            "available": sorted({k for k in _REGISTRY if "_" in k}),
        }
    if args is not None and not isinstance(args, dict):
        return {"ok": False, "error": "args must be a dict"}
    return _REGISTRY[tool_name](args or {})


def list_tools() -> list[str]:
    return sorted({k for k in _REGISTRY if k.endswith("_tool")})


__all__ = ["run_legal_tool", "list_tools"]
