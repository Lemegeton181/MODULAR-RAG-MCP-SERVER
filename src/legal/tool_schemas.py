"""JSON-schema-style descriptors for the legal agent tools (Phase K).

Kept framework-agnostic on purpose: the same dicts can feed an MCP
``tools/list`` response, an Anthropic / OpenAI function-calling spec, or
an internal Agent registry. Wiring belongs to whichever framework
consumes this module, not here.
"""
from __future__ import annotations

from typing import Any, Dict, List


_COMMON_DB = {
    "type": "string",
    "description": "Path to the legal SQLite DB.",
    "default": "data/db/legal.db",
}

_MODE = {
    "type": "string",
    "enum": ["fts", "semantic", "hybrid"],
    "default": "hybrid",
    "description": "Retrieval mode.",
}


LEGAL_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "legal_search_tool",
        "description": (
            "Hybrid retrieval over the legal store. Returns top-k passages "
            "with file_name / page_no / snippet / source. No LLM, no answer "
            "composition — pure retrieval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "db_path": _COMMON_DB,
                "mode": _MODE,
                "limit": {"type": "integer", "default": 5, "minimum": 1},
                "use_query_understanding": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    {
        "name": "legal_answer_tool",
        "description": (
            "Evidence-based answer with citation_check. Routes reasoning "
            "queries away from party-role fields. Optional local LLM via "
            "llama.cpp; default is extractive fallback. Supports minimal "
            "multi-turn via case_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "db_path": _COMMON_DB,
                "mode": _MODE,
                "limit": {"type": "integer", "default": 5, "minimum": 1},
                "llm": {
                    "type": "string",
                    "enum": ["none", "local"],
                    "default": "none",
                },
                "model_path": {"type": ["string", "null"], "default": None},
                "case_id": {"type": ["string", "null"], "default": None},
                "use_context": {"type": "boolean", "default": False},
                "save_context": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "legal_reindex_fields_tool",
        "description": (
            "Re-extract legal_fields for every doc in the DB without "
            "re-ingesting PDFs. Use after FIELD_DEFS upgrades."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"db_path": _COMMON_DB},
        },
    },
    {
        "name": "legal_eval_tool",
        "description": (
            "Run the Golden Set retrieval-proxy evaluation and return "
            "hit@1 / hit@3 / page_hit_rate / keyword_hit_rate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db_path": _COMMON_DB,
                "golden_path": {
                    "type": "string",
                    "default": "evaluation/legal_golden_queries.jsonl",
                },
                "mode": _MODE,
                "limit": {"type": "integer", "default": 5, "minimum": 1},
            },
        },
    },
    {
        "name": "legal_self_check_tool",
        "description": (
            "Offline environment self-check (Python / SQLite FTS5 / tables / "
            "optional deps / local LLM model). Returns PASS/WARN/FAIL summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "db_path": _COMMON_DB,
                "model_path": {"type": ["string", "null"], "default": None},
            },
        },
    },
]


def get_schema(name: str) -> Dict[str, Any] | None:
    for s in LEGAL_TOOL_SCHEMAS:
        if s["name"] == name:
            return s
    return None


__all__ = ["LEGAL_TOOL_SCHEMAS", "get_schema"]
