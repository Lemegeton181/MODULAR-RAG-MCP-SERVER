"""Agent-facing tool wrappers for the legal RAG stack (Phase K).

Each ``*_tool`` function:

* takes a single ``args: dict`` (Agent / MCP / function-calling friendly);
* returns a JSON-serializable ``dict`` with ``ok`` and either domain
  payload or ``error`` — never prints, never raises;
* delegates to the existing retrieval / answer / eval / self-check code
  paths; this module never reimplements business logic.

Schemas live in :mod:`src.legal.tool_schemas`; central dispatch in
:mod:`src.legal.tool_runner`.
"""
from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.legal.answer import answer_legal_question, citation_check
from src.legal.case_context import (
    load_case_context,
    resolve_followup_query,
    save_case_context,
)
from src.legal.hybrid_search import hybrid_search


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _safe(fn: Callable[..., Dict[str, Any]]) -> Callable[..., Dict[str, Any]]:
    """Wrap a tool so every exception becomes ``{"ok": False, "error": ...}``."""

    def _inner(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        a = dict(args or {})
        try:
            out = fn(a)
            out.setdefault("ok", True)
            return out
        except Exception as exc:  # noqa: BLE001 — boundary
            return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    _inner.__name__ = fn.__name__
    _inner.__doc__ = fn.__doc__
    return _inner


# ---------------------------------------------------------------------------
# 1. legal_search_tool
# ---------------------------------------------------------------------------


def _build_embedder_if_needed(mode: str):
    if mode == "fts":
        return None
    try:
        from src.legal.embeddings import (  # noqa: WPS433
            DEFAULT_ST_MODEL,
            SentenceTransformerEmbeddingProvider,
        )
        return SentenceTransformerEmbeddingProvider(model=DEFAULT_ST_MODEL)
    except ImportError:
        return None


@_safe
def legal_search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not query:
        return {"ok": False, "error": "missing required arg: query"}
    db_path = args.get("db_path", "data/db/legal.db")
    mode = args.get("mode", "hybrid")
    if mode not in ("fts", "semantic", "hybrid"):
        return {"ok": False, "error": f"unknown mode: {mode!r}"}
    limit = int(args.get("limit", 5))
    use_qu = bool(args.get("use_query_understanding", True))

    embedder = _build_embedder_if_needed(mode)
    effective_mode = mode if (mode == "fts" or embedder is not None) else "fts"

    hits = hybrid_search(
        db_path,
        query,
        embedder=embedder,
        mode=effective_mode,
        limit=limit,
        use_query_understanding=use_qu,
    )
    return {
        "ok": True,
        "results": [
            {
                "file_name": h.get("file_name"),
                "page_no": h.get("page_no"),
                "page_type": h.get("page_type"),
                "snippet": h.get("snippet"),
                "score": h.get("score"),
                "source": h.get("source"),
                "matched_field_name": h.get("matched_field_name"),
                "matched_field_value": h.get("matched_field_value"),
            }
            for h in hits
        ],
        "mode": effective_mode,
        "downgraded": effective_mode != mode,
    }


# ---------------------------------------------------------------------------
# 2. legal_answer_tool
# ---------------------------------------------------------------------------


@_safe
def legal_answer_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not query:
        return {"ok": False, "error": "missing required arg: query"}
    db_path = args.get("db_path", "data/db/legal.db")
    mode = args.get("mode", "hybrid")
    limit = int(args.get("limit", 5))
    llm_kind = args.get("llm", "none")
    model_path = args.get("model_path")
    case_id = args.get("case_id")
    use_ctx = bool(args.get("use_context", False))
    save_ctx = bool(args.get("save_context", False))

    effective_query = query
    context_used = False
    if use_ctx and case_id:
        ctx = load_case_context(db_path, case_id)
        if ctx:
            rewritten = resolve_followup_query(query, ctx)
            if rewritten != query:
                effective_query = rewritten
                context_used = True

    embedder = _build_embedder_if_needed(mode)
    effective_mode = mode if (mode == "fts" or embedder is not None) else "fts"

    llm_client = None
    if llm_kind == "local":
        from src.legal.local_llm import (  # noqa: WPS433
            LocalLLMUnavailable,
            build_local_llm_client,
        )
        try:
            llm_client = build_local_llm_client(model_path)
        except LocalLLMUnavailable as e:
            return {"ok": False, "error": f"local_llm_unavailable: {e}"}

    result = answer_legal_question(
        db_path,
        effective_query,
        llm_client=llm_client,
        embedder=embedder,
        mode=effective_mode,
        limit=limit,
    )
    citation_check(result)

    if save_ctx and case_id:
        save_case_context(db_path, case_id, query, result)

    result["ok"] = True
    result["context_used"] = context_used
    result["mode"] = effective_mode
    return result


# ---------------------------------------------------------------------------
# 3. legal_reindex_fields_tool
# ---------------------------------------------------------------------------


@_safe
def legal_reindex_fields_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    from src.legal.field_extractor import (  # noqa: WPS433
        extract_legal_fields_for_all_docs,
        list_doc_ids,
    )

    db_path = args.get("db_path", "data/db/legal.db")
    if not Path(db_path).exists():
        return {"ok": False, "error": f"db_not_found: {db_path}"}
    docs = len(list_doc_ids(db_path))
    extracted = extract_legal_fields_for_all_docs(db_path)
    return {"ok": True, "docs": docs, "extracted_fields": extracted}


# ---------------------------------------------------------------------------
# 4. legal_eval_tool
# ---------------------------------------------------------------------------


@_safe
def legal_eval_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    import json

    from scripts.legal_eval import evaluate  # type: ignore  # noqa: WPS433

    db_path = args.get("db_path", "data/db/legal.db")
    golden_path = Path(
        args.get("golden_path", "evaluation/legal_golden_queries.jsonl")
    )
    mode = args.get("mode", "hybrid")
    limit = int(args.get("limit", 5))

    if not Path(db_path).exists():
        return {"ok": False, "error": f"db_not_found: {db_path}"}
    if not golden_path.exists():
        return {"ok": False, "error": f"golden_not_found: {golden_path}"}

    cases = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))

    embedder = _build_embedder_if_needed(mode)
    effective_mode = mode if (mode == "fts" or embedder is not None) else "fts"

    report = evaluate(
        db_path,
        cases,
        mode=effective_mode,
        limit=limit,
        embedder=embedder,
        use_query_understanding=True,
    )
    report["ok"] = True
    report["mode"] = effective_mode
    return report


# ---------------------------------------------------------------------------
# 5. legal_self_check_tool
# ---------------------------------------------------------------------------


def _load_self_check_module():
    spec = importlib.util.spec_from_file_location(
        "_legal_self_check_loaded",
        _REPO_ROOT / "scripts" / "legal_self_check.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@_safe
def legal_self_check_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    db_path = args.get("db_path", "data/db/legal.db")
    model_path = args.get("model_path")

    sc = _load_self_check_module()
    buf = io.StringIO()
    saved_argv = sys.argv[:]
    sys.argv = ["legal_self_check.py", "--db", db_path]
    if model_path:
        sys.argv += ["--model-path", model_path]
    try:
        with redirect_stdout(buf):
            rc = sc.main()
    finally:
        sys.argv = saved_argv

    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    summary_line = next((ln for ln in lines if ln.startswith("summary:")), "")
    checks = []
    for ln in lines:
        if ln.startswith("[") and "]" in ln:
            lvl = ln[1: ln.index("]")]
            body = ln[ln.index("]") + 1:].strip()
            checks.append({"level": lvl, "line": body})
    return {
        "ok": rc == 0,
        "exit_code": rc,
        "summary": summary_line,
        "checks": checks,
    }


__all__ = [
    "legal_search_tool",
    "legal_answer_tool",
    "legal_reindex_fields_tool",
    "legal_eval_tool",
    "legal_self_check_tool",
]
