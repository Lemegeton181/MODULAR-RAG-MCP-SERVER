"""Minimal case-level multi-turn memory.

Phase I — one-row-per-case memory of the last query, last answer and last
citations. No LLM, no vector memory, no long-term storage growth. The
table is created on demand so this module never has to be wired into
:func:`src.legal.db.init_legal_db` and never touches existing rows.

Public API:

* :func:`save_case_context` — upsert the most recent (query, answer)
  pair for a ``case_id``.
* :func:`load_case_context` — read it back; returns ``None`` if absent.
* :func:`resolve_followup_query` — rule-based follow-up rewrite, e.g.
  「还有哪些理由？」 + previous query 「为什么申请人认为裁决应撤销？」
  becomes the previous query + "\\n追问：" + the follow-up.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS legal_case_context (
    case_id TEXT PRIMARY KEY,
    last_query TEXT NOT NULL,
    last_answer TEXT NOT NULL,
    last_citations_json TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
"""

_FOLLOWUP_MARKERS = ("这个", "上述", "这些", "它", "继续", "还有吗", "还有哪些")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_TABLE_SQL)


def save_case_context(
    db_path: str | Path,
    case_id: str,
    query: str,
    answer_result: Dict[str, Any],
) -> None:
    """Persist the most recent turn for ``case_id`` (upsert)."""
    if not case_id:
        raise ValueError("case_id is required")
    citations = answer_result.get("citations") or []
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO legal_case_context
                (case_id, last_query, last_answer, last_citations_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                last_query=excluded.last_query,
                last_answer=excluded.last_answer,
                last_citations_json=excluded.last_citations_json,
                updated_at=excluded.updated_at
            """,
            (
                case_id,
                query or "",
                answer_result.get("answer") or "",
                json.dumps(citations, ensure_ascii=False),
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_case_context(
    db_path: str | Path, case_id: str
) -> Optional[Dict[str, Any]]:
    """Return ``{last_query, last_answer, last_citations, updated_at}`` or None."""
    if not case_id:
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT last_query, last_answer, last_citations_json, updated_at
            FROM legal_case_context WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    last_query, last_answer, citations_json, updated_at = row
    try:
        citations = json.loads(citations_json) if citations_json else []
    except json.JSONDecodeError:
        citations = []
    return {
        "case_id": case_id,
        "last_query": last_query,
        "last_answer": last_answer,
        "last_citations": citations,
        "updated_at": updated_at,
    }


def resolve_followup_query(
    query: str, context: Optional[Dict[str, Any]]
) -> str:
    """Rule-based follow-up rewrite. No LLM, no embeddings.

    If ``query`` contains a follow-up marker and ``context`` carries a
    ``last_query``, return ``"{last_query}\\n追问：{query}"``. Otherwise
    return the original query unchanged.
    """
    q = (query or "").strip()
    if not q or not context:
        return query or ""
    last_query = (context.get("last_query") or "").strip()
    if not last_query:
        return query
    if any(marker in q for marker in _FOLLOWUP_MARKERS):
        return f"{last_query}\n追问：{q}"
    return query
