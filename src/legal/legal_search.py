"""Keyword search facade over the legal SQLite + FTS5 store.

:func:`search_legal_chunks` runs the FTS5 / LIKE retrieval defined in
:mod:`src.legal.fts_store` and then hydrates each hit with the
Phase F per-page metadata (``page_type`` and ``page_image_path``) by
joining against the ``pages`` table.

Phase G: each hit also carries ``score`` (1/rank) and ``source="fts"``
so callers can treat the legacy keyword API and the new
:mod:`src.legal.hybrid_search` API uniformly.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from src.legal.fts_store import search_chunks


def search_legal_chunks(
    db_path: str | Path,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Run a keyword query and return enriched hits.

    Each hit dict has: ``file_name``, ``page_no``, ``page_type``,
    ``snippet``, ``page_image_path``, ``score``, ``source``.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Legal DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        hits = search_chunks(conn, query, limit=limit)
        return [
            _hydrate_page_meta(conn, hit, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]
    finally:
        conn.close()


def _hydrate_page_meta(
    conn: sqlite3.Connection, hit: Dict[str, Any], *, rank: int = 1
) -> Dict[str, Any]:
    """Attach ``page_type`` / ``page_image_path`` from the pages table."""
    row = conn.execute(
        """
        SELECT p.page_type, p.page_image_path
        FROM pages p
        JOIN documents d ON d.doc_id = p.doc_id
        WHERE d.file_name = ? AND p.page_no = ?
        LIMIT 1
        """,
        (hit["file_name"], hit["page_no"]),
    ).fetchone()
    page_type = row[0] if row else "text"
    page_image_path = row[1] if row else None
    return {
        "file_name": hit["file_name"],
        "page_no": hit["page_no"],
        "page_type": page_type or "text",
        "snippet": hit["snippet"],
        "page_image_path": page_image_path,
        "score": 1.0 / rank,
        "source": "fts",
    }
