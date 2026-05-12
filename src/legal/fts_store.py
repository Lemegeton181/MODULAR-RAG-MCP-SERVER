"""FTS5 keyword indexing and search for legal chunks.

:func:`search_chunks` first tries the FTS5 MATCH path (works well for
English and for CJK queries that contain at least one trigram). If MATCH
returns no rows or raises a syntax error (e.g. CJK query shorter than a
trigram, or the table was created with a non-trigram tokenizer), the
function falls back to a LIKE scan over the ``chunks`` table joined with
``documents`` and builds the snippet locally.
"""
from __future__ import annotations

import sqlite3
from typing import Any, List, Dict


def index_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    doc_id: str,
    file_name: str,
    page_no: int,
    chunk_text: str,
) -> None:
    """Insert a chunk row into the chunks_fts virtual table."""
    conn.execute(
        "INSERT INTO chunks_fts (chunk_id, doc_id, file_name, page_no, chunk_text) "
        "VALUES (?, ?, ?, ?, ?)",
        (chunk_id, doc_id, file_name, page_no, chunk_text),
    )


def search_chunks(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search chunks_fts (and fall back to LIKE) for ``query``.

    Returns ``[{file_name, page_no, snippet}, ...]`` ordered by FTS5 rank
    when MATCH succeeds, or by chunk insertion order when the LIKE fallback
    is used. Returns an empty list if nothing matches.
    """
    query = query.strip()
    if not query:
        return []

    rows = _fts_match(conn, query, limit)
    if rows:
        return rows

    return _like_fallback(conn, query, limit)


def _fts_match(
    conn: sqlite3.Connection, query: str, limit: int
) -> List[Dict[str, Any]]:
    """Use FTS5 MATCH as the index, build the snippet locally.

    The FTS5 built-in ``snippet()`` wraps the *matched tokens*. Under the
    trigram tokenizer that means it wraps a 3-char window in the middle of
    the real word (e.g. ``<b>borr</b>owed``) which breaks downstream
    substring checks. We instead fetch the chunk text and let
    :func:`_build_snippet` highlight the literal query substring.
    """
    fts_query = '"' + query.replace('"', '""') + '"'
    try:
        cursor = conn.execute(
            """
            SELECT file_name, page_no, chunk_text
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "file_name": row[0],
            "page_no": row[1],
            "snippet": _build_snippet(row[2], query),
        }
        for row in rows
    ]


def _like_fallback(
    conn: sqlite3.Connection, query: str, limit: int
) -> List[Dict[str, Any]]:
    pattern = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    try:
        rows = conn.execute(
            """
            SELECT d.file_name, c.page_no, c.chunk_text
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_text LIKE ? ESCAPE '\\'
            ORDER BY c.rowid
            LIMIT ?
            """,
            (pattern, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "file_name": row[0],
            "page_no": row[1],
            "snippet": _build_snippet(row[2], query),
        }
        for row in rows
    ]


def _build_snippet(text: str, query: str, ctx: int = 20) -> str:
    """Build a short context snippet around the first match of ``query``."""
    if not text:
        return ""
    lower_text = text.lower()
    lower_q = query.lower()
    idx = lower_text.find(lower_q)
    if idx < 0:
        return text[:80]
    end_match = idx + len(query)
    start = max(0, idx - ctx)
    end = min(len(text), end_match + ctx)
    pre = "..." if start > 0 else ""
    post = "..." if end < len(text) else ""
    before = text[start:idx]
    matched = text[idx:end_match]
    after = text[end_match:end]
    return f"{pre}{before}<b>{matched}</b>{after}{post}"
