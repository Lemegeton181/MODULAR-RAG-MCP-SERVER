"""FTS5 keyword indexing and search for legal chunks."""
from __future__ import annotations

import sqlite3
from typing import Any


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
) -> list[dict[str, Any]]:
    """Run an FTS5 MATCH query against chunks_fts.

    Returns a list of dicts with keys: file_name, page_no, snippet.
    Results are ordered by FTS5 rank (best match first).
    """
    rows = conn.execute(
        """
        SELECT
            file_name,
            page_no,
            snippet(chunks_fts, 4, '<b>', '</b>', '...', 16) AS snippet
        FROM chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [
        {"file_name": row[0], "page_no": row[1], "snippet": row[2]}
        for row in rows
    ]
