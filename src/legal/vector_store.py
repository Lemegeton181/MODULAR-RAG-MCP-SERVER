"""SQLite-backed vector store for legal chunks.

Vectors live in the ``legal_vectors`` table created by
:func:`src.legal.db.init_legal_db`. Each row stores the chunk's L2-normalized
float vector as a JSON string plus its model identifier and dimension.

The store deliberately keeps everything in SQLite — no extra dependency,
no separate vector DB. For the kind of corpus a single-laptop legal RAG
system handles (a few cases, tens of thousands of chunks), an in-Python
cosine scan is fast enough and removes a whole class of moving parts.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.legal.embeddings import EmbeddingProvider, cosine_similarity

try:  # numpy is optional but already in the repo's runtime
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore
    _HAS_NUMPY = False


def upsert_chunk_vector(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    doc_id: str,
    page_no: int,
    embedding: Sequence[float],
    model: str,
) -> None:
    """Insert or replace a single chunk's vector row."""
    conn.execute(
        "INSERT OR REPLACE INTO legal_vectors "
        "(chunk_id, doc_id, page_no, model, dim, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            doc_id,
            page_no,
            model,
            len(embedding),
            json.dumps(list(embedding)),
        ),
    )


def delete_doc_vectors(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM legal_vectors WHERE doc_id = ?", (doc_id,))


def count_vectors(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM legal_vectors").fetchone()[0]


def semantic_search(
    db_path: str | Path,
    query: str,
    embedder: EmbeddingProvider,
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Embed ``query``, scan ``legal_vectors`` for top-k cosine neighbors.

    Returns ``[{file_name, page_no, snippet, chunk_id, score}, ...]``
    ordered by descending similarity. ``snippet`` is the raw chunk text.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Legal DB not found: {db_path}")

    query = (query or "").strip()
    if not query:
        return []

    q_vec = embedder.embed([query])[0]

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT v.chunk_id, v.embedding, c.chunk_text, c.page_no, d.file_name
            FROM legal_vectors v
            JOIN chunks c ON c.chunk_id = v.chunk_id
            JOIN documents d ON d.doc_id = v.doc_id
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    scored: List[Dict[str, Any]] = []
    if _HAS_NUMPY:
        q_np = _np.asarray(q_vec, dtype=float)
        q_norm = float(_np.linalg.norm(q_np)) or 1.0
        for chunk_id, emb_json, chunk_text, page_no, file_name in rows:
            emb = _np.asarray(json.loads(emb_json), dtype=float)
            denom = float(_np.linalg.norm(emb)) * q_norm
            score = float(_np.dot(q_np, emb) / denom) if denom else 0.0
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "page_no": page_no,
                    "snippet": chunk_text,
                    "score": score,
                }
            )
    else:
        for chunk_id, emb_json, chunk_text, page_no, file_name in rows:
            emb = json.loads(emb_json)
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": file_name,
                    "page_no": page_no,
                    "snippet": chunk_text,
                    "score": cosine_similarity(q_vec, emb),
                }
            )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def lookup_chunk_id(
    conn: sqlite3.Connection, file_name: str, page_no: int, snippet_text: str
) -> Optional[str]:
    """Best-effort lookup of a chunk_id from FTS hit metadata.

    The FTS hit carries ``file_name`` + ``page_no`` + an HTML-decorated
    snippet, but not ``chunk_id``. To merge with semantic results we need a
    stable id. We resolve via a substring match against ``chunks``: strip
    ``<b>``/``</b>``/``...`` from the snippet and look up the first chunk
    on that page whose text contains the cleaned string.
    """
    cleaned = (
        (snippet_text or "")
        .replace("<b>", "")
        .replace("</b>", "")
        .replace("...", "")
        .strip()
    )
    if not cleaned:
        return None

    row = conn.execute(
        """
        SELECT c.chunk_id
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE d.file_name = ? AND c.page_no = ? AND instr(c.chunk_text, ?) > 0
        ORDER BY c.rowid
        LIMIT 1
        """,
        (file_name, page_no, cleaned),
    ).fetchone()
    return row[0] if row else None
