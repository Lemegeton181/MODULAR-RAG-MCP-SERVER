"""End-to-end legal ingestion into SQLite + FTS5.

Two entry points:

* :func:`ingest_pages` — primary MVP path. Accepts an in-memory
  ``[{"page_no", "text"}, ...]`` list (typically produced by an
  :class:`~src.legal.ocr.OCRProvider`).
* :func:`ingest_pdf` — convenience wrapper that extracts pages from a PDF
  via PyMuPDF and then delegates to :func:`ingest_pages`.

Both writes are idempotent per ``doc_id`` (pages / chunks / chunks_fts rows
for that doc are cleared before the new rows land).
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.legal.chunker import chunk_page_text
from src.legal.db import init_legal_db
from src.legal.fts_store import index_chunk
from src.legal.pdf_pages import extract_pdf_pages


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _hash_pages(file_name: str, pages: Iterable[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(file_name.encode("utf-8"))
    for p in pages:
        h.update(b"\x00")
        h.update(str(p.get("page_no", "")).encode("utf-8"))
        h.update(b"\x00")
        h.update((p.get("text") or "").encode("utf-8"))
    return h.hexdigest()


def ingest_pages(
    db_path: str | Path,
    file_name: str,
    pages: List[Dict[str, Any]],
    *,
    file_path: Optional[str] = None,
    file_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest pre-extracted page text into the legal SQLite database.

    ``pages`` must be ``[{"page_no": int, "text": str}, ...]``.

    Performs:
        1. ``init_legal_db(db_path)``
        2. Insert / replace one ``documents`` row
        3. For each page: insert into ``pages``, then chunk and mirror into
           ``chunks`` + ``chunks_fts``

    Returns a summary: ``{"doc_id", "file_name", "n_pages", "n_chunks"}``.
    """
    db_path = Path(db_path)
    init_legal_db(db_path)

    pages = list(pages)
    if file_hash is None:
        file_hash = _hash_pages(file_name, pages)
    doc_id = f"doc_{file_hash[:16]}"

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(doc_id, file_name, file_path, file_hash) VALUES (?, ?, ?, ?)",
            (doc_id, file_name, file_path or "", file_hash),
        )

        conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))

        n_chunks = 0
        for page in pages:
            page_no = page["page_no"]
            page_text = page.get("text", "") or ""
            page_id = f"{doc_id}_p{page_no}"

            conn.execute(
                "INSERT INTO pages "
                "(page_id, doc_id, page_no, page_text) VALUES (?, ?, ?, ?)",
                (page_id, doc_id, page_no, page_text),
            )

            for chunk in chunk_page_text(doc_id, page_no, page_text):
                conn.execute(
                    "INSERT INTO chunks "
                    "(chunk_id, doc_id, page_no, chunk_text, start_char, end_char) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk["chunk_id"],
                        chunk["doc_id"],
                        chunk["page_no"],
                        chunk["chunk_text"],
                        chunk["start_char"],
                        chunk["end_char"],
                    ),
                )
                index_chunk(
                    conn,
                    chunk_id=chunk["chunk_id"],
                    doc_id=doc_id,
                    file_name=file_name,
                    page_no=page_no,
                    chunk_text=chunk["chunk_text"],
                )
                n_chunks += 1

        conn.commit()
    finally:
        conn.close()

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "n_pages": len(pages),
        "n_chunks": n_chunks,
    }


def ingest_pdf(db_path: str | Path, pdf_path: str | Path) -> Dict[str, Any]:
    """Ingest a single PDF: extract per-page text via PyMuPDF, then ingest."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    file_hash = _file_sha256(pdf_path)
    pages = extract_pdf_pages(pdf_path)

    return ingest_pages(
        db_path,
        pdf_path.name,
        pages,
        file_path=str(pdf_path),
        file_hash=file_hash,
    )
