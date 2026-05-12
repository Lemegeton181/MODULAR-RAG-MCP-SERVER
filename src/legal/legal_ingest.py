"""End-to-end legal ingestion into SQLite + FTS5.

Three entry points:

* :func:`ingest_pages` — primary loader. Accepts an in-memory page list
  ``[{"page_no", "text", "page_type"?, "page_image_path"?}, ...]`` (typically
  produced by an :class:`~src.legal.ocr.OCRProvider` or by
  :func:`~src.legal.pdf_analyzer.analyze_pdf`).
* :func:`ingest_pdf_with_analysis` — Phase F path. Streams a PDF page-by-page,
  classifies each page (text / scanned / mixed / table_like), extracts text or
  runs OCR as needed, and writes everything via :func:`ingest_pages`.
* :func:`ingest_pdf` — legacy text-layer-only PDF path (Phase E and earlier).

All writes are idempotent per ``doc_id`` (pages / chunks / chunks_fts rows
for that doc are cleared before the new rows land).
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.legal.chunker import chunk_page_text
from src.legal.db import init_legal_db
from src.legal.field_extractor import extract_legal_fields_for_doc
from src.legal.fts_store import index_chunk
from src.legal.pdf_pages import extract_pdf_pages
from src.legal.vector_store import delete_doc_vectors, upsert_chunk_vector


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
    embedder: Any = None,
) -> Dict[str, Any]:
    """Ingest pre-extracted page rows into the legal SQLite database.

    Each ``pages`` row may carry the optional Phase F fields
    ``page_type`` (default ``"text"``) and ``page_image_path``.

    If ``embedder`` is provided (anything with ``.embed(list[str]) ->
    list[list[float]]`` and ``.model`` / ``.dim`` attributes), the chunk
    texts are also embedded and written into ``legal_vectors`` in the same
    transaction. Existing vectors for the same ``doc_id`` are wiped first
    so re-ingestion stays idempotent.

    Returns ``{"doc_id", "file_name", "n_pages", "n_chunks", "n_vectors"}``.
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
        delete_doc_vectors(conn, doc_id)
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))

        # Materialize chunks first so we can batch-embed all their texts.
        page_chunks: List[List[Dict[str, Any]]] = []
        all_chunks: List[Dict[str, Any]] = []
        for page in pages:
            page_no = page["page_no"]
            page_text = page.get("text", "") or ""
            chunks = chunk_page_text(doc_id, page_no, page_text)
            page_chunks.append(chunks)
            all_chunks.extend(chunks)

        embeddings: List[List[float]] = []
        if embedder is not None and all_chunks:
            embeddings = embedder.embed([c["chunk_text"] for c in all_chunks])
            if len(embeddings) != len(all_chunks):
                raise ValueError(
                    "embedder returned wrong number of vectors: "
                    f"{len(embeddings)} for {len(all_chunks)} chunks"
                )

        n_chunks = 0
        n_vectors = 0
        emb_iter = iter(embeddings)
        for page, chunks in zip(pages, page_chunks):
            page_no = page["page_no"]
            page_text = page.get("text", "") or ""
            page_type = page.get("page_type") or "text"
            page_image_path = page.get("page_image_path")
            page_id = f"{doc_id}_p{page_no}"

            conn.execute(
                "INSERT INTO pages "
                "(page_id, doc_id, page_no, page_text, page_type, page_image_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (page_id, doc_id, page_no, page_text, page_type, page_image_path),
            )

            for chunk in chunks:
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
                if embeddings:
                    vec = next(emb_iter)
                    upsert_chunk_vector(
                        conn,
                        chunk_id=chunk["chunk_id"],
                        doc_id=doc_id,
                        page_no=page_no,
                        embedding=vec,
                        model=getattr(embedder, "model", "unknown"),
                    )
                    n_vectors += 1
                n_chunks += 1

        conn.commit()
    finally:
        conn.close()

    # Phase H 前置：自动抽取 legal_fields（regex + evidence keywords）。
    # 失败不应阻塞 ingest 主链路 —— 抽不到字段是常态，但运行时错误要可见。
    n_fields = extract_legal_fields_for_doc(db_path, doc_id)

    return {
        "doc_id": doc_id,
        "file_name": file_name,
        "n_pages": len(pages),
        "n_chunks": n_chunks,
        "n_vectors": n_vectors,
        "n_fields": n_fields,
    }


def ingest_pdf(
    db_path: str | Path,
    pdf_path: str | Path,
    *,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Legacy: ingest a PDF using only its text layer (no OCR, no analysis)."""
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
        embedder=embedder,
    )


def ingest_pdf_with_analysis(
    db_path: str | Path,
    pdf_path: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    ocr_provider: Any = None,
    analyzer: Any = None,
    embedder: Any = None,
) -> Dict[str, Any]:
    """Phase F: stream a PDF, classify each page, and ingest.

    ``analyzer`` is a callable with the same signature as
    :func:`src.legal.pdf_analyzer.analyze_pdf`. Tests can pass a fake
    callable to avoid touching real PDFs / RapidOCR.

    ``ocr_provider`` is forwarded to the analyzer; only ``scanned`` pages
    consume it. If a ``scanned`` page is found and no provider is given,
    its text remains empty (still indexed; ``page_image_path`` is set).

    ``embedder`` is forwarded to :func:`ingest_pages`; when given, every
    chunk is also embedded and persisted in ``legal_vectors``.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if analyzer is None:
        from src.legal.pdf_analyzer import analyze_pdf
        analyzer = analyze_pdf

    if image_dir is None:
        image_dir = pdf_path.parent / "_legal_page_images" / pdf_path.stem

    file_hash = _file_sha256(pdf_path)
    pages = analyzer(
        pdf_path,
        image_dir=image_dir,
        ocr_provider=ocr_provider,
    )

    summary = ingest_pages(
        db_path,
        pdf_path.name,
        pages,
        file_path=str(pdf_path),
        file_hash=file_hash,
        embedder=embedder,
    )
    summary["page_types"] = _count_page_types(pages)
    return summary


def _count_page_types(pages: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in pages:
        t = p.get("page_type") or "text"
        counts[t] = counts.get(t, 0) + 1
    return counts
