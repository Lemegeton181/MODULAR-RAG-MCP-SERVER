"""Phase C/D MVP test: pages-dir -> ingest_pages -> search_legal_chunks.

Covers the full minimal loop required for Phase C/D:

    test_docs/legal_pages/*.txt
        -> SimpleTextOCRProvider.extract_pages()
        -> ingest_pages()
        -> documents / pages / chunks / chunks_fts
        -> search_legal_chunks()
        -> [{file_name, page_no, snippet}, ...]
"""
import sqlite3

from src.legal.legal_ingest import ingest_pages
from src.legal.legal_search import search_legal_chunks
from src.legal.ocr import SimpleTextOCRProvider


def _write_pages(pages_dir):
    pages_dir.mkdir()
    (pages_dir / "page_001.txt").write_text(
        "Page one contains the loan agreement keyword.",
        encoding="utf-8",
    )
    (pages_dir / "page_002.txt").write_text(
        "Page two discusses the court ruling and interest.",
        encoding="utf-8",
    )


def test_simple_text_ocr_extracts_pages_in_order(tmp_path):
    pages_dir = tmp_path / "legal_pages"
    _write_pages(pages_dir)

    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    assert [p["page_no"] for p in pages] == [1, 2]
    assert "loan" in pages[0]["text"].lower()
    assert "court" in pages[1]["text"].lower()


def test_pages_dir_ingest_and_search_minimal_loop(tmp_path):
    pages_dir = tmp_path / "legal_pages"
    _write_pages(pages_dir)

    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    db_path = tmp_path / "legal.db"
    result = ingest_pages(db_path, "case.pdf", pages)

    assert result["file_name"] == "case.pdf"
    assert result["n_pages"] == 2
    assert result["n_chunks"] >= 2

    conn = sqlite3.connect(db_path)
    try:
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_pages_db = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        n_chunks_db = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    finally:
        conn.close()

    assert n_docs == 1
    assert n_pages_db == 2
    assert n_chunks_db == result["n_chunks"]
    assert n_fts == result["n_chunks"]

    hits = search_legal_chunks(db_path, "loan", limit=10)
    assert len(hits) >= 1
    top = hits[0]
    assert top["file_name"] == "case.pdf"
    assert top["page_no"] == 1
    assert "loan" in top["snippet"].lower()


def test_ingest_pages_is_idempotent_per_doc(tmp_path):
    """Re-ingesting the same file must not duplicate chunks_fts rows."""
    pages_dir = tmp_path / "legal_pages"
    _write_pages(pages_dir)
    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    db_path = tmp_path / "legal.db"

    first = ingest_pages(db_path, "case.pdf", pages)
    second = ingest_pages(db_path, "case.pdf", pages)

    assert first["doc_id"] == second["doc_id"]
    assert first["n_chunks"] == second["n_chunks"]

    conn = sqlite3.connect(db_path)
    try:
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_pages_db = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        n_chunks_db = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    finally:
        conn.close()

    assert n_docs == 1
    assert n_pages_db == 2
    assert n_chunks_db == second["n_chunks"]
    assert n_fts == second["n_chunks"]


def test_search_legal_chunks_returns_empty_on_no_match(tmp_path):
    pages_dir = tmp_path / "legal_pages"
    _write_pages(pages_dir)
    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    db_path = tmp_path / "legal.db"
    ingest_pages(db_path, "case.pdf", pages)

    assert search_legal_chunks(db_path, "spaceship", limit=10) == []
