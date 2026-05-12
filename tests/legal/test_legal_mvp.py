"""Phase C/D + E MVP tests for the legal retrieval pipeline.

Covers:

    test_docs/legal_pages/*.txt
        -> SimpleTextOCRProvider.extract_pages()
        -> ingest_pages()
        -> documents / pages / chunks / chunks_fts
        -> search_legal_chunks()
        -> [{file_name, page_no, snippet}, ...]

Plus a Chinese-substring search that exercises the FTS5 trigram path (or
its LIKE fallback) introduced in Phase E.
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


def _write_chinese_pages(pages_dir):
    pages_dir.mkdir()
    (pages_dir / "page_001.txt").write_text(
        "张三向李四借款50000元，双方约定2023年5月6日还款。",
        encoding="utf-8",
    )
    (pages_dir / "page_002.txt").write_text(
        "本页为银行转账记录，显示李四向张三转账50000元。",
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


def test_chinese_substring_query_hits_via_fts5_or_like_fallback(tmp_path):
    """``--query "借款"`` must hit page 1 of the Chinese sample."""
    pages_dir = tmp_path / "legal_pages_cn"
    _write_chinese_pages(pages_dir)
    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    db_path = tmp_path / "legal_cn.db"
    ingest_pages(db_path, "case.pdf", pages)

    hits = search_legal_chunks(db_path, "借款", limit=10)
    assert len(hits) >= 1
    top = hits[0]
    assert top["file_name"] == "case.pdf"
    assert top["page_no"] == 1
    assert "借款" in top["snippet"]


def test_chinese_longer_query_also_hits(tmp_path):
    """A >=3 char CJK query should hit via the FTS5 trigram path."""
    pages_dir = tmp_path / "legal_pages_cn2"
    _write_chinese_pages(pages_dir)
    pages = SimpleTextOCRProvider().extract_pages(pages_dir)

    db_path = tmp_path / "legal_cn2.db"
    ingest_pages(db_path, "case.pdf", pages)

    hits = search_legal_chunks(db_path, "银行转账", limit=10)
    assert len(hits) >= 1
    assert hits[0]["file_name"] == "case.pdf"
    assert hits[0]["page_no"] == 2
    assert "银行转账" in hits[0]["snippet"]
