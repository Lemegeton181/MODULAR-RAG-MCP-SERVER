import sqlite3

import fitz  # PyMuPDF

from src.legal.fts_store import search_chunks
from src.legal.legal_ingest import ingest_pdf


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_populates_tables_and_supports_keyword_search(tmp_path):
    pdf_path = tmp_path / "case.pdf"
    db_path = tmp_path / "legal.db"

    _make_pdf(
        pdf_path,
        [
            "Page one contains the loan agreement keyword.",
            "Page two discusses the court ruling.",
        ],
    )

    result = ingest_pdf(db_path, pdf_path)

    assert result["n_pages"] == 2
    assert result["n_chunks"] >= 2

    conn = sqlite3.connect(db_path)
    try:
        n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_pages = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

        assert n_docs == 1
        assert n_pages == 2
        assert n_chunks >= 2
        assert n_fts == n_chunks

        results = search_chunks(conn, "loan", limit=10)
        assert len(results) >= 1
        hit = results[0]
        assert hit["file_name"] == "case.pdf"
        assert hit["page_no"] == 1
        assert "loan" in hit["snippet"].lower()
    finally:
        conn.close()
