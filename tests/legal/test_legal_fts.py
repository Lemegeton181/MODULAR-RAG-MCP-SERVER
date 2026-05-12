import sqlite3

from src.legal.db import init_legal_db
from src.legal.fts_store import index_chunk, search_chunks


def _open_db(tmp_path):
    db_path = tmp_path / "legal.db"
    init_legal_db(db_path)
    return sqlite3.connect(db_path)


def test_index_chunk_writes_row_to_chunks_fts(tmp_path):
    conn = _open_db(tmp_path)

    index_chunk(
        conn,
        chunk_id="chunk1",
        doc_id="doc1",
        file_name="case.pdf",
        page_no=1,
        chunk_text="loan agreement defendant plaintiff",
    )
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert count == 1


def test_search_chunks_returns_file_name_page_no_snippet(tmp_path):
    conn = _open_db(tmp_path)

    index_chunk(
        conn,
        chunk_id="chunk1",
        doc_id="doc1",
        file_name="case.pdf",
        page_no=3,
        chunk_text="The defendant borrowed fifty thousand yuan from plaintiff",
    )
    index_chunk(
        conn,
        chunk_id="chunk2",
        doc_id="doc1",
        file_name="case.pdf",
        page_no=5,
        chunk_text="The court ruled that interest should be paid",
    )
    conn.commit()

    results = search_chunks(conn, "borrowed", limit=10)

    assert len(results) == 1
    hit = results[0]
    assert hit["file_name"] == "case.pdf"
    assert hit["page_no"] == 3
    assert "borrowed" in hit["snippet"].lower()


def test_search_chunks_returns_empty_when_no_match(tmp_path):
    conn = _open_db(tmp_path)

    index_chunk(
        conn,
        chunk_id="chunk1",
        doc_id="doc1",
        file_name="case.pdf",
        page_no=1,
        chunk_text="The court ruled in favor of plaintiff",
    )
    conn.commit()

    results = search_chunks(conn, "spaceship", limit=10)
    assert results == []


def test_search_chunks_chinese_token_match(tmp_path):
    conn = _open_db(tmp_path)

    index_chunk(
        conn,
        chunk_id="chunk1",
        doc_id="doc1",
        file_name="案件.pdf",
        page_no=2,
        chunk_text="被告借款五万元",
    )
    conn.commit()

    # Default unicode61 tokenizer treats the continuous CJK run as one token,
    # so we match it exactly to verify Chinese text round-trips through FTS5.
    results = search_chunks(conn, "被告借款五万元", limit=10)

    assert len(results) == 1
    assert results[0]["file_name"] == "案件.pdf"
    assert results[0]["page_no"] == 2


def test_search_chunks_respects_limit(tmp_path):
    conn = _open_db(tmp_path)

    for i in range(5):
        index_chunk(
            conn,
            chunk_id=f"chunk{i}",
            doc_id="doc1",
            file_name="case.pdf",
            page_no=i + 1,
            chunk_text=f"keyword content number {i}",
        )
    conn.commit()

    results = search_chunks(conn, "keyword", limit=2)
    assert len(results) == 2
