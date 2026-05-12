import sqlite3

from src.legal.db import init_legal_db


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
    ).fetchall()
    return {row[0] for row in rows}


def test_init_legal_db_creates_required_tables(tmp_path):
    db_path = tmp_path / "legal.db"

    init_legal_db(db_path)

    conn = sqlite3.connect(db_path)
    tables = _tables(conn)

    assert "documents" in tables
    assert "pages" in tables
    assert "chunks" in tables
    assert "chunks_fts" in tables


def test_insert_document_page_chunk(tmp_path):
    db_path = tmp_path / "legal.db"
    init_legal_db(db_path)

    conn = sqlite3.connect(db_path)

    conn.execute(
        "INSERT INTO documents (doc_id, file_name, file_path, file_hash) VALUES (?, ?, ?, ?)",
        ("doc1", "case.pdf", "D:/case.pdf", "hash1"),
    )
    conn.execute(
        "INSERT INTO pages (page_id, doc_id, page_no, page_text) VALUES (?, ?, ?, ?)",
        ("page1", "doc1", 1, "张三向李四借款50000元。"),
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, doc_id, page_no, chunk_text, start_char, end_char) VALUES (?, ?, ?, ?, ?, ?)",
        ("chunk1", "doc1", 1, "张三向李四借款50000元。", 0, 13),
    )
    conn.commit()

    row = conn.execute("SELECT chunk_text FROM chunks WHERE chunk_id = ?", ("chunk1",)).fetchone()

    assert row[0] == "张三向李四借款50000元。"
