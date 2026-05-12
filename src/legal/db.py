"""SQLite schema initialization for the legal retrieval store.

The ``chunks_fts`` virtual table prefers the FTS5 ``trigram`` tokenizer so
that CJK substring queries (e.g. ``借款`` against ``张三向李四借款五万元``)
can hit. If the running SQLite build does not ship the trigram tokenizer
(SQLite < 3.34), the table is created with the default ``unicode61``
tokenizer instead; :func:`src.legal.fts_store.search_chunks` then falls back
to a LIKE-based scan for short / CJK queries.
"""
from pathlib import Path
import sqlite3


_FTS_COLUMNS = (
    "chunk_id UNINDEXED, "
    "doc_id UNINDEXED, "
    "file_name UNINDEXED, "
    "page_no UNINDEXED, "
    "chunk_text"
)


def init_legal_db(db_path: str | Path) -> None:
    """Initialize SQLite tables for legal document retrieval."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                page_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                page_text TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )

        _create_chunks_fts(conn)

        conn.commit()
    finally:
        conn.close()


def _create_chunks_fts(conn: sqlite3.Connection) -> None:
    """Create the chunks_fts virtual table, preferring the trigram tokenizer."""
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            f"USING fts5({_FTS_COLUMNS}, tokenize='trigram')"
        )
    except sqlite3.OperationalError:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            f"USING fts5({_FTS_COLUMNS})"
        )
