"""SQLite schema initialization for the legal retrieval store.

Tables:

* ``documents`` — one row per ingested file.
* ``pages`` — one row per page; carries ``page_text`` plus the Phase F
  fields ``page_type`` (text / scanned / mixed / table_like) and
  ``page_image_path`` (set when the page was rendered to an image, e.g.
  for OCR or for mixed-page evidence preview).
* ``chunks`` — fixed-size character chunks of ``page_text``.
* ``chunks_fts`` — FTS5 virtual table mirroring ``chunks`` for keyword
  search. Prefers the ``trigram`` tokenizer so that CJK substring queries
  (e.g. ``借款`` against ``...借款五万元...``) hit; falls back to the
  default ``unicode61`` tokenizer when the running SQLite build does not
  ship trigram. :func:`src.legal.fts_store.search_chunks` then has its
  own LIKE fallback for queries that even trigram can't match.

:func:`init_legal_db` is idempotent and ALTERs ``pages`` in place to add
new columns when upgrading from an older Phase C/D / E database.
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
    """Initialize / upgrade SQLite tables for legal document retrieval."""
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
                page_type TEXT NOT NULL DEFAULT 'text',
                page_image_path TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        _ensure_pages_columns(conn)

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


def _ensure_pages_columns(conn: sqlite3.Connection) -> None:
    """Add Phase F columns to a pre-existing ``pages`` table if missing."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(pages)").fetchall()}
    if "page_type" not in existing:
        conn.execute(
            "ALTER TABLE pages ADD COLUMN page_type TEXT NOT NULL DEFAULT 'text'"
        )
    if "page_image_path" not in existing:
        conn.execute("ALTER TABLE pages ADD COLUMN page_image_path TEXT")


def _create_chunks_fts(conn: sqlite3.Connection) -> None:
    """Create chunks_fts, preferring the trigram tokenizer (with fallback)."""
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
