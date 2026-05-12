"""Keyword search facade over the legal SQLite + FTS5 store."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from src.legal.fts_store import search_chunks


def search_legal_chunks(
    db_path: str | Path,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Run an FTS5 keyword query and return ``[{file_name, page_no, snippet}, ...]``.

    Results are ordered by FTS5 rank (best match first).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Legal DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        return search_chunks(conn, query, limit=limit)
    finally:
        conn.close()
