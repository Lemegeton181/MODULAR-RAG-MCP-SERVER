#!/usr/bin/env python
"""CLI for keyword search over the legal SQLite + FTS5 store.

Usage:
    python scripts/legal_search.py --query "loan" --db data/db/legal.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.legal_search import search_legal_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keyword search over the legal SQLite + FTS5 store."
    )
    parser.add_argument("--query", required=True, help="FTS5 query string.")
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10).",
    )
    args = parser.parse_args()

    results = search_legal_chunks(args.db, args.query, limit=args.limit)
    if not results:
        print(f"no results for query={args.query!r}")
        return 0
    for i, hit in enumerate(results, start=1):
        print(f"[{i}]")
        print(f"file_name={hit['file_name']}")
        print(f"page_no={hit['page_no']}")
        print(f"snippet={hit['snippet']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
