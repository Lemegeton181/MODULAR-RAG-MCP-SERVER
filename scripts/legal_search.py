#!/usr/bin/env python
"""CLI for keyword / semantic / hybrid search over the legal store.

Usage:
    # Default: hybrid (FTS5 + semantic embeddings, RRF fused)
    python scripts/legal_search.py --query "loan" --db data/db/legal.db

    # Keyword only (no embedding model required)
    python scripts/legal_search.py --query "借款" --mode fts --db data/db/legal.db

    # Semantic only (requires sentence-transformers)
    python scripts/legal_search.py --query "贷款合同" --mode semantic --db data/db/legal.db

Per-hit output:
    file_name=...
    page_no=...
    page_type=text|scanned|mixed|table_like
    snippet=...
    page_image_path=... (or - if the page was not rendered)
    score=...
    source=fts|semantic|hybrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.embeddings import (  # noqa: E402
    DEFAULT_ST_MODEL,
    ST_INSTALL_HINT,
    SentenceTransformerEmbeddingProvider,
)
from src.legal.hybrid_search import hybrid_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search the legal SQLite store (FTS / semantic / hybrid)."
    )
    parser.add_argument("--query", required=True, help="Search string.")
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    parser.add_argument(
        "--mode",
        choices=["fts", "semantic", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10).",
    )
    parser.add_argument(
        "--st-model",
        default=DEFAULT_ST_MODEL,
        help=(
            "sentence-transformers model name for --mode semantic / hybrid "
            f"(default: {DEFAULT_ST_MODEL})."
        ),
    )
    parser.add_argument(
        "--no-query-understanding",
        dest="use_query_understanding",
        action="store_false",
        default=True,
        help=(
            "Disable rule-based query understanding + field-first overlay. "
            "Falls back to plain Phase G hybrid behavior."
        ),
    )
    args = parser.parse_args()

    embedder = None
    if args.mode in ("semantic", "hybrid"):
        try:
            embedder = SentenceTransformerEmbeddingProvider(model=args.st_model)
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            print(f"hint: {ST_INSTALL_HINT}", file=sys.stderr)
            return 2

    results = hybrid_search(
        args.db,
        args.query,
        embedder=embedder,
        mode=args.mode,
        limit=args.limit,
        use_query_understanding=args.use_query_understanding,
    )
    if not results:
        print(f"no results for query={args.query!r} mode={args.mode}")
        return 0
    for i, hit in enumerate(results, start=1):
        print(f"[{i}]")
        print(f"file_name={hit['file_name']}")
        print(f"page_no={hit['page_no']}")
        print(f"page_type={hit['page_type']}")
        print(f"snippet={hit['snippet']}")
        print(f"page_image_path={hit.get('page_image_path') or '-'}")
        print(f"score={hit['score']:.6f}")
        print(f"source={hit['source']}")
        if hit.get("source") == "field":
            print(f"matched_field_name={hit.get('matched_field_name', '-')}")
            print(f"matched_field_value={hit.get('matched_field_value', '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
