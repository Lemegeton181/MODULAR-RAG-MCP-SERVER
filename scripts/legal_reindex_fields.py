#!/usr/bin/env python
"""Re-extract legal_fields for every document already in the legal DB.

Useful after upgrading the field schema (FIELD_DEFS / EVIDENCE_KEYWORDS)
without re-ingesting the underlying PDFs.

Usage:
    python scripts/legal_reindex_fields.py --db data/db/legal.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.field_extractor import (  # noqa: E402
    extract_legal_fields_for_all_docs,
    list_doc_ids,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-extract legal_fields for every doc in the DB."
    )
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"error: db not found: {args.db}", file=sys.stderr)
        return 2

    n_docs = len(list_doc_ids(args.db))
    n_fields = extract_legal_fields_for_all_docs(args.db)
    print(f"docs={n_docs}")
    print(f"extracted_fields={n_fields}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
