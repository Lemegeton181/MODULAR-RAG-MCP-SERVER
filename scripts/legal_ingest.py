#!/usr/bin/env python
"""CLI for ingesting legal pages into the legal SQLite + FTS5 store.

Primary MVP mode (pre-OCR'd text folder):

    python scripts/legal_ingest.py \\
        --pages-dir test_docs/legal_pages \\
        --file-name case.pdf \\
        --db data/db/legal.db

Optional PDF mode:

    python scripts/legal_ingest.py --path xxx.pdf --db data/db/legal.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.legal_ingest import ingest_pages, ingest_pdf  # noqa: E402
from src.legal.ocr import SimpleTextOCRProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest legal pages or a PDF.")
    parser.add_argument(
        "--pages-dir",
        help="Directory of pre-OCR'd page text files (*.txt).",
    )
    parser.add_argument(
        "--file-name",
        help="Logical file name to record (required with --pages-dir).",
    )
    parser.add_argument(
        "--path",
        help="Path to a single PDF file (alternative to --pages-dir).",
    )
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    args = parser.parse_args()

    if args.pages_dir:
        if not args.file_name:
            parser.error("--file-name is required when --pages-dir is used")
        provider = SimpleTextOCRProvider()
        pages = provider.extract_pages(args.pages_dir)
        result = ingest_pages(args.db, args.file_name, pages)
    elif args.path:
        result = ingest_pdf(args.db, args.path)
    else:
        parser.error("either --pages-dir (with --file-name) or --path is required")
        return 2  # unreachable; parser.error exits

    print(
        f"ingested file_name={result['file_name']} "
        f"pages={result['n_pages']} chunks={result['n_chunks']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
