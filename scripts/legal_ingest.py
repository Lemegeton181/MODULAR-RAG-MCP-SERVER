#!/usr/bin/env python
"""CLI for ingesting legal pages into the legal SQLite + FTS5 store.

Three input modes (mutually exclusive):

    # Pre-OCR'd text folder (Phase C/D, fastest):
    python scripts/legal_ingest.py \\
        --pages-dir test_docs/legal_pages \\
        --file-name case.pdf \\
        --db data/db/legal.db

    # Single image, real local OCR via RapidOCR (Phase E):
    python scripts/legal_ingest.py \\
        --image test_docs/page_001.png \\
        --file-name case.pdf \\
        --db data/db/legal.db

    # Single PDF (text-layer extraction):
    python scripts/legal_ingest.py --path xxx.pdf --db data/db/legal.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.legal_ingest import ingest_pages, ingest_pdf  # noqa: E402
from src.legal.ocr import RapidOCRProvider, SimpleTextOCRProvider  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest legal pages, an image, or a PDF.")
    parser.add_argument(
        "--pages-dir",
        help="Directory of pre-OCR'd page text files (*.txt).",
    )
    parser.add_argument(
        "--image",
        help="Path to a single image file (OCR'd via local RapidOCR).",
    )
    parser.add_argument(
        "--file-name",
        help="Logical file name to record (required with --pages-dir or --image).",
    )
    parser.add_argument(
        "--path",
        help="Path to a single PDF file (alternative to --pages-dir / --image).",
    )
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    args = parser.parse_args()

    modes = sum(1 for x in (args.pages_dir, args.image, args.path) if x)
    if modes == 0:
        parser.error("one of --pages-dir, --image, or --path is required")
    if modes > 1:
        parser.error("--pages-dir, --image, and --path are mutually exclusive")

    if args.pages_dir:
        if not args.file_name:
            parser.error("--file-name is required when --pages-dir is used")
        provider = SimpleTextOCRProvider()
        pages = provider.extract_pages(args.pages_dir)
        result = ingest_pages(
            args.db,
            args.file_name,
            pages,
            file_path=str(Path(args.pages_dir).resolve()),
        )
    elif args.image:
        if not args.file_name:
            parser.error("--file-name is required when --image is used")
        provider = RapidOCRProvider()
        pages = provider.extract_pages(args.image)
        result = ingest_pages(
            args.db,
            args.file_name,
            pages,
            file_path=str(Path(args.image).resolve()),
        )
    else:
        result = ingest_pdf(args.db, args.path)

    print(
        f"ingested file_name={result['file_name']} "
        f"pages={result['n_pages']} chunks={result['n_chunks']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
