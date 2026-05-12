"""Per-page text extraction for legal PDFs using PyMuPDF."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF


def extract_pdf_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """Extract plain text from each page of a PDF.

    Returns a list of dicts: [{"page_no": 1, "text": "..."}, ...].
    Page numbers are 1-indexed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text() or ""
            pages.append({"page_no": i + 1, "text": text})
    finally:
        doc.close()
    return pages
