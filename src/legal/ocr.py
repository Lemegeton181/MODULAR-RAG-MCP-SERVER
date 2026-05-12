"""OCR provider abstraction for legal page text extraction.

The MVP only ships :class:`SimpleTextOCRProvider`, which reads pre-OCR'd page
text files from a directory to simulate downstream OCR output. Real OCR
backends (Tesseract / PaddleOCR) will live alongside this class in later
phases.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class OCRProvider(ABC):
    """Abstract provider that turns a source path into per-page text rows."""

    @abstractmethod
    def extract_pages(self, source: str | Path) -> List[Dict[str, Any]]:
        """Return ``[{"page_no": int, "text": str}, ...]`` for ``source``."""


class SimpleTextOCRProvider(OCRProvider):
    """Read pre-OCR'd page text files from a directory.

    Each matched file is treated as one page. Files are sorted by name. The
    page number is read from the *last* digit run in the file stem when
    present (e.g. ``page_001.txt`` -> 1), otherwise falls back to the
    1-indexed sorted position.
    """

    def __init__(self, encoding: str = "utf-8", pattern: str = "*.txt") -> None:
        self.encoding = encoding
        self.pattern = pattern

    def extract_pages(self, source: str | Path) -> List[Dict[str, Any]]:
        directory = Path(source)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Pages directory not found: {directory}")

        files = sorted(directory.glob(self.pattern))
        pages: List[Dict[str, Any]] = []
        for idx, fp in enumerate(files, start=1):
            page_no = _parse_page_no(fp.stem)
            if page_no is None:
                page_no = idx
            text = fp.read_text(encoding=self.encoding)
            pages.append({"page_no": page_no, "text": text})
        return pages


def _parse_page_no(stem: str) -> int | None:
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    return int(nums[-1])
