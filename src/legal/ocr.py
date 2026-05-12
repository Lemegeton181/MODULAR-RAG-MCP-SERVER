"""OCR provider abstraction for legal page text extraction.

Providers:

* :class:`SimpleTextOCRProvider` — reads pre-OCR'd page text files from a
  directory. Used by the pages-dir CLI path for the Phase C/D loop.
* :class:`RapidOCRProvider` — invokes the local RapidOCR engine on a single
  image and returns its real recognition output. Requires the ``rapidocr``
  package (and an ONNX runtime). Raises a clear, actionable ImportError if
  the package is not installed.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

# Shown in the error message when RapidOCR is not installed. Kept as a
# module-level constant so tests can assert against the exact hint.
RAPID_OCR_INSTALL_HINT = (
    ".venv\\Scripts\\python.exe -m pip install rapidocr onnxruntime"
)


class OCRProvider(ABC):
    """Abstract provider that turns a source path into per-page text rows."""

    @abstractmethod
    def extract_pages(self, source: str | Path) -> List[Dict[str, Any]]:
        """Return ``[{"page_no": int, "text": str}, ...]`` for ``source``."""


class SimpleTextOCRProvider(OCRProvider):
    """Read pre-OCR'd page text files from a directory.

    Each matched file is treated as one page. Files are sorted by name. The
    page number is read from the last digit run in the file stem when
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


class RapidOCRProvider(OCRProvider):
    """Run the local RapidOCR engine on a single image.

    Construction performs an immediate import probe so that a missing
    ``rapidocr`` package surfaces an actionable error before any work is
    done. The actual engine instance is built lazily on first use.
    """

    INSTALL_HINT = RAPID_OCR_INSTALL_HINT

    def __init__(self) -> None:
        try:
            self._engine_cls = _import_rapid_ocr_class()
        except ImportError as e:
            raise ImportError(
                "RapidOCR is not installed. "
                f"Install it with: {self.INSTALL_HINT}"
            ) from e
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = self._engine_cls()
        return self._engine

    def extract_pages(self, source: str | Path) -> List[Dict[str, Any]]:
        image_path = Path(source)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        engine = self._get_engine()
        result = engine(str(image_path))
        text = _extract_text_from_rapid_result(result)
        return [{"page_no": 1, "text": text}]


def _import_rapid_ocr_class():
    """Import the RapidOCR engine class from either supported distribution."""
    try:
        from rapidocr import RapidOCR  # type: ignore
        return RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        return RapidOCR


def _extract_text_from_rapid_result(result: Any) -> str:
    """Normalize the various result shapes that RapidOCR may return.

    Supports:
        * New ``rapidocr`` package — object with a ``txts`` attribute (list[str]).
        * Legacy ``rapidocr_onnxruntime`` — ``(detections, elapse)`` tuple
          where each detection is ``[box, text, score]``.
        * Plain list of ``[box, text, score]`` detections.
    """
    if result is None:
        return ""

    txts = getattr(result, "txts", None)
    if txts is not None:
        return "\n".join(t for t in txts if t)

    if isinstance(result, tuple) and len(result) >= 1:
        detections = result[0]
    else:
        detections = result

    if not detections:
        return ""

    texts: List[str] = []
    for det in detections:
        if isinstance(det, (list, tuple)) and len(det) >= 2:
            texts.append(str(det[1]))
        else:
            txt_attr = getattr(det, "txt", None) or getattr(det, "text", None)
            if txt_attr:
                texts.append(str(txt_attr))
    return "\n".join(texts)
