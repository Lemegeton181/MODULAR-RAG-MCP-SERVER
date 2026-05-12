"""Per-page analysis and type classification for legal PDFs.

The Phase F ingestion path streams a PDF page-by-page (no full-document
buffering). For each page we:

1. extract its text layer via ``page.get_text()``;
2. count its embedded image objects via ``page.get_images()``;
3. classify it into one of four ``page_type`` values:

   * ``text``      — has a meaningful text layer (default).
   * ``scanned``   — has image objects and (almost) no text. Needs OCR.
   * ``mixed``     — has both a text layer AND image objects. Text is
                    used as-is; the rendered page image is also kept so
                    later phases can show the image evidence.
   * ``table_like`` — text-layer page that looks tabular (multi-column
                    digit/currency rows). First-pass we only flag it; no
                    structured-table extraction yet.

4. for ``scanned`` and ``mixed`` we render the page to a PNG so OCR /
   evidence preview have something to consume; ``text`` and ``table_like``
   pages skip rendering entirely.

PyMuPDF is required at import time. If missing, we raise ImportError with
the exact ``pip install pymupdf`` hint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PYMUPDF_INSTALL_HINT = ".venv\\Scripts\\python.exe -m pip install pymupdf"

try:
    import fitz  # PyMuPDF
except ImportError as e:  # pragma: no cover - env-dependent
    raise ImportError(
        "PyMuPDF (fitz) is not installed. "
        f"Install it with: {PYMUPDF_INSTALL_HINT}"
    ) from e


# ---------------------------------------------------------------------------
# Tunables. Kept module-level so tests can monkeypatch precisely without
# depending on default values drifting.
# ---------------------------------------------------------------------------

#: Below this many non-whitespace chars, the text layer is considered too
#: thin to be the page's primary content.
TEXT_MIN_CHARS = 40

#: Render DPI used when rasterizing scanned / mixed pages for OCR / preview.
RENDER_DPI = 200

PAGE_TYPE_TEXT = "text"
PAGE_TYPE_SCANNED = "scanned"
PAGE_TYPE_MIXED = "mixed"
PAGE_TYPE_TABLE_LIKE = "table_like"


@dataclass
class PageAnalysis:
    """Per-page analysis result fed into the ingestion pipeline."""

    page_no: int
    page_type: str
    text: str
    n_images: int
    page_image_path: Optional[str] = None


def classify_page_type(text: str, n_images: int) -> str:
    """Classify a single page given its text layer and image-object count.

    Rules (checked in order):

    * ``n_images > 0`` and text below the threshold  → ``scanned``.
    * ``n_images > 0`` and text above the threshold  → ``mixed``.
    * No images, text above threshold, looks tabular → ``table_like``.
    * Otherwise                                       → ``text``.
    """
    text_len = len((text or "").strip())
    has_text = text_len >= TEXT_MIN_CHARS

    if n_images > 0 and not has_text:
        return PAGE_TYPE_SCANNED
    if n_images > 0 and has_text:
        return PAGE_TYPE_MIXED
    if has_text and _looks_tabular(text):
        return PAGE_TYPE_TABLE_LIKE
    return PAGE_TYPE_TEXT


# Currency / amount patterns common in Chinese legal docs.
_AMOUNT_RE = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,})"
)
_CN_AMOUNT_RE = re.compile(r"[¥￥]\s*\d|\d+\s*元")


def _looks_tabular(text: str) -> bool:
    """Heuristic: a text-layer page that looks like an amount table.

    Triggered when the page has many short lines and a high density of
    numeric / currency tokens, OR many lines containing 2+ whitespace-runs
    that look like column separators.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False

    amount_lines = 0
    multi_col_lines = 0
    for ln in lines:
        if _AMOUNT_RE.search(ln) or _CN_AMOUNT_RE.search(ln):
            amount_lines += 1
        # A line with at least two runs of >=2 whitespace looks columnar.
        if len(re.findall(r"\s{2,}", ln)) >= 2:
            multi_col_lines += 1

    if amount_lines >= max(3, len(lines) // 2):
        return True
    if multi_col_lines >= max(3, len(lines) // 2):
        return True
    return False


# ---------------------------------------------------------------------------
# Page iteration
# ---------------------------------------------------------------------------


def iter_pdf_pages(
    pdf_path: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    render_dpi: int = RENDER_DPI,
    ocr_provider: Any = None,
) -> Iterator[PageAnalysis]:
    """Yield one :class:`PageAnalysis` per page of ``pdf_path``.

    Parameters
    ----------
    pdf_path
        PDF file to analyze. Pages are streamed one at a time; we do not
        buffer the whole document.
    image_dir
        Directory in which to write rendered page PNGs for ``scanned`` /
        ``mixed`` pages. Created on demand. If ``None``, no images are
        written (and ``page_image_path`` stays ``None``); ``scanned``
        pages then have empty text.
    render_dpi
        Raster DPI for ``scanned`` / ``mixed`` pages.
    ocr_provider
        Object exposing ``extract_pages(image_path)`` returning
        ``[{"page_no": int, "text": str}]``. Required to recover text
        from ``scanned`` pages. ``mixed`` pages prefer the text layer
        and skip OCR.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    image_dir_path = Path(image_dir) if image_dir else None
    if image_dir_path is not None:
        image_dir_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page_no = i + 1
            page = doc[i]
            text = page.get_text() or ""
            try:
                n_images = len(page.get_images(full=True))
            except Exception:
                n_images = 0

            page_type = classify_page_type(text, n_images)

            page_image_path: Optional[str] = None
            if page_type in (PAGE_TYPE_SCANNED, PAGE_TYPE_MIXED) and image_dir_path:
                page_image_path = _render_page(
                    page, image_dir_path, pdf_path.stem, page_no, render_dpi
                )

            if page_type == PAGE_TYPE_SCANNED and ocr_provider is not None and page_image_path:
                ocr_pages = ocr_provider.extract_pages(page_image_path)
                if ocr_pages:
                    text = ocr_pages[0].get("text", "") or ""

            yield PageAnalysis(
                page_no=page_no,
                page_type=page_type,
                text=text,
                n_images=n_images,
                page_image_path=page_image_path,
            )
    finally:
        doc.close()


def analyze_pdf(
    pdf_path: str | Path,
    *,
    image_dir: Optional[str | Path] = None,
    render_dpi: int = RENDER_DPI,
    ocr_provider: Any = None,
) -> List[Dict[str, Any]]:
    """Materialize :func:`iter_pdf_pages` into a list of dicts.

    Each dict has keys: ``page_no``, ``page_type``, ``text``, ``n_images``,
    ``page_image_path``.
    """
    pages: List[Dict[str, Any]] = []
    for analysis in iter_pdf_pages(
        pdf_path,
        image_dir=image_dir,
        render_dpi=render_dpi,
        ocr_provider=ocr_provider,
    ):
        pages.append(
            {
                "page_no": analysis.page_no,
                "page_type": analysis.page_type,
                "text": analysis.text,
                "n_images": analysis.n_images,
                "page_image_path": analysis.page_image_path,
            }
        )
    return pages


def _render_page(page, image_dir: Path, stem: str, page_no: int, dpi: int) -> str:
    """Render one PyMuPDF page to a PNG and return the absolute path."""
    out = image_dir / f"{stem}_p{page_no:04d}.png"
    pix = page.get_pixmap(dpi=dpi)
    pix.save(str(out))
    return str(out)
