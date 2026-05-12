"""Phase F tests: PDF page analysis, dispatch, and search hydration.

Covers:

* :func:`classify_page_type` rules — text / scanned / mixed / table_like.
* :func:`analyze_pdf` on a real text-layer PDF — no OCR call expected.
* :func:`ingest_pdf_with_analysis` end-to-end with a fake analyzer that
  returns one page of each type, so we can assert:
    - documents / pages / chunks / chunks_fts all populated
    - scanned page text comes from the fake OCR provider, not from text layer
    - search results carry ``page_type`` and ``page_image_path``
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import fitz  # PyMuPDF

from src.legal.legal_ingest import ingest_pdf_with_analysis
from src.legal.legal_search import search_legal_chunks
from src.legal.pdf_analyzer import (
    PAGE_TYPE_MIXED,
    PAGE_TYPE_SCANNED,
    PAGE_TYPE_TABLE_LIKE,
    PAGE_TYPE_TEXT,
    analyze_pdf,
    classify_page_type,
)


# ---------------------------------------------------------------------------
# classify_page_type — pure-function rules
# ---------------------------------------------------------------------------


def test_classify_text_page_when_text_layer_is_substantial_and_no_images():
    text = "本页是法院判决书的正文部分。" * 5
    assert classify_page_type(text, n_images=0) == PAGE_TYPE_TEXT


def test_classify_scanned_page_when_images_present_and_text_too_thin():
    assert classify_page_type("", n_images=1) == PAGE_TYPE_SCANNED
    assert classify_page_type("  \n  ", n_images=2) == PAGE_TYPE_SCANNED


def test_classify_mixed_page_when_both_text_and_images_present():
    text = "本页既有文本说明又附带证据图片。" * 3
    assert classify_page_type(text, n_images=1) == PAGE_TYPE_MIXED


def test_classify_table_like_page_with_amount_rows():
    table_text = (
        "项目             金额(元)        日期\n"
        "借款本金         50,000.00       2023-05-06\n"
        "还款一期         10,000.00       2023-06-06\n"
        "还款二期         10,000.00       2023-07-06\n"
        "还款三期         10,000.00       2023-08-06\n"
        "还款四期         10,000.00       2023-09-06\n"
    )
    assert classify_page_type(table_text, n_images=0) == PAGE_TYPE_TABLE_LIKE


# ---------------------------------------------------------------------------
# analyze_pdf — real PDF, text page only, must NOT call OCR
# ---------------------------------------------------------------------------


def _make_text_pdf(path: Path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class _OCRSpy:
    """OCR provider stand-in. Records calls; raising on use catches mistakes."""

    def __init__(self):
        self.calls = []

    def extract_pages(self, source):
        self.calls.append(str(source))
        return [{"page_no": 1, "text": "OCR_FALLBACK"}]


def test_analyze_pdf_text_page_does_not_call_ocr(tmp_path):
    pdf_path = tmp_path / "case.pdf"
    _make_text_pdf(
        pdf_path,
        [
            "Page one is the loan agreement signed by both parties on 2023-05-06.",
            "Page two records the court ruling and the interest schedule.",
        ],
    )
    spy = _OCRSpy()

    pages = analyze_pdf(pdf_path, image_dir=tmp_path / "img", ocr_provider=spy)

    assert len(pages) == 2
    for p in pages:
        assert p["page_type"] == PAGE_TYPE_TEXT
        assert p["page_image_path"] is None
        assert p["text"].strip()
    assert spy.calls == []


# ---------------------------------------------------------------------------
# ingest_pdf_with_analysis — fake analyzer covering all four types
# ---------------------------------------------------------------------------


def _make_blank_pdf(path: Path):
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_with_analysis_dispatches_per_page_type(tmp_path):
    """Use a fake analyzer + fake OCR to exercise the full dispatch matrix."""
    pdf_path = tmp_path / "case.pdf"
    _make_blank_pdf(pdf_path)

    spy = _OCRSpy()

    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    scanned_image = image_dir / "case_p2.png"
    scanned_image.write_bytes(b"fake-png-bytes")
    mixed_image = image_dir / "case_p3.png"
    mixed_image.write_bytes(b"fake-png-bytes")

    def fake_analyzer(pdf_path, *, image_dir, ocr_provider, render_dpi=None):
        # Page 1: pure text layer.
        # Page 2: scanned — analyzer renders to image + invokes OCR for text.
        # Page 3: mixed — text layer kept, image saved alongside.
        # Page 4: table_like — text layer kept, no image.
        ocr_text = ""
        if ocr_provider is not None:
            ocr_text = ocr_provider.extract_pages(str(scanned_image))[0]["text"]
        return [
            {
                "page_no": 1,
                "page_type": PAGE_TYPE_TEXT,
                "text": "Page one contains the loan agreement keyword.",
                "n_images": 0,
                "page_image_path": None,
            },
            {
                "page_no": 2,
                "page_type": PAGE_TYPE_SCANNED,
                "text": ocr_text,
                "n_images": 1,
                "page_image_path": str(scanned_image),
            },
            {
                "page_no": 3,
                "page_type": PAGE_TYPE_MIXED,
                "text": "Page three has both text and an attached evidence photo.",
                "n_images": 1,
                "page_image_path": str(mixed_image),
            },
            {
                "page_no": 4,
                "page_type": PAGE_TYPE_TABLE_LIKE,
                "text": (
                    "项目     金额(元)    日期\n"
                    "借款本金 50,000.00   2023-05-06\n"
                    "还款一期 10,000.00   2023-06-06\n"
                    "还款二期 10,000.00   2023-07-06\n"
                ),
                "n_images": 0,
                "page_image_path": None,
            },
        ]

    db_path = tmp_path / "legal.db"
    result = ingest_pdf_with_analysis(
        db_path,
        pdf_path,
        image_dir=image_dir,
        ocr_provider=spy,
        analyzer=fake_analyzer,
    )

    assert result["n_pages"] == 4
    assert result["n_chunks"] >= 4
    assert result["page_types"] == {
        PAGE_TYPE_TEXT: 1,
        PAGE_TYPE_SCANNED: 1,
        PAGE_TYPE_MIXED: 1,
        PAGE_TYPE_TABLE_LIKE: 1,
    }

    # Only the scanned page should have invoked OCR.
    assert spy.calls == [str(scanned_image)]

    # Tables populated, with page_type / page_image_path stored.
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT page_no, page_type, page_image_path "
            "FROM pages ORDER BY page_no"
        ).fetchall()
    finally:
        conn.close()

    by_no = {r[0]: (r[1], r[2]) for r in rows}
    assert by_no[1] == (PAGE_TYPE_TEXT, None)
    assert by_no[2] == (PAGE_TYPE_SCANNED, str(scanned_image))
    assert by_no[3] == (PAGE_TYPE_MIXED, str(mixed_image))
    assert by_no[4] == (PAGE_TYPE_TABLE_LIKE, None)

    # Search hits hydrate page_type and page_image_path.
    hits = search_legal_chunks(db_path, "loan", limit=5)
    assert hits and hits[0]["page_no"] == 1
    assert hits[0]["page_type"] == PAGE_TYPE_TEXT
    assert hits[0]["page_image_path"] is None

    ocr_hits = search_legal_chunks(db_path, "OCR_FALLBACK", limit=5)
    assert ocr_hits and ocr_hits[0]["page_no"] == 2
    assert ocr_hits[0]["page_type"] == PAGE_TYPE_SCANNED
    assert ocr_hits[0]["page_image_path"] == str(scanned_image)

    mixed_hits = search_legal_chunks(db_path, "evidence", limit=5)
    assert mixed_hits and mixed_hits[0]["page_no"] == 3
    assert mixed_hits[0]["page_type"] == PAGE_TYPE_MIXED
    assert mixed_hits[0]["page_image_path"] == str(mixed_image)

    table_hits = search_legal_chunks(db_path, "借款本金", limit=5)
    assert table_hits and table_hits[0]["page_no"] == 4
    assert table_hits[0]["page_type"] == PAGE_TYPE_TABLE_LIKE
    assert table_hits[0]["page_image_path"] is None


def test_ingest_pdf_with_analysis_smoke_on_real_text_pdf(tmp_path):
    """No fake analyzer: real PyMuPDF + real classifier on a text PDF.

    Verifies the integration path end-to-end without depending on RapidOCR.
    Chinese retrieval is covered by dedicated FTS/MVP tests; this smoke test
    uses ASCII text to avoid PDF font/extraction instability.
    """
    pdf_path = tmp_path / "case.pdf"
    _make_text_pdf(
        pdf_path,
        [
            "loan agreement page one amount 50000 yuan.",
            "bank transfer record page two amount 50000 yuan.",
        ],
    )
    db_path = tmp_path / "legal.db"
    image_dir = tmp_path / "imgs"

    result = ingest_pdf_with_analysis(
        db_path,
        pdf_path,
        image_dir=image_dir,
        ocr_provider=None,  # safe: no scanned pages expected
    )

    assert result["n_pages"] == 2
    assert result["page_types"].get(PAGE_TYPE_TEXT, 0) >= 1

    hits = search_legal_chunks(db_path, "loan", limit=5)
    assert hits
    assert hits[0]["file_name"] == "case.pdf"
    assert hits[0]["page_no"] == 1
    assert hits[0]["page_type"] in (PAGE_TYPE_TEXT, PAGE_TYPE_MIXED)
    assert "loan" in hits[0]["snippet"].lower()
