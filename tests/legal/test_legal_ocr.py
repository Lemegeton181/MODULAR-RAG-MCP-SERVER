"""Tests for legal OCR providers.

Covers:

* :class:`SimpleTextOCRProvider` – directory scanning, page-no parsing.
* :class:`RapidOCRProvider` – clear ImportError with install hint when the
  ``rapidocr`` / ``rapidocr_onnxruntime`` packages are not installed.
"""
from __future__ import annotations

import sys

import pytest


def test_simple_text_ocr_provider_parses_page_no_from_stem(tmp_path):
    from src.legal.ocr import SimpleTextOCRProvider

    (tmp_path / "page_010.txt").write_text("ten", encoding="utf-8")
    (tmp_path / "page_002.txt").write_text("two", encoding="utf-8")
    (tmp_path / "page_001.txt").write_text("one", encoding="utf-8")

    pages = SimpleTextOCRProvider().extract_pages(tmp_path)

    # Files are sorted by name (page_001, page_002, page_010), and the
    # page_no is taken from the trailing digit run in the stem.
    assert [p["page_no"] for p in pages] == [1, 2, 10]
    assert [p["text"] for p in pages] == ["one", "two", "ten"]


def test_simple_text_ocr_provider_raises_for_missing_dir(tmp_path):
    from src.legal.ocr import SimpleTextOCRProvider

    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        SimpleTextOCRProvider().extract_pages(missing)


def test_rapid_ocr_provider_raises_with_install_hint_when_missing(monkeypatch):
    """When neither rapidocr distribution is importable, the provider must
    raise ImportError with the exact ``pip install rapidocr onnxruntime``
    hint so the user knows what to do."""
    # Force both candidate module names to look unimportable, regardless of
    # whether they are actually installed on the current machine.
    monkeypatch.setitem(sys.modules, "rapidocr", None)
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)

    from src.legal.ocr import RAPID_OCR_INSTALL_HINT, RapidOCRProvider

    with pytest.raises(ImportError) as excinfo:
        RapidOCRProvider()

    msg = str(excinfo.value)
    assert "RapidOCR is not installed" in msg
    assert "pip install" in msg
    assert "rapidocr" in msg
    assert "onnxruntime" in msg
    assert RAPID_OCR_INSTALL_HINT in msg


def test_rapid_ocr_install_hint_is_the_documented_command():
    """Pin the install hint so docs and error message stay in sync."""
    from src.legal.ocr import RAPID_OCR_INSTALL_HINT

    assert (
        RAPID_OCR_INSTALL_HINT
        == ".venv\\Scripts\\python.exe -m pip install rapidocr onnxruntime"
    )


def test_rapid_ocr_extract_text_normalizes_legacy_tuple_shape():
    """``rapidocr_onnxruntime`` historically returned ``(detections, elapse)``.

    Each detection is ``[box, text, score]``. The provider must join the
    ``text`` field across detections in order.
    """
    from src.legal.ocr import _extract_text_from_rapid_result

    fake_result = (
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], "张三向李四借款50000元", 0.99],
            [[[0, 20], [10, 20], [10, 30], [0, 30]], "约定2023年5月6日还款", 0.98],
        ],
        0.123,
    )

    text = _extract_text_from_rapid_result(fake_result)
    assert "张三向李四借款50000元" in text
    assert "约定2023年5月6日还款" in text


def test_rapid_ocr_extract_text_normalizes_new_object_shape():
    """The new ``rapidocr`` package returns an object with a ``txts`` list."""
    from src.legal.ocr import _extract_text_from_rapid_result

    class _FakeOut:
        txts = ["借款合同", "金额五万元"]

    text = _extract_text_from_rapid_result(_FakeOut())
    assert "借款合同" in text
    assert "金额五万元" in text


def test_rapid_ocr_extract_text_handles_empty_result():
    from src.legal.ocr import _extract_text_from_rapid_result

    assert _extract_text_from_rapid_result(None) == ""
    assert _extract_text_from_rapid_result((None, 0.0)) == ""
    assert _extract_text_from_rapid_result([]) == ""
