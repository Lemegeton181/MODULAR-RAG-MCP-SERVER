"""Phase H — citation_check tests.

Pure-function tests for :func:`src.legal.answer.citation_check`. No DB,
no retrieval. The check enforces the answer/citation contract:

* ``no_evidence=True`` ⇒ citations must be empty.
* ``no_evidence=False`` ⇒ citations must be non-empty AND every
  citation must carry ``file_name`` / ``page_no`` / ``snippet``.
* Any violation downgrades ``confidence`` to ``"low"``.
"""
from __future__ import annotations

from src.legal.answer import citation_check


def _ok_citation(**overrides):
    base = {"file_name": "case.pdf", "page_no": 1, "snippet": "...", "source": "fts"}
    base.update(overrides)
    return base


def test_no_evidence_with_no_citations_is_clean():
    r = {
        "answer": "未在当前材料中找到依据。",
        "citations": [],
        "confidence": "low",
        "no_evidence": True,
    }
    citation_check(r)
    assert r["citation_issues"] == []
    assert r["confidence"] == "low"


def test_no_evidence_but_has_citations_is_flagged():
    r = {
        "answer": "x",
        "citations": [_ok_citation()],
        "confidence": "high",
        "no_evidence": True,
    }
    citation_check(r)
    assert "no_evidence_but_has_citations" in r["citation_issues"]
    assert r["confidence"] == "low"


def test_has_evidence_but_no_citations_is_flagged():
    r = {
        "answer": "依据材料...",
        "citations": [],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert "missing_citations" in r["citation_issues"]
    assert r["confidence"] == "low"


def test_citation_missing_file_name_is_flagged():
    r = {
        "answer": "x",
        "citations": [_ok_citation(file_name="")],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert any("file_name_missing" in issue for issue in r["citation_issues"])
    assert r["confidence"] == "low"


def test_citation_missing_page_no_is_flagged():
    r = {
        "answer": "x",
        "citations": [_ok_citation(page_no=None)],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert any("page_no_missing" in issue for issue in r["citation_issues"])
    assert r["confidence"] == "low"


def test_citation_missing_snippet_is_flagged():
    r = {
        "answer": "x",
        "citations": [_ok_citation(snippet="")],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert any("snippet_missing" in issue for issue in r["citation_issues"])
    assert r["confidence"] == "low"


def test_clean_result_is_unchanged_in_confidence():
    r = {
        "answer": "案号：X字（2025）1号",
        "citations": [_ok_citation()],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert r["citation_issues"] == []
    assert r["confidence"] == "high"


def test_multiple_issues_collected_in_one_pass():
    r = {
        "answer": "x",
        "citations": [
            _ok_citation(file_name=""),
            _ok_citation(snippet=""),
        ],
        "confidence": "high",
        "no_evidence": False,
    }
    citation_check(r)
    assert len(r["citation_issues"]) >= 2
    assert r["confidence"] == "low"


def test_citation_check_returns_result_for_chaining():
    r = {
        "answer": "x",
        "citations": [_ok_citation()],
        "confidence": "high",
        "no_evidence": False,
    }
    out = citation_check(r)
    assert out is r
