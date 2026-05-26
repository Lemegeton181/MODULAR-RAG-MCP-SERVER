"""Phase I — minimal case-level context memory tests.

Pure SQLite + pure rules. No retrieval, no LLM, no embeddings.
"""
from __future__ import annotations

from src.legal.case_context import (
    load_case_context,
    resolve_followup_query,
    save_case_context,
)


def _result(answer="答案", citations=None):
    return {
        "answer": answer,
        "citations": citations
        or [
            {
                "file_name": "case.pdf",
                "page_no": 2,
                "snippet": "适用法律错误...",
                "source": "fts",
            }
        ],
        "confidence": "medium",
        "no_evidence": False,
        "query_type": "evidence_search",
    }


def test_save_and_load_roundtrip(tmp_path):
    db = tmp_path / "ctx.db"
    save_case_context(db, "demo", "为什么申请人认为裁决应撤销？", _result())
    ctx = load_case_context(db, "demo")
    assert ctx is not None
    assert ctx["last_query"] == "为什么申请人认为裁决应撤销？"
    assert ctx["last_answer"] == "答案"
    assert ctx["last_citations"]
    assert ctx["last_citations"][0]["page_no"] == 2


def test_load_missing_case_returns_none(tmp_path):
    db = tmp_path / "ctx.db"
    # Touch the table by saving a different case first.
    save_case_context(db, "other", "q", _result())
    assert load_case_context(db, "demo") is None


def test_save_is_upsert_keeps_only_latest_turn(tmp_path):
    db = tmp_path / "ctx.db"
    save_case_context(db, "demo", "q1", _result(answer="a1"))
    save_case_context(db, "demo", "q2", _result(answer="a2"))
    ctx = load_case_context(db, "demo")
    assert ctx["last_query"] == "q2"
    assert ctx["last_answer"] == "a2"


def test_resolve_followup_rewrites_when_marker_present():
    ctx = {"last_query": "为什么申请人认为裁决应撤销？", "last_answer": "...", "last_citations": []}
    out = resolve_followup_query("还有哪些理由？", ctx)
    assert "为什么申请人认为裁决应撤销？" in out
    assert "追问：还有哪些理由？" in out


def test_resolve_followup_passes_through_when_no_marker():
    ctx = {"last_query": "为什么申请人认为裁决应撤销？", "last_answer": "...", "last_citations": []}
    out = resolve_followup_query("案号是什么", ctx)
    assert out == "案号是什么"


def test_resolve_followup_passes_through_when_no_context():
    out = resolve_followup_query("还有哪些理由？", None)
    assert out == "还有哪些理由？"


def test_resolve_followup_passes_through_when_context_has_no_last_query():
    out = resolve_followup_query("还有哪些理由？", {"last_query": ""})
    assert out == "还有哪些理由？"


def test_save_requires_case_id(tmp_path):
    db = tmp_path / "ctx.db"
    import pytest

    with pytest.raises(ValueError):
        save_case_context(db, "", "q", _result())
