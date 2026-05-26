"""Phase K — tool wrapper tests.

We exercise each *_tool against tmp_path DBs ingested via the existing
``ingest_pages`` helper. No real LLM. No external API.
"""
from __future__ import annotations

import pytest

from src.legal.legal_ingest import ingest_pages
from src.legal.tools import (
    legal_answer_tool,
    legal_reindex_fields_tool,
    legal_search_tool,
    legal_self_check_tool,
)


PAGES = [
    {
        "page_no": 1,
        "text": (
            "案号：哈劳人仲字（2025）1077号\n"
            "申请人：张三\n"
            "被申请人：某科技公司\n"
        ),
    },
    {
        "page_no": 2,
        "text": (
            "申请人认为裁决应予撤销，理由如下：一、适用法律错误。"
            "二、二倍工资属于惩罚性赔偿。三、事实认定与法律结论脱节。"
        ),
    },
]


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / "legal.db"
    ingest_pages(p, "case.pdf", PAGES)
    return str(p)


# ---------------------------------------------------------------------------
# legal_search_tool
# ---------------------------------------------------------------------------


def test_search_tool_returns_ok_with_results(db_path):
    out = legal_search_tool({"query": "惩罚性赔偿", "db_path": db_path, "mode": "fts"})
    assert out["ok"] is True
    assert isinstance(out["results"], list)
    assert out["results"], "expected at least one hit for 惩罚性赔偿"
    for r in out["results"]:
        assert "file_name" in r and "page_no" in r and "snippet" in r


def test_search_tool_missing_query_returns_error():
    out = legal_search_tool({})
    assert out["ok"] is False
    assert "query" in out["error"]


def test_search_tool_unknown_mode_is_captured(db_path):
    out = legal_search_tool({"query": "x", "db_path": db_path, "mode": "bogus"})
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# legal_answer_tool
# ---------------------------------------------------------------------------


def test_answer_tool_returns_citations_for_real_query(db_path):
    out = legal_answer_tool(
        {"query": "案号是什么", "db_path": db_path, "mode": "fts"}
    )
    assert out["ok"] is True
    assert out["no_evidence"] is False
    assert out["citations"]
    assert out.get("query_type")


def test_answer_tool_returns_no_evidence_for_unsupported_query(db_path):
    out = legal_answer_tool(
        {"query": "外星人入侵地球的证据", "db_path": db_path, "mode": "fts"}
    )
    assert out["ok"] is True
    assert out["no_evidence"] is True
    assert out["citations"] == []


def test_answer_tool_round_trips_case_context(db_path):
    first = legal_answer_tool(
        {
            "query": "为什么申请人认为裁决应撤销？",
            "db_path": db_path, "mode": "fts",
            "case_id": "demo", "save_context": True,
        }
    )
    assert first["ok"] is True
    follow = legal_answer_tool(
        {
            "query": "还有哪些理由？",
            "db_path": db_path, "mode": "fts",
            "case_id": "demo", "use_context": True,
        }
    )
    assert follow["ok"] is True
    assert follow["context_used"] is True


def test_answer_tool_local_llm_without_model_path_errors(db_path, monkeypatch):
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    out = legal_answer_tool(
        {"query": "案号", "db_path": db_path, "mode": "fts", "llm": "local"}
    )
    assert out["ok"] is False
    assert "local_llm_unavailable" in out["error"]


# ---------------------------------------------------------------------------
# legal_reindex_fields_tool
# ---------------------------------------------------------------------------


def test_reindex_tool_reports_counts(db_path):
    out = legal_reindex_fields_tool({"db_path": db_path})
    assert out["ok"] is True
    assert out["docs"] >= 1
    assert out["extracted_fields"] >= 0


def test_reindex_tool_missing_db_is_captured(tmp_path):
    out = legal_reindex_fields_tool({"db_path": str(tmp_path / "missing.db")})
    assert out["ok"] is False
    assert "db_not_found" in out["error"]


# ---------------------------------------------------------------------------
# legal_self_check_tool
# ---------------------------------------------------------------------------


def test_self_check_tool_returns_summary(db_path, monkeypatch):
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    out = legal_self_check_tool({"db_path": db_path})
    assert "summary" in out
    assert "FAIL=0" in out["summary"]
    assert out["checks"], "expected at least one check line"
    assert any(c["level"] == "PASS" for c in out["checks"])
