"""Phase H — evidence-based answer tests.

Covers :func:`src.legal.answer.answer_legal_question` end-to-end with:

* extractive fallback (no LLM injected),
* field-overlay-driven answers (top hit comes from ``legal_fields``),
* no-evidence handling (hard-no answer + empty citations),
* injected fake LLM that obeys the evidence-only contract.
"""
from __future__ import annotations

import pytest

from src.legal.answer import (
    _HARD_NO_ANSWER,
    answer_legal_question,
    build_evidence_prompt,
)
from src.legal.legal_ingest import ingest_pages


SAMPLE_PAGES = [
    {
        "page_no": 1,
        "text": (
            "哈尔滨市劳动人事争议仲裁委员会\n"
            "劳动仲裁裁决书\n"
            "案号：哈劳人仲字（2025）1077号\n"
            "申请人：张三\n"
            "被申请人：北京某某科技有限公司\n"
            "申请事项：请求被申请人支付二倍工资差额23453元。\n"
        ),
    },
    {
        "page_no": 2,
        "text": (
            "本仲裁委员会认为，根据查明的事实，被申请人应当依法支付"
            "二倍工资差额。裁决日期：2025年6月15日。"
        ),
    },
]


def _ingest(tmp_path):
    db_path = tmp_path / "legal.db"
    ingest_pages(db_path, "case.pdf", SAMPLE_PAGES)
    return db_path


# ---------------------------------------------------------------------------
# Extractive fallback (no llm_client)
# ---------------------------------------------------------------------------


def test_extractive_answer_for_field_query_returns_field_value(tmp_path):
    db_path = _ingest(tmp_path)
    result = answer_legal_question(db_path, "案号是什么", mode="fts", limit=3)
    assert result["no_evidence"] is False
    assert result["citations"], "field-lookup query must return citations"
    # Top citation should be the field overlay hit.
    top = result["citations"][0]
    assert top["file_name"] == "case.pdf"
    assert top["page_no"] == 1
    assert top["source"] == "field"
    assert "哈劳人仲字（2025）1077号" in result["answer"]
    assert result["confidence"] == "high"


def test_extractive_answer_for_keyword_query_uses_top_snippet(tmp_path):
    db_path = _ingest(tmp_path)
    result = answer_legal_question(
        db_path, "二倍工资差额", mode="fts", limit=5
    )
    assert result["no_evidence"] is False
    assert result["citations"]
    # Snippet text must be HTML-stripped in citations.
    for c in result["citations"]:
        assert "<b>" not in c["snippet"]
        assert "</b>" not in c["snippet"]


def test_extractive_answer_strips_highlight_tags(tmp_path):
    db_path = _ingest(tmp_path)
    result = answer_legal_question(db_path, "二倍工资差额", mode="fts", limit=3)
    assert "<b>" not in result["answer"]
    assert "</b>" not in result["answer"]


# ---------------------------------------------------------------------------
# No-evidence path
# ---------------------------------------------------------------------------


def test_no_evidence_returns_hard_no_and_empty_citations(tmp_path):
    db_path = _ingest(tmp_path)
    # Query has no chance of matching any text in SAMPLE_PAGES.
    result = answer_legal_question(
        db_path, "外星人入侵地球的证据", mode="fts", limit=5
    )
    assert result["no_evidence"] is True
    assert result["citations"] == []
    assert result["answer"] == _HARD_NO_ANSWER
    assert result["confidence"] == "low"


def test_no_evidence_query_does_not_invent_citations_via_llm(tmp_path):
    """Even with an LLM injected, if retrieval returns nothing the
    citations list must stay empty."""
    db_path = _ingest(tmp_path)

    class HallucinatingLLM:
        def generate(self, prompt: str) -> str:
            return "依据 [99] 第99页，外星人于2025年6月15日入侵。"

    result = answer_legal_question(
        db_path,
        "外星人入侵地球的证据",
        llm_client=HallucinatingLLM(),
        mode="fts",
        limit=5,
    )
    assert result["no_evidence"] is True
    assert result["citations"] == []
    assert result["answer"] == _HARD_NO_ANSWER


# ---------------------------------------------------------------------------
# Injected LLM
# ---------------------------------------------------------------------------


class _RecordingFakeLLM:
    """Captures the prompt; returns a fixed answer string."""

    def __init__(self, response: str = "依据证据[1]，案号为哈劳人仲字（2025）1077号。"):
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_injected_llm_receives_evidence_only_prompt(tmp_path):
    db_path = _ingest(tmp_path)
    llm = _RecordingFakeLLM()
    result = answer_legal_question(
        db_path, "案号是什么", llm_client=llm, mode="fts", limit=3
    )
    assert llm.last_prompt is not None
    # The prompt must include the hard-no instruction and at least one
    # evidence line; it must not include the user query alone with no context.
    assert _HARD_NO_ANSWER in llm.last_prompt
    assert "证据：" in llm.last_prompt
    assert "案号" in llm.last_prompt
    # Citations come from retrieval, NOT from the LLM text — they're real.
    assert result["citations"]
    assert all(
        c["file_name"] and c["page_no"] and c["snippet"]
        for c in result["citations"]
    )
    # The LLM's text becomes the answer.
    assert result["answer"] == llm.response


def test_empty_llm_response_falls_back_to_extractive(tmp_path):
    db_path = _ingest(tmp_path)

    class EmptyLLM:
        def generate(self, prompt: str) -> str:
            return "   \n  "

    result = answer_legal_question(
        db_path, "案号是什么", llm_client=EmptyLLM(), mode="fts", limit=3
    )
    assert result["answer"] != ""
    assert result["answer"] != _HARD_NO_ANSWER  # we DO have evidence
    assert "哈劳人仲字（2025）1077号" in result["answer"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def test_build_evidence_prompt_includes_all_hits():
    hits = [
        {
            "file_name": "a.pdf",
            "page_no": 1,
            "snippet": "snippet one",
            "source": "field",
            "matched_field_name": "案号",
            "matched_field_value": "X字（2025）1号",
        },
        {
            "file_name": "a.pdf",
            "page_no": 2,
            "snippet": "snippet two",
            "source": "fts",
        },
    ]
    prompt = build_evidence_prompt("案号是什么", hits)
    assert "a.pdf 第1页" in prompt
    assert "a.pdf 第2页" in prompt
    assert "snippet one" in prompt
    assert "snippet two" in prompt
    assert "案号是什么" in prompt


def test_build_evidence_prompt_strips_highlight_tags():
    hits = [
        {
            "file_name": "a.pdf",
            "page_no": 1,
            "snippet": "context <b>match</b> tail",
            "source": "fts",
        }
    ]
    prompt = build_evidence_prompt("q", hits)
    assert "<b>" not in prompt
    assert "match" in prompt


# ---------------------------------------------------------------------------
# Mode handling
# ---------------------------------------------------------------------------


def test_unknown_mode_raises(tmp_path):
    db_path = _ingest(tmp_path)
    with pytest.raises(ValueError):
        answer_legal_question(db_path, "案号", mode="bogus")


def test_semantic_mode_without_embedder_downgrades_to_fts(tmp_path):
    """The CLI passes mode='hybrid' by default; if the embedder is
    unavailable the answer module must still produce a result rather
    than crashing."""
    db_path = _ingest(tmp_path)
    result = answer_legal_question(db_path, "案号", mode="hybrid", limit=3)
    assert result["citations"]


# ---------------------------------------------------------------------------
# Phase I — reasoning query must not be answered with a party-role field
# ---------------------------------------------------------------------------


REASONING_PAGES = [
    {
        "page_no": 1,
        "text": (
            "案号：哈劳人仲字（2025）1077号\n"
            "申请人：中科智运（云南）供应链科技有限公司\n"
            "被申请人：李某\n"
            "申请事项：请求撤销劳动仲裁裁决。\n"
        ),
    },
    {
        "page_no": 2,
        "text": (
            "申请人认为裁决应予撤销，主要理由如下：\n"
            "一、适用法律错误。仲裁委员会同时援引互斥的法律依据，"
            "构成法律逻辑互斥。\n"
            "二、二倍工资属于惩罚性赔偿，与本案补偿性请求不能并存。\n"
            "三、事实认定与法律结论脱节，符合法定撤销情形。\n"
            "综上，申请撤销该裁决。"
        ),
    },
]


def _ingest_reasoning(tmp_path):
    db_path = tmp_path / "legal_reasoning.db"
    ingest_pages(db_path, "case_reasoning.pdf", REASONING_PAGES)
    return db_path


def test_reasoning_query_does_not_use_party_field_as_primary_answer(tmp_path):
    db_path = _ingest_reasoning(tmp_path)
    result = answer_legal_question(
        db_path, "为什么申请人认为裁决应撤销？", mode="fts", limit=5
    )
    assert result["no_evidence"] is False
    assert result["query_type"] in ("evidence_search", "reasoning_search")
    answer = result["answer"]
    # Must NOT be 「申请人：中科智运...」 as the primary answer.
    assert not answer.startswith("申请人："), (
        f"reasoning question was answered with a party-role field: {answer!r}"
    )
    # Must include at least one reasoning anchor from the spec.
    assert any(
        kw in answer
        for kw in ("适用法律错误", "法律逻辑互斥", "惩罚性赔偿", "撤销")
    ), f"answer missing reasoning anchor: {answer!r}"
    assert result["citations"]
    assert all(c.get("page_no") for c in result["citations"])


def test_field_query_still_answers_with_field_value(tmp_path):
    db_path = _ingest_reasoning(tmp_path)
    result = answer_legal_question(
        db_path, "申请人是谁", mode="fts", limit=3
    )
    assert result["query_type"] == "field_lookup"
    assert "申请人" in result["answer"]


def test_query_type_present_in_no_evidence_path(tmp_path):
    db_path = _ingest_reasoning(tmp_path)
    result = answer_legal_question(
        db_path, "外星人入侵地球的证据", mode="fts", limit=3
    )
    assert result["no_evidence"] is True
    assert "query_type" in result
