"""Tests for the per-document legal field extractor + field-first search."""
from __future__ import annotations

import sqlite3

import pytest

from src.legal.field_extractor import (
    _normalize_case_no,
    extract_fields_from_text,
    extract_legal_fields_for_doc,
    get_doc_field_catalog,
)
from src.legal.hybrid_search import hybrid_search
from src.legal.legal_ingest import ingest_pages


SAMPLE_TEXT = (
    "哈尔滨市劳动人事争议仲裁委员会\n"
    "劳动仲裁裁决书\n"
    "案号：哈劳人仲字（2025）1077号\n"
    "申请人：张三\n"
    "被申请人：北京某某科技有限公司\n"
    "法定代表人：李四\n"
    "统一社会信用代码：91110101MA00ABCD1X\n"
    "申请事项：1. 请求被申请人支付二倍工资差额23453元；\n"
    "          2. 请求支付经济补偿金10000元。\n"
    "二倍工资金额：23453元\n"
    "本案有证人证言、银行转账记录和微信聊天记录作为证据。\n"
    "本仲裁委员会受理本案后，依法组成仲裁庭进行了审理。\n"
    "裁决日期：2025年6月15日\n"
)


def _ingest_sample(tmp_path):
    db_path = tmp_path / "legal.db"
    pages = [
        {"page_no": 1, "text": SAMPLE_TEXT},
        {
            "page_no": 2,
            "text": "本页继续：综上，本仲裁委员会依法作出如下裁决：被申请人于本裁决生效之日起十日内向申请人支付二倍工资差额23453元。",
        },
    ]
    ingest_pages(db_path, "case.pdf", pages)
    return db_path


def test_extract_fields_from_text_finds_high_confidence_fields():
    rows = extract_fields_from_text("doc1", 1, SAMPLE_TEXT)
    by_name = {r["field_name"] for r in rows}

    assert "案号" in by_name
    assert "申请人" in by_name
    assert "被申请人" in by_name
    assert "法定代表人" in by_name
    assert "统一社会信用代码" in by_name
    assert "二倍工资金额" in by_name
    assert "申请事项" in by_name

    # Source text + page_no must be set on every row.
    for r in rows:
        assert r["page_no"] == 1
        assert r["doc_id"] == "doc1"
        assert r["source_text"]


def test_extract_fields_captures_expected_values():
    rows = extract_fields_from_text("doc1", 1, SAMPLE_TEXT)
    by_name = {r["field_name"]: r for r in rows}

    assert "哈劳人仲字（2025）1077号" == by_name["案号"]["field_value"]
    assert by_name["申请人"]["field_value"].startswith("张三")
    assert by_name["被申请人"]["field_value"].startswith("北京某某科技有限公司")
    assert by_name["法定代表人"]["field_value"].startswith("李四")
    assert by_name["统一社会信用代码"]["field_value"] == "91110101MA00ABCD1X"
    assert "23453" in by_name["二倍工资金额"]["field_value"]


def test_extract_fields_keyword_rows_for_evidence_have_no_invented_value():
    rows = extract_fields_from_text("doc1", 1, SAMPLE_TEXT)
    evidence_rows = [r for r in rows if r["source_type"] == "keyword"]
    assert evidence_rows
    # Evidence rows record the keyword as field_value, not a fabricated answer.
    values = {r["field_value"] for r in evidence_rows}
    assert any(v in {"证人证言", "证言"} for v in values)
    # transfer group could match either 转账 / 银行流水 / 汇款凭证 etc.
    assert any("转账" in v or "银行流水" in v for v in values)
    # chat group
    assert any("微信" in v or "聊天记录" in v or "短信" in v for v in values)


def test_does_not_fabricate_fields_when_absent():
    rows = extract_fields_from_text("doc1", 1, "本页只是一段无关法律字段的文本，没有任何标签。")
    # No structured regex should match.
    assert all(r["source_type"] != "regex" for r in rows)


def test_ingest_pages_auto_runs_field_extraction(tmp_path):
    db_path = _ingest_sample(tmp_path)

    # Verify n_fields is exposed via ingest_pages summary.
    summary = ingest_pages(
        db_path,
        "case.pdf",
        [{"page_no": 1, "text": SAMPLE_TEXT}],
    )
    assert summary["n_fields"] >= 1

    conn = sqlite3.connect(db_path)
    try:
        n_rows = conn.execute("SELECT COUNT(*) FROM legal_fields").fetchone()[0]
    finally:
        conn.close()
    assert n_rows >= 5


def test_get_doc_field_catalog_returns_field_names_and_values(tmp_path):
    db_path = _ingest_sample(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        doc_id = conn.execute("SELECT doc_id FROM documents LIMIT 1").fetchone()[0]
    finally:
        conn.close()

    catalog = get_doc_field_catalog(db_path, doc_id)
    assert "案号" in catalog["field_names"]
    assert "申请人" in catalog["field_names"]
    assert "evidence" in catalog["categories"]
    assert catalog["values_by_field"]["案号"][0] == "哈劳人仲字（2025）1077号"


def test_reindex_replaces_old_rows(tmp_path):
    db_path = _ingest_sample(tmp_path)

    conn = sqlite3.connect(db_path)
    try:
        doc_id = conn.execute("SELECT doc_id FROM documents LIMIT 1").fetchone()[0]
        before = conn.execute(
            "SELECT COUNT(*) FROM legal_fields WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    # Re-run extractor — count should remain equal (idempotent).
    n_after = extract_legal_fields_for_doc(db_path, doc_id)
    assert n_after == before
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM legal_fields WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == before


# ---------------------------------------------------------------------------
# Field-first search
# ---------------------------------------------------------------------------


def test_field_lookup_query_prepends_field_overlay(tmp_path):
    db_path = _ingest_sample(tmp_path)
    hits = hybrid_search(db_path, "案号是什么", mode="fts", limit=5)
    assert hits, "expected at least one hit"
    assert hits[0]["source"] == "field"
    assert hits[0]["matched_field_name"] == "案号"
    assert hits[0]["matched_field_value"] == "哈劳人仲字（2025）1077号"


def test_field_lookup_does_not_drop_hybrid_candidates(tmp_path):
    """Even when the field overlay fires, plain hybrid hits should still
    appear underneath. We use a query that both regex-matches a field
    AND has many keyword matches downstream."""
    db_path = _ingest_sample(tmp_path)
    hits = hybrid_search(db_path, "申请人是谁", mode="fts", limit=10)
    sources = [h["source"] for h in hits]
    assert sources[0] == "field"
    assert any(s == "fts" for s in sources[1:])


def test_field_lookup_applicant_returns_applicant_value(tmp_path):
    db_path = _ingest_sample(tmp_path)
    hits = hybrid_search(db_path, "申请人是谁", mode="fts", limit=3)
    assert hits[0]["matched_field_name"] == "申请人"
    assert "张三" in hits[0]["matched_field_value"]


def test_double_salary_amount_returns_23453(tmp_path):
    db_path = _ingest_sample(tmp_path)
    hits = hybrid_search(db_path, "二倍工资金额", mode="fts", limit=3)
    assert hits
    assert hits[0]["source"] == "field"
    assert "23453" in hits[0]["matched_field_value"]


def test_no_query_understanding_falls_back_to_plain_hybrid(tmp_path):
    db_path = _ingest_sample(tmp_path)
    hits = hybrid_search(
        db_path, "案号是什么", mode="fts", limit=5, use_query_understanding=False
    )
    assert hits
    # No field overlay when query understanding is off.
    assert all(h["source"] != "field" for h in hits)


def test_reindex_after_ingest_keeps_field_lookup_working(tmp_path):
    db_path = _ingest_sample(tmp_path)
    conn = sqlite3.connect(db_path)
    try:
        doc_id = conn.execute("SELECT doc_id FROM documents LIMIT 1").fetchone()[0]
    finally:
        conn.close()
    n = extract_legal_fields_for_doc(db_path, doc_id)
    assert n > 0
    hits = hybrid_search(db_path, "案号是什么", mode="fts", limit=3)
    assert hits[0]["source"] == "field"


# ---------------------------------------------------------------------------
# Case-number normalization (Phase H 收口)
# ---------------------------------------------------------------------------


def test_normalize_case_no_strips_leading_sentence_context():
    """Free-text leakage like 「会做出的哈劳人仲字（2025）1077号」 must collapse
    to the canonical 「哈劳人仲字（2025）1077号」 stem."""
    assert (
        _normalize_case_no("会做出的哈劳人仲字（2025）1077号")
        == "哈劳人仲字（2025）1077号"
    )
    assert (
        _normalize_case_no("裁委员会做出的哈劳人仲字（2025）1077号")
        == "哈劳人仲字（2025）1077号"
    )


def test_normalize_case_no_keeps_already_clean_values_intact():
    assert (
        _normalize_case_no("哈劳人仲字（2025）1077号")
        == "哈劳人仲字（2025）1077号"
    )
    # Stems with embedded digits like 京01民初 must survive untouched.
    assert (
        _normalize_case_no("京01民初字（2024）888号")
        == "京01民初字（2024）888号"
    )


def test_normalize_case_no_leaves_unrecognized_text_alone():
    assert _normalize_case_no("xxx not a case no") == "xxx not a case no"
    assert _normalize_case_no("") == ""


def test_extract_case_no_normalizes_within_pipeline(tmp_path):
    """End-to-end: a sentence-shaped fragment must store the clean stem."""
    leaky = (
        "本仲裁委员会做出的哈劳人仲字（2025）1077号裁决书，依法应当履行。\n"
    )
    rows = extract_fields_from_text("doc1", 1, leaky)
    case_no_rows = [r for r in rows if r["field_name"] == "案号"]
    assert case_no_rows
    values = {r["field_value"] for r in case_no_rows}
    assert "哈劳人仲字（2025）1077号" in values
