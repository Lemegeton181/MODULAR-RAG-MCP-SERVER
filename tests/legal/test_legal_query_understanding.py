"""Tests for the rule-based legal query understanding layer."""
from __future__ import annotations

from src.legal.query_understanding import analyze_legal_query


def test_case_no_query_is_field_lookup_and_yields_case_no_candidate():
    a = analyze_legal_query("案号是什么")
    assert a["query_type"] == "field_lookup"
    assert "案号" in a["field_candidates"]


def test_doc_number_synonym_also_lands_on_case_no_field():
    a = analyze_legal_query("文书编号是多少")
    assert a["query_type"] == "field_lookup"
    assert "案号" in a["field_candidates"]


def test_dangshiren_umbrella_expands_to_all_party_roles():
    a = analyze_legal_query("当事人")
    expected = {
        "申请人", "被申请人", "原告", "被告", "第三人",
        "上诉人", "被上诉人", "法定代表人", "委托代理人",
        "辩护人", "证人",
    }
    assert expected <= set(a["field_candidates"])
    assert a["query_type"] == "field_lookup"


def test_witness_testimony_is_evidence_search_not_field_lookup():
    a = analyze_legal_query("有没有证人证言")
    assert a["query_type"] == "evidence_search"
    # Expanded terms must include the testimony synonyms.
    assert "证人证言" in a["expanded_terms"] or "证言" in a["expanded_terms"]


def test_transfer_evidence_query_maps_to_evidence_search_with_synonyms():
    a = analyze_legal_query("转账凭证在哪")
    assert a["query_type"] == "evidence_search"
    # Sibling transfer keywords should appear in expanded_terms.
    assert "银行流水" in a["expanded_terms"] or "汇款凭证" in a["expanded_terms"]


def test_chat_evidence_query_maps_to_evidence_search():
    a = analyze_legal_query("聊天记录在哪")
    assert a["query_type"] == "evidence_search"
    assert any("聊天" in t or "短信" in t or "微信" in t for t in a["expanded_terms"] + [a["original_query"]])


def test_double_salary_amount_query_resolves_to_amount_field():
    a = analyze_legal_query("二倍工资金额")
    assert a["query_type"] == "field_lookup"
    assert "二倍工资金额" in a["field_candidates"]


def test_law_application_error_resolves_to_fact_field():
    a = analyze_legal_query("适用法律是否错误")
    assert a["query_type"] == "field_lookup"
    assert "适用法律错误" in a["field_candidates"]


def test_general_query_falls_through_to_general_search():
    a = analyze_legal_query("张三李四之间的故事")
    assert a["query_type"] == "general_search"
    assert a["field_candidates"] == []


def test_empty_query_returns_general_search_with_empty_fields():
    a = analyze_legal_query("")
    assert a["query_type"] == "general_search"
    assert a["field_candidates"] == []
    assert a["expanded_terms"] == []


def test_doc_catalog_can_add_field_candidates_not_in_base_triggers():
    """If the doc actually carries '公司名称' and the user mentions it,
    catalog augmentation should still surface it even without a base
    trigger match."""
    catalog = {
        "doc_id": "d1",
        "field_names": ["公司名称"],
        "categories": {"identity": ["公司名称"]},
        "values_by_field": {"公司名称": ["北京某某科技有限公司"]},
    }
    a = analyze_legal_query("北京公司名称是什么", doc_field_catalog=catalog)
    assert "公司名称" in a["field_candidates"]


# ---------------------------------------------------------------------------
# Phase I — reasoning routing
# ---------------------------------------------------------------------------


def test_why_question_with_party_role_is_not_field_lookup():
    """「为什么申请人认为裁决应撤销？」 contains the role keyword 申请人
    but the user is asking about reasoning, not who the applicant is."""
    a = analyze_legal_query("为什么申请人认为裁决应撤销？")
    assert a["query_type"] != "field_lookup"
    assert a["query_type"] in ("evidence_search", "reasoning_search")


def test_who_is_applicant_remains_field_lookup():
    a = analyze_legal_query("申请人是谁")
    assert a["query_type"] == "field_lookup"
    assert "申请人" in a["field_candidates"]


def test_reasoning_trigger_with_fact_field_keeps_field_lookup():
    """Fact fields (e.g. 适用法律错误) are NOT downgraded by 是否/为什么 —
    the answer is still a field, just framed as a yes/no question."""
    a = analyze_legal_query("适用法律是否错误")
    assert a["query_type"] == "field_lookup"
    assert "适用法律错误" in a["field_candidates"]


def test_multiple_reasoning_triggers_with_role_field_routes_to_evidence():
    a = analyze_legal_query("被申请人有哪些理由抗辩？")
    assert a["query_type"] == "evidence_search"
