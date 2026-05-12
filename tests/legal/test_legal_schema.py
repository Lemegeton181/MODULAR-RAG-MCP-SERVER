"""Tests for the base legal field registry."""
from __future__ import annotations

from src.legal.legal_schema import (
    ALL_CATEGORIES,
    CATEGORY_AMOUNT,
    CATEGORY_DATE,
    CATEGORY_DOC_BASIC,
    CATEGORY_EVIDENCE,
    CATEGORY_FACT,
    CATEGORY_IDENTITY,
    CATEGORY_INSTITUTION,
    CATEGORY_LAW,
    CATEGORY_PARTY,
    CATEGORY_REQUEST,
    EVIDENCE_KEYWORDS,
    FIELD_DEFS,
    PARTY_ROLE_FIELDS,
    PARTY_UMBRELLA_TRIGGERS,
    evidence_keyword_index,
    fields_by_category,
    is_party_umbrella_query,
)


def test_field_defs_cover_every_required_category():
    cats = {definition["category"] for definition in FIELD_DEFS.values()}
    # Evidence has no FIELD_DEFS rows (it lives in EVIDENCE_KEYWORDS), so
    # we don't require it here.
    required = {
        CATEGORY_DOC_BASIC,
        CATEGORY_INSTITUTION,
        CATEGORY_PARTY,
        CATEGORY_IDENTITY,
        CATEGORY_REQUEST,
        CATEGORY_FACT,
        CATEGORY_AMOUNT,
        CATEGORY_DATE,
        CATEGORY_LAW,
    }
    assert required <= cats


def test_evidence_keywords_include_the_documented_groups():
    expected_groups = {
        "contract", "iou", "receipt", "transfer", "chat", "av",
        "attendance", "salary", "social_security", "delivery",
        "testimony", "appraisal",
    }
    assert expected_groups <= set(EVIDENCE_KEYWORDS.keys())
    # Each group has at least one keyword.
    for group, kws in EVIDENCE_KEYWORDS.items():
        assert kws, f"evidence group {group} has no keywords"


def test_party_role_fields_resolve_to_existing_field_defs():
    for fid in PARTY_ROLE_FIELDS:
        assert fid in FIELD_DEFS
        assert FIELD_DEFS[fid]["category"] == CATEGORY_PARTY


def test_evidence_keyword_index_maps_every_alias_to_its_group():
    idx = evidence_keyword_index()
    for group, kws in EVIDENCE_KEYWORDS.items():
        for kw in kws:
            assert idx[kw] == group


def test_party_umbrella_query_detector():
    assert is_party_umbrella_query("当事人")
    assert is_party_umbrella_query("案件当事人")
    assert is_party_umbrella_query("诉讼当事人是谁")
    assert not is_party_umbrella_query("申请人是谁")
    assert not is_party_umbrella_query("")
    for t in PARTY_UMBRELLA_TRIGGERS:
        assert is_party_umbrella_query(t)


def test_fields_by_category_returns_only_matching_fields():
    party_fields = fields_by_category(CATEGORY_PARTY)
    assert "applicant" in party_fields
    assert "respondent" in party_fields
    for fid in party_fields:
        assert FIELD_DEFS[fid]["category"] == CATEGORY_PARTY


def test_evidence_category_constant_is_listed():
    # Even though FIELD_DEFS doesn't carry evidence rows, the category
    # constant exists for downstream consumers (e.g. legal_fields rows
    # written by field_extractor.py).
    assert CATEGORY_EVIDENCE in ALL_CATEGORIES
