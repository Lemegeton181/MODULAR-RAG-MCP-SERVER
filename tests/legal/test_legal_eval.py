"""Tests for the Golden Set evaluation runner (scripts/legal_eval.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from src.legal.legal_ingest import ingest_pages

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_legal_eval():
    """Load scripts/legal_eval.py as a module without invoking its CLI."""
    if "legal_eval" in sys.modules:
        return sys.modules["legal_eval"]
    spec = importlib.util.spec_from_file_location(
        "legal_eval", _REPO_ROOT / "scripts" / "legal_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["legal_eval"] = module
    spec.loader.exec_module(module)
    return module


SAMPLE_TEXT = (
    "哈尔滨市劳动人事争议仲裁委员会\n"
    "劳动仲裁裁决书\n"
    "案号：哈劳人仲字（2025）1077号\n"
    "申请人：张三\n"
    "被申请人：北京某某科技有限公司\n"
    "法定代表人：李四\n"
    "申请事项：1. 请求被申请人支付二倍工资差额23453元\n"
    "二倍工资金额：23453元\n"
    "本案有证人证言、银行转账记录和微信聊天记录作为证据。\n"
    "劳动关系认定：双方存在事实劳动关系。\n"
    "用工主体责任：被申请人应承担用工主体责任。\n"
    "适用法律错误：原裁决适用法律错误。\n"
    "惩罚性赔偿：5000元\n"
)


def _ingest(tmp_path):
    db_path = tmp_path / "legal.db"
    ingest_pages(db_path, "case.pdf", [{"page_no": 1, "text": SAMPLE_TEXT}])
    return db_path


def test_golden_file_exists_and_is_jsonl():
    p = _REPO_ROOT / "evaluation" / "legal_golden_queries.jsonl"
    assert p.exists()
    legal_eval = _load_legal_eval()
    cases = legal_eval._load_golden(p)
    # At least 15 cases as the prompt requested.
    assert len(cases) >= 15
    for c in cases:
        assert "query" in c
        assert "expected_keywords" in c
        assert "category" in c


def test_evaluate_returns_all_required_metric_keys(tmp_path):
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()

    cases = [
        {"id": "q1", "query": "案号是什么", "expected_keywords": ["哈劳人仲字"], "expected_pages": [1]},
        {"id": "q2", "query": "申请人是谁", "expected_keywords": ["张三"], "expected_pages": None},
        {"id": "q3", "query": "二倍工资金额", "expected_keywords": ["23453"], "expected_pages": None},
    ]
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)

    for k in [
        "total",
        "scored",
        "hit@1",
        "hit@3",
        "page_hit_rate",
        "keyword_hit_rate",
        "failed_cases",
        "optional_cases_no_hit",
        "metric_kind",
    ]:
        assert k in report
    assert report["total"] == 3
    assert report["scored"] == 3
    assert report["metric_kind"] == "retrieval_proxy"


def test_allow_no_hit_case_that_misses_is_optional_not_failed(tmp_path):
    """A case marked allow_no_hit=true must not contaminate metrics or
    show up in failed_cases when its keywords are absent from the corpus."""
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()

    cases = [
        # Real, hittable case: anchors hit@1=1.0 against the corpus.
        {
            "id": "real",
            "query": "案号是什么",
            "expected_keywords": ["哈劳人仲字"],
            "expected_pages": [1],
        },
        # Optional case whose target evidence does not exist in the sample.
        {
            "id": "absent_evidence",
            "query": "聊天记录在哪",
            "expected_keywords": ["这串绝对不会出现在样本里的关键词"],
            "expected_pages": None,
            "allow_no_hit": True,
            "notes": "test: not in corpus",
        },
    ]
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)

    assert report["total"] == 2
    assert report["scored"] == 1  # the optional miss is excluded
    assert report["hit@1"] == 1.0  # denominator is scored, not total
    assert report["hit@3"] == 1.0
    assert report["keyword_hit_rate"] == 1.0
    assert len(report["optional_cases_no_hit"]) == 1
    assert report["optional_cases_no_hit"][0]["id"] == "absent_evidence"
    # The optional miss MUST NOT appear in failed_cases.
    failed_ids = [fc["id"] for fc in report["failed_cases"]]
    assert "absent_evidence" not in failed_ids


def test_allow_no_hit_case_that_does_hit_still_counts_normally(tmp_path):
    """When an allow_no_hit case actually hits, we credit it like any
    other case — we never throw away real signal."""
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()

    cases = [
        {
            "id": "absent_but_now_hits",
            "query": "案号是什么",
            "expected_keywords": ["哈劳人仲字"],
            "expected_pages": None,
            "allow_no_hit": True,
            "notes": "test: would be optional but corpus actually has it",
        },
    ]
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)

    assert report["total"] == 1
    assert report["scored"] == 1
    assert report["hit@1"] == 1.0
    assert report["optional_cases_no_hit"] == []


def test_field_lookup_query_scores_hit_at_1(tmp_path):
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()

    cases = [
        {
            "id": "case_no",
            "query": "案号是什么",
            "expected_keywords": ["哈劳人仲字", "号"],
            "expected_pages": [1],
        },
    ]
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)
    assert report["hit@1"] == 1.0
    assert report["hit@3"] == 1.0
    assert report["page_hit_rate"] == 1.0
    assert report["keyword_hit_rate"] == 1.0


def test_unsatisfiable_query_lands_in_failed_cases(tmp_path):
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()

    cases = [
        {
            "id": "unsat",
            "query": "案号是什么",
            "expected_keywords": ["这个字符串绝对不会出现"],
            "expected_pages": None,
        },
    ]
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)
    assert report["hit@1"] == 0.0
    assert report["hit@3"] == 0.0
    assert len(report["failed_cases"]) == 1
    assert report["failed_cases"][0]["id"] == "unsat"


def test_evaluate_runs_against_real_golden_file(tmp_path):
    db_path = _ingest(tmp_path)
    legal_eval = _load_legal_eval()
    cases = legal_eval._load_golden(_REPO_ROOT / "evaluation" / "legal_golden_queries.jsonl")
    report = legal_eval.evaluate(str(db_path), cases, mode="fts", limit=5)
    assert report["total"] == len(cases)
    # We expect at least some hits on the sample corpus; we don't pin a
    # specific number because golden / sample co-evolve. But hit@3 should
    # be strictly above zero.
    assert report["hit@3"] > 0.0
