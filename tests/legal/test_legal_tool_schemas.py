"""Phase K — tool schema descriptor tests."""
from __future__ import annotations

from src.legal.tool_schemas import LEGAL_TOOL_SCHEMAS, get_schema


def test_five_tools_declared():
    names = [s["name"] for s in LEGAL_TOOL_SCHEMAS]
    assert set(names) == {
        "legal_search_tool",
        "legal_answer_tool",
        "legal_reindex_fields_tool",
        "legal_eval_tool",
        "legal_self_check_tool",
    }
    assert len(names) == len(set(names)), "tool names must be unique"


def test_every_schema_has_required_keys():
    for s in LEGAL_TOOL_SCHEMAS:
        assert s.get("name")
        assert s.get("description")
        schema = s.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_query_tools_require_query():
    for name in ("legal_search_tool", "legal_answer_tool"):
        s = get_schema(name)
        assert s is not None
        assert "query" in s["input_schema"]["required"]


def test_get_schema_returns_none_for_unknown():
    assert get_schema("nope") is None


def test_answer_schema_advertises_local_llm_and_case_id():
    s = get_schema("legal_answer_tool")
    props = s["input_schema"]["properties"]
    assert props["llm"]["enum"] == ["none", "local"]
    assert "case_id" in props
    assert "model_path" in props
