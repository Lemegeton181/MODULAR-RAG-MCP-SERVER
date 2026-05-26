"""Phase K — central tool dispatcher tests."""
from __future__ import annotations

from src.legal.tool_runner import list_tools, run_legal_tool


def test_unknown_tool_returns_ok_false():
    out = run_legal_tool("not_a_tool")
    assert out["ok"] is False
    assert "unknown_tool" in out["error"]
    assert "available" in out


def test_args_must_be_dict():
    out = run_legal_tool("legal_search_tool", "not a dict")  # type: ignore[arg-type]
    assert out["ok"] is False


def test_list_tools_returns_five_full_names():
    tools = list_tools()
    expected = {
        "legal_search_tool",
        "legal_answer_tool",
        "legal_reindex_fields_tool",
        "legal_eval_tool",
        "legal_self_check_tool",
    }
    assert expected == set(tools)


def test_short_alias_dispatches_to_same_tool(tmp_path):
    # Both names should route to the same wrapper. We don't need the DB to
    # exist — missing-arg error suffices to prove dispatch works.
    a = run_legal_tool("search", {})
    b = run_legal_tool("legal_search_tool", {})
    assert a["ok"] is False and b["ok"] is False
    assert a["error"] == b["error"]


def test_empty_tool_name_is_unknown():
    out = run_legal_tool("", {})
    assert out["ok"] is False
    assert "unknown_tool" in out["error"]
