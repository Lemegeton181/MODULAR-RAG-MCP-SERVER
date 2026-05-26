"""Phase J — self-check script tests.

Drives :mod:`scripts.legal_self_check` as a module so we can assert the
PASS / WARN / FAIL contract on a clean tmp DB without depending on the
real `data/db/legal.db`.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_self_check():
    spec = importlib.util.spec_from_file_location(
        "legal_self_check", _REPO_ROOT / "scripts" / "legal_self_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_min_db(db_path: Path) -> None:
    """Create a minimal DB with chunks_fts + legal_fields so checks PASS."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(x)")
        conn.execute("CREATE TABLE legal_fields (field_id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def test_self_check_passes_on_valid_db(tmp_path, capsys, monkeypatch):
    db = tmp_path / "legal.db"
    _make_min_db(db)
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    sc = _load_self_check()
    monkeypatch.setattr(sys, "argv", ["legal_self_check.py", "--db", str(db)])
    rc = sc.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[PASS] sqlite_fts5" in out
    assert "[PASS] table:chunks_fts" in out
    assert "[PASS] table:legal_fields" in out
    # Without LLM model: WARN, not FAIL.
    assert "[WARN] local_llm_model" in out
    assert "FAIL=0" in out


def test_self_check_warns_on_missing_db(tmp_path, capsys, monkeypatch):
    db = tmp_path / "missing.db"
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    sc = _load_self_check()
    monkeypatch.setattr(sys, "argv", ["legal_self_check.py", "--db", str(db)])
    rc = sc.main()
    out = capsys.readouterr().out
    # Missing DB ⇒ WARN, not FAIL.
    assert rc == 0
    assert "[WARN] legal_db_present" in out


def test_self_check_picks_up_model_path_arg(tmp_path, capsys, monkeypatch):
    db = tmp_path / "legal.db"
    _make_min_db(db)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\x00")
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    sc = _load_self_check()
    monkeypatch.setattr(
        sys, "argv",
        ["legal_self_check.py", "--db", str(db), "--model-path", str(gguf)],
    )
    rc = sc.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "[PASS] local_llm_model" in out


def test_readme_legal_exists_and_has_run_commands():
    p = _REPO_ROOT / "README_LEGAL.md"
    assert p.exists(), "README_LEGAL.md must be present for delivery"
    text = p.read_text(encoding="utf-8")
    assert "legal_answer.py" in text
    assert "legal_self_check.py" in text
    assert "models\\llm" in text or "models/llm" in text
