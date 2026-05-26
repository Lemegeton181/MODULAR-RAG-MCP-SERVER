#!/usr/bin/env python
"""Offline self-check for the legal RAG pipeline (Phase J).

Prints one line per check with PASS / WARN / FAIL. The exit code is
non-zero only when a FAIL check is recorded — missing optional pieces
(LLM model, RapidOCR, sentence-transformers) are WARNs by design so
the script can run on a freshly-cloned, fully offline machine.

Usage::

    .venv\\Scripts\\python.exe scripts\\legal_self_check.py \\
        --db data\\db\\legal.db [--model-path models\\llm\\xxx.gguf]
"""
from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _emit(level: str, name: str, detail: str = "") -> None:
    line = f"[{level}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)


def _check_python() -> str:
    v = sys.version_info
    if v < (3, 9):
        _emit("FAIL", "python>=3.9", f"got {v.major}.{v.minor}")
        return "FAIL"
    _emit("PASS", "python", f"{v.major}.{v.minor}.{v.micro}")
    return "PASS"


def _check_sqlite_fts5() -> str:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
    except sqlite3.OperationalError as e:
        _emit("FAIL", "sqlite_fts5", str(e))
        return "FAIL"
    _emit("PASS", "sqlite_fts5")
    return "PASS"


def _check_db(db_path: Path) -> str:
    if not db_path.exists():
        _emit("WARN", "legal_db_present", f"{db_path} not found")
        return "WARN"
    _emit("PASS", "legal_db_present", str(db_path))
    return "PASS"


def _check_table(db_path: Path, table: str, fts: bool = False) -> str:
    if not db_path.exists():
        _emit("WARN", f"table:{table}", "db missing")
        return "WARN"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = ?", (table,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        _emit("FAIL", f"table:{table}", "missing")
        return "FAIL"
    _emit("PASS", f"table:{table}")
    return "PASS"


def _check_optional(module: str, label: str) -> str:
    try:
        importlib.import_module(module)
    except ImportError:
        _emit("WARN", label, "not installed (optional)")
        return "WARN"
    _emit("PASS", label)
    return "PASS"


def _check_local_llm(model_path: str | None) -> str:
    p = model_path or os.environ.get("LEGAL_LLM_MODEL_PATH")
    if not p:
        _emit("WARN", "local_llm_model", "no --model-path / $LEGAL_LLM_MODEL_PATH")
        return "WARN"
    if not Path(p).exists():
        _emit("WARN", "local_llm_model", f"path not found: {p}")
        return "WARN"
    _emit("PASS", "local_llm_model", p)
    return "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description="Legal RAG offline self-check.")
    parser.add_argument("--db", default="data/db/legal.db")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    results = [
        _check_python(),
        _check_sqlite_fts5(),
        _check_db(db_path),
        _check_table(db_path, "chunks_fts"),
        _check_table(db_path, "legal_fields"),
        _check_optional("rapidocr_onnxruntime", "rapidocr"),
        _check_optional("sentence_transformers", "sentence_transformers"),
        _check_optional("llama_cpp", "llama_cpp"),
        _check_local_llm(args.model_path),
    ]

    fails = sum(1 for r in results if r == "FAIL")
    warns = sum(1 for r in results if r == "WARN")
    print(f"summary: PASS={results.count('PASS')} WARN={warns} FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
