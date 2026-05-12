"""Per-document legal field extraction.

The extractor runs against the per-page text already stored in
``pages.page_text``. For each page it:

1. Walks every structured field in
   :data:`src.legal.legal_schema.FIELD_DEFS` and runs its
   ``extract_patterns`` (compiled once per process). The captured
   ``value`` group is normalized lightly (whitespace + trailing
   punctuation) and persisted into ``legal_fields`` with
   ``source_type='regex'``.
2. Scans the page for evidence keywords from
   :data:`src.legal.legal_schema.EVIDENCE_KEYWORDS`. Each hit becomes a
   ``legal_fields`` row with ``field_category='evidence'``,
   ``source_type='keyword'``, ``field_value=<keyword>`` — i.e. we record
   *that* a piece of evidence is mentioned on a page, without
   pretending to know its canonical value.

Design rules:

* If we cannot extract a value, we DO NOT write a row. No fabrication.
* Re-running on the same ``doc_id`` first deletes its old field rows so
  reindex stays idempotent.
* All matches keep ``page_no`` and a ``source_text`` window so a
  downstream caller can always show where the value came from.
"""
from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.legal.legal_schema import (
    EVIDENCE_KEYWORDS,
    FIELD_DEFS,
    evidence_keyword_index,
)


# Maximum length of the captured value after stripping. Conservative —
# legal docs use full-width punctuation inconsistently and some
# patterns like 申请人:... can run very long lines.
_MAX_VALUE_LEN = 80
_SOURCE_WINDOW = 60  # chars before / after the match for source_text


# ---------------------------------------------------------------------------
# Case-number normalization
#
# Free-text matches like 「...裁委员会做出的哈劳人仲字（2025）1077号」 or 「字（2025）
# 1077号」 carry leaking prefix / suffix. We normalize by locating the
# 「字（YYYY）N号」 anchor and walking backwards to grab a short prefix that
# looks like a 字号 stem (e.g. 「哈劳人仲」, 「京01民初」, 「沪一民」).
#
# Heuristic — kept conservative to avoid distorting genuine matches:
# 1. Find the 「字（YYYY）...N号」 anchor.
# 2. The prefix is the run of Chinese chars / digits / latin letters
#    immediately before 「字」, limited to PREFIX_MAX chars. We pick the
#    last 2..PREFIX_MAX chars of that run as the stem — Chinese case-no
#    stems are conventionally 2-5 chars long, and capping the tail
#    discards anything that came from sentence context like 「做出的」.
# 3. If we cannot find an anchor we leave the value untouched.
# ---------------------------------------------------------------------------
_CASE_NO_ANCHOR_RE = re.compile(
    r"(?P<prefix_run>[一-鿿0-9A-Za-z]*?)字\s*"
    r"[（(]\s*(?P<year>\d{4})\s*[)）]\s*"
    r"(?P<middle>[一-鿿0-9A-Za-z]{0,12}?)\s*"
    r"(?P<number>\d+)\s*号"
)

# Common function words / verbs that should never appear inside a 字号 stem.
# When we walk back from 「字」 to pick the prefix, hitting any of these
# characters terminates the stem — they are sentence connectors leaked
# from free text like 「做出的」, 「依法作出」, 「向某某发出的」.
_CASE_NO_PREFIX_STOP_CHARS = set("的之是由在经过后将与和及对从向到为以等所被")

_CASE_NO_PREFIX_MAX = 6
_CASE_NO_PREFIX_MIN = 1


def _normalize_case_no(value: str) -> str:
    """Trim free-text leakage around the canonical 「X字（YYYY）N号」 token.

    Returns the normalized value, or the original ``value`` when no anchor
    is found (so we never invent or distort a real match).
    """
    if not value:
        return value
    m = _CASE_NO_ANCHOR_RE.search(value)
    if not m:
        return value
    full_prefix = m.group("prefix_run") or ""
    # Walk back from 「字」 over the contiguous prefix run, stop at the
    # first connector char (e.g. 「的」), then cap at PREFIX_MAX.
    trimmed_chars: List[str] = []
    for ch in reversed(full_prefix):
        if ch in _CASE_NO_PREFIX_STOP_CHARS:
            break
        trimmed_chars.append(ch)
        if len(trimmed_chars) >= _CASE_NO_PREFIX_MAX:
            break
    prefix = "".join(reversed(trimmed_chars))
    return f"{prefix}字（{m.group('year')}）{m.group('middle')}{m.group('number')}号"


@lru_cache(maxsize=1)
def _compiled_field_patterns() -> List[Tuple[str, str, str, "re.Pattern[str]"]]:
    """Return ``(field_id, canonical_name, category, regex)`` for every field."""
    out: List[Tuple[str, str, str, "re.Pattern[str]"]] = []
    for field_id, definition in FIELD_DEFS.items():
        for pat in definition.get("extract_patterns", []) or []:
            out.append(
                (
                    field_id,
                    definition["canonical_name"],
                    definition["category"],
                    re.compile(pat),
                )
            )
    return out


def _clean_value(raw: str) -> str:
    if raw is None:
        return ""
    v = raw.strip()
    # Trim trailing common separators / brackets / quotes that creep in
    # from full-width forms.
    v = v.rstrip("。,，；;:：、 \t\r\n")
    if len(v) > _MAX_VALUE_LEN:
        v = v[:_MAX_VALUE_LEN].rstrip()
    return v


def _source_window(text: str, span: Tuple[int, int]) -> str:
    start = max(0, span[0] - _SOURCE_WINDOW)
    end = min(len(text), span[1] + _SOURCE_WINDOW)
    return text[start:end].replace("\n", " ").strip()


def extract_fields_from_text(
    doc_id: str, page_no: int, text: str
) -> List[Dict[str, Any]]:
    """Return all field rows mined from a single page's text.

    Each dict is shaped for direct insertion into ``legal_fields``::

        {doc_id, field_name, field_value, field_category,
         page_no, source_text, confidence, source_type}
    """
    if not text:
        return []

    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    # 1. Structured (regex) fields ----------------------------------------
    for field_id, canonical, category, regex in _compiled_field_patterns():
        for m in regex.finditer(text):
            try:
                raw_value = m.group("value")
            except (IndexError, KeyError):
                continue
            value = _clean_value(raw_value)
            if canonical == "案号":
                value = _normalize_case_no(value)
            if not value:
                continue
            key = (canonical, value)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "doc_id": doc_id,
                    "field_name": canonical,
                    "field_value": value,
                    "field_category": category,
                    "page_no": page_no,
                    "source_text": _source_window(text, m.span()),
                    "confidence": 0.8,
                    "source_type": "regex",
                }
            )

    # 2. Evidence keywords (no value invention; presence only) ------------
    kw_index = evidence_keyword_index()
    for kw, group in kw_index.items():
        idx = text.find(kw)
        if idx < 0:
            continue
        key = ("evidence:" + group, kw)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "doc_id": doc_id,
                "field_name": f"evidence:{group}",
                "field_value": kw,
                "field_category": "evidence",
                "page_no": page_no,
                "source_text": _source_window(text, (idx, idx + len(kw))),
                "confidence": 0.5,
                "source_type": "keyword",
            }
        )

    return rows


def upsert_legal_fields(
    db_path: str | Path, fields: List[Dict[str, Any]], *, doc_id: Optional[str] = None
) -> int:
    """Insert ``fields`` rows. If ``doc_id`` is given, replace its prior rows.

    Returns the number of rows inserted.
    """
    if not fields and doc_id is None:
        return 0

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        if doc_id is not None:
            conn.execute("DELETE FROM legal_fields WHERE doc_id = ?", (doc_id,))
        for f in fields:
            conn.execute(
                """
                INSERT INTO legal_fields
                (doc_id, field_name, field_value, field_category,
                 page_no, source_text, confidence, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f["doc_id"],
                    f["field_name"],
                    f["field_value"],
                    f["field_category"],
                    f["page_no"],
                    f["source_text"],
                    float(f.get("confidence", 0.5)),
                    f.get("source_type", "regex"),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return len(fields)


def extract_legal_fields_for_doc(db_path: str | Path, doc_id: str) -> int:
    """Re-extract fields for one document from its already-stored page text.

    Wipes any prior ``legal_fields`` rows for ``doc_id`` first.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT page_no, page_text FROM pages WHERE doc_id = ? ORDER BY page_no",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    fields: List[Dict[str, Any]] = []
    for page_no, page_text in rows:
        fields.extend(extract_fields_from_text(doc_id, page_no, page_text or ""))

    return upsert_legal_fields(db_path, fields, doc_id=doc_id)


def extract_legal_fields_for_all_docs(db_path: str | Path) -> int:
    """Re-extract fields for every document. Returns total rows written."""
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        doc_ids = [row[0] for row in conn.execute("SELECT doc_id FROM documents").fetchall()]
    finally:
        conn.close()

    total = 0
    for doc_id in doc_ids:
        total += extract_legal_fields_for_doc(db_path, doc_id)
    return total


def get_doc_field_catalog(db_path: str | Path, doc_id: str) -> Dict[str, Any]:
    """Return the per-document field catalog used by query understanding.

    Shape::

        {
          "doc_id": "...",
          "field_names": ["案号", "申请人", ...],
          "categories": {"doc_basic": [...], "evidence": [...], ...},
          "values_by_field": {"案号": ["哈劳人仲字（2025）1077号"], ...},
        }
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT field_name, field_value, field_category "
            "FROM legal_fields WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    finally:
        conn.close()

    field_names: List[str] = []
    categories: Dict[str, List[str]] = {}
    values_by_field: Dict[str, List[str]] = {}
    seen_names: set[str] = set()
    for name, value, category in rows:
        if name not in seen_names:
            field_names.append(name)
            seen_names.add(name)
        categories.setdefault(category, [])
        if name not in categories[category]:
            categories[category].append(name)
        values_by_field.setdefault(name, [])
        if value not in values_by_field[name]:
            values_by_field[name].append(value)

    return {
        "doc_id": doc_id,
        "field_names": field_names,
        "categories": categories,
        "values_by_field": values_by_field,
    }


def list_doc_ids(db_path: str | Path) -> List[str]:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT doc_id FROM documents").fetchall()]
    finally:
        conn.close()
