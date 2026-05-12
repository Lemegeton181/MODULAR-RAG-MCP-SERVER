"""Rule-based legal query understanding.

No LLM. The classifier walks the user query against:

* :data:`src.legal.legal_schema.FIELD_DEFS` — every field's
  ``query_triggers`` is matched as a substring;
* :data:`src.legal.legal_schema.EVIDENCE_KEYWORDS` — the document-side
  evidence vocabulary, also used as user-side triggers;
* an optional per-document ``doc_field_catalog`` produced by
  :func:`src.legal.field_extractor.get_doc_field_catalog` — if a field
  has actually been extracted from THIS doc, we boost it (and we still
  emit it as a candidate even when the base registry does not match,
  because the registry covers the common cases not all of them).

The output drives :mod:`src.legal.hybrid_search`:

* ``query_type='field_lookup'`` — try ``legal_fields`` first;
* ``query_type='evidence_search'`` — feed ``expanded_terms`` into hybrid
  retrieval;
* ``query_type='general_search'`` — pure hybrid as before.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from src.legal.legal_schema import (
    EVIDENCE_KEYWORDS,
    FIELD_DEFS,
    PARTY_ROLE_FIELDS,
    evidence_keyword_index,
    is_party_umbrella_query,
)


class QueryAnalysis(TypedDict):
    original_query: str
    expanded_terms: List[str]
    field_candidates: List[str]  # canonical field names, ordered
    query_type: str  # field_lookup | evidence_search | general_search


def analyze_legal_query(
    query: str, doc_field_catalog: Optional[Dict[str, Any]] = None
) -> QueryAnalysis:
    """Classify the user query and produce field / term expansions.

    The function is pure — same input ⇒ same output. ``doc_field_catalog``
    is optional; when provided, fields actually present in the current
    document are added to ``field_candidates`` even if the base triggers
    did not fire.
    """
    q = (query or "").strip()
    expanded: List[str] = []
    field_candidates: List[str] = []
    seen_fields: set[str] = set()

    if not q:
        return QueryAnalysis(
            original_query=query or "",
            expanded_terms=[],
            field_candidates=[],
            query_type="general_search",
        )

    # ---- 1. evidence-keyword detection (does NOT yet decide query_type) -
    evidence_groups: List[str] = []
    evidence_terms: List[str] = []
    kw_index = evidence_keyword_index()
    for kw, group in kw_index.items():
        if kw in q:
            if group not in evidence_groups:
                evidence_groups.append(group)
            if kw not in evidence_terms:
                evidence_terms.append(kw)

    # If the user query mentions an evidence keyword, expand to siblings
    # so hybrid retrieval can also catch the same evidence under a
    # synonymous label (e.g. 转账 -> 银行流水).
    for group in evidence_groups:
        for kw in EVIDENCE_KEYWORDS.get(group, []):
            if kw not in expanded and kw not in q:
                expanded.append(kw)

    # ---- 2. structured-field trigger detection --------------------------
    for field_id, definition in FIELD_DEFS.items():
        for trigger in definition.get("query_triggers", []):
            if trigger and trigger in q:
                name = definition["canonical_name"]
                if name not in seen_fields:
                    field_candidates.append(name)
                    seen_fields.add(name)
                break

    # 「当事人」 umbrella expansion — does NOT short-circuit specific roles
    if is_party_umbrella_query(q):
        for fid in PARTY_ROLE_FIELDS:
            name = FIELD_DEFS[fid]["canonical_name"]
            if name not in seen_fields:
                field_candidates.append(name)
                seen_fields.add(name)

    # ---- 3. doc-catalog augmentation ------------------------------------
    catalog_field_names: List[str] = []
    if doc_field_catalog:
        catalog_field_names = list(doc_field_catalog.get("field_names") or [])
        # Boost: any field present in the doc whose canonical_name appears
        # as a substring of the query, but which we missed above.
        for name in catalog_field_names:
            if name in q and name not in seen_fields:
                field_candidates.append(name)
                seen_fields.add(name)

    # ---- 4. query_type decision ----------------------------------------
    # Structured field triggers always win over evidence keywords. Without
    # this guard, a query like 「二倍工资金额」 would be classified as
    # evidence_search because it contains 「工资」 (an evidence keyword
    # for the salary group). The user intent there is a field lookup;
    # evidence-search is only the right answer when the query is about
    # whether a piece of evidence is *present*, not about a numeric or
    # entity field. We carve out a single exception for testimony, which
    # the spec mandates must land as evidence_search even though 「证人」
    # is also a party-role field.
    has_testimony_intent = any(
        kw in q for kw in EVIDENCE_KEYWORDS["testimony"] + ["有没有证人", "是否有证人"]
    )

    if field_candidates and not has_testimony_intent:
        query_type = "field_lookup"
    elif has_testimony_intent or evidence_groups:
        query_type = "evidence_search"
        # Make sure the evidence terms themselves are first in expanded.
        for term in evidence_terms:
            if term not in expanded:
                expanded.insert(0, term)
    else:
        query_type = "general_search"

    return QueryAnalysis(
        original_query=q,
        expanded_terms=expanded,
        field_candidates=field_candidates,
        query_type=query_type,
    )
