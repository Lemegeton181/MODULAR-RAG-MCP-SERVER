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


# Reasoning / why-style triggers. When one of these appears together with a
# party-role field candidate, the user's true intent is to look for the
# reasoning / evidence behind a party's claim, not the party's name itself.
# Fact-field candidates (e.g. "适用法律错误") are NOT affected — those
# remain field_lookup so 「适用法律是否错误」 keeps working.
REASONING_TRIGGERS: List[str] = [
    "为什么", "为何", "原因", "理由", "依据",
    "哪些理由", "哪些事实", "哪些证据",
    "是否构成", "是否存在", "如何证明",
    "是否", "能否",
]

PARTY_ROLE_CANONICAL_NAMES: set[str] = {
    "申请人", "被申请人", "原告", "被告", "第三人",
    "上诉人", "被上诉人", "法定代表人", "委托代理人",
    "辩护人", "证人",
}


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

    # Reasoning / why-style override: if the user is clearly asking WHY a
    # party did X (not WHO the party is), the answer should come from the
    # evidence layer, not from the legal_fields overlay. Only role fields
    # trigger this — fact fields like 「适用法律错误」 stay field_lookup
    # even when the question phrases them as 「适用法律是否错误」.
    has_reasoning_intent = any(t in q for t in REASONING_TRIGGERS)
    has_role_field = any(
        name in PARTY_ROLE_CANONICAL_NAMES for name in field_candidates
    )
    force_evidence = has_testimony_intent or (
        has_reasoning_intent and has_role_field
    )

    if field_candidates and not force_evidence:
        query_type = "field_lookup"
    elif force_evidence or evidence_groups:
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
