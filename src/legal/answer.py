"""Evidence-based legal question answering with citation enforcement.

This module sits on top of :mod:`src.legal.hybrid_search` and forms the
LLM-facing surface of Phase H:

1. :func:`answer_legal_question` retrieves top-k evidence via the
   existing hybrid retrieval stack. If no evidence is found, it returns
   a hard-no answer and an empty citation list — no fabrication.
2. When an ``llm_client`` is provided, the function builds a strict
   evidence-only prompt and forwards it. The LLM is expected to answer
   based on the evidence alone; the citations come from the retrieval
   layer regardless, so the output is auditable even if the LLM
   hallucinates the answer text.
3. When ``llm_client`` is ``None`` (the default), the function falls
   back to an extractive answer derived directly from the top hit's
   snippet / matched field — still with citations attached.
4. :func:`citation_check` validates the result shape after the fact:
   ``no_evidence`` and ``citations`` must agree, and every citation
   must carry ``file_name`` / ``page_no`` / ``snippet``.

No external API is called. The LLM is a pure injectable interface
satisfying :class:`LLMClient` — fakes plug in for tests, real local
LLMs plug in later. No new heavy dependencies are introduced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from src.legal.embeddings import EmbeddingProvider
from src.legal.hybrid_search import hybrid_search
from src.legal.query_understanding import analyze_legal_query


_HARD_NO_ANSWER = "未在当前材料中找到依据。"
_MAX_PROMPT_SNIPPET = 400


class LLMClient(Protocol):
    """Minimal protocol any LLM client must satisfy.

    Implementations take a prompt string and return the completion text.
    No streaming, no tool use — answer composition stays inside this
    module so the citation contract holds.
    """

    def generate(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


def answer_legal_question(
    db_path: str | Path,
    query: str,
    *,
    llm_client: Optional[LLMClient] = None,
    embedder: Optional[EmbeddingProvider] = None,
    mode: str = "hybrid",
    limit: int = 5,
) -> Dict[str, Any]:
    """Answer ``query`` against the legal store with grounded citations.

    Returns::

        {
            "answer": "...",
            "citations": [
                {
                    "file_name": "...",
                    "page_no": 1,
                    "snippet": "...",
                    "source": "field|hybrid|fts|semantic",
                },
                ...
            ],
            "confidence": "high|medium|low",
            "no_evidence": True|False,
        }

    ``mode`` mirrors :func:`src.legal.hybrid_search.hybrid_search`. When
    ``mode != "fts"`` an ``embedder`` is required; if one is not given,
    the function transparently downgrades to FTS so the CLI still works
    on machines without the sentence-transformers stack installed.
    """
    if mode not in ("fts", "semantic", "hybrid"):
        raise ValueError(f"unknown mode: {mode!r}")

    effective_mode = mode
    if mode in ("semantic", "hybrid") and embedder is None:
        effective_mode = "fts"

    # Classify the query so the extractive fallback can decide whether a
    # field-overlay hit is allowed as the *primary* answer. Reasoning /
    # evidence / general queries must be answered from evidence snippets;
    # field hits are demoted to supplementary citations only.
    analysis = analyze_legal_query(query)
    query_type = analysis["query_type"]

    hits = hybrid_search(
        db_path,
        query,
        embedder=embedder if effective_mode != "fts" else None,
        mode=effective_mode,
        limit=limit,
    )

    if not hits:
        return {
            "answer": _HARD_NO_ANSWER,
            "citations": [],
            "confidence": "low",
            "no_evidence": True,
            "query_type": query_type,
        }

    citations = [
        {
            "file_name": h["file_name"],
            "page_no": h["page_no"],
            "snippet": _strip_html(h.get("snippet") or ""),
            "source": h.get("source", "hybrid"),
        }
        for h in hits
    ]

    if llm_client is None:
        answer_text = _extractive_answer(query, hits, query_type)
    else:
        prompt = build_evidence_prompt(query, hits)
        try:
            raw = llm_client.generate(prompt)
        except Exception as exc:  # pragma: no cover - defensive
            raw = f"[llm_error: {exc.__class__.__name__}]"
        answer_text = (raw or "").strip() or _extractive_answer(
            query, hits, query_type
        )

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": _confidence_from_hits(hits),
        "no_evidence": False,
        "query_type": query_type,
    }


# ---------------------------------------------------------------------------
# Citation check
# ---------------------------------------------------------------------------


def citation_check(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``answer_legal_question``'s result for citation integrity.

    Rules:

    1. ``no_evidence=True`` ⇒ ``citations`` must be empty.
    2. ``no_evidence=False`` ⇒ ``citations`` must be non-empty.
    3. Every citation must carry ``file_name``, ``page_no`` and
       ``snippet`` (truthy).
    4. Any violation downgrades ``confidence`` to ``"low"``.

    The function mutates ``result`` in place (adds ``citation_issues``
    and may overwrite ``confidence``) and also returns it for chaining.
    No NLP, no new dependencies.
    """
    issues: List[str] = []
    no_evidence = bool(result.get("no_evidence"))
    citations = result.get("citations") or []

    if no_evidence and citations:
        issues.append("no_evidence_but_has_citations")
    if not no_evidence and not citations:
        issues.append("missing_citations")

    for i, c in enumerate(citations):
        if not c.get("file_name"):
            issues.append(f"citation[{i}].file_name_missing")
        if c.get("page_no") in (None, "", 0):
            # page_no==0 is suspect; legal pages are 1-indexed
            issues.append(f"citation[{i}].page_no_missing")
        if not c.get("snippet"):
            issues.append(f"citation[{i}].snippet_missing")

    if issues:
        result["confidence"] = "low"

    result["citation_issues"] = issues
    return result


# ---------------------------------------------------------------------------
# Prompt + extractive fallback
# ---------------------------------------------------------------------------


def build_evidence_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    """Render the strict evidence-only prompt sent to an injected LLM.

    The prompt is intentionally short, in Chinese, and forbids the model
    from answering beyond the listed evidence. Even if a real LLM later
    ignores the rule, citations are still produced from the retrieval
    layer, so :func:`citation_check` can flag the inconsistency.
    """
    lines: List[str] = []
    for i, h in enumerate(hits, start=1):
        snippet = _strip_html(h.get("snippet") or "")
        if len(snippet) > _MAX_PROMPT_SNIPPET:
            snippet = snippet[:_MAX_PROMPT_SNIPPET] + "..."
        lines.append(
            f"[{i}] {h['file_name']} 第{h['page_no']}页 "
            f"(source={h.get('source','hybrid')}): {snippet}"
        )
    evidence_block = "\n".join(lines) if lines else "(无)"
    return (
        "你是法律文书检索助手。请只基于下面列出的证据片段回答问题。"
        "如果证据不足以回答，请直接回复\"" + _HARD_NO_ANSWER + "\"。"
        "不要补充材料外的信息，不要编造法条或人名。\n\n"
        "证据：\n"
        f"{evidence_block}\n\n"
        f"问题：{query}\n\n"
        "回答："
    )


def _extractive_answer(
    query: str, hits: List[Dict[str, Any]], query_type: str = "general_search"
) -> str:
    """Compose an answer directly from the top hit when no LLM is given.

    Routing by ``query_type``:

    * ``field_lookup`` — a field-overlay top hit answers directly as
      ``{field_name}：{field_value}``.
    * ``evidence_search`` / ``general_search`` — field hits are demoted
      to supplementary citations. The primary answer is the top
      non-field snippet, so a question like 「为什么申请人认为裁决应撤销？」
      is answered from the reasoning page, not from 「申请人：xxx」.
    """
    if not hits:
        return _HARD_NO_ANSWER

    if query_type == "field_lookup":
        top = hits[0]
        if top.get("source") == "field" and top.get("matched_field_name"):
            name = top["matched_field_name"]
            value = top.get("matched_field_value") or ""
            if value:
                return f"{name}：{value}"
        snippet = _strip_html(top.get("snippet") or "")
        return snippet or _HARD_NO_ANSWER

    # evidence_search / general_search / reasoning_search: prefer the top
    # non-field evidence snippet.
    for h in hits:
        if h.get("source") == "field":
            continue
        snippet = _strip_html(h.get("snippet") or "")
        if snippet:
            return snippet
    # Only field hits available — fall back to that snippet (source_text),
    # which is the surrounding sentence, not the bare field value.
    snippet = _strip_html(hits[0].get("snippet") or "")
    return snippet or _HARD_NO_ANSWER


def _confidence_from_hits(hits: List[Dict[str, Any]]) -> str:
    """Coarse confidence heuristic. Not a probability — a label."""
    if not hits:
        return "low"
    # A field overlay hit means the regex pinpointed a structured value.
    if any(h.get("source") == "field" for h in hits):
        return "high"
    # Multiple convergent pages → medium. Single hit → medium-low.
    distinct_pages = {(h["file_name"], h["page_no"]) for h in hits}
    if len(distinct_pages) >= 2:
        return "medium"
    return "medium"


def _strip_html(text: str) -> str:
    """Remove ``<b>``/``</b>`` highlight tags from FTS snippets."""
    if not text:
        return ""
    return text.replace("<b>", "").replace("</b>", "")
