"""Hybrid (FTS + semantic) retrieval over the legal store.

Layered retrieval (Phase H 前置):

1. **Query understanding** (optional, default on) — see
   :mod:`src.legal.query_understanding`. Classifies the query into
   ``field_lookup`` / ``evidence_search`` / ``general_search`` and
   produces ``expanded_terms`` + ``field_candidates``.
2. **Field-first overlay** — for ``field_lookup`` queries we look up the
   matching ``legal_fields`` rows and PREPEND them to the result list as
   ``source="field"``. We do NOT drop hybrid candidates: a wrong regex
   match would otherwise hide a correct keyword/semantic hit.
3. **FTS / semantic / hybrid** — unchanged retrieval modes from Phase G.
   For ``evidence_search`` queries, ``expanded_terms`` widens the FTS
   query so synonyms (转账 vs 银行流水) can match.

All results carry::

    file_name, page_no, page_type, snippet, page_image_path,
    score, source ∈ {field, fts, semantic, hybrid},
    matched_field_name?, matched_field_value?      # only when source=field
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.legal.embeddings import EmbeddingProvider
from src.legal.field_extractor import get_doc_field_catalog, list_doc_ids
from src.legal.fts_store import search_chunks
from src.legal.query_understanding import analyze_legal_query
from src.legal.vector_store import lookup_chunk_id, semantic_search

RRF_K = 60

# Score given to field-overlay hits. Higher than any RRF score so they
# sort to the top, but not so high they're treated as the only answer
# (we still emit hybrid candidates beneath them).
FIELD_OVERLAY_SCORE = 10.0


def hybrid_search(
    db_path: str | Path,
    query: str,
    *,
    embedder: Optional[EmbeddingProvider] = None,
    mode: str = "hybrid",
    limit: int = 10,
    candidate_limit: int = 50,
    use_query_understanding: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve from the legal store with optional field-first overlay.

    ``embedder`` is required for ``semantic`` and ``hybrid`` modes; for
    ``fts`` mode it may be ``None``. Results are hydrated with
    ``page_type`` / ``page_image_path``.
    """
    if mode not in ("fts", "semantic", "hybrid"):
        raise ValueError(f"unknown mode: {mode!r}")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Legal DB not found: {db_path}")

    if mode in ("semantic", "hybrid") and embedder is None:
        raise ValueError(
            f"mode={mode!r} requires an embedder; pass embedder=... or "
            "use mode='fts'"
        )

    # --- Query understanding (optional) ---------------------------------
    analysis = None
    if use_query_understanding:
        catalog = _collect_doc_catalog(db_path)
        analysis = analyze_legal_query(query, doc_field_catalog=catalog)

    # --- Effective FTS / semantic query string --------------------------
    effective_query = query
    if analysis and analysis["query_type"] == "evidence_search" and analysis["expanded_terms"]:
        effective_query = " OR ".join([query] + analysis["expanded_terms"][:5])

    conn = sqlite3.connect(db_path)
    try:
        fts_hits: List[Dict[str, Any]] = []
        sem_hits: List[Dict[str, Any]] = []

        if mode in ("fts", "hybrid"):
            for h in search_chunks(conn, effective_query, limit=candidate_limit):
                h["chunk_id"] = lookup_chunk_id(
                    conn, h["file_name"], h["page_no"], h["snippet"]
                )
                fts_hits.append(h)

        if mode in ("semantic", "hybrid"):
            sem_hits = semantic_search(
                db_path, query, embedder, limit=candidate_limit
            )

        if mode == "fts":
            ranked = _rank_fts_only(fts_hits)
        elif mode == "semantic":
            ranked = _rank_semantic_only(sem_hits)
        else:
            ranked = _rrf_fuse(fts_hits, sem_hits)

        # --- Field-first overlay ---------------------------------------
        field_overlay: List[Dict[str, Any]] = []
        if analysis and analysis["query_type"] == "field_lookup":
            field_overlay = _field_lookup_hits(conn, analysis["field_candidates"])

        merged = _merge_field_overlay(field_overlay, ranked, limit=limit)
        return [_hydrate(conn, r) for r in merged]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Field overlay
# ---------------------------------------------------------------------------


def _collect_doc_catalog(db_path: Path) -> Dict[str, Any]:
    """Merge per-doc field catalogs into a single union for query analysis.

    A query is run against the whole DB, not a single doc. The union
    keeps the catalog signal useful (any field present in any doc can
    boost a query candidate) without over-coupling to one doc.
    """
    union = {
        "doc_id": "__union__",
        "field_names": [],
        "categories": {},
        "values_by_field": {},
    }
    for doc_id in list_doc_ids(db_path):
        cat = get_doc_field_catalog(db_path, doc_id)
        for name in cat["field_names"]:
            if name not in union["field_names"]:
                union["field_names"].append(name)
        for c, names in cat["categories"].items():
            bucket = union["categories"].setdefault(c, [])
            for n in names:
                if n not in bucket:
                    bucket.append(n)
        for name, values in cat["values_by_field"].items():
            bucket = union["values_by_field"].setdefault(name, [])
            for v in values:
                if v not in bucket:
                    bucket.append(v)
    return union


def _field_lookup_hits(
    conn: sqlite3.Connection, field_candidates: List[str]
) -> List[Dict[str, Any]]:
    """Find legal_fields rows whose field_name is in ``field_candidates``.

    Returns one hit per (doc, field, value) triple, ordered by the
    candidate's position so the most-confident user-intent field comes
    first.
    """
    if not field_candidates:
        return []

    out: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    for name in field_candidates:
        rows = conn.execute(
            """
            SELECT lf.field_name, lf.field_value, lf.page_no,
                   lf.source_text, lf.source_type, lf.confidence,
                   d.file_name, d.doc_id
            FROM legal_fields lf
            JOIN documents d ON d.doc_id = lf.doc_id
            WHERE lf.field_name = ?
            ORDER BY lf.confidence DESC, lf.page_no ASC
            """,
            (name,),
        ).fetchall()
        for (
            field_name,
            field_value,
            page_no,
            source_text,
            source_type,
            confidence,
            file_name,
            doc_id,
        ) in rows:
            key = (file_name, page_no, field_name, field_value)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "chunk_id": f"__field__::{doc_id}::{field_name}::{field_value}",
                    "file_name": file_name,
                    "page_no": page_no,
                    "snippet": source_text or field_value,
                    "score": FIELD_OVERLAY_SCORE + float(confidence or 0.0),
                    "source": "field",
                    "matched_field_name": field_name,
                    "matched_field_value": field_value,
                    "field_source_type": source_type,
                }
            )
    return out


def _merge_field_overlay(
    field_overlay: List[Dict[str, Any]],
    hybrid_ranked: List[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Prepend field hits, then append hybrid hits, dedup by (file, page)."""
    out: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    for h in field_overlay + hybrid_ranked:
        # Field overlay rows are unique by (file, page, field_name+value).
        # Hybrid rows dedup on (file, page) so we don't show the same page
        # twice when both keyword and semantic ranked it. We DO keep
        # field rows even when the same page is also a hybrid hit — they
        # carry different metadata.
        if h.get("source") == "field":
            key = (
                "field",
                h["file_name"],
                h["page_no"],
                h.get("matched_field_name"),
                h.get("matched_field_value"),
            )
        else:
            key = ("hybrid", h["file_name"], h["page_no"], h.get("snippet", "")[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Rankers (Phase G — unchanged behavior, kept here for cohesion)
# ---------------------------------------------------------------------------


def _rank_fts_only(fts_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rank, h in enumerate(fts_hits, start=1):
        cid = h.get("chunk_id") or _synthetic_id(h)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "chunk_id": cid,
                "file_name": h["file_name"],
                "page_no": h["page_no"],
                "snippet": h["snippet"],
                "score": 1.0 / rank,
                "source": "fts",
            }
        )
    return out


def _rank_semantic_only(sem_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for h in sem_hits:
        cid = h.get("chunk_id") or _synthetic_id(h)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "chunk_id": cid,
                "file_name": h["file_name"],
                "page_no": h["page_no"],
                "snippet": h["snippet"],
                "score": float(h["score"]),
                "source": "semantic",
            }
        )
    return out


def _rrf_fuse(
    fts_hits: List[Dict[str, Any]], sem_hits: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over fts + semantic candidates."""
    fused: Dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(fts_hits, start=1):
        cid = h.get("chunk_id") or _synthetic_id(h)
        entry = fused.setdefault(
            cid,
            {
                "chunk_id": cid,
                "file_name": h["file_name"],
                "page_no": h["page_no"],
                "snippet": h["snippet"],
                "score": 0.0,
                "source": "hybrid",
            },
        )
        entry["score"] += 1.0 / (RRF_K + rank)

    for rank, h in enumerate(sem_hits, start=1):
        cid = h.get("chunk_id") or _synthetic_id(h)
        entry = fused.get(cid)
        if entry is None:
            entry = fused.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "file_name": h["file_name"],
                    "page_no": h["page_no"],
                    "snippet": h["snippet"],
                    "score": 0.0,
                    "source": "hybrid",
                },
            )
        entry["score"] += 1.0 / (RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda r: r["score"], reverse=True)
    return ordered


def _synthetic_id(hit: Dict[str, Any]) -> str:
    return f"__syn__::{hit['file_name']}::p{hit['page_no']}::{hit.get('snippet', '')[:64]}"


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


def _hydrate(conn: sqlite3.Connection, hit: Dict[str, Any]) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT p.page_type, p.page_image_path
        FROM pages p
        JOIN documents d ON d.doc_id = p.doc_id
        WHERE d.file_name = ? AND p.page_no = ?
        LIMIT 1
        """,
        (hit["file_name"], hit["page_no"]),
    ).fetchone()
    page_type = row[0] if row else "text"
    page_image_path = row[1] if row else None
    out: Dict[str, Any] = {
        "file_name": hit["file_name"],
        "page_no": hit["page_no"],
        "page_type": page_type or "text",
        "snippet": hit["snippet"],
        "page_image_path": page_image_path,
        "score": float(hit.get("score", 0.0)),
        "source": hit.get("source", "fts"),
    }
    if hit.get("source") == "field":
        out["matched_field_name"] = hit.get("matched_field_name")
        out["matched_field_value"] = hit.get("matched_field_value")
    return out
