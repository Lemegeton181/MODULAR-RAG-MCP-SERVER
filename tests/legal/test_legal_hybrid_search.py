"""Phase G — hybrid_search tests.

Exercises the FTS-only / semantic-only / hybrid (RRF-fused) modes through
:func:`hybrid_search`, plus the RRF math itself.
"""
from __future__ import annotations

import pytest

from src.legal.embeddings import DeterministicEmbeddingProvider
from src.legal.hybrid_search import RRF_K, _rrf_fuse, hybrid_search
from src.legal.legal_ingest import ingest_pages


def _ingest_corpus(tmp_path):
    embedder = DeterministicEmbeddingProvider(dim=64)
    db_path = tmp_path / "legal.db"
    pages = [
        {"page_no": 1, "text": "张三向李四借款50000元，双方约定2023年5月6日还款。"},
        {"page_no": 2, "text": "本页为银行转账记录，显示李四向张三转账50000元。"},
        {"page_no": 3, "text": "法院判决书：被告应当承担违约责任。"},
        {"page_no": 4, "text": "Page four contains the loan agreement keyword in English."},
    ]
    ingest_pages(db_path, "case.pdf", pages, embedder=embedder)
    return db_path, embedder


def test_unknown_mode_raises(tmp_path):
    db_path, embedder = _ingest_corpus(tmp_path)
    with pytest.raises(ValueError):
        hybrid_search(db_path, "借款", embedder=embedder, mode="bogus")


def test_semantic_or_hybrid_without_embedder_raises(tmp_path):
    db_path, _ = _ingest_corpus(tmp_path)
    with pytest.raises(ValueError):
        hybrid_search(db_path, "借款", mode="semantic")
    with pytest.raises(ValueError):
        hybrid_search(db_path, "借款", mode="hybrid")


def test_fts_mode_returns_fts_source_and_no_embedder_required(tmp_path):
    db_path, _ = _ingest_corpus(tmp_path)
    hits = hybrid_search(db_path, "借款", mode="fts", limit=5)
    assert hits
    assert all(h["source"] == "fts" for h in hits)
    top = hits[0]
    assert top["page_no"] == 1
    assert top["page_type"] == "text"
    assert "借款" in top["snippet"]
    # Scores are 1/rank, strictly descending.
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_semantic_mode_uses_embedding_similarity(tmp_path):
    db_path, embedder = _ingest_corpus(tmp_path)
    # Re-embed a verbatim chunk text so the deterministic provider gives
    # us a near-1.0 cosine on the matching chunk.
    hits = hybrid_search(
        db_path,
        "本页为银行转账记录，显示李四向张三转账50000元。",
        embedder=embedder,
        mode="semantic",
        limit=3,
    )
    assert hits
    assert all(h["source"] == "semantic" for h in hits)
    assert hits[0]["page_no"] == 2
    assert hits[0]["score"] > 0.9
    assert hits[0]["page_type"] == "text"


def test_hybrid_mode_fuses_fts_and_semantic_and_dedupes(tmp_path):
    db_path, embedder = _ingest_corpus(tmp_path)
    hits = hybrid_search(
        db_path,
        "借款",
        embedder=embedder,
        mode="hybrid",
        limit=5,
    )
    assert hits
    assert all(h["source"] == "hybrid" for h in hits)
    # Same chunk_id (file + page) must not appear twice.
    seen = set()
    for h in hits:
        key = (h["file_name"], h["page_no"], h["snippet"])
        assert key not in seen
        seen.add(key)
    # Page 1 should win or be in the top 2 since both rankers favor it.
    top_pages = [h["page_no"] for h in hits[:2]]
    assert 1 in top_pages


def test_hybrid_mode_returns_page_metadata(tmp_path):
    db_path, embedder = _ingest_corpus(tmp_path)
    hits = hybrid_search(
        db_path, "借款", embedder=embedder, mode="hybrid", limit=3
    )
    assert hits
    for h in hits:
        assert "page_type" in h
        assert "page_image_path" in h  # may be None
        assert "score" in h
        assert "source" in h


# ---------------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------------


def _hit(chunk_id, page_no=1, file_name="case.pdf", snippet="snip"):
    return {
        "chunk_id": chunk_id,
        "file_name": file_name,
        "page_no": page_no,
        "snippet": snippet,
        "score": 0.0,
    }


def test_rrf_fuse_uses_1_over_k_plus_rank():
    fts = [_hit("a"), _hit("b"), _hit("c")]
    sem = [_hit("c"), _hit("a"), _hit("d")]

    fused = _rrf_fuse(fts, sem)
    by_id = {f["chunk_id"]: f for f in fused}

    # a: rank 1 in fts, rank 2 in sem
    expected_a = 1 / (RRF_K + 1) + 1 / (RRF_K + 2)
    # c: rank 3 in fts, rank 1 in sem
    expected_c = 1 / (RRF_K + 3) + 1 / (RRF_K + 1)
    # b: only fts rank 2
    expected_b = 1 / (RRF_K + 2)
    # d: only sem rank 3
    expected_d = 1 / (RRF_K + 3)

    assert by_id["a"]["score"] == pytest.approx(expected_a)
    assert by_id["b"]["score"] == pytest.approx(expected_b)
    assert by_id["c"]["score"] == pytest.approx(expected_c)
    assert by_id["d"]["score"] == pytest.approx(expected_d)

    # Ordering: a and c tied near the top, then b/d.
    ordered_ids = [f["chunk_id"] for f in fused]
    assert set(ordered_ids[:2]) == {"a", "c"}
    assert set(ordered_ids[2:]) == {"b", "d"}


def test_rrf_fuse_dedupes_chunk_id_within_a_single_ranker():
    fts = [_hit("a"), _hit("a")]  # synthetic edge case
    sem = []
    fused = _rrf_fuse(fts, sem)
    # The fuse uses dict-keyed accumulation, so duplicates within fts
    # accumulate into a single entry — still a single output row.
    ids = [f["chunk_id"] for f in fused]
    assert ids.count("a") == 1
