"""Phase G — vector store + semantic search tests.

Uses :class:`DeterministicEmbeddingProvider` so we never download a model
at test time, while still exercising the full
ingest_pages → upsert → semantic_search loop end-to-end.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.legal.embeddings import (
    DEFAULT_ST_MODEL,
    ST_INSTALL_HINT,
    DeterministicEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    cosine_similarity,
)
from src.legal.legal_ingest import ingest_pages
from src.legal.vector_store import (
    count_vectors,
    lookup_chunk_id,
    semantic_search,
)


def _ingest_corpus(tmp_path, embedder):
    db_path = tmp_path / "legal.db"
    pages = [
        {"page_no": 1, "text": "张三向李四借款50000元，双方约定2023年5月6日还款。"},
        {"page_no": 2, "text": "本页为银行转账记录，显示李四向张三转账50000元。"},
        {"page_no": 3, "text": "法院判决书：被告应当承担违约责任。"},
    ]
    summary = ingest_pages(db_path, "case.pdf", pages, embedder=embedder)
    return db_path, summary


def test_legal_vectors_table_is_created(tmp_path):
    """init_legal_db must create the legal_vectors table with the expected columns."""
    from src.legal.db import init_legal_db

    db_path = tmp_path / "legal.db"
    init_legal_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(legal_vectors)").fetchall()
        }
    finally:
        conn.close()

    assert {"chunk_id", "doc_id", "page_no", "model", "dim", "embedding"} <= cols


def test_ingest_pages_writes_one_vector_per_chunk(tmp_path):
    embedder = DeterministicEmbeddingProvider(dim=32)
    db_path, summary = _ingest_corpus(tmp_path, embedder)

    assert summary["n_vectors"] == summary["n_chunks"]
    assert summary["n_vectors"] >= 3

    conn = sqlite3.connect(db_path)
    try:
        n_vec = count_vectors(conn)
        sample = conn.execute(
            "SELECT model, dim FROM legal_vectors LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert n_vec == summary["n_vectors"]
    assert sample[0] == embedder.model
    assert sample[1] == embedder.dim


def test_ingest_pages_without_embedder_writes_zero_vectors(tmp_path):
    db_path = tmp_path / "legal.db"
    pages = [{"page_no": 1, "text": "Page one contains the loan agreement keyword."}]
    summary = ingest_pages(db_path, "case.pdf", pages)
    assert summary["n_vectors"] == 0


def test_re_ingest_same_doc_replaces_vectors_idempotently(tmp_path):
    embedder = DeterministicEmbeddingProvider(dim=32)
    db_path, first = _ingest_corpus(tmp_path, embedder)
    db_path2, second = _ingest_corpus(tmp_path, embedder)

    assert db_path == db_path2
    assert first["doc_id"] == second["doc_id"]
    assert first["n_vectors"] == second["n_vectors"]

    conn = sqlite3.connect(db_path)
    try:
        n_vec = count_vectors(conn)
    finally:
        conn.close()
    assert n_vec == second["n_vectors"]


def test_semantic_search_ranks_query_chunk_at_top(tmp_path):
    """The chunk that exactly matches the query string must rank #1."""
    embedder = DeterministicEmbeddingProvider(dim=64)
    db_path, _ = _ingest_corpus(tmp_path, embedder)

    hits = semantic_search(
        db_path,
        "本页为银行转账记录，显示李四向张三转账50000元。",
        embedder,
        limit=3,
    )
    assert hits
    assert hits[0]["page_no"] == 2
    assert "银行转账" in hits[0]["snippet"]
    # Top score should be near 1.0 because we re-embed the exact text.
    assert hits[0]["score"] > 0.9


def test_semantic_search_returns_empty_when_no_vectors(tmp_path):
    db_path = tmp_path / "legal.db"
    pages = [{"page_no": 1, "text": "无向量内容"}]
    ingest_pages(db_path, "case.pdf", pages)  # no embedder => no vectors

    embedder = DeterministicEmbeddingProvider(dim=16)
    assert semantic_search(db_path, "借款", embedder, limit=5) == []


def test_semantic_search_empty_query_returns_empty(tmp_path):
    embedder = DeterministicEmbeddingProvider(dim=16)
    db_path, _ = _ingest_corpus(tmp_path, embedder)
    assert semantic_search(db_path, "   ", embedder, limit=5) == []


def test_lookup_chunk_id_resolves_fts_snippet_back_to_chunk_id(tmp_path):
    embedder = DeterministicEmbeddingProvider(dim=16)
    db_path, _ = _ingest_corpus(tmp_path, embedder)

    conn = sqlite3.connect(db_path)
    try:
        chunk_id = lookup_chunk_id(
            conn, "case.pdf", 1, "...<b>借款</b>50000元..."
        )
    finally:
        conn.close()
    assert chunk_id is not None
    assert chunk_id.endswith("_p1_c0")


def test_cosine_similarity_basic_properties():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    c = [1.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)
    assert cosine_similarity(a, c) == pytest.approx(1.0)
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 1.0]) == 0.0


def test_sentence_transformer_provider_install_hint_is_documented():
    """Pin the install hint so docs / CLI / error message stay aligned."""
    assert SentenceTransformerEmbeddingProvider.INSTALL_HINT == ST_INSTALL_HINT
    assert ST_INSTALL_HINT == (
        ".venv\\Scripts\\python.exe -m pip install sentence-transformers"
    )
    assert DEFAULT_ST_MODEL == "paraphrase-multilingual-MiniLM-L12-v2"


def test_sentence_transformer_provider_raises_when_st_missing(monkeypatch):
    """When sentence_transformers is not importable, construction must
    raise ImportError with the documented hint — no model download attempted."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing sentence_transformers")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        SentenceTransformerEmbeddingProvider()

    msg = str(excinfo.value)
    assert "sentence-transformers is not installed" in msg
    assert ST_INSTALL_HINT in msg
