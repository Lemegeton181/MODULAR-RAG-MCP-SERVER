"""Local embedding providers for the legal retrieval store.

Two providers ship in Phase G:

* :class:`SentenceTransformerEmbeddingProvider` — real local embedding
  backend powered by ``sentence-transformers``. Lazily imported so the
  rest of the legal module remains usable without the dependency. Raises
  a clear ImportError with the exact ``pip install`` hint when missing.
* :class:`DeterministicEmbeddingProvider` — model-free deterministic
  hash-based vectors. Used by the test suite so we never download a
  model at test time, while still exercising the full
  ingest → vector_store → semantic_search → hybrid_search loop.
"""
from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Sequence

DEFAULT_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

ST_INSTALL_HINT = (
    ".venv\\Scripts\\python.exe -m pip install sentence-transformers"
)


class EmbeddingProvider(ABC):
    """Abstract embedding backend.

    Implementations must expose a stable ``model`` identifier and a fixed
    ``dim``, and must accept a list of strings.
    """

    model: str
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one float vector per input text, length ``self.dim``."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Real local embedding backend via the ``sentence-transformers`` package.

    Construction performs an import-only probe so a missing package surfaces
    a single, actionable error before any model download is attempted. The
    model itself is loaded lazily on first :meth:`embed` call.
    """

    INSTALL_HINT = ST_INSTALL_HINT

    def __init__(self, model: str = DEFAULT_ST_MODEL) -> None:
        try:
            self._st_module = _import_sentence_transformers()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                f"Install it with: {self.INSTALL_HINT}"
            ) from e
        self.model = model
        self._model_obj = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:  # type: ignore[override]
        if self._dim is None:
            # Force lazy load so we know the dimension; embed an empty-ish
            # sample once.
            vec = self.embed(["_"])[0]
            self._dim = len(vec)
        return self._dim

    def _get_model(self):
        if self._model_obj is None:
            self._model_obj = self._st_module.SentenceTransformer(self.model)
        return self._model_obj

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        out = model.encode(list(texts), show_progress_bar=False)
        # Normalize numpy array / list-of-lists into pure Python floats.
        result: List[List[float]] = []
        for row in out:
            row_list = list(row)
            result.append([float(x) for x in row_list])
        if self._dim is None and result:
            self._dim = len(result[0])
        return result


def _import_sentence_transformers():
    import importlib

    return importlib.import_module("sentence_transformers")


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hash-derived deterministic embeddings — no downloads, no model.

    Used by tests to exercise the full vector pipeline. Vectors are L2
    normalized so cosine similarity equals the dot product. The same text
    always maps to the same vector across runs, and similar substrings
    share components, which is enough to give the test suite a stable
    semantic-ranking signal.
    """

    def __init__(self, dim: int = 64, model: str = "deterministic-hash-v1") -> None:
        self.model = model
        self._dim_value = int(dim)

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self._dim_value

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dim_value
        # Use overlapping char trigrams so semantically-near strings share
        # components without being identical.
        norm = (text or "").strip().lower()
        if not norm:
            return _l2_normalize([1.0] + [0.0] * (self._dim_value - 1))

        tokens: List[str] = []
        if len(norm) < 3:
            tokens.append(norm)
        else:
            for i in range(len(norm) - 2):
                tokens.append(norm[i : i + 3])

        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for j in range(0, min(len(digest), 32), 4):
                slot = struct.unpack(">I", digest[j : j + 4])[0] % self._dim_value
                # Sign bit derived from the high byte to spread signs.
                sign = 1.0 if (digest[j] & 0x80) == 0 else -1.0
                vec[slot] += sign

        return _l2_normalize(vec)


def _l2_normalize(vec: Iterable[float]) -> List[float]:
    v = list(vec)
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain cosine similarity over two equal-length float sequences."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)
