"""Local LLM adapter (Phase J).

Pure adapter — no model auto-download, no external API. Designed to plug
into :func:`src.legal.answer.answer_legal_question` as an ``LLMClient``.

The only supported backend right now is ``llama_cpp`` (GGUF on CPU/GPU
via llama.cpp). The dependency is *optional*: this module imports it
lazily inside :meth:`LlamaCppClient.generate` / ``__init__`` so importing
``src.legal.local_llm`` itself never fails on a clean install.

Model path resolution order:

1. ``model_path`` passed to the constructor.
2. ``LEGAL_LLM_MODEL_PATH`` environment variable.

Neither is set ⇒ :class:`LocalLLMUnavailable` with the user-facing hint
``请设置 LEGAL_LLM_MODEL_PATH 或使用 --model-path 指定本地 GGUF 模型。``
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


_MISSING_DEP_MSG = (
    "llama_cpp 未安装。请运行：\n"
    "    .venv\\Scripts\\python.exe -m pip install llama-cpp-python"
)
_MISSING_PATH_MSG = (
    "请设置 LEGAL_LLM_MODEL_PATH 或使用 --model-path 指定本地 GGUF 模型。"
)


class LocalLLMUnavailable(RuntimeError):
    """Raised when local LLM cannot be initialized (no dep / no model)."""


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Pick the GGUF path; raise :class:`LocalLLMUnavailable` if absent."""
    candidate = model_path or os.environ.get("LEGAL_LLM_MODEL_PATH")
    if not candidate:
        raise LocalLLMUnavailable(_MISSING_PATH_MSG)
    p = Path(candidate)
    if not p.exists():
        raise LocalLLMUnavailable(
            f"本地模型文件不存在: {p}\n{_MISSING_PATH_MSG}"
        )
    return str(p)


class LlamaCppClient:
    """``LLMClient``-compatible wrapper around ``llama_cpp.Llama``.

    Construction validates the model path and the optional dependency
    eagerly so the CLI can surface a clear error before retrieval runs.
    The ``llama_cpp`` symbol is imported lazily so test suites can
    monkey-patch :func:`_load_llama` to inject a fake.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        llama_cls = _load_llama()
        kwargs: dict[str, Any] = {"model_path": self.model_path, "n_ctx": n_ctx}
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        self._llm = llama_cls(**kwargs)

    def generate(self, prompt: str) -> str:
        out = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>"],
        )
        # llama_cpp returns OpenAI-style choices; be defensive.
        choices = (out or {}).get("choices") or []
        if not choices:
            return ""
        text = choices[0].get("text") or ""
        return text.strip()


def _load_llama():
    """Lazy import; tests monkey-patch this to inject a fake Llama class."""
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise LocalLLMUnavailable(_MISSING_DEP_MSG) from exc
    return Llama


def build_local_llm_client(model_path: Optional[str] = None) -> LlamaCppClient:
    """Public factory used by the CLI. Always raises LocalLLMUnavailable on failure."""
    return LlamaCppClient(model_path=model_path)
