"""Phase J — local LLM adapter tests.

No real GGUF model required. The llama_cpp loader is monkey-patched
through :func:`src.legal.local_llm._load_llama` so we can drive the full
adapter path with a fake Llama class.
"""
from __future__ import annotations

import pytest

from src.legal import local_llm
from src.legal.local_llm import (
    LlamaCppClient,
    LocalLLMUnavailable,
    build_local_llm_client,
    resolve_model_path,
)


class _FakeLlama:
    def __init__(self, model_path, n_ctx=4096, n_threads=None):
        self.model_path = model_path
        self.n_ctx = n_ctx

    def __call__(self, prompt, max_tokens=512, temperature=0.1, stop=None):
        return {"choices": [{"text": f"FAKE::{prompt[:8]}"}]}


def _patch_llama(monkeypatch):
    monkeypatch.setattr(local_llm, "_load_llama", lambda: _FakeLlama)


# ---------------------------------------------------------------------------
# Model-path resolution
# ---------------------------------------------------------------------------


def test_resolve_model_path_missing_raises(monkeypatch):
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    with pytest.raises(LocalLLMUnavailable) as exc:
        resolve_model_path(None)
    assert "LEGAL_LLM_MODEL_PATH" in str(exc.value)
    assert "--model-path" in str(exc.value)


def test_resolve_model_path_nonexistent_path_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("LEGAL_LLM_MODEL_PATH", raising=False)
    bogus = tmp_path / "no-such-model.gguf"
    with pytest.raises(LocalLLMUnavailable) as exc:
        resolve_model_path(str(bogus))
    assert "不存在" in str(exc.value)


def test_resolve_model_path_env_var(monkeypatch, tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00")
    monkeypatch.setenv("LEGAL_LLM_MODEL_PATH", str(p))
    assert resolve_model_path(None) == str(p)


def test_resolve_model_path_explicit_overrides_env(monkeypatch, tmp_path):
    env_p = tmp_path / "env.gguf"
    env_p.write_bytes(b"\x00")
    explicit = tmp_path / "explicit.gguf"
    explicit.write_bytes(b"\x00")
    monkeypatch.setenv("LEGAL_LLM_MODEL_PATH", str(env_p))
    assert resolve_model_path(str(explicit)) == str(explicit)


# ---------------------------------------------------------------------------
# Dependency missing
# ---------------------------------------------------------------------------


def test_missing_llama_cpp_dependency_raises(monkeypatch, tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00")

    def _raise():
        raise LocalLLMUnavailable(
            "llama_cpp 未安装。请运行：\n"
            "    .venv\\Scripts\\python.exe -m pip install llama-cpp-python"
        )

    monkeypatch.setattr(local_llm, "_load_llama", _raise)
    with pytest.raises(LocalLLMUnavailable) as exc:
        LlamaCppClient(model_path=str(p))
    msg = str(exc.value)
    assert "llama_cpp" in msg
    assert "pip install llama-cpp-python" in msg


# ---------------------------------------------------------------------------
# Adapter end-to-end with fake Llama
# ---------------------------------------------------------------------------


def test_llama_client_generate_uses_fake(monkeypatch, tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00")
    _patch_llama(monkeypatch)
    c = LlamaCppClient(model_path=str(p))
    out = c.generate("案号是什么")
    assert out.startswith("FAKE::")


def test_build_local_llm_client_factory(monkeypatch, tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00")
    _patch_llama(monkeypatch)
    c = build_local_llm_client(str(p))
    assert isinstance(c, LlamaCppClient)
    assert c.model_path == str(p)


def test_llama_client_empty_choices_returns_empty(monkeypatch, tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"\x00")

    class _Empty(_FakeLlama):
        def __call__(self, *a, **kw):
            return {"choices": []}

    monkeypatch.setattr(local_llm, "_load_llama", lambda: _Empty)
    c = LlamaCppClient(model_path=str(p))
    assert c.generate("x") == ""
