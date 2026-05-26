#!/usr/bin/env python
"""CLI: evidence-based legal answer with citation check.

Usage:
    .venv\\Scripts\\python.exe scripts\\legal_answer.py \\
        --query "案号是什么" --db data\\db\\legal.db --mode hybrid

    # No-evidence path (returns the hard-no answer with citations=[]):
    .venv\\Scripts\\python.exe scripts\\legal_answer.py \\
        --query "有没有外星人证据？" --db data\\db\\legal.db --mode fts

The CLI never calls an external API. ``answer_legal_question`` runs
without an ``llm_client`` (extractive fallback) and the result is then
validated by ``citation_check``. If ``--mode`` requires embeddings but
``sentence-transformers`` is missing, the run transparently downgrades
to ``fts`` rather than crashing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.answer import answer_legal_question, citation_check  # noqa: E402
from src.legal.case_context import (  # noqa: E402
    load_case_context,
    resolve_followup_query,
    save_case_context,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer a legal question with citations (no external API)."
    )
    parser.add_argument("--query", required=True, help="Legal question to answer.")
    parser.add_argument(
        "--db",
        default="data/db/legal.db",
        help="Path to the legal SQLite database (default: data/db/legal.db).",
    )
    parser.add_argument(
        "--mode",
        choices=["fts", "semantic", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k evidence to retrieve (default: 5).",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Case identifier for minimal multi-turn memory (Phase I).",
    )
    parser.add_argument(
        "--use-context",
        action="store_true",
        help="Resolve follow-up against last turn of --case-id before retrieval.",
    )
    parser.add_argument(
        "--save-context",
        action="store_true",
        help="Persist this turn (query + answer + citations) under --case-id.",
    )
    parser.add_argument(
        "--llm",
        choices=["none", "local"],
        default="none",
        help="LLM backend: 'none' (extractive fallback, default) or 'local' (llama.cpp).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a local GGUF model (overrides $LEGAL_LLM_MODEL_PATH).",
    )
    args = parser.parse_args()

    if (args.use_context or args.save_context) and not args.case_id:
        print(
            "warning: --use-context/--save-context require --case-id; ignoring.",
            file=sys.stderr,
        )

    context_used = False
    effective_query = args.query
    if args.use_context and args.case_id:
        ctx = load_case_context(args.db, args.case_id)
        if ctx:
            rewritten = resolve_followup_query(args.query, ctx)
            if rewritten != args.query:
                effective_query = rewritten
                context_used = True

    embedder = None
    if args.mode in ("semantic", "hybrid"):
        try:
            from src.legal.embeddings import (  # noqa: WPS433
                DEFAULT_ST_MODEL,
                SentenceTransformerEmbeddingProvider,
            )
            embedder = SentenceTransformerEmbeddingProvider(model=DEFAULT_ST_MODEL)
        except ImportError as e:
            print(
                f"warning: embeddings unavailable ({e}); falling back to mode=fts",
                file=sys.stderr,
            )
            args.mode = "fts"

    llm_client = None
    if args.llm == "local":
        from src.legal.local_llm import (  # noqa: WPS433
            LocalLLMUnavailable,
            build_local_llm_client,
        )
        try:
            llm_client = build_local_llm_client(args.model_path)
        except LocalLLMUnavailable as e:
            print(f"error: local LLM unavailable: {e}", file=sys.stderr)
            return 2

    result = answer_legal_question(
        args.db,
        effective_query,
        llm_client=llm_client,
        embedder=embedder,
        mode=args.mode,
        limit=args.limit,
    )
    citation_check(result)

    if args.save_context and args.case_id:
        save_case_context(args.db, args.case_id, args.query, result)

    print(f"answer={result['answer']}")
    print("citations:")
    if not result["citations"]:
        print("- (none)")
    else:
        for c in result["citations"]:
            print(
                f"- file_name={c['file_name']} "
                f"page_no={c['page_no']} "
                f"source={c.get('source', '-')} "
                f"snippet={c['snippet']}"
            )
    print(f"confidence={result['confidence']}")
    print(f"no_evidence={'true' if result['no_evidence'] else 'false'}")
    print(f"query_type={result.get('query_type', '-')}")
    print(f"context_used={'true' if context_used else 'false'}")
    if result.get("citation_issues"):
        print(f"citation_issues={','.join(result['citation_issues'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
