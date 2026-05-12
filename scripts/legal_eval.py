"""Golden Set evaluation runner for the legal retrieval store.

This is a **retrieval proxy** evaluator. We do NOT call an LLM and do
NOT score natural-language answers. We score whether retrieval surfaces
the page / keyword that a human-curated answer would have to live on:

    hit@1               result[0] is a hit
    hit@3               any of result[0..2] is a hit
    page_hit_rate       fraction of cases where expected_pages contains
                        result[0].page_no (None means N/A — excluded)
    keyword_hit_rate    fraction of cases where any expected_keyword
                        appears in any of the top-3 snippets/values

A "hit" for hit@1/hit@3 is defined as either (a) the result's snippet
or matched_field_value contains an expected_keyword, or (b) its page_no
is in expected_pages.

These numbers are useful for tracking *retrieval health* under code
changes — they are NOT a substitute for end-to-end QA accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.legal.embeddings import (  # noqa: E402
    DEFAULT_ST_MODEL,
    ST_INSTALL_HINT,
    SentenceTransformerEmbeddingProvider,
)
from src.legal.hybrid_search import hybrid_search  # noqa: E402


def _load_golden(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def _result_hit(
    result: Dict[str, Any],
    expected_keywords: List[str],
    expected_pages: Optional[List[int]],
) -> bool:
    hay = " ".join(
        [
            result.get("snippet", "") or "",
            result.get("matched_field_value", "") or "",
            result.get("matched_field_name", "") or "",
        ]
    )
    if expected_keywords and any(kw in hay for kw in expected_keywords):
        return True
    if expected_pages and result.get("page_no") in expected_pages:
        return True
    return False


def evaluate(
    db_path: str,
    cases: List[Dict[str, Any]],
    *,
    mode: str = "hybrid",
    limit: int = 5,
    embedder: Any = None,
    use_query_understanding: bool = True,
) -> Dict[str, Any]:
    """Run the golden set and return a metrics dict.

    Cases marked ``allow_no_hit: true`` (e.g. evidence queries whose
    target evidence is not actually present in the corpus) are treated
    specially:

    * If they DO hit, they count normally in hit@1 / hit@3 /
      keyword_hit_rate — we never penalize real ability.
    * If they DON'T hit, they are excluded from the denominator of the
      rate metrics and listed in ``optional_cases_no_hit`` instead of
      ``failed_cases``. This prevents corpus-specific absences from
      contaminating retrieval health numbers.

    Rate metrics therefore reflect only the cases the corpus actually
    can be expected to answer. The total case count remains visible.
    """
    total = len(cases)
    hit1 = 0
    hit3 = 0
    page_hit_total = 0
    page_hit_eligible = 0
    keyword_hit = 0
    scored_cases = 0  # denominator that excludes optional-no-hit cases
    failed: List[Dict[str, Any]] = []
    optional_no_hit: List[Dict[str, Any]] = []

    for case in cases:
        results = hybrid_search(
            db_path,
            case["query"],
            embedder=embedder,
            mode=mode,
            limit=limit,
            use_query_understanding=use_query_understanding,
        )

        top1 = results[:1]
        top3 = results[:3]

        expected_keywords = case.get("expected_keywords") or []
        expected_pages = case.get("expected_pages")  # None or list[int]
        allow_no_hit = bool(case.get("allow_no_hit", False))

        is_hit1 = any(_result_hit(r, expected_keywords, expected_pages) for r in top1)
        is_hit3 = any(_result_hit(r, expected_keywords, expected_pages) for r in top3)
        is_kw = any(
            any(
                kw in (r.get("snippet", "") or "")
                or kw in (r.get("matched_field_value", "") or "")
                for kw in expected_keywords
            )
            for r in top3
        )

        # Optional cases that didn't hit are excluded from rate metrics.
        if allow_no_hit and not is_hit3:
            optional_no_hit.append(
                {
                    "id": case.get("id", case["query"]),
                    "query": case["query"],
                    "reason": case.get("notes")
                    or "allow_no_hit=true; not present in current corpus",
                }
            )
            continue

        scored_cases += 1
        if is_hit1:
            hit1 += 1
        if is_hit3:
            hit3 += 1
        if expected_pages:
            page_hit_eligible += 1
            if top1 and top1[0].get("page_no") in expected_pages:
                page_hit_total += 1
        if is_kw:
            keyword_hit += 1

        if not is_hit3:
            failed.append(
                {
                    "id": case.get("id", case["query"]),
                    "query": case["query"],
                    "expected_keywords": expected_keywords,
                    "expected_pages": expected_pages,
                    "top3": [
                        {
                            "file_name": r.get("file_name"),
                            "page_no": r.get("page_no"),
                            "source": r.get("source"),
                            "snippet": (r.get("snippet") or "")[:120],
                            "matched_field_name": r.get("matched_field_name"),
                            "matched_field_value": r.get("matched_field_value"),
                        }
                        for r in top3
                    ]
                    or [],
                }
            )

    denom = scored_cases or 1  # guard div-by-zero when every case is optional
    return {
        "total": total,
        "scored": scored_cases,
        "hit@1": hit1 / denom if scored_cases else 0.0,
        "hit@3": hit3 / denom if scored_cases else 0.0,
        "page_hit_rate": (
            page_hit_total / page_hit_eligible if page_hit_eligible else None
        ),
        "keyword_hit_rate": keyword_hit / denom if scored_cases else 0.0,
        "failed_cases": failed,
        "optional_cases_no_hit": optional_no_hit,
        "metric_kind": "retrieval_proxy",
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"total={report['total']}")
    print(f"scored={report['scored']}")
    print(f"hit@1={report['hit@1']:.3f}")
    print(f"hit@3={report['hit@3']:.3f}")
    if report["page_hit_rate"] is None:
        print("page_hit_rate=n/a (no cases with expected_pages)")
    else:
        print(f"page_hit_rate={report['page_hit_rate']:.3f}")
    print(f"keyword_hit_rate={report['keyword_hit_rate']:.3f}")
    print(f"metric_kind={report['metric_kind']} (retrieval proxy, not QA accuracy)")
    optional = report.get("optional_cases_no_hit") or []
    print(f"optional_cases_no_hit={len(optional)} (excluded from rate metrics)")
    for oc in optional:
        print(f"  ~ {oc['id']}: {oc['query']!r}  reason={oc['reason']!r}")
    print(f"failed_cases={len(report['failed_cases'])}")
    for fc in report["failed_cases"]:
        print(f"  - {fc['id']}: {fc['query']!r}")
        for i, t in enumerate(fc["top3"], start=1):
            print(
                f"      [{i}] p{t['page_no']} src={t['source']} "
                f"field={t['matched_field_name']} "
                f"snippet={t['snippet']!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the legal retrieval store on a golden set."
    )
    parser.add_argument("--db", required=True, help="Path to the legal SQLite DB.")
    parser.add_argument(
        "--golden",
        default="evaluation/legal_golden_queries.jsonl",
        help="Path to the golden JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=["fts", "semantic", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid).",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--st-model",
        default=DEFAULT_ST_MODEL,
        help="sentence-transformers model for --mode semantic / hybrid.",
    )
    parser.add_argument(
        "--no-query-understanding",
        dest="use_query_understanding",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object instead of a human report.",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"error: golden file not found: {golden_path}", file=sys.stderr)
        return 2
    cases = _load_golden(golden_path)

    embedder = None
    if args.mode in ("semantic", "hybrid"):
        try:
            embedder = SentenceTransformerEmbeddingProvider(model=args.st_model)
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            print(f"hint: {ST_INSTALL_HINT}", file=sys.stderr)
            return 2

    report = evaluate(
        args.db,
        cases,
        mode=args.mode,
        limit=args.limit,
        embedder=embedder,
        use_query_understanding=args.use_query_understanding,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
