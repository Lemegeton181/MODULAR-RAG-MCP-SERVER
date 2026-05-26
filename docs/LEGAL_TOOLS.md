# LEGAL_TOOLS — Agent / MCP-ready tool layer (Phase K)

A thin, framework-agnostic wrapper around the legal RAG stack so it can
be driven by any tool-using LLM (Claude tool use / OpenAI function calling
/ MCP server / homegrown Agent loop) without leaking implementation details.

## 5 tools

| Name | Purpose |
| --- | --- |
| `legal_search_tool` | Top-k hybrid retrieval. Returns file/page/snippet, no LLM. |
| `legal_answer_tool` | Evidence-based answer + citation check, with optional local LLM and case_id memory. |
| `legal_reindex_fields_tool` | Re-extract `legal_fields` for all docs after schema upgrades. |
| `legal_eval_tool` | Golden Set retrieval-proxy metrics (`hit@1` / `hit@3` / `page_hit_rate` / `keyword_hit_rate`). |
| `legal_self_check_tool` | PASS/WARN/FAIL environment check (Python, SQLite FTS5, tables, optional deps, LLM model). |

## I/O contract

Every tool takes a single `args: dict` and returns a JSON-serializable
`dict` containing **at least** `ok: bool`. On failure: `{"ok": false,
"error": "<ExceptionClass>: <message>"}`. Tools never raise, never print.

### legal_search_tool — example

Input:
```json
{"query": "惩罚性赔偿", "db_path": "data/db/legal.db", "mode": "hybrid", "limit": 3}
```
Output (truncated):
```json
{
  "ok": true,
  "results": [
    {"file_name": "case.pdf", "page_no": 2, "snippet": "...惩罚性赔偿...",
     "score": 10.8, "source": "field",
     "matched_field_name": "二倍工资金额", "matched_field_value": "23453元"}
  ],
  "mode": "hybrid", "downgraded": false
}
```

### legal_answer_tool — example

Input:
```json
{"query": "为什么申请人认为裁决应撤销？", "db_path": "data/db/legal.db",
 "mode": "hybrid", "case_id": "demo", "save_context": true}
```
Output (shape):
```json
{
  "ok": true,
  "answer": "...",
  "citations": [{"file_name": "...", "page_no": 2, "snippet": "...", "source": "fts"}],
  "confidence": "medium",
  "no_evidence": false,
  "query_type": "evidence_search",
  "citation_issues": [],
  "context_used": false,
  "mode": "hybrid"
}
```

### legal_self_check_tool — example

Output:
```json
{"ok": true, "exit_code": 0, "summary": "summary: PASS=7 WARN=2 FAIL=0",
 "checks": [{"level": "PASS", "line": "python — 3.11.9"}, ...]}
```

## How to wire it in

- **MCP server**: in `tools/list`, return `LEGAL_TOOL_SCHEMAS`. In
  `tools/call`, route `name` + `arguments` into
  `run_legal_tool(name, arguments)`.
- **Claude / OpenAI function calling**: feed each schema as a tool
  declaration; on tool_use, call `run_legal_tool`.
- **Internal Agent loop**: same as above — `run_legal_tool` is the only
  entry point needed.

## Why a thin wrapper, not an MCP server here

- Decouples the legal pipeline from any single tool protocol; today MCP,
  tomorrow plain function calling — both reuse the same code.
- Keeps test surface tiny — wrappers can be tested without a stdio
  server, an event loop, or a network mock.
- Avoids reinventing what `src/mcp_server/` already owns for the
  modular-RAG side. Phase K intentionally does NOT touch
  `src/mcp_server/`.

## TODO (next phase candidates)

- Wire `LEGAL_TOOL_SCHEMAS` into an MCP server entry under
  `src/mcp_server/`, exposing `tools/list` + `tools/call`.
- Add streaming variants for `legal_answer_tool` once a real local LLM
  is in place.
- Per-tool latency / error telemetry hooks for observability.
