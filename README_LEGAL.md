# Legal RAG — 本地法律文书检索与带引用问答（MVP）

## 一句话

一个**完全离线**的法律文书检索 + 带页码引用问答系统，PDF/扫描件 → 字段 + 证据，
回答只基于检索到的材料，不联网、不编造。

## 系统解决什么问题

律师/法务/合规人员需要在卷宗里快速回答：
"案号是什么？""为什么申请人认为裁决应撤销？""有没有转账凭证？"
传统做法靠人翻 PDF。本系统把卷宗变成可检索、可引用、可校验的知识库，
**回答必须落到页码**，无依据时直接拒答。

## 当前已实现能力

- PDF 文本/扫描页面分流（page_type=text/scanned/mixed/table_like）；
- 真实轻量 OCR（RapidOCR）；
- SQLite FTS5（trigram，CJK 友好）+ 本地 embedding + Hybrid RRF 融合；
- 动态法律字段体系（FIELD_DEFS + EVIDENCE_KEYWORDS）+ 字段优先 overlay；
- Golden Set 评测（hit@1/hit@3/page_hit_rate/keyword_hit_rate）；
- Evidence-based answer + citation_check（无证据强制拒答，禁止幻觉引用）；
- Reasoning 路由：「为什么…」类问题归到 evidence_search，不会误答成「申请人：xxx」；
- 案件级最小多轮上下文（case_id + 跟随问改写）；
- 本地 LLM 适配器（llama.cpp / GGUF），可选；不联网。

## 技术架构链路

```
PDF / 图片
   ↓ pdfplumber / RapidOCR
pages (page_type + page_image_path)
   ↓ 切分
chunks
   ↓
FTS5 (trigram)  ─┐
embedding 向量库 ─┼─► hybrid_search (RRF)
field overlay   ─┘
   ↓
answer_legal_question (extractive | local LLM)
   ↓
citation_check (file_name / page_no / snippet)
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# 可选（启用本地 LLM）：
pip install llama-cpp-python
```

## 运行（典型流程）

```powershell
python scripts\ingest.py --input test_docs --db data\db\legal.db
python scripts\legal_reindex_fields.py --db data\db\legal.db
python scripts\legal_answer.py --query "案号是什么" --db data\db\legal.db --mode hybrid
python scripts\legal_answer.py --query "为什么申请人认为裁决应撤销？" --db data\db\legal.db --mode hybrid
python scripts\legal_eval.py --db data\db\legal.db
python scripts\legal_self_check.py --db data\db\legal.db
```

一键演示：`start_legal_demo.bat`

## 真实验证命令

```powershell
:: 拒答（无依据）
python scripts\legal_answer.py --query "外星人入侵地球的证据" --db data\db\legal.db --mode hybrid

:: 多轮（case_id）
python scripts\legal_answer.py --query "为什么申请人认为裁决应撤销？" --db data\db\legal.db --mode hybrid --case-id demo --save-context
python scripts\legal_answer.py --query "还有哪些理由？" --db data\db\legal.db --mode hybrid --case-id demo --use-context --save-context

:: 本地 LLM（可选）
python scripts\legal_answer.py --query "..." --db data\db\legal.db --mode hybrid --llm local --model-path models\llm\xxx.gguf
```

## legal_eval 指标

- `hit@1 / hit@3` — Golden Set 的 top-1 / top-3 文件命中率；
- `page_hit_rate` — top-3 内是否命中正确页码；
- `keyword_hit_rate` — top-3 snippet 是否覆盖参考关键词；
- `allow_no_hit` — 标记本就该拒答的 query，命中即视为正例。

## 本地模型说明

- LLM：放 `models/llm/*.gguf`，通过 `LEGAL_LLM_MODEL_PATH` 或 `--model-path` 指向；
- embedding：sentence-transformers 模型可缓存到 `models/embedding/`；
- OCR：RapidOCR 自带轻量模型。

## 离线部署边界

- 不联网、不调用第三方 API；
- 不自动下载模型，权重不入仓；
- 仅依赖本机 SQLite / llama.cpp / sentence-transformers / RapidOCR。

## 当前限制

- 单机 SQLite，不适合大规模并发；
- 暂未做 reranker；
- 暂无 UI 和 MCP 工具暴露；
- case_context 仅最小规则改写，不做语义记忆。

## 下一步 TODO

- reranker（bge-reranker / cross-encoder）；
- MCP tool（`legal_search_tool` / `citation_check_tool`）；
- Streamlit / Gradio 演示 UI；
- PyInstaller / 一键 installer；
- 多案件 case_id 隔离（doc_id ↔ case_id 映射）。

## 1 分钟讲法（面试版）

> 我做了一个**完全离线**的法律文书 RAG MVP。链路是 PDF/OCR → SQLite FTS5 + 本地
> embedding → RRF hybrid，再叠一层动态法律字段 overlay。回答层强制引用页码，
> 没有命中就拒答；问"为什么"类问题时路由到证据检索，不会用「申请人是谁」糊弄。
> 案件级最小多轮上下文用一张 SQLite 表实现。LLM 用 llama.cpp 接 GGUF，纯本地。
> 评测用 Golden Set 跑 hit@k / page_hit_rate / keyword_hit_rate，目前命中率 0.73+。
> 全程不联网、不下载模型、不依赖外部 API。
