# LEGAL_DEV_SPEC

## 1. 项目最终目标

本项目最终目标是：**离线、本地、8G 显存笔记本可运行、一键部署的法律卷宗 RAG Agent 与证据检索系统**。

目标场景：

律师上传大量法律卷宗 PDF。卷宗通常是扫描件，单个 PDF 可能接近 1GB，内容包含文字、表格和图片证据。系统需要在本地完成 OCR、页码级索引、证据检索，并支持后续基于 RAG 的带引用问答。

最终整体链路：

```text
扫描件 PDF
→ 按页处理
→ 本地 OCR
→ 表格文本化
→ 图片证据保留
→ 页码级索引
→ FTS5 + 向量混合检索
→ RAG 带引用回答
→ 案件级上下文管理
→ MCP / Agent 工具化
→ Windows 一键部署
```

核心原则：

大模型不直接读取整份卷宗。检索系统先定位证据片段，大模型只基于 top-k 证据片段做摘要、问答、时间线整理和引用校验。

---

## 2. 当前 MVP 边界

当前 MVP 目标是先跑通一个**法律卷宗检索纵向闭环**：

```text
已 OCR 页面文本 / 单个 PDF
→ page_no + page_text
→ documents / pages / chunks / chunks_fts
→ FTS5 检索
→ CLI 返回 file_name / page_no / snippet
```

当前阶段重点不是完整 OCR，而是把后续 OCR 得到的 `page_text` 正确入库、切分、索引和检索。

当前 MVP 采用两种输入路径：

1. **已 OCR 页面文本输入**
   - 用文本文件夹模拟扫描件 OCR 结果。
   - 先验证主链路：page_text → SQLite → FTS5 → search。

2. **PDF 输入预留**
   - 后续接真实 OCR。
   - 文字版 PDF 可走文本层提取。
   - 扫描件 PDF 后续走本地 OCR。

当前阶段必须完成：

- 初始化法律检索数据库；
- 保存 document、page、chunk；
- 建立 SQLite FTS5 全文索引；
- 支持页面文本导入；
- 支持关键词检索；
- 输出 `file_name`、`page_no`、`snippet`；
- 提供 `legal_ingest.py` 和 `legal_search.py` 两个 CLI。

当前阶段暂缓：

- 不接 LLM；
- 不接 embedding / 向量检索；
- 不接 MCP / Agent；
- 不做 Dashboard / UI；
- 不做复杂表格结构化；
- 不做图片证据裁剪和存档；
- 不做 Windows 安装包；
- 不改原有 `scripts/ingest.py` 和 `scripts/query.py`。

---

## 3. 架构原则

### 3.1 检索优先

系统先通过 OCR、FTS5、metadata、向量索引定位证据。  
LLM 只在检索结果之上做解释和组织，不直接读取整份卷宗。

### 3.2 大文件按页处理

单个 PDF 可能接近 1GB。系统必须按页处理，避免整本 PDF 一次性进入内存或模型上下文。

### 3.3 OCR 结果持久化

每页 OCR 结果必须写入 SQLite。后续检索、重建索引、生成回答时复用已有 OCR 文本，避免重复 OCR。

### 3.4 结果可追溯

所有检索结果必须能定位回原文：

```text
file_name
page_no
chunk_id
source_text 或 snippet
```

### 3.5 案件隔离

后续引入多案件时，所有检索、上下文、记忆和日志都必须按 `case_id` 隔离，防止 A 案件内容进入 B 案件结果。

### 3.6 分阶段增强

当前先做 FTS5 检索闭环。  
后续再接 OCR、图片、表格、embedding、RAG、MCP、部署。

---

## 4. 当前新增目录设计

```text
src/legal/
  __init__.py
  db.py
  fts_store.py
  ocr.py
  chunker.py
  legal_ingest.py
  legal_search.py

scripts/
  legal_ingest.py
  legal_search.py

tests/legal/
  test_legal_db.py
  test_legal_fts.py
  test_legal_mvp.py
```

---

## 5. 模块职责

### src/legal/db.py

负责 SQLite 初始化和建表。

当前已实现：

- `documents`
- `pages`
- `chunks`
- `chunks_fts`

### src/legal/fts_store.py

负责 FTS5 写入与关键词检索。

当前已实现：

- 写入 chunk 到 `chunks_fts`
- 根据关键词查询
- 返回 `file_name`、`page_no`、`snippet`

### src/legal/ocr.py

负责 OCR 抽象。

当前 MVP 不直接接重型 OCR，只定义统一接口：

```text
OCRProvider
```

第一版实现：

```text
SimpleTextOCRProvider
```

它读取已 OCR 好的文本，用来模拟真实 OCR 输出，保证主链路先跑通。

后续再新增：

```text
TesseractOCRProvider
PaddleOCRProvider
```

### src/legal/chunker.py

负责把页面文本切成 chunk。

输入：

```text
doc_id
page_no
page_text
```

输出：

```text
chunk_id
doc_id
page_no
chunk_text
start_char
end_char
```

### src/legal/legal_ingest.py

负责法律卷宗导入主流程。

当前 MVP 需要支持：

```text
ingest_pages(db_path, file_name, pages)
```

其中 `pages` 是：

```python
[
    {"page_no": 1, "text": "..."},
    {"page_no": 2, "text": "..."}
]
```

导入流程：

```text
init_legal_db
→ 插入 documents
→ 插入 pages
→ page_text 切 chunk
→ 插入 chunks
→ 写入 chunks_fts
```

### src/legal/legal_search.py

负责法律卷宗关键词检索。

当前 MVP 提供：

```text
search_legal_chunks(db_path, query, limit)
```

返回：

```python
[
    {
        "file_name": "...",
        "page_no": 1,
        "snippet": "..."
    }
]
```

### scripts/legal_ingest.py

CLI 导入入口。

当前 MVP 支持从文本文件夹导入，模拟 OCR 结果：

```powershell
python scripts/legal_ingest.py --pages-dir test_docs/legal_pages --file-name case.pdf --db data/db/legal.db
```

目录示例：

```text
test_docs/legal_pages/
  page_001.txt
  page_002.txt
  page_003.txt
```

### scripts/legal_search.py

CLI 搜索入口：

```powershell
python scripts/legal_search.py --query "借款" --db data/db/legal.db
```

输出必须包含：

```text
file_name
page_no
snippet
```

---

## 6. 数据表设计

### documents

| 字段 | 说明 |
|---|---|
| doc_id | 文档 ID |
| file_name | 文件名 |
| file_path | 文件路径，可为空 |
| file_hash | 文件 hash，可由 file_name + 内容生成 |
| created_at | 导入时间 |

### pages

| 字段 | 说明 |
|---|---|
| page_id | 页 ID |
| doc_id | 文档 ID |
| page_no | 页码 |
| page_text | 页面 OCR / 文本层内容 |

### chunks

| 字段 | 说明 |
|---|---|
| chunk_id | chunk ID |
| doc_id | 文档 ID |
| page_no | 页码 |
| chunk_text | chunk 文本 |
| start_char | 起始字符位置 |
| end_char | 结束字符位置 |

### chunks_fts

SQLite FTS5 虚拟表，用于关键词检索。

| 字段 | 说明 |
|---|---|
| chunk_id | chunk ID |
| doc_id | 文档 ID |
| file_name | 文件名 |
| page_no | 页码 |
| chunk_text | 检索文本 |

---

## 7. 当前 MVP 运行方式

### 7.1 准备模拟 OCR 页面文本

```powershell
New-Item -ItemType Directory -Force test_docs\legal_pages

"张三向李四借款50000元，双方约定2023年5月6日还款。" | Set-Content -Encoding utf8 test_docs\legal_pages\page_001.txt

"本页为银行转账记录，显示李四向张三转账50000元。" | Set-Content -Encoding utf8 test_docs\legal_pages\page_002.txt
```

### 7.2 导入

```powershell
.venv\Scripts\python.exe scripts\legal_ingest.py --pages-dir test_docs\legal_pages --file-name case.pdf --db data\db\legal.db
```

预期输出：

```text
ingested file_name=case.pdf pages=2 chunks=...
```

### 7.3 搜索

```powershell
.venv\Scripts\python.exe scripts\legal_search.py --query "借款" --db data\db\legal.db
```

预期输出：

```text
file_name=case.pdf
page_no=1
snippet=...借款...
```

---

## 8. 阶段规划

| Phase | 主题 | 说明 |
|---|---|---|
| Phase A | 项目结构与 spec | 已完成 |
| Phase B | SQLite + FTS5 文本检索 | 已完成基础能力 |
| Phase C | 页面文本导入闭环 | 当前阶段，用文本模拟 OCR 输出 |
| Phase D | 卷宗搜索 CLI | 当前阶段，和 Phase C 一次跑通 |
| Phase E | 真实 OCR 接入 | Tesseract / PaddleOCR，本地 OCR |
| Phase F | 表格和图片证据保存 | 页面截图、图片证据、表格文本化 |
| Phase G | 本地 embedding + Hybrid RAG | 本地 embedding + FTS5 / BM25 / 向量融合 |
| Phase H | LLM 带引用回答 + citation check | 本地 LLM 基于检索片段回答 |
| Phase I | 案件级 context memory | 当前案件、上一轮查询、选中证据 |
| Phase J | MCP / Agent 工具化 | 暴露 legal_search_tool / citation_check_tool |
| Phase K | Windows 一键部署 | PyInstaller / start.bat / self_check |

---

## 9. 进度跟踪

状态说明：

- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成

| 任务 | 状态 | 完成日期 | 备注 |
|---|---|---|---|
| Phase A - 项目结构与依赖准备 | [x] | - | 已完成 |
| Phase B / B1 - SQLite schema 设计 | [x] | - | 已完成 |
| Phase B / B2 - init_legal_db 实现 | [x] | - | 已完成 |
| Phase B / B3 - chunks_fts 写入 | [x] | 2026-05-11 | 已实现 chunks_fts 写入 |
| Phase B / B4 - 关键词检索 | [x] | 2026-05-11 | 已实现关键词检索并返回 file_name/page_no/snippet |
| Phase C / D - 页面文本导入 + 搜索 CLI 最小闭环 | [~] | - | 当前推进 |
| Phase E - 真实 OCR 接入 | [ ] | - | 待开始 |
| Phase F - 表格和图片证据保存 | [ ] | - | 待开始 |
| Phase G - 本地 embedding + Hybrid RAG | [ ] | - | 待开始 |
| Phase H - LLM 带引用回答 + citation check | [ ] | - | 待开始 |
| Phase I - 案件级 context memory | [ ] | - | 待开始 |
| Phase J - MCP / Agent 工具化 | [ ] | - | 待开始 |
| Phase K - Windows 一键部署 | [ ] | - | 待开始 |

---

## 10. 当前 Claude 执行边界

当前只允许 Claude 实现：

```text
Phase C / D - 页面文本导入 + 搜索 CLI 最小闭环
```

允许新增或修改：

```text
src/legal/ocr.py
src/legal/chunker.py
src/legal/legal_ingest.py
src/legal/legal_search.py
scripts/legal_ingest.py
scripts/legal_search.py
tests/legal/test_legal_mvp.py
LEGAL_DEV_SPEC.md
```

当前实现目标：

```text
已 OCR 页面文本文件夹
→ ingest_pages()
→ SQLite documents/pages/chunks/chunks_fts
→ search_legal_chunks()
→ legal_search.py CLI
→ file_name/page_no/snippet
```

测试目标：

```powershell
pytest tests/legal/test_legal_mvp.py -v
```

测试通过后，才能把 Phase C / D 状态改为 `[x]`。

---

## 11. Claude 执行完成后的验收

Claude 完成 Phase C / D 后，只输出：

1. 修改了哪些文件；
2. 当前 MVP 怎么运行；
3. 我应该运行哪条 pytest 命令；
4. pytest 通过后，CLI 验证命令是什么。

用户本地运行 pytest。  
pytest 通过后，再更新本文件进度表。