# LEGAL_DEV_SPEC

## 1. 项目最终目标

本项目最终目标是：**离线、本地、8G 显存笔记本可运行、一键部署的法律卷宗 RAG Agent 与证据检索系统**。

目标场景：
律师上传大量扫描件 PDF，单个 PDF 可能接近 1GB，包含文字、表格和图片证据。系统需要在本地完成 OCR、页码级索引、证据检索，并支持后续基于 RAG 的带引用问答。

整体处理链路：

扫描件 PDF → 按页 OCR → 表格文本化 → 图片证据保留 → 页码级索引 → FTS5 + 向量混合检索 → RAG 带引用回答 → 本地部署。

## 2. 当前 MVP 边界

完整能力分多个 Phase 实现（见 §5 阶段规划）。当前 MVP 边界仅覆盖 **Phase B + Phase C**，目标是把 PDF 按页解析、OCR 入库并支持关键词检索。其余能力（向量、LLM、Agent、MCP、Dashboard、表格 / 图片证据、一键部署）由后续 Phase 推进，当前阶段不实现。

### 当前阶段做什么

- 输入单个 PDF 文件（含扫描件）
- 按页提取文本，扫描件按页调用本地 OCR
- 保存 document、page、chunk 数据
- 建立 SQLite FTS5 全文索引
- 支持关键词检索，返回 file_name、page_no、snippet

### 当前阶段不做什么

- 不接 LLM
- 不接 embedding / 向量检索
- 不接 MCP / Agent
- 不做 Dashboard / UI
- 不做表格抽取与图片证据存档（留给 Phase E）
- 不做一键部署
- 不改原有 scripts/ingest.py 和 scripts/query.py

## 3. 架构原则

- **检索优先**：不让大模型直接读取整份卷宗，所有上下文必须先经过检索过滤。
- **大文件按页处理**：单个 PDF 可能接近 1GB，必须按页流式处理，不允许整本一次性加载到内存或模型。
- **OCR 结果必须持久化**：扫描件 OCR 结果写入 SQLite，下次检索 / 重建索引不允许重复 OCR。
- **所有检索结果必须可定位到原文**：每条命中必须返回 file_name、page_no、source_text 或 snippet。
- **不允许跨案件污染上下文**：检索、记忆、RAG 上下文必须按案件 / 卷宗隔离，不允许 A 案件查询命中 B 案件内容。

## 4. 新增目录设计

```text
src/legal/
  __init__.py
  db.py
  models.py
  pdf_pages.py
  chunker.py
  fts_store.py
  legal_ingest.py
  legal_search.py

scripts/
  legal_ingest.py
  legal_search.py

tests/legal/
  test_legal_db.py
  test_legal_fts.py
  test_legal_ingest.py
  test_legal_search.py
```

## 5. 阶段规划

| Phase | 主题 |
|-------|------|
| Phase B | SQLite + FTS5 文本检索 |
| Phase C | PDF 按页解析 + OCR |
| Phase D | 卷宗搜索 CLI |
| Phase E | 表格和图片证据保存 |
| Phase F | 本地 embedding + Hybrid RAG |
| Phase G | LLM 带引用回答 + citation check |
| Phase H | 案件级 context memory |
| Phase I | MCP / Agent 工具化 |
| Phase J | Windows 一键部署 |

当前正在推进 **Phase C**，其余未启动的 Phase 不动代码、不预先搭骨架。

## 6. 进度跟踪

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|---------|------|
| Phase A - 项目结构与依赖准备 | [x] | - | 已完成 |
| Phase B / B1 - SQLite schema 设计 | [x] | - | 已完成 |
| Phase B / B2 - init_legal_db 实现 | [x] | - | 已完成 |
| Phase B / B3 - chunks_fts 写入 | [x] | 2026-05-11 | 已实现 chunks_fts 写入 |
| Phase B / B4 - 关键词检索 | [x] | 2026-05-11 | 已实现关键词检索并返回 file_name/page_no/snippet |
| Phase C - PDF 按页解析 + OCR | [ ] | - | 进行中（当前阶段） |
| Phase D - 卷宗搜索 CLI | [ ] | - | 待开始 |
| Phase E - 表格和图片证据保存 | [ ] | - | 待开始 |
| Phase F - 本地 embedding + Hybrid RAG | [ ] | - | 待开始 |
| Phase G - LLM 带引用回答 + citation check | [ ] | - | 待开始 |
| Phase H - 案件级 context memory | [ ] | - | 待开始 |
| Phase I - MCP / Agent 工具化 | [ ] | - | 待开始 |
| Phase J - Windows 一键部署 | [ ] | - | 待开始 |
