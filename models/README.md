# models/

本目录用于**本地存放**离线模型文件。本项目默认不自动下载任何大模型，
也不在仓库内提交模型权重，请按需要自行准备。

## 目录约定

- `models/llm/` — 本地 LLM GGUF 文件（llama.cpp 兼容），例如
  `qwen2.5-7b-instruct.Q4_K_M.gguf`。供 `legal_answer.py --llm local
  --model-path models\\llm\\xxx.gguf` 使用。
- `models/embedding/` — sentence-transformers 模型权重（如需离线缓存）。
- `models/ocr/` — RapidOCR / Tesseract 等 OCR 模型文件（如需）。

## 不做的事

- 不自动下载大模型；
- 不在仓库内提交权重；
- 不依赖联网。

## 离线部署

将上述目录中的模型文件随项目一起拷贝到目标机器，
设置环境变量 `LEGAL_LLM_MODEL_PATH` 指向 GGUF 文件，
或在 CLI 中用 `--model-path` 显式传入。
