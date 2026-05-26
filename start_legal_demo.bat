@echo off
REM start_legal_demo.bat — local demo launcher (Phase J).
REM Activates .venv, runs self-check, prints the common command set.
REM Does NOT start any dashboard / UI / external service.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [warn] .venv not found; using system python
)

echo === self check ===
python scripts\legal_self_check.py --db data\db\legal.db
echo.

echo === common commands ===
echo PDF ingest:
echo     python scripts\ingest.py --input test_docs --db data\db\legal.db
echo Field reindex:
echo     python scripts\legal_reindex_fields.py --db data\db\legal.db
echo Hybrid search:
echo     python scripts\query.py --db data\db\legal.db --query "案号"
echo Evidence-based answer (extractive):
echo     python scripts\legal_answer.py --query "为什么申请人认为裁决应撤销？" --db data\db\legal.db --mode hybrid
echo Evidence-based answer (local LLM):
echo     python scripts\legal_answer.py --query "为什么申请人认为裁决应撤销？" --db data\db\legal.db --mode hybrid --llm local --model-path models\llm\model.gguf
echo Golden-set evaluation:
echo     python scripts\legal_eval.py --db data\db\legal.db
echo.
