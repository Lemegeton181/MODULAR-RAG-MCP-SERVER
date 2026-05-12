"""Fixed-size character-window chunker for legal page text."""
from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_CHUNK_SIZE = 500


def chunk_page_text(
    doc_id: str,
    page_no: int,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    """Split one page's text into fixed-size character chunks.

    Each chunk dict contains:
        chunk_id, doc_id, page_no, chunk_text, start_char, end_char.

    Whitespace-only chunks are skipped.
    """
    if not text:
        return []

    chunks: List[Dict[str, Any]] = []
    idx = 0
    pos = 0
    length = len(text)
    while pos < length:
        end = min(pos + chunk_size, length)
        segment = text[pos:end]
        if segment.strip():
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_p{page_no}_c{idx}",
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "chunk_text": segment,
                    "start_char": pos,
                    "end_char": end,
                }
            )
            idx += 1
        pos = end
    return chunks
