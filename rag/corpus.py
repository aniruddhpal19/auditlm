"""Load the passage store (chunks.jsonl) for retrieval.

Read-only over the existing corpus — RAG never re-chunks. Each record carries
`id`, `text`, `citation`, `source`, `doc_type`, `title`, `url`, `metadata`.
"""

from __future__ import annotations

import json
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKS = RAG_DIR.parent / "data" / "corpus" / "chunks.jsonl"


def load_chunks(path: Path | str = DEFAULT_CHUNKS) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build the corpus first (the segmenter "
            f"writes data/corpus/chunks.jsonl).")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(chunks: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for c in chunks}


# How many times a chunk's title/subject is repeated in the BM25 blob (a term-frequency
# boost). A standard's own subject lives in its `title` ("Consideration of Fraud…") but its
# body intro chunks are cross-reference-heavy and don't restate it, so a natural "which
# standard governs fraud?" lookup missed it entirely. Indexing the title fixes that; the
# weight is the lowest that surfaces subject lookups without disturbing working queries.
TITLE_WEIGHT = 2


def lexical_blob(chunk: dict) -> str:
    """Text used for LEXICAL (BM25) indexing — the passage body, its citation identifiers,
    AND the standard's title/subject (repeated TITLE_WEIGHT times). Identifiers let an
    exact-identifier query ("AS 2110", "ASC 842") surface the passage; the title lets a
    natural subject query ("which standard governs fraud?") surface it even though the body
    intro chunks don't restate their own subject. (Used by BM25 only; dense embeds `text`.)"""
    md = chunk.get("metadata") or {}
    ids = [chunk.get("citation"), md.get("standard_id"), md.get("paragraph"),
           md.get("asc_topic") and f"ASC {md['asc_topic']}",
           md.get("regulation"), md.get("section"),
           md.get("sab_number") and f"SAB {md['sab_number']}"]
    prefix = " ".join(str(x) for x in ids if x)
    base = f"{prefix}\n{chunk.get('text', '')}" if prefix else chunk.get("text", "")
    title = (chunk.get("title") or "").strip()
    return base + (f"\n{title}" * TITLE_WEIGHT if title else "")
