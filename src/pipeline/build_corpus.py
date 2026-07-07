"""Corpus pipeline: normalize -> clean -> dedup -> filter -> segment.

Reads the per-source JSONL corpora, unifies them into one schema, removes exact
and near-duplicate records, drops empty/tiny/garbled fragments, and segments long
records into overlapping passages that keep their citation/section metadata (for
RAG). Emits both a full-document corpus and a chunked passage store, plus token
counts from the Qwen2.5 tokenizer, and writes the corpus statistics report.

    python -m src.pipeline.build_corpus

Outputs:
    data/corpus/unified.jsonl   normalized, deduped, filtered full documents
    data/corpus/chunks.jsonl    overlapping passages with citation metadata
    CORPUS_STATS.md             the corpus statistics report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zlib
from pathlib import Path

import polars as pl
from tokenizers import Tokenizer

# Fixed universal-hashing coefficients so MinHash (and thus dedup) is deterministic
# across runs — reproducibility is a first-class goal.
_NUM_PERM = 48
_PRIME = (1 << 61) - 1
_COEFFS = [(r.randrange(1, _PRIME), r.randrange(0, _PRIME))
           for r in [random.Random(42)] for _ in range(_NUM_PERM)]

SOURCES = [
    "pcaob_standards", "sec_filings", "sec_regulations",
    "sec_sab", "fasb_asc_topics", "gao_yellowbook",
]
CORE = {"id", "source", "doc_type", "title", "citation", "text", "url",
        "metadata", "collected_at"}
TOKENIZER_PATH = Path("data/raw/tokenizer/qwen2.5_tokenizer.json")

CHUNK_TOKENS = 800       # target passage size
CHUNK_OVERLAP = 120      # passage overlap (context continuity for RAG)
MIN_TOKENS = 12          # quality floor — drop fragments below this


def normalize_record(r: dict) -> dict:
    """Map a source record to the unified schema; fold extras into metadata."""
    meta = dict(r.get("metadata") or {})
    for k, v in r.items():
        if k not in CORE:
            meta[k] = v
    title = r.get("title") or r.get("section_title") or r.get("chapter_title")
    return {
        "id": r["id"],
        "source": r["source"],
        "doc_type": r["doc_type"],
        "title": title,
        "citation": r.get("citation"),
        "text": re.sub(r"[ \t]+", " ", (r["text"] or "")).strip(),
        "url": r.get("url"),
        "metadata": meta,
        "collected_at": r.get("collected_at"),
    }


def load_unified(corpus_dir: Path) -> list[dict]:
    recs = []
    for name in SOURCES:
        path = corpus_dir / f"{name}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                recs.append(normalize_record(json.loads(line)))
    return recs


# --- dedup ----------------------------------------------------------------
def _norm_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingles: set[str]) -> tuple[int, ...]:
    base = [zlib.crc32(s.encode()) for s in shingles]  # deterministic shingle hashes
    return tuple(min((a * h + b) % _PRIME for h in base) for a, b in _COEFFS)


def dedup(recs: list[dict]) -> tuple[list[dict], dict]:
    """Remove exact duplicates everywhere, then near-duplicates within EDGAR 10-Ks.

    Exact dedup (normalized-text hash) is safe for all sources. Near-dup (MinHash-
    LSH, Jaccard >= 0.92) is applied **only to 10-K filing records**, where filers
    repeat boilerplate heavily — it is NOT applied to the standards (PCAOB AS,
    GAGAS, ASC, S-X/S-K, SABs), whose short, formulaic paragraphs are distinct
    authoritative units that a similarity test would wrongly collapse.
    """
    seen_exact: set[str] = set()
    exact_removed = 0
    staged = []
    for r in recs:
        h = hashlib.sha1(_norm_for_hash(r["text"]).encode()).hexdigest()
        if h in seen_exact:
            exact_removed += 1
            continue
        seen_exact.add(h)
        staged.append(r)

    edgar = [i for i, r in enumerate(staged) if r["doc_type"] == "10-K"]
    bands = 12          # 4 rows per band over _NUM_PERM (48) permutations
    rows = _NUM_PERM // bands
    sigs = {i: _minhash(_shingles(staged[i]["text"])) for i in edgar}
    buckets: dict[tuple, list[int]] = {}
    for i in edgar:
        for b in range(bands):
            buckets.setdefault((b, sigs[i][b * rows:(b + 1) * rows]), []).append(i)
    parent = {i: i for i in edgar}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def jacc(a, b):
        return sum(1 for x, y in zip(sigs[a], sigs[b]) if x == y) / _NUM_PERM

    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        base = idxs[0]
        for j in idxs[1:]:
            if jacc(base, j) >= 0.92:
                parent[find(j)] = find(base)
    seen_root, near_removed_ids = set(), set()
    for i in edgar:
        root = find(i)
        if root in seen_root:
            near_removed_ids.add(i)
        else:
            seen_root.add(root)
    keep = [r for i, r in enumerate(staged) if i not in near_removed_ids]
    return keep, {"exact_removed": exact_removed, "near_removed": len(near_removed_ids)}


# --- quality filter -------------------------------------------------------
def quality_ok(r: dict, token_count: int) -> bool:
    if token_count < MIN_TOKENS or not r["text"].strip():
        return False
    # Require real prose — at least 8 alphabetic words. This drops pure-number
    # fragments and garbled extraction without punishing legitimately number-heavy
    # financial notes (flattened tables), which still carry plenty of words.
    return len(re.findall(r"[A-Za-z]{2,}", r["text"])) >= 8


# --- segmentation ---------------------------------------------------------
def segment(rec: dict, ids: list[int], tok: Tokenizer) -> list[dict]:
    """Split a record's tokens into overlapping passages, carrying its metadata."""
    if len(ids) <= CHUNK_TOKENS:
        windows = [ids]
    else:
        step = CHUNK_TOKENS - CHUNK_OVERLAP
        windows = [ids[i:i + CHUNK_TOKENS] for i in range(0, len(ids), step)]
        windows = [w for w in windows if w]
    chunks = []
    for ci, w in enumerate(windows):
        chunks.append({
            "id": f"{rec['id']}-c{ci}" if len(windows) > 1 else rec["id"],
            "parent_id": rec["id"],
            "chunk_index": ci,
            "n_chunks": len(windows),
            "source": rec["source"],
            "doc_type": rec["doc_type"],
            "title": rec["title"],
            "citation": rec["citation"],
            "text": tok.decode(w).strip(),
            "token_count": len(w),
            "url": rec["url"],
            "metadata": rec["metadata"],
        })
    return chunks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the unified corpus.")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--stats", type=Path, default=Path("CORPUS_STATS.md"))
    args = ap.parse_args(argv)

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    raw = load_unified(args.corpus)
    n_raw = len(raw)
    deduped, dedup_stats = dedup(raw)

    # tokenize (batch) and quality-filter
    token_lists = [e.ids for e in tok.encode_batch([r["text"] for r in deduped])]
    unified, tok_counts, dropped_quality = [], [], 0
    kept_ids = []
    for r, ids in zip(deduped, token_lists):
        if not quality_ok(r, len(ids)):
            dropped_quality += 1
            continue
        r = {**r, "token_count": len(ids)}
        unified.append(r)
        tok_counts.append(len(ids))
        kept_ids.append(ids)

    # write full-document corpus
    with (args.corpus / "unified.jsonl").open("w", encoding="utf-8") as f:
        for r in unified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # segment -> chunked passage store
    chunks = []
    for r, ids in zip(unified, kept_ids):
        chunks.extend(segment(r, ids, tok))
    with (args.corpus / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    write_stats(args.stats, n_raw, dedup_stats, dropped_quality, unified, chunks)
    print(f"[done] {n_raw} raw -> {len(unified)} unified docs, {len(chunks)} chunks; "
          f"stats -> {args.stats}")
    return 0


def write_stats(path, n_raw, dedup_stats, dropped_quality, unified, chunks):
    df = pl.DataFrame([{k: r[k] for k in ("source", "doc_type", "token_count")}
                       for r in unified])
    cdf = pl.DataFrame([{"source": c["source"], "token_count": c["token_count"]}
                        for c in chunks])
    by_source = (df.group_by("source")
                 .agg(pl.len().alias("docs"),
                      pl.col("token_count").sum().alias("tokens"),
                      pl.col("token_count").median().alias("median_tok"),
                      pl.col("token_count").max().alias("max_tok"))
                 .sort("tokens", descending=True))
    chunk_by_source = (cdf.group_by("source").agg(pl.len().alias("chunks"))
                       .sort("chunks", descending=True))
    total_tokens = int(df["token_count"].sum())
    n = len(unified)

    lines = ["# AuditLM — Corpus Statistics", "",
             f"Unified corpus of **{n:,} documents** and **{total_tokens:,} tokens** "
             "(Qwen2.5 tokenizer), segmented into "
             f"**{len(chunks):,} passages**.", "",
             "## Pipeline", "",
             f"- Loaded **{n_raw:,}** raw records from 6 source collectors.",
             f"- Exact duplicates removed: **{dedup_stats['exact_removed']}**.",
             f"- Near-duplicates removed within EDGAR 10-Ks (MinHash-LSH, "
             f"Jaccard ≥ 0.92): **{dedup_stats['near_removed']}**.",
             f"- Dropped by quality filter (tiny/garbled, < {MIN_TOKENS} tokens or "
             f"non-English): **{dropped_quality}**.",
             f"- Final: **{n:,}** documents.", "",
             "## Per-source (documents, tokens)", "",
             "| source | docs | tokens | median tok/doc | max tok/doc |",
             "|---|---|---|---|---|"]
    for row in by_source.iter_rows(named=True):
        lines.append(f"| {row['source']} | {row['docs']:,} | {row['tokens']:,} | "
                     f"{int(row['median_tok'])} | {int(row['max_tok']):,} |")
    lines += ["", "## Per-source (passages after segmentation)", "",
              "| source | passages |", "|---|---|"]
    for row in chunk_by_source.iter_rows(named=True):
        lines.append(f"| {row['source']} | {row['chunks']:,} |")

    # chunk size distribution
    ct = cdf["token_count"]
    lines += ["", "## Passage size (tokens)", "",
              f"- count: {len(ct):,} | median: {int(ct.median())} | "
              f"mean: {int(ct.mean())} | max: {int(ct.max())}",
              f"- passages at the {CHUNK_TOKENS}-token cap (i.e. from split "
              f"records): {int((ct >= CHUNK_TOKENS - 5).sum()):,}", ""]
    # by doc_type
    bt = (df.group_by("doc_type").agg(pl.len().alias("docs"),
          pl.col("token_count").sum().alias("tokens")).sort("tokens", descending=True))
    lines += ["## Per doc_type", "", "| doc_type | docs | tokens |", "|---|---|---|"]
    for row in bt.iter_rows(named=True):
        lines.append(f"| {row['doc_type']} | {row['docs']:,} | {row['tokens']:,} |")

    # sample record per source (one with a citation where the source has them)
    lines += ["", "## Sample records (one per source)", ""]
    for src in ["PCAOB", "SEC", "FASB", "GAO"]:
        pool = [r for r in unified if r["source"] == src]
        sample = next((r for r in pool if r["citation"]), pool[0]) if pool else None
        if sample:
            snippet = re.sub(r"\s+", " ", sample["text"])[:200]
            cit = sample["citation"] or sample["title"] or sample["id"]
            lines += [f"**{src} — {cit}**  ", f"> {snippet}…", ""]

    lines += ["## Quality flags / known limitations", "",
              "- **GAGAS currency:** GAO content is the **2018 revision** "
              "(GAO-21-368G); the 2024 revision is the current effective version — "
              "a v1.1 refresh candidate (clean machine-readable 2024 PDF not yet "
              "available).",
              "- **EDGAR notes (6 of 91 filings):** name-only / index-only note "
              "headings (BofA, Goldman, IBM, McDonald's, BlackRock, Disney) yield "
              "no footnotes (reports + CAMs still captured) — deferred to "
              "LLM-assisted extraction.",
              "- **Flattened tables:** financial-statement tables lose column "
              "structure in text extraction; note prose is preserved.",
              "- **SAB 114:** a broad multi-topic codification update, so its single "
              "extracted SAB-Codification topic is approximate.",
              "- **FASB ASC:** topic/subtopic **number** scaffolding only (from the "
              "public XBRL taxonomy); the licensed/bot-protected ASC prose is not "
              "reproduced.",
              "- **Wells Fargo** 10-K skipped (cover-page primary document; "
              "financials incorporated by reference).", ""]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
