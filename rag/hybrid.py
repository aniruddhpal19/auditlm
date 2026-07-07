"""Hybrid retrieval — dense (nomic+FAISS) + BM25 lexical, fused via RRF.

BM25 indexes `corpus.lexical_blob` (passage body PLUS its citation identifiers —
citation / standard_id / paragraph / asc_topic / regulation / SAB number), so an
exact-identifier query ("AS 2110", "ASC 842") surfaces that exact passage even when
the token is sparse in the body. Dense handles semantics; BM25 handles exact tokens;
reciprocal rank fusion (RRF) combines their rankings without score calibration.

    python hybrid.py query "AS 2301"            # hybrid top-k
    python hybrid.py compare "AS 2301"          # dense-only vs hybrid, side by side
"""

from __future__ import annotations

import argparse
import re

from corpus import DEFAULT_CHUNKS, by_id, lexical_blob, load_chunks
from dense import DenseRetriever, _fmt

# Tokenizer for BM25: split into atomic alphanumeric tokens (drop dots/hyphens), so
# a standard-level query "AS 2301" -> ["as","2301"] matches the paragraph passage
# "AS 2301.05" (its citation tokenizes to ["as","2301","05"]). Keeping "2301.05" as
# one token would WRONGLY miss it for an "AS 2301" query — the standard number is the
# discriminative token, and IDF down-weights the noise ("as", "05").
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(lexical_blob(c)) for c in chunks])

    def scores(self, query: str) -> dict[str, float]:
        arr = self.bm25.get_scores(tokenize(query))
        return {self.ids[i]: float(arr[i]) for i in range(len(arr))}

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = sorted(s, key=s.get, reverse=True)[:k]
        return [(cid, s[cid]) for cid in top if s[cid] > 0]


# --- STRUCTURED-IDENTIFIER EXACT-MATCH PATH ---
# When a query contains an explicit standard identifier (AS NNNN / ASC NNN), we match
# it against the structured standard_id / asc_topic metadata the corpus pipeline keeps and
# rank THAT standard's own passages first — like a standards-research tool (type
# "AS 2110", get AS 2110). It is deterministic, transparent, and scoped to identifier
# queries ONLY; concept queries (no identifier) skip this entirely and use pure
# dense+BM25 RRF (which is why concept retrieval does not regress).
_ID_PATTERNS = [(re.compile(r"\bAS\s?(\d{3,4})\b", re.I), "AS"),
                (re.compile(r"\bASC\s?(\d{3,4})\b", re.I), "ASC")]


def detect_identifiers(query: str) -> list[tuple[str, str]]:
    out = []
    for pat, kind in _ID_PATTERNS:
        out += [(kind, m.group(1)) for m in pat.finditer(query)]
    return out


def _matches_identifier(chunk: dict, ids: list[tuple[str, str]]) -> bool:
    md = chunk.get("metadata") or {}
    sid, asc, cit = md.get("standard_id") or "", str(md.get("asc_topic") or ""), chunk.get("citation") or ""
    for kind, num in ids:
        if kind == "AS" and sid == f"AS {num}":
            return True
        if kind == "ASC" and (asc == num or cit == f"ASC {num}"):
            return True
    return False


def rrf(rankings: list[list[str]], k_rrf: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each list contributes 1/(k_rrf + rank) per id."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """Dense + BM25 fused via RRF. `pool` is how deep each retriever goes before
    fusion (so a result strong in one modality isn't lost); `k` is the fused cut."""

    def __init__(self, chunks_path=DEFAULT_CHUNKS, device: str = "cpu", pool: int = 20):
        chunks = load_chunks(chunks_path)
        self.chunks = by_id(chunks)
        self.dense = DenseRetriever(chunks_path, device=device)
        self.bm25 = BM25Retriever(chunks)
        self.pool = pool

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        bm_scores = self.bm25.scores(query)
        d = [cid for cid, _ in self.dense.search(query, self.pool)]
        b = sorted(bm_scores, key=bm_scores.get, reverse=True)[:self.pool]
        fused = rrf([d, b])
        ids = detect_identifiers(query)
        if not ids:
            return fused[:k]
        # identifier query: rank THIS standard's own passages first (by lexical score),
        # then fill from the fused list. Uses standard_id/asc_topic metadata exactly.
        exact = sorted((c for c in self.bm25.ids if _matches_identifier(self.chunks[c], ids)),
                       key=lambda c: bm_scores[c], reverse=True)
        seen = set(exact)
        ordered = [(c, bm_scores[c]) for c in exact] + [(c, s) for c, s in fused if c not in seen]
        return ordered[:k]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid (dense + BM25 + RRF) retrieval.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("query", "compare"):
        p = sub.add_parser(name)
        p.add_argument("text")
        p.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)

    h = HybridRetriever()
    if args.cmd == "query":
        print(f"HYBRID: {args.text!r}")
        for cid, score in h.search(args.text, args.k):
            print(f"  {score:.4f}  {_fmt(h.chunks[cid])}")
        return 0
    # compare: dense-only vs hybrid
    print(f"=== {args.text!r} ===")
    print("DENSE-ONLY:")
    for cid, score in h.dense.search(args.text, args.k):
        print(f"  {score:.3f}  {_fmt(h.chunks[cid])}")
    print("HYBRID (dense+BM25+RRF):")
    for cid, score in h.search(args.text, args.k):
        print(f"  {score:.4f}  {_fmt(h.chunks[cid])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
