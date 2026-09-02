"""Dense retrieval — nomic-embed-text-v1.5 + FAISS (local, cached).

nomic-embed requires task-instruction prefixes (NOT interchangeable): passages are
embedded with `search_document:`, queries with `search_query:`. Embeddings are
L2-normalized and indexed with FAISS inner-product, so search returns cosine
similarity. Everything is cached under rag/index/ so re-runs don't re-embed.

    python dense.py build              # embed all passages, build + cache the index
    python dense.py query "going concern"   # top-k dense retrievals (validation)
"""

from __future__ import annotations

import os

# torch and faiss each bundle their own OpenMP runtime; on macOS loading both
# aborts/segfaults. Allow the duplicate runtime (standard workaround) before either
# is imported. We also keep import order torch-before-faiss in the retriever.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from corpus import DEFAULT_CHUNKS, by_id, load_chunks

RAG_DIR = Path(__file__).resolve().parent
INDEX_DIR = RAG_DIR / "index"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"   # Nomic AI (US), Apache 2.0, open
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def _auto_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, trust_remote_code=True,
                               device=device or _auto_device())


# The full-corpus build runs on CPU: the nomic custom kernel segfaults on MPS over a
# 12K batch (single-query encode on MPS is fine). The build is a one-time cached cost
# (~7 min on an M3 Max), so stability beats speed here.
def build(chunks_path: Path = DEFAULT_CHUNKS, batch_size: int = 32,
          device: str = "cpu") -> dict:
    chunks = load_chunks(chunks_path)
    ids = [c["id"] for c in chunks]
    texts = [DOC_PREFIX + (c.get("text") or "") for c in chunks]

    print(
        f"[dense] embedding {len(texts)} passages with {MODEL_NAME} "
        f"on {device} ...",
        file=sys.stderr,
    )

    model = load_model(device)

    # Encode in checkpointed groups so an interruption doesn't lose
    # the entire build.
    checkpoint_dir = INDEX_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_size = 256
    embeddings = []

    for start in range(0, len(texts), checkpoint_size):
        end = min(start + checkpoint_size, len(texts))
        checkpoint = checkpoint_dir / f"embeddings_{start:05d}_{end:05d}.npy"

        if checkpoint.exists():
            print(
                f"[dense] loading checkpoint {start}:{end}",
                file=sys.stderr,
            )
            emb = np.load(checkpoint)
        else:
            print(
                f"[dense] encoding {start}:{end} "
                f"({end - start} passages)...",
                file=sys.stderr,
            )

            emb = model.encode(
                texts[start:end],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")

            np.save(checkpoint, emb)

            print(
                f"[dense] checkpoint saved: {checkpoint}",
                file=sys.stderr,
            )

        embeddings.append(emb)

    emb = np.concatenate(embeddings, axis=0)

    # Import FAISS only after embedding.
    import faiss

    INDEX_DIR.mkdir(exist_ok=True)

    np.save(INDEX_DIR / "embeddings.npy", emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(
        index,
        str(INDEX_DIR / "dense.faiss"),
    )

    (INDEX_DIR / "ids.json").write_text(
        json.dumps(ids),
        encoding="utf-8",
    )

    meta = {
        "model": MODEL_NAME,
        "dim": int(emb.shape[1]),
        "count": len(ids),
        "doc_prefix": DOC_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "normalized": True,
        "metric": "inner_product",
    }

    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print(
        f"[dense] built index: {len(ids)} vectors, "
        f"dim {emb.shape[1]} -> {INDEX_DIR}",
        file=sys.stderr,
    )

    return meta

class DenseRetriever:
    def __init__(self, chunks_path: Path = DEFAULT_CHUNKS, device: str = "cpu"):
        # Load the torch model FIRST (before faiss) — import order matters for the
        # OpenMP coexistence. Query encoding runs on CPU: stable, and a single short
        # query is fast (~0.1s), so MPS's custom-kernel instability is avoided.
        self._model = load_model(device)
        import faiss
        self.meta = json.loads((INDEX_DIR / "meta.json").read_text())
        self.ids = json.loads((INDEX_DIR / "ids.json").read_text())
        self.index = faiss.read_index(str(INDEX_DIR / "dense.faiss"))
        self.chunks = by_id(load_chunks(chunks_path))

    @property
    def model(self):
        return self._model

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(q, k)
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


def _fmt(chunk: dict) -> str:
    cite = chunk.get("citation") or f"[{chunk['source']}/{chunk['doc_type']}]"
    body = " ".join((chunk.get("text") or "").split())[:150]
    return f"{cite:22} ({chunk['source']:5}) {body}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dense (nomic + FAISS) index build / query.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=5)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build()
        return 0
    r = DenseRetriever()
    print(f"QUERY: {args.text!r}")
    for cid, score in r.search(args.text, args.k):
        print(f"  {score:.3f}  {_fmt(r.chunks[cid])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
