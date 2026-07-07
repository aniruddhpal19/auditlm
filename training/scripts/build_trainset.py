"""Assemble the training set and run the CONTAMINATION GATE (hard stop).

1. Normalize the hand-curated safety demos into the grounded-prompt training format and
   verify their PCAOB/ASC citations with the SAME corpus verifier as the capability demos.
2. Merge capability (clean) + safety -> the training set.
3. Contamination gate: 8-gram shingle overlap of every training demo's (question+answer)
   vs ALL 166 test items (question+reference_answer). Any demo that near-duplicates a test
   item is REMOVED. Training cannot start until clean. Writes data/train.jsonl.

    python build_trainset.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ground import TIERED_POLICY        # noqa: E402  (the deployment grounding policy)
import corpus_verify as cv               # noqa: E402

DATA = ROOT / "training" / "data"
CAP = DATA / "capability_demos.jsonl"
SAF = ROOT / "training" / "safety_sft_pilot_v2.jsonl"
TRAIN = DATA / "train.jsonl"
AB = Path(os.environ.get("ASSURANCEBENCH") or ROOT.parent / "assurancebench")

SYSTEM = ("You are an assistant for US external audit (PCAOB standards) and US "
          "GAAP accounting. Answer precisely and cite standards by their identifiers "
          "(e.g. AS 2301.05, ASC 606, 17 CFR 210.2-01) where relevant.")

WORD = re.compile(r"[a-z0-9]+")
K = 8
FLAG = 0.40   # >=40% shingle overlap (either direction) with a single test item = near-dup


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def shingles(text):
    t = WORD.findall(text.lower())
    return {hash(tuple(t[i:i + K])) for i in range(len(t) - K + 1)}


# ---- normalize safety + verify citations ------------------------------------------
def normalize_safety(r):
    grounded = (f"{TIERED_POLICY}\n\nRETRIEVED PASSAGES:\n{r['retrieved_context']}\n\n"
                f"QUESTION:\n{r['question']}\n\nAnswer:")
    return {"id": r["id"], "suite": "safety", "task_category": r["task_category"],
            "zone": r.get("zone"), "severity": r.get("severity"), "question": r["question"],
            "answer": r["ideal_answer"],
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": grounded},
                         {"role": "assistant", "content": r["ideal_answer"]}],
            "provenance": "hand-curated"}


def normalize_capability(r):
    return {"id": r["id"], "suite": "capability", "task_category": r["task_category"],
            "question": r["question"], "answer": r["messages"][2]["content"],
            "messages": r["messages"], "provenance": r.get("provenance")}


def main():
    idx = cv.build_index()
    cap = [normalize_capability(r) for r in load(CAP)]
    saf_raw = load(SAF)

    print("=== SAFETY-DEMO CITATION VERIFICATION (same corpus verifier) ===")
    saf = []
    for r in saf_raw:
        v = cv.verify_text(r["ideal_answer"], idx)
        flag = "" if v["clean"] else f"  <-- UNVERIFIABLE {v['unverifiable']}"
        print(f"  {r['id']:20} verified={v['verified']}{flag}")
        saf.append(normalize_safety(r))
    n_saf_clean = sum(1 for r in saf_raw if cv.verify_text(r["ideal_answer"], idx)["clean"])
    print(f"  -> {n_saf_clean}/{len(saf_raw)} safety demos have all citations corpus-verified")

    train = cap + saf
    print(f"\n=== MERGE === capability {len(cap)} + safety {len(saf)} = {len(train)} training demos")

    # ---- contamination gate --------------------------------------------------------
    test = [it for f in (AB / "items").glob("*.jsonl") for it in load(f) if it.get("split") == "test"]
    test_sh = [(it["id"], shingles(it["question"] + " " + (it.get("reference_answer") or ""))) for it in test]
    print(f"\n=== CONTAMINATION GATE === {len(train)} training demos vs {len(test)} test items "
          f"(8-gram shingles, flag >= {FLAG:.0%} overlap either direction)")

    flagged, rows = [], []
    for d in train:
        ds = shingles(d["question"] + " " + d["answer"])
        best, best_id = 0.0, None
        for tid, ts in test_sh:
            inter = len(ds & ts)
            if not inter:
                continue
            ov = max(inter / len(ds) if ds else 0, inter / len(ts) if ts else 0)
            if ov > best:
                best, best_id = ov, tid
        rows.append((best, d["id"], best_id))
        if best >= FLAG:
            flagged.append((d["id"], best_id, best))
    rows.sort(reverse=True)
    print("top 8 overlaps (overlap | demo | nearest test item):")
    for ov, did, tid in rows[:8]:
        print(f"  {ov:6.1%}  {did:18} ~ {tid}")

    clean = [d for d in train if d["id"] not in {f[0] for f in flagged}]
    print(f"\nflagged (>= {FLAG:.0%}): {len(flagged)} | removed: {len(flagged)} | "
          f"FINAL CLEAN TRAINING SET: {len(clean)}")
    if flagged:
        print("REMOVED (near-duplicate of a test item):")
        for did, tid, ov in flagged:
            print(f"  {did} ~ {tid} ({ov:.1%})")
    else:
        print("RESULT: CLEAN — training set verified disjoint from the test split. Cleared to train.")

    TRAIN.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in clean) + "\n", encoding="utf-8")
    print(f"\nwritten -> {TRAIN} ({len(clean)} demos)")


if __name__ == "__main__":
    main()
