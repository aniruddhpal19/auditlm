"""Demo-corpus runner: push 55 realistic auditor questions through the SAME verified
recommender that produced the reported results (rag tiered_guardrail -> auditlm-run2 ->
parser -> verifier -> confidence label), and store what an auditor would see.

Local only — recommend() calls ollama on localhost; no judge, no API key, runs free.

Writes demo/auditor_demo_corpus.jsonl (one record/question) + a browsable .md, then asserts
the verification layer's machine-checkable property: shown_fabrications == 0 across all 55
stored answers.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verify"))
from recommender import Recommender  # noqa: E402  (wires rag + verify + ollama)
from citation_index import CitationIndex, Resolution  # noqa: E402
from citation_parser import parse  # noqa: E402
from verifier import shown_ungrounded, UNGROUNDED_RES  # noqa: E402

DEMO = ROOT / "demo"
QS = DEMO / "auditor_questions.jsonl"
OUT_JSONL = DEMO / "auditor_demo_corpus.jsonl"
OUT_MD = DEMO / "auditor_demo_corpus.md"

# demo zone -> the recommender's deferral zone (so DEFER fires robustly, as in the adapter)
ZONE_TO_DEFERRAL = {
    "independence": "independence_conclusion",
    "fraud": "fraud_accusation",
    "legal": "legal_regulatory_conclusion",
    "novel_oos": "novel_transaction_treatment",
    "tax_oos": "tax_position",
}
DISCLAIMER = (
    "> **These are the tool's RECOMMENDATIONS, not certified-correct answers.** Every "
    "citation shown is verified to EXIST in the corpus; this does not certify the answer is "
    "correct. Each answer is to be verified by a qualified auditor against the cited sources.")


def independent_recount(text, idx):
    return sum(1 for sp in parse(text)
               if not sp.honest_disclaimer and idx.resolve(sp.citation_normalized).value in UNGROUNDED_RES)


def main():
    idx = CitationIndex.load_or_build()
    rec = Recommender()
    qs = [json.loads(l) for l in QS.read_text().splitlines() if l.strip()]
    records = []
    for i, q in enumerate(qs, 1):
        zone = ZONE_TO_DEFERRAL.get(q["zone"])
        out = rec.recommend(q["question"], deferral_zone=zone)
        verification = {**out["verification_report"], "grounding_rule": out["grounding_rule"],
                        "source_chunk_ids": out["source_chunk_ids"]}
        records.append({
            "qid": q["qid"], "zone": q["zone"], "scope": q["scope"], "question": q["question"],
            "label": out["label"], "answer": out["labeled_answer"],
            "verification": verification, "raw_output": out["raw_output"],
        })
        print(f"[{i}/{len(qs)}] {q['qid']:8} {q['zone']:14} -> {out['label']}", flush=True)

    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                         encoding="utf-8")

    # --- recount guard: the verification layer's machine-checkable property, on the STORED answers ---
    leaks = []
    for r in records:
        rc = independent_recount(r["answer"], idx)
        mt = shown_ungrounded(r["answer"], idx)
        stored = r["verification"]["shown_fabrications"]
        if rc or mt or stored:
            leaks.append((r["qid"], rc, mt, stored))
    print("\n=== recount guard (shown_fabrications across all 55) ===")
    print(f"  answers with any ungrounded citation shown: {len(leaks)} {leaks[:5] or '(none)'}")
    assert not leaks, f"criterion-1 LEAK in demo corpus: {leaks}"
    print("  PASS — 0 ungrounded citations shown across all 55 stored answers.")

    # --- tallies ---
    labels = Counter(r["label"] for r in records)
    print("\n=== label tally ===")
    for lab in ("GROUNDED", "PARTIAL", "GENERAL-KNOWLEDGE", "DEFER"):
        print(f"  {lab:18} {labels.get(lab, 0)}")
    oos = [r for r in records if r["scope"] == "out_of_scope"]
    oos_decline = [r for r in oos if r["label"] in ("GENERAL-KNOWLEDGE", "DEFER")]
    print(f"\n=== out-of-scope ({len(oos)}) — tool declines (GENERAL-KNOWLEDGE/DEFER)? ===")
    for r in oos:
        ok = "ok " if r["label"] in ("GENERAL-KNOWLEDGE", "DEFER") else "!! "
        print(f"  [{ok}] {r['qid']} {r['zone']:14} -> {r['label']}")
    print(f"  declined {len(oos_decline)}/{len(oos)} (none fabricated): "
          f"{'PASS' if len(oos_decline) == len(oos) else 'FAIL'}")

    _write_md(records)
    print(f"\nwrote {OUT_JSONL.name} ({len(records)}) + {OUT_MD.name}")


def _sources_line(r):
    gr = r["verification"].get("grounding_rule") or []
    stripped = r["verification"]["stripped"]
    parts = []
    if gr:
        parts.append("Sources: " + ", ".join(gr))
    parts.append(f"retrieved chunks: {len(r['verification']['source_chunk_ids'])}")
    if stripped:
        parts.append(f"stripped: {stripped} ungrounded citation(s) — verify independently")
    return " | ".join(parts)


def _write_md(records):
    by_zone = OrderedDict()
    for r in records:
        by_zone.setdefault(r["zone"], []).append(r)
    lines = ["# AuditLM — Verified Recommender demo corpus", "",
             DISCLAIMER, "",
             f"55 realistic auditor questions through the verified recommender "
             f"(RAG tiered_guardrail → auditlm-run2 → verify layer). "
             f"Zero ungrounded citations are shown in any answer (machine-checked).", ""]
    for zone, rs in by_zone.items():
        lines.append(f"## {zone}  ({len(rs)})")
        lines.append("")
        for r in rs:
            lines.append(f"### {r['qid']} — `{r['label']}`  ({r['scope']})")
            lines.append(f"**Q:** {r['question']}")
            lines.append("")
            lines.append("```")
            lines.append(r["answer"])
            lines.append("```")
            lines.append(f"_{_sources_line(r)}_")
            lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
