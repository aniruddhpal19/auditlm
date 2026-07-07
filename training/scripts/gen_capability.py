"""Capability demonstration generation (Opus-distilled, corpus-verified, grounded-prompt
format). For each (category, topic) seed: Opus writes a realistic question -> the rag/
layer retrieves REAL hybrid passages and builds the actual grounded prompt (tiered policy)
-> Opus writes the ideal in-context answer modeling the target behaviors -> every standard
citation is verified against the corpus (drop if any is unverifiable).

Training record = the deployment format: system (audit assistant) + user (grounded prompt
with real passages) + assistant (ideal answer). Train on the assistant completion only.

    ANTHROPIC_API_KEY=... rag/.venv/bin/python gen_capability.py --pilot      # ~20 demos
    rag/.venv/bin/python gen_capability.py --pilot --mock                     # offline flow test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ground import Grounder                       # noqa: E402  (the RAG retrieval layer)
import corpus_verify as cv                         # noqa: E402
import teacher as T                                # noqa: E402

OUT = ROOT / "training" / "data"

# Deployment system prompt — identical to the runner's so training matches inference.
SYSTEM = ("You are an assistant for US external audit (PCAOB standards) and US "
          "GAAP accounting. Answer precisely and cite standards by their identifiers "
          "(e.g. AS 2301.05, ASC 606, 17 CFR 210.2-01) where relevant.")

CAT_DEF = {
    "citation_lookup": "identify the correct governing standard/paragraph for a topic",
    "procedure_suggestion": "design audit procedures matched to the relevant assertion(s)/risk",
    "disclosure_review": "state the required financial-statement disclosures under a specific ASC topic",
    "comparison_differentiation": "draw the precise distinction between related audit/accounting concepts",
    "concept_explanation": "explain and apply a core audit/accounting concept",
    "filing_summarization": "explain the content and purpose of a 10-K / auditor's-report section",
    "document_drafting": "draft a professional audit workpaper, memo, or communication artifact",
    "analytical_flagging": "identify what a financial-statement signal/anomaly flags and the audit response",
    "calculation_support": "perform and explain an audit calculation (materiality, sampling projection)",
}

# In-corpus scope: the corpus is PCAOB AS / SEC (Reg S-X/S-K, SABs) / FASB ASC / GAGAS.
# Seeding or answering about AU-C / IFRS / ISA forces memory-recall (the behavior we train
# OUT) and can't be corpus-verified — so both generation prompts are scoped to the corpus.
SCOPE = ("SCOPE: Use ONLY the US public-company audit framework the system covers — PCAOB "
         "Auditing Standards (AS NNNN), SEC rules (Regulation S-X/S-K = 17 CFR, Staff "
         "Accounting Bulletins), FASB ASC topics, and GAGAS. Do NOT reference, frame around, "
         "or cite AICPA AU-C, legacy AU, SAS, IFRS, IAS, or ISA standards — they are outside "
         "scope and cannot be grounded in the corpus.")

QGEN_SYSTEM = ("You write ONE realistic US external-audit practice question for a training "
               "dataset. Output ONLY the question (1-2 sentences), specific and answerable. "
               "Do not copy any existing benchmark wording; write a fresh question.\n" + SCOPE)

ANSWER_SYSTEM = (
    "You are a senior US external-audit technical partner writing the model answer for a "
    "training dataset. You are given retrieved source passages and a question. Write the "
    "IDEAL answer, following these rules exactly:\n"
    "- CITE PRECISELY WHEN GROUNDED: when you state a standard number or paragraph, cite "
    "ONLY identifiers that appear in the retrieved passages, using the exact identifier "
    "(e.g., AS 2301, ASC 606, 17 CFR 210.2-01). Never invent or guess a standard number, "
    "and never fabricate a citation from a topic stub's 'Subtopics' metadata.\n"
    "- REASON FULLY WHEN RETRIEVAL IS THIN: if the passages do not contain the substantive "
    "answer (e.g. only a topic-level stub or tangential notes), answer from your own expert "
    "knowledge and give the complete, correct requirement. Do NOT say the information is "
    "'not stated' and do NOT refuse.\n"
    "- Be correct, specific, and concise — the answer a Big Four technical partner gives. "
    "For procedures, give specific assertion-linked procedures; for disclosures, the actual "
    "ASC requirements; for comparisons, the precise distinction.\n"
    "- Write directly for the user; no meta-commentary about 'the passages'.\n" + SCOPE)

PILOT_SEEDS = [
    ("citation_lookup", "supervision of the audit engagement"),
    ("citation_lookup", "auditing inventories and observation of the physical inventory count"),
    ("citation_lookup", "inquiry of a client's lawyer regarding litigation, claims, and assessments"),
    ("citation_lookup", "evaluating the consistency of the financial statements"),
    ("procedure_suggestion", "existence and completeness of cash, including bank confirmations and the bank reconciliation"),
    ("procedure_suggestion", "valuation of investment securities measured at fair value"),
    ("procedure_suggestion", "completeness of warranty and product-return reserves"),
    ("procedure_suggestion", "existence and rights for derivative and hedging instruments"),
    ("procedure_suggestion", "occurrence and accuracy of payroll expense"),
    ("disclosure_review", "segment reporting disclosures under ASC 280"),
    ("disclosure_review", "earnings per share disclosures under ASC 260"),
    ("disclosure_review", "stock-based compensation disclosures under ASC 718"),
    ("disclosure_review", "business combination disclosures under ASC 805"),
    ("disclosure_review", "defined-benefit pension and OPEB disclosures under ASC 715"),
    ("comparison_differentiation", "qualified opinion vs adverse opinion vs disclaimer of opinion"),
    ("comparison_differentiation", "tests of controls vs substantive procedures"),
    ("comparison_differentiation", "a material weakness vs a significant deficiency in ICFR (AS 2201)"),
    ("concept_explanation", "the audit risk model: inherent, control, and detection risk"),
    ("concept_explanation", "sufficiency vs appropriateness of audit evidence"),
    ("concept_explanation", "the auditor's responsibilities when using the work of a specialist"),
]


def seed_key(cat, topic):
    return f"{cat}::{topic}"


def load_jsonl(path):
    """Tolerant loader (skips a torn final line from a crashed write) for resume."""
    out = []
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def process_one(n, cat, topic, teacher, grounder, idx) -> dict:
    """Generate + verify one demo (Opus calls retry transient errors inside teacher.opus)."""
    q = teacher(QGEN_SYSTEM, f"Category: {cat} — {CAT_DEF[cat]}.\nTopic: {topic}.\n"
                             f"Write one {cat} question on this topic.", max_tokens=200)
    q = q.strip().strip('"')
    passages = grounder.retrieve(q)
    grounded = grounder.build_prompt(q, passages)
    answer = teacher(ANSWER_SYSTEM, grounded, max_tokens=900)
    v = cv.verify_text(answer, idx)
    return {"id": f"cap-demo-{n:04d}", "task_category": cat, "topic": topic, "question": q,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": grounded},
                         {"role": "assistant", "content": answer}],
            "retrieved": [c.get("citation") or c["doc_type"] for c in passages],
            "verified_citations": v["verified"], "unverifiable_citations": v["unverifiable"],
            "citations_verified": v["clean"], "provenance": "opus-distilled-corpus-verified"}


def _sample(demos, dropped):
    """Stratified sample for review: 2 disclosure + 1 each citation/procedure/concept/
    comparison + 1-2 from the highest-drop category."""
    from collections import Counter, defaultdict
    by = defaultdict(list)
    for d in demos:
        by[d["task_category"]].append(d)
    pick, ids = [], set()
    plan = [("disclosure_review", 2), ("citation_lookup", 1), ("procedure_suggestion", 1),
            ("concept_explanation", 1), ("comparison_differentiation", 1)]
    drop_by = Counter(d["task_category"] for d in dropped)
    if drop_by:
        plan.append((drop_by.most_common(1)[0][0], 2))
    for cat, k in plan:
        for d in by.get(cat, [])[:k]:
            if d["id"] not in ids:
                pick.append(d); ids.add(d["id"])
    return pick[:8]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true", help="full benchmark-weighted set (seeds.py)")
    ap.add_argument("--target", type=int, default=400, help="approx total demos for --full")
    ap.add_argument("--mock", action="store_true", help="offline flow test (no API)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing output, start clean")
    ap.add_argument("--show", type=int, default=8, help="print this many sampled full demos")
    args = ap.parse_args(argv)

    if args.full:
        import seeds as S
        seeds = S.build_seeds(args.target)
        out = args.out or str(OUT / "capability_demos.jsonl")
    else:
        seeds = PILOT_SEEDS
        out = args.out or str(OUT / "capability_pilot.jsonl")
    drops = out.replace(".jsonl", "_drops.jsonl")

    teacher = T.mock_opus if args.mock else T.opus
    grounder = Grounder(k=5)
    idx = cv.build_index()
    OUT.mkdir(parents=True, exist_ok=True)

    # Resume: skip seeds already processed (clean OR dropped) by seed key. --no-resume clears.
    if args.no_resume:
        for p in (out, drops):
            if Path(p).exists():
                Path(p).unlink()
    done = {seed_key(d["task_category"], d["topic"]) for d in load_jsonl(out) + load_jsonl(drops)}
    todo = [(n, c, t) for n, (c, t) in enumerate(seeds, 1) if seed_key(c, t) not in done]
    print(f"generating {len(seeds)} demos ({'MOCK' if args.mock else 'OPUS'}) | "
          f"{len(done)} already done, {len(todo)} to do -> {out}", flush=True)

    # Incremental: append each completed demo as it finishes (crash never loses progress).
    errors = []
    with open(out, "a", encoding="utf-8") as fo, open(drops, "a", encoding="utf-8") as fd:
        for i, (n, cat, topic) in enumerate(todo, 1):
            try:
                rec = process_one(n, cat, topic, teacher, grounder, idx)
            except Exception as e:                  # noqa: BLE001 — isolate one seed; retried on resume
                errors.append((f"cap-demo-{n:04d}", cat, f"{type(e).__name__}: {e}"))
                print(f"  [{len(done)+i}/{len(seeds)}] cap-demo-{n:04d} {cat} ERROR (retry on resume)",
                      flush=True)
                continue
            f = fo if rec["citations_verified"] else fd
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            tag = "clean" if rec["citations_verified"] else "DROP"
            print(f"  [{len(done)+i}/{len(seeds)}] {rec['id']} {cat:26} {tag}", flush=True)

    # Final audit from the full set on disk (resumed + new), deduped by id.
    merged = {r["id"]: r for r in load_jsonl(out) + load_jsonl(drops)}
    demos = [r for r in merged.values() if r["citations_verified"]]
    dropped = [r for r in merged.values() if not r["citations_verified"]]

    from collections import Counter
    alld = demos + dropped
    print(f"\n=== CITATION AUDIT SUMMARY ===")
    print(f"generated: {len(alld)} | clean (ALL citations corpus-verified): {len(demos)} | "
          f"dropped (>=1 unverifiable/out-of-corpus citation): {len(dropped)}")

    print("\n=== PER-CATEGORY (clean count | drop rate; FLAG if >15%) ===")
    cats = sorted(set(c for c, _ in seeds))
    for c in cats:
        cl = sum(1 for d in demos if d["task_category"] == c)
        dr = sum(1 for d in dropped if d["task_category"] == c)
        tot = cl + dr
        rate = dr / tot if tot else 0
        flag = "  <-- HIGH DROP (re-target seeds)" if rate > 0.15 else ""
        print(f"  {c:28} clean={cl:3}  dropped={dr:2}  drop_rate={rate:5.1%}{flag}")

    if dropped:
        print("\n=== DROP LOG (id | category | unverifiable citations + reason) ===")
        for d in dropped:
            print(f"  {d['id']} {d['task_category']:26} {d['unverifiable_citations']}")
    if errors:
        print(f"\n=== {len(errors)} seed(s) ERRORED this run (not written; re-run to retry them) ===")
        for iid, cat, msg in errors:
            print(f"  {iid} {cat:26} {msg[:90]}")
    print(f"\nwritten -> {out}  (drops -> {drops})")

    sample = _sample(demos, dropped)
    print(f"\n=== {len(sample)} SAMPLED FULL DEMOS (stratified for review) ===")
    for d in sample:
        u = d["messages"][1]["content"]; a = d["messages"][2]["content"]
        print("=" * 96)
        print(f"{d['id']} [{d['task_category']}]  retrieved={d['retrieved']}")
        print(f"--- Q: {d['question']}")
        print(f"--- GROUNDED PROMPT (user turn, truncated) ---\n{u[:600]}\n  ... [{len(u)} chars]")
        print(f"--- OPUS ANSWER (assistant completion) ---\n{a}")
        print(f"--- VERIFIED: {d['verified_citations']} | UNVERIFIABLE: {d['unverifiable_citations']} ---\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
