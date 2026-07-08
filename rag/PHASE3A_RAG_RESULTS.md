# Phase 3A — Base + RAG: Recorded Result

The documented outcome of Phase 3A (retrieval-augmented generation over the
corpus, evaluated on the frozen AssuranceBench v1.0 **`test`** split, 166 items (superseded by the v1.0.1 correction — 164 items; these results are preserved as-run), with
the Claude judge). **Primary headline: RAG-strict.** **Documented ablation: RAG-tiered.**

- **Base** — Llama 3.1-8B, no retrieval (the frozen v1.0 baseline).
- **RAG-strict** — hybrid retrieval + grounding prompt strict everywhere (cite only
  retrieved passages; say "not found" if the source is absent). `AUDITLM_GROUNDING=strict`.
- **RAG-tiered** — strict on citations, soft on substance (when retrieval is thin, the
  model answers from its own knowledge). `AUDITLM_GROUNDING=tiered` (default).

Retrieval (both configs): nomic-embed-text-v1.5 dense + FAISS, BM25 over a
citation-identifier lexical field, RRF fusion, and a structured-identifier exact-match
path for explicit citation queries. Same retrieval for both; only the grounding prompt
differs (see `ground.py`: `STRICT_POLICY` / `TIERED_POLICY`).

## Capability — three-way (mean score per task category)

| category | Base | RAG-strict | RAG-tiered | strict Δ | tiered Δ |
|---|---|---|---|---|---|
| **citation_lookup** | 0.476 | **0.800** | 0.717 | **+0.324** | +0.241 |
| **comparison_differentiation** | 0.285 | **0.660** | 0.574 | **+0.375** | +0.289 |
| concept_explanation | 0.588 | 0.662 | 0.678 | +0.075 | +0.090 |
| analytical_flagging | 0.593 | 0.525 | 0.695 | −0.068 | +0.103 |
| document_drafting | 0.525 | 0.480 | 0.535 | −0.045 | +0.010 |
| calculation_support | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 |
| filing_summarization | 0.526 | 0.458 | 0.505 | −0.068 | −0.022 |
| procedure_suggestion | 0.411 | 0.352 | 0.393 | −0.059 | −0.018 |
| **disclosure_review** | 0.445 | **0.209** | 0.432 | **−0.236** | −0.014 |
| **CAPABILITY OVERALL** | 0.494 | 0.576 | **0.589** | +0.082 | +0.095 |

*RAG-strict is the **fresh archive run** (complete, 11/11 disclosure incl. `cap-disc-111`).
It **reproduced** the original strict run — headline numbers stable (citation 0.80,
comparison 0.66, disclosure 0.21, overall 0.61); only minor per-category LLM-judge variance
on a few free-form tasks (e.g. filing/analytical ±0.02). The original strict run had
`cap-disc-111` error on a transient Anthropic 529 (10/11); the fresh run completed it.
Stability/reproducibility of the Phase 3A result is confirmed.*

## Safety — three-way (gate / dangerous conclusions / per-zone mean)

| metric | Base | RAG-strict | RAG-tiered |
|---|---|---|---|
| **Dangerous conclusions** (zero-tolerance) | **0** | **0** | **0** |
| Gate | FAIL | FAIL | FAIL |
| Non-dangerous pass-rate | 75% | 79% | 75% |
| independence_conclusion | 0.778 | 0.556 | 0.556 |
| fraud_accusation | 0.833 | 1.000 | 0.833 |
| legal_regulatory_conclusion | 0.600 | 0.800 | 1.000 |
| novel_transaction_treatment | 1.000 | 1.000 | 0.750 |
| tax_position | 0.500 | 0.750 | 0.750 |

## Key findings

1. **Citation is the thesis-confirming win, where the corpus holds the authoritative
   answer.** RAG-strict **+0.324** (0.476→0.800; pass-rate 49%→81%); RAG-tiered +0.241
   (→0.717). The corpus contains the PCAOB AS text, retrieval surfaces it, the model
   grounds the citation. Strict is the clean headline and the defensible audit-tool
   default (don't confabulate standard numbers). Comparison moves the same way
   (+0.375 strict / +0.289 tiered).

2. **Disclosure is an architectural-boundary finding — RAG hurts where the corpus has
   only fragments.** RAG-strict **−0.236** (0.445→0.209). The corpus has **no licensed
   FASB disclosure prose** — only 85 topic stubs; every disclosure query retrieves the
   stub + tangential 10-K/SAB notes that never enumerate the requirement (11/11 lead
   with the stub). Under strict grounding the model suppresses its own knowledge and
   answers "not explicitly stated". *The "model = skills, RAG = facts" thesis holds only
   where RAG actually holds the facts.*

3. **The drop is mostly a recoverable grounding amplifier — at a citation cost (the
   strict/soft tension).** RAG-tiered recovers disclosure to **−0.014** (≈Base) and
   filing toward Base (0.441→0.505) by letting parametric knowledge back in — but loses
   citation precision (0.800→0.717) and comparison (0.660→0.574). So: **cite precisely
   (strict) vs reason fully (soft) is a real tradeoff a single prompt cannot win.**

4. **Procedure is unaffected by grounding — it is a reasoning skill, not a retrieval
   gap.** Near-floor in both (0.352 / 0.393, from Base 0.411 at 3% pass). RAG can't teach
   procedure design; it is an SFT target, and the softening neither rescued nor harmed it.

5. **Safety is stable-but-uncalibrated, with ZERO dangerous conclusions in all three
   configs.** The gate fails everywhere on the calibrated-deferral threshold, never on a
   dangerous conclusion. **Independence regresses identically (−0.222 vs Base) under BOTH
   grounding policies** — handed the precise SEC reg (17 CFR 210.2-01), the model concludes
   ("No, it is not allowed") instead of deferring. That this is policy-independent makes it
   an architectural RAG/safety tension, not a prompt artifact.

## Conclusion — the tension motivates SFT

RAG alone cannot simultaneously (a) cite precisely, (b) reason fully when retrieval is
thin, and (c) defer the conclusion even when handed an authoritative rule — strict wins
(a) and loses (b); tiered wins (b) at the cost of (a); neither fixes (c). These are three
behaviors a single grounding prompt cannot reconcile, and they are exactly what the **SFT
run** targets: teach the model to do strict-citation **as a skill** (keep the citation
win), retain the soft-reasoning fallback (keep disclosure/filing), and exercise calibrated
professional deferral even with the rule in hand (fix independence). The disclosure
architectural boundary (missing FASB prose) is separate and persists until that content is
licensed/added — it bounds what RAG can deliver and is an honest result of record.

## Reproducibility

- Default grounding is **tiered** (the SFT run uses it). Reproduce either config:
  ```bash
  # tiered (default)
  AUDITLM_RAG=<auditlm> rag/.venv/bin/python -m src.runner \
    --split test --judge --model "rag:ollama:llama3.1:8b" --no-resume
  # strict (ablation) — to a separate --out so it doesn't overwrite the tiered file
  AUDITLM_GROUNDING=strict AUDITLM_RAG=<auditlm> rag/.venv/bin/python -m src.runner \
    --split test --judge --model "rag:ollama:llama3.1:8b" --no-resume --out runs/strict
  ```
- Run from the `assurancebench` directory. **Both configs' raw artifacts are archived:**
  - tiered: `runs/rag_ollama_llama3.1_8b_test_{results.jsonl,scorecard.md}`
  - strict: `runs/strict/rag_ollama_llama3.1_8b_test_{results.jsonl,scorecard.md}`
  - committed scorecards (in this repo): `rag/results/rag_{strict,tiered}_test_scorecard.md`
- The original strict run was overwritten by the `--no-resume` tiered run, then
  **re-run fresh to `runs/strict/` and reproduced exactly** (stability confirmed — see the
  capability-table note). Both configs are now preserved side by side for the ablation.
