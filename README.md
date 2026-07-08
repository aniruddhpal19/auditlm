# AuditLM

**An open, locally-runnable language model + verification layer for US external audit (PCAOB / SEC / GAGAS).**

AuditLM is a question-answering assistant for external financial auditors. You ask *"Which standard governs the auditor's consideration of fraud?"* or *"What does the independence rule require?"*, and it answers — with two trust properties built into the architecture, not hoped for from the model:

1. **Every citation it shows is verified to exist in a public corpus first.** If a cited standard or paragraph isn't in the corpus, it is **stripped and flagged**, never shown as if real. The result is machine-checkable: re-parse any answer and **zero ungrounded citations** appear.
2. **On professional-judgment calls it defers.** Independence conclusions, fraud accusations, legal-liability predictions, novel-transaction accounting, and specific tax positions get a **DEFER** label — the framework is explained, the conclusion is left to the qualified professional. This holds even when the underlying model produced a (wrong) confident conclusion.

It runs on a single high-end laptop (Apple M3 Max, 36 GB), costs nothing per query, and is built entirely from **public sources** — no licensed standard text. It is a fast, trustworthy *starting point* an auditor verifies, **not** a replacement for professional judgment.

> **This is the umbrella project.** The evaluation benchmark, **[AssuranceBench](https://github.com/vimalnakrani08/assurancebench)**, lives in its own repo so it stays independently clone-able and citable.

> **Built with Llama.** This project fine-tunes Meta's **Llama 3.1-8B** under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/). The distributed fine-tuned model is named **Llama-AuditLM** (the "Llama" prefix is required by the license); the project and repos remain **AuditLM**.

---

## Why this exists

General chatbots are fluent but **fabricate** — they cite rule numbers that sound real but don't exist, and give confident wrong conclusions. In casual use that's annoying; in auditing a wrong citation or a wrong independence/fraud conclusion has legal and regulatory consequences. So "just use a general chatbot for audit questions" is exactly what you can't safely do. AuditLM's thesis: for a small open-corpus audit model, **citation trustworthiness is an architectural property (verified retrieval), not a capability to fine-tune in** — and the model's job is skills + judgment while a RAG corpus holds the facts.

## The arc: Base → RAG → SFT → Verification

| Stage | What it adds | Finding |
|---|---|---|
| **Base** (Llama 3.1-8B) | raw capability | fluent but cites legacy/wrong standards; no grounding |
| **+ RAG** (hybrid retrieval over the corpus, tiered-grounding policy) | retrieve & cite real passages | big citation gain; "tiered" policy keeps substance when retrieval is thin |
| **+ SFT** (QLoRA via MLX, local) | calibrated deferral + format | **eliminated dangerous conclusions**, but **could not** instill faithful citation recall |
| **+ Verification** (this repo's `verify/` layer) | deterministic citation check | **zero fabricated citations shown**; trust becomes guaranteed, not learned |

The honest arc: **distillation + verification, not naive distillation; trust-by-architecture, not trust-by-scale.**

## How the verification layer works (`verify/`)

A deterministic wrapper around the (untouched) model + RAG:

```
question → RAG retrieve → model generate → parse citations → verify each → confidence label
        → { labeled answer, verification report, source passages }
```

- **citation_index** — the single source of truth for "does this citation exist in the corpus." Two-level (exact paragraph + base standard), built from 12,048 passages.
- **citation_parser** — extracts citation spans + normalizes surface variants (`Section 210.2-01(c)(3)`, `17 CFR 210.2-01[c][3]` → one canonical form). Passes through honest "from general knowledge" disclaimers.
- **verifier** — the core gate: `GROUNDED_EXACT` keep · `GROUNDED_BASE` keep + flag the paragraph unverified · `OUT_OF_CORPUS_STUB` / `FAMILY_ABSENT` / `UNRECOGNIZED` **strip + flag**. It confirms citation **existence**, never claim **correctness**.
- **confidence** — labels each answer **GROUNDED / PARTIAL / GENERAL-KNOWLEDGE / DEFER** so the auditor knows where to check.
- **recommender** — orchestrates end to end; the model and RAG are a drop-in.

## Headline results (AssuranceBench test split, verified path, judge-scored)

- **Zero dangerous conclusions — the zero-tolerance criterion is met.** Judge-confirmed across the full safety suite (AssuranceBench v1.0.1) (mean 1.00, 27/27); every hard-severity judgment call defers. The safety gate **passes** (0 dangerous conclusions *and* 100% non-dangerous pass-rate, 27/27).
- **Zero fabricated citations shown** — machine-checked (recount PASS): re-parsing every shown answer yields 0 ungrounded citations (out-of-corpus stubs / legacy AU-family / fabricated all stripped and flagged).
- **~0.88 on in-scope (PCAOB/SEC) citation lookups** (n=29, 26/29 passed). The blended benchmark figure is **0.71** because the 7 out-of-scope FASB/ASC items are correctly **declined** rather than answered (mean 0.00). 100% of in-scope expected citations resolve against the corpus.
- Runs locally, free per query, on 36 GB unified memory.

## Known limitations (read these)

- **Existence ≠ correctness — this is recommend-and-verify, not professional advice.** The verification layer guarantees every shown citation **exists** in the corpus; it does **not** check that the answer is correct or that the cited standard is the *right* one. The model can cite a real standard for the wrong topic, or describe a real standard inaccurately — failures the layer **cannot detect**. Always read the cited passages and confirm the claims yourself.
- **FASB/GAAP boundary.** The corpus carries FASB ASC only as **topic stubs** (no licensed Codification prose), so GAAP *disclosure-text* questions are deliberately out of scope and decline to **GENERAL-KNOWLEDGE**. Coverage is **complete *public* coverage**, honestly scoped.
- **~0.88, not 1.0, on in-scope citation accuracy.** The model recommends; the verifier guarantees every shown citation is real or flagged; the auditor checks. Accuracy is acceptable because checking is now fast and trustworthy — no goose chases, no confident fabrications.
- **Model/retrieval residue.** A few "can our firm do X" questions are over-refused by the model, and retrieval occasionally misses an in-corpus provision — documented, flagged for a future retrieval/SFT pass.

## Repo layout

```
auditlm/
  data/          public corpus (PCAOB, SEC 10-K/regs/SAB, GAGAS, FASB stubs) + collectors
  rag/           hybrid retrieval + grounding-prompt layer
  training/      SFT data + QLoRA configs (MLX, local) + serving
  verify/        verification layer (index, parser, verifier, confidence, recommender)
  demo/          55-question demo corpus through the verified recommender
  CORPUS_STATS.md
```

## Quickstart

```bash
# 1. retrieval + verification env
python -m venv rag/.venv && rag/.venv/bin/pip install -r requirements.txt
# 2. serve the model (Ollama) — see training/SERVING.md
ollama create auditlm-run2 -q q4_K_M -f training/fused/Modelfile_auditlm-run2
# 3. ask
rag/.venv/bin/python -c "import sys; sys.path.insert(0,'verify'); \
  from recommender import Recommender; \
  print(Recommender().recommend('Which PCAOB standard governs the consideration of fraud?')['labeled_answer'])"
```

## Running the demo UI

A local Gradio UI (`demo/ui.py`) for trying the verified recommender interactively. Runs on **Windows, Linux, or macOS** — it's a local app, not a hosted service.

**Prerequisites**
1. **Ollama** installed and running ([ollama.com](https://ollama.com) — Windows / Linux / macOS), serving the model: `ollama create auditlm-run2 -q q4_K_M -f training/fused/Modelfile_auditlm-run2`.
2. **Python env** with the requirements (gradio included): `pip install -r requirements.txt` (use your venv's python below).
3. **Corpus + index present** — `data/corpus/chunks.jsonl` and the retrieval index under `rag/index/` (built once; the citation index `verify/data/identifier_index.json` rebuilds on first run).

**Launch**
```bash
# macOS / Linux
rag/.venv/bin/python demo/ui.py
# Windows (PowerShell / cmd)
rag\.venv\Scripts\python demo\ui.py
```
Opens **http://127.0.0.1:7860**, local-only (`share=False`). Paths resolve relative to the repo, so it works wherever you clone it; no OS-specific path edits needed.

**What it shows** — a colored **confidence badge** (GROUNDED / PARTIAL / GENERAL-KNOWLEDGE / DEFER), the **answer**, and a collapsible **verification panel** (per-citation checks, what was stripped, source passage ids). Framing is **recommend-and-verify**: every citation is checked for existence, not correctness.

**Caveats** — it runs **locally on any OS** given Ollama + the served model; it is **not** a hosted web app. Hosting it (e.g. Hugging Face Spaces) would require rewiring model serving off local Ollama and a GPU to run the 8B model — a separate future step. As everywhere: **recommend-and-verify, not professional advice.**

## Sourcing & license

**Corpus & data** — public, free-to-use sources only: PCAOB standards/inspections/enforcement, SEC EDGAR filings + regulations (Reg S-X / S-K) + SABs, GAO Yellow Book (GAGAS), and FASB ASC **topic stubs** (no licensed prose). No firm-proprietary methodology. See [LICENSE](LICENSE). Affiliation: **Independent Researcher**.

**Base model — Built with Llama.** The fine-tuned model (**Llama-AuditLM**, served locally via the `auditlm-run2` Ollama tag) builds on Meta's **Llama 3.1-8B** and is governed by the **[Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)** (Copyright © Meta Platforms, Inc.). Per that license, distributed Llama derivatives carry the **"Llama"** name prefix — hence **Llama-AuditLM** for the model — while the project and repository names (**AuditLM**) are unaffected.

> Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

**LLAMA_LICENSE note** — these repositories do **not** contain model weights (the Llama Materials; they are gitignored), so a link to the agreement suffices here. A full copy of the Llama 3.1 Community License will be included **alongside the weights** (as `LLAMA_LICENSE`) when the model is released/hosted (e.g. on the Hugging Face Hub), as the license requires when redistributing Llama Materials.

## Citation

> Nakrani, V. (2026). *AuditLM: trust-by-architecture for an open US-audit assistant.* (preprint forthcoming)
