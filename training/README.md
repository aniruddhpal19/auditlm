# SFT + QLoRA (Base + RAG + SFT)

Fine-tune Llama 3.1-8B (QLoRA) on audit-task demonstrations to teach the five target
behaviors identified in the RAG baseline (see [`rag/PHASE3A_RAG_RESULTS.md`](../rag/PHASE3A_RAG_RESULTS.md)),
then evaluate **Base + RAG + SFT** on the frozen AssuranceBench `test` split.

## Target behaviors
1. **Cite precisely when grounded** (keep the citation win, never confabulate a number).
2. **Reason fully when retrieval is thin** (keep the disclosure/filing recovery).
3. **Defer the conclusion even with the rule in hand** (fix the −0.22 independence regression).
4. **Procedure-design reasoning** (base near-floor; RAG can't teach it).
5. **Disclosure knowledge as a skill** (corpus lacks FASB prose → must know it parametrically).

## Base model (apples-to-apples)
`meta-llama/Llama-3.1-8B-Instruct` — the same model the baseline serves as
`ollama:llama3.1:8b`. SFT trains a LoRA adapter on it; eval serves the merged result
through Ollama with the RAG layer (tiered grounding), so Base → Base+RAG+SFT is a clean
comparison.

## Fine-tuning stack

**bitsandbytes 4-bit QLoRA is CUDA-only**, so the two paths differ by hardware:

| | **Local (M3 Max) — MLX-LM** *(used for run 1)* | **Cloud (A100/H100) — HF QLoRA** |
|---|---|---|
| tool | `mlx-lm` (`mlx_lm.lora`) — Apple Metal on this hardware | `peft` + `trl` SFTTrainer + `bitsandbytes` (or `unsloth`) |
| base | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (un-gated 4-bit) | `meta-llama/Llama-3.1-8B-Instruct` (HF-gated → license + `HF_TOKEN`) |
| quant | 4-bit base + LoRA (QLoRA-equivalent) | 4-bit nf4 base + LoRA (true bnb QLoRA) |
| cost / time | $0 / ~30–90 min, fits 36 GB | ~$1.5–2.5/hr × ~1–2 hr (<$20) + setup |
| friction | install `mlx-lm`, ~5 GB download | rent GPU, HF license, upload data |

Both train a LoRA adapter on Llama 3.1-8B Instruct; serving converges:
**fuse/merge → GGUF (llama.cpp) → quantize Q4_K_M → `ollama create auditlm` → eval
`rag:ollama:auditlm`**. Run 1 went with local MLX-LM: zero cost, no gating, matches the
local-first ethos, and it's ample for ~500 demos.

## QLoRA config (run 1)
See [`configs/qlora_run1.json`](configs/qlora_run1.json): r=16, alpha=32, dropout 0.05,
target = attention + MLP projections; **2 epochs**, LR 2e-4 cosine, warmup 0.03, effective
batch 8 (1×grad-accum 8), **max_seq_len 4096**, **train on completions only** (mask the
prompt), seed 42, Llama-3.1 chat template. Modest by design — guards overfitting at this
data scale.

## Demo format
Demos use the **grounded-prompt format**: input = the full grounded prompt (tiered
policy + retrieved hybrid passages + question), completion = the ideal answer. Behaviors
1–3 are about *how to use retrieved passages*, so the demos have to match the deployment
format: the teacher (Opus) sees the same grounded context the student will, and models
cite-from-passages / reason-when-thin / defer-even-with-rule in-context. The lighter
alternative — question → answer with no passages — teaches answer style but not the
in-context grounding behavior, which is why it was rejected.

## Layout
```
training/
  README.md                 # this doc
  configs/qlora_run1.json    # run-1 hyperparameters
  data/                      # capability_demos.jsonl, safety_demos.jsonl (merged), train.jsonl
  scripts/                   # generation, corpus-verification, contamination gate, train, serve
  adapter/                   # trained LoRA adapter (gitignored — large)
```

## Contamination
Capability demos are generated from the **task taxonomy + corpus**, never from the 164 test items (v1.0.1). The dev split (45) is **style-anchor only** (read for format, never trained on). The
contamination checker (shingle overlap vs all 164 test items) is a **hard stop before
training** — training cannot start until the set is verified clean.

## Honest framing
Distillation bounds capability **near (below) the teacher** — target is "close most of the
gap to Opus, run locally/cheap," **not** match Opus. Safety is where it can **exceed** Opus
(hand-curated deferral demos > the teacher's own deferral, which fails the gate).
