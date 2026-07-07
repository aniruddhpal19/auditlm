# Serving the SFT model for evaluation

The trained LoRA adapters are served through **Ollama** so the existing AssuranceBench
runner evaluates **Base + RAG + SFT** exactly like the prior configs — same test split,
same judge, same scoring, **tiered grounding** (the SFT default), apples-to-apples
with the RAG-tiered baseline.

## Serving path
`mlx_lm.fuse` (dequantize the 4-bit base + fuse the LoRA → f16 HF model) → `llama.cpp`
`convert_hf_to_gguf` (f16 GGUF) → `ollama create -q q4_K_M` with the **exact llama3.1:8b
chat template + stop params**. (mlx's own GGUF export is broken for this model — "can only
serialize row-major arrays" — so llama.cpp does the conversion.) The result is a Q4_K_M
Ollama model the **same 4.9 GB size as the baseline `llama3.1:8b`**.

Reproduce with [`scripts/build_ollama_model.sh`](scripts/build_ollama_model.sh):
```bash
scripts/build_ollama_model.sh adapter_1ep auditlm        # 1-epoch (primary)
scripts/build_ollama_model.sh adapter_2ep auditlm-2ep    # 2-epoch (ablation)
```
(adapter_1ep / adapter_2ep each = `adapters.safetensors` (the chosen checkpoint) +
`adapter_config.json`, copied from `adapter/0000179_*` and `adapter/0000358_*`.)

## Models created (verified serving)
| ollama model | adapter | eval spec |
|---|---|---|
| `auditlm` | 1-epoch (best val) — **primary** | `rag:ollama:auditlm` |
| `auditlm-2ep` | 2-epoch — **ablation** | `rag:ollama:auditlm-2ep` |

Smoke-tested end-to-end through the runner (`rag:ollama:auditlm` on a citation dev item):
retrieval → tiered grounded prompt → SFT model → exact_citation PASS. (Direct generate
without RAG hallucinates — confirming SFT learned to *use* the retrieved passages.)

## Run the evaluations (the judge needs ANTHROPIC_API_KEY)
From the `assurancebench` directory, with `AUDITLM_RAG` set and the rag venv:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export AUDITLM_RAG=/path/to/auditlm                        # your auditlm checkout
"$AUDITLM_RAG"/rag/.venv/bin/python -m src.runner \
  --split test --judge --model "rag:ollama:auditlm"        # PRIMARY (1-epoch)

"$AUDITLM_RAG"/rag/.venv/bin/python -m src.runner \
  --split test --judge --model "rag:ollama:auditlm-2ep"    # ABLATION (2-epoch)
```
Resumable/incremental (same resilient runner). Outputs:
`runs/rag_ollama_auditlm_test_{results.jsonl,scorecard.md}` and the `-2ep` equivalents.

## Note (minor confound, documented honestly)
The QLoRA base is 4-bit MLX, so the served model went 4-bit → dequant-f16 → Q4_K_M, vs the
baseline's f16 → Q4_K_M. This adds minor quantization noise relative to the no-SFT configs;
the SFT delta dominates, and it's inherent to the QLoRA-on-4-bit local path. Eliminating it
(train on f16) is a run-two option.

## Artifacts
- `adapter/`, `adapter_1ep/`, `adapter_2ep/`, `fused/` — gitignored (large; regenerable).
- Ollama models `auditlm`, `auditlm-2ep` live in the local Ollama store.
