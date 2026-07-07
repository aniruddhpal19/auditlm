# QLoRA Run 1 (training record)

Local MLX-LM QLoRA on **Llama-3.1-8B-Instruct** (4-bit), the same base the baseline serves
as `ollama:llama3.1:8b`. Completions-only SFT on **357 demos** (343 capability + 14 safety;
30 capability held out for validation) from the contamination-cleared training set.

## Config (`configs/mlx_lora_run1.yaml`)
- base: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- LoRA: rank 16, scale 20, dropout 0.05, on attention+MLP of the last 16 layers
  (20.97M trainable params, 0.26%); `mask_prompt: true` (train on the completion only)
- optimizer adamw, lr 1e-5 (constant), batch 2, max_seq 4096, grad-checkpoint on, seed 42
- 358 iters = 2 epochs. Saved at iter 179 (1 epoch) and 358 (2 epochs).
- **MLX conventions:** mlx `scale` is a direct LoRA multiplier (not HF `alpha/r`); lr 1e-5
  is the mlx-LoRA default (HF QLoRA's 2e-4 is a bitsandbytes convention). bitsandbytes is
  CUDA-only, so MLX is the M3 Max local path. grad-checkpoint required —
  it OOM'd at 36 GB without it; peak mem with it was 12 GB.
- time: ~2h20m on the M3 Max (~22 s/iter on long grounded-prompt sequences).

## Loss curve

| iter | train | val |
|---|---|---|
| 1 | 1.63 | 1.55 |
| 60 | 1.38 | **1.24** |
| 120 | 1.34 | 1.43 |
| **179/180 (1 epoch)** | 1.43 | **1.20  ← best val** |
| 240 | 1.10 | 1.40 |
| 300 | 1.07 | 1.30 |
| **358 (2 epochs)** | **0.89** | 1.27 |

(train sampled every 10 iters; val at 1/60/120/180/240/300/358, val_batches 8 ≈ 16 items.)

## Read — 2 epochs OVERFIT; epoch 1 is the better-generalizing checkpoint
In **epoch 2, train loss fell ~0.5 (1.43 → 0.89) while val loss did not improve past its
epoch-1 best (1.20), ending worse (1.27)**. The train–val gap widened from 0.23 (iter 180)
to 0.38 (iter 358) — the model began memorizing the 357 demos. **Best val = 1.199 at the
end of epoch 1.** So 2 epochs was one epoch too many at this data scale.

Caveat: the held-out set is small/noisy (16 items), so val loss bounces; the *pattern*
(widening gap, no epoch-2 val gain) is the reliable signal, and the benchmark eval
is the final arbiter.

**Recommendation:** evaluate the **1-epoch adapter** (`adapter/0000179_adapters.safetensors`)
as primary in the benchmark eval; optionally eval the 2-epoch as a quick ablation to confirm.

## Adapters (`training/adapter/`, gitignored — large; regenerable from this recipe)
- `0000179_adapters.safetensors` — 1 epoch (recommended)
- `0000358_adapters.safetensors` = `adapters.safetensors` — 2 epochs
- `adapter_config.json`
- Load: LoRA adapter on `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`. Serving for eval:
  fuse the chosen adapter → GGUF (llama.cpp) → Q4_K_M → `ollama create auditlm` → eval
  `rag:ollama:auditlm` (tiered grounding).
