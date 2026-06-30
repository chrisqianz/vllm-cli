# vllm-cli v0.2.9.7 Release Notes

## Overview

vllm-cli v0.2.9.7 adds full support for **vLLM v0.24.0**, including the new Streaming Parser Engine, Model Runner V2, DeepEP v2, and Diffusion LLM support.

## Key Changes

### BREAKING: Device Selection Change
vLLM v0.24.0 no longer sets `CUDA_VISIBLE_DEVICES` internally. Use `--device-ids` instead:
```bash
# Old (deprecated)
CUDA_VISIBLE_DEVICES=0,1 vllm-cli serve --model ...

# New (v0.24.0+)
vllm-cli serve --model ... --device-ids 0,1
```

### Streaming Parser Engine
Unified tool-call/reasoning parsing across models. New parsers available:
- `qwen3` — Qwen3 series
- `minimax_m2` — MiniMax-M2
- `glm47` / `glm51` / `glm52` — GLM 4.7/5.1/5.2
- `nemotron_v3` — Nemotron V3

### 60 New CLI Arguments
| Category | Count | Highlights |
|----------|-------|------------|
| Data Parallel | 8 | `--data-parallel-backend`, `--data-parallel-external-lb` |
| LoRA | 6 | `--lora-dtype`, `--lora-target-modules`, `--fully-sharded-loras` |
| Multimodal | 13 | `--mm-encoder-attn-backend`, `--mm-processor-cache-gb` |
| Model Loading | 5 | `--safetensors-load-strategy`, `--model-weights` |
| KV Cache | 4 | `--kv-sharing-fast-prefill`, `--num-gpu-blocks-override` |
| NUMA | 2 | `--numa-bind-cpus`, `--numa-bind-nodes` |
| Scheduler | 3 | `--prefill-schedule-interval`, `--sliding-window` |
| Monitoring | 3 | `--aggregate-engine-logging`, `--collect-detailed-traces` |
| Other | 16 | `--diffusion-config`, `--device-ids`, `--tokenizer-revision` |

### 13 Deprecated Arguments
These parameters are removed in vLLM v0.24.0 but remain in schema for backward compatibility:
`--swap-space`, `--cpu-offload-space`, `--num-lookahead-slots`, `--num-speculative-tokens`, `--speculative-model`, `--rope-scaling`, `--rope-theta`, `--guided-decoding-backend`, `--generation-config-override`, `--limit-per-prompt`, `--tq-max-kv-splits-for-cuda-graph`, `--disable-async-output-proc`, `--disable-frontend-multiprocessing`

### New Profiles
- `diffusion_gemma` — For DiffusionGemma and diffusion-based LLMs
- `deepseek_v4` — Optimized for DeepSeek-V4 with FlashInfer sparse index cache

## vLLM v0.24.0 Highlights

- **Model Runner V2**: Quantized models by default, GraniteMoE default, DFlash speculative decoding
- **DeepEP v2**: Integrated for expert parallelism
- **Rust Frontend**: API server migrated to Rust with new endpoints
- **Diffusion LLMs**: DiffusionGemma support with CPU path
- **DeepSeek-V4**: FlashInfer sparse index cache, prefill chunk-planning optimization
- **Gemma 4**: Unified FlashAttention (FA4) across all layers

## Schema Updates
- Schema version: 2.0.2 → 2.1.0
- Total arguments: 216 → 276
- Last synced: v0.23.0 → v0.24.0
- vLLM versions supported: 0.20.0 - 0.24.0

## Migration Guide

### For Users with Existing Configs
Your existing configurations will continue to work. However:
1. Replace `CUDA_VISIBLE_DEVICES` with `--device-ids` for GPU selection
2. Profiles using `swap_space` should migrate to `kv_cache_memory_bytes`
3. `num_speculative_tokens` is now controlled via `speculative_config`

### For New Deployments
```bash
# Install
pip install vllm-cli

# Sync CLI args from v0.24.0
vllm-cli recipes --sync-args --tag v0.24.0 --apply-args

# Serve with device_ids
vllm-cli serve --model Qwen3-32B --device-ids 0 --tool-call-parser qwen3
```

## Files Changed
- `src/vllm_cli/schemas/argument_schema.json` — 60 new args, 13 deprecated
- `src/vllm_cli/schemas/default_profiles.json` — 2 new profiles
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.24.0 to supported versions
- `VERSION` — 0.2.9.6 → 0.2.9.7
- `CHANGELOG.md` — Added v0.2.9.7 entry
