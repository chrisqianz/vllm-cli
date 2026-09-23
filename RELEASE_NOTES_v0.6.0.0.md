# vllm-cli v0.6.0.0 Release Notes (2026-09-23)

## Overview

vllm-cli v0.6.0.0 adds full support for **vLLM v0.30.0** (762 commits from 315 contributors, 104 new; published 2026-09-22). Argument schema updated to **v2.7.0** with **329 arguments** (9 new, 2 deprecated) and synced against `vllm/engine/arg_utils.py` (EngineArgs, 243 fields) and `vllm/entrypoints/launchers/cli_args.py` (FrontendArgs, 58 fields) at tag v0.30.0.

This is a **major version bump** (0.5.0.0 → 0.6.0.0), matching the vLLM minor-version increment to 0.30.0.

## Highlights

### 🤖 New Models

- **DeepSeek-V4.1-Flash**: whole KV stored in MXFP8 through the FlashMLA V4.1 record on SM100, DeepGEMM Mega-mHC, async Engram prefetch with Engram DP sharding, XGrammar V4.1 schema constraints for strict tool parameters
- **DeepSeek-V4-Flash-Vision-Exp**: multimodal variant, ROCm support and LoRA
- **DeepSeek-V4 CPU backend**: AVX512/AMX sparse MLA, indexer, mHC and compressor kernels
- **GLM-5.3-Flash**: EPLB, FlashKDA for KDA chunked prefill (1.7-3.8x faster than Triton chunk path), sparse prefill for NoPE layout
- **K2-Horizon** (with dedicated reasoning + tool parsers), **Cohere Compass**, **Bailing V3 VL** (with MTP), **Nanbeige4.2** via Transformers backend

### ⚡ Fast Start: Weight-Cache Daemon

A persistent per-GPU weight-cache daemon keeps post-quantized, TP-sharded weights in GPU memory; restarting engines map them over CUDA IPC with `--load-format ipc_cache` instead of reloading from disk. Now covers FP4 checkpoints and multi-node TP.

### 💧 Watermarking (New)

Gumbel-max watermarked generation and detection with a keyed PRF, per-request opt-out, and an example detection endpoint. Dual-key Gumbel-max keeps it compatible with speculative decoding; configured through `--watermark-config` (JSON).

### 🗄️ HiSparse: Host-Resident KV Tier

Sparse-MLA decode can spill KV pages to pinned host memory under GPU pressure and serve top-k misses from a per-request GPU hot buffer via `HiSparseConnector`, with Prometheus counters and a host cache shared across TP ranks.

### 🏗️ Model Runner V2

Dual-batch overlap in eager mode and with FULL CUDA graphs; MTP and EAGLE3/DFlash/DSpark speculative decoding under pipeline parallelism; adaptive verification for every draft-model speculator via an online acceptance estimator; GC frozen during graph capture cuts capture time 12s→2s and engine init 28.9s→8.2s (H200).

### 🧠 Kimi K3 & Qwen3.8-Flash-Next Performance

- Kimi K3: KDA mixed-batch gather/scatter removal (5.2-7.7% E2E throughput), grouped FP8 MLA cache insertion (4-6x at small batch), FlashInfer KDA kernels, overlapped TP8 KDA projections, native CUDA AttnRes default on SM100, symmetric DCP disaggregation for hybrid Mamba models
- Qwen3.8-Flash-Next: separate prefill/decode QSA indexer kernels, fused PLE kernels, FP8 indexer cache, padded-index skipping in sparse GQA, UVA PLE offload and Engram tensor parallelism via `--engram-config`

### 🌐 Large Scale Serving

PCP+DCP on sparse-MLA models, elastic EP scaling (`--elastic-ep-max-dp-size`), DP step sync tuning (`--dp-sync-interval`), NCCL communicator suspend/resume (`--enable-nccl-comm-suspend`), scale-out endpoint gating (`--enable-scale-out`).

## Schema Changes (v2.6.0 → v2.7.0)

### New CLI Arguments (9)

| Argument | Type | Notes |
|---|---|---|
| `--engram-config` | JSON string | Engram N-gram lookup: UVA offload, tensor/DP sharding |
| `--watermark-config` | JSON string | Gumbel-max watermarking policy |
| `--kda-decode-backend` | choice | auto / native / flashinfer / triton |
| `--sparse-indexer-topk-backend` | choice | auto / deep_select / cooperative / persistent / per_row / flashinfer / torch |
| `--dp-sync-interval` | int (16) | Steps between data-parallel rank syncs |
| `--elastic-ep-max-dp-size` | int | Max DP size for elastic EP scaling |
| `--enable-mamba-fine-grained-prefix-cache` | bool | Fine-grained prefix caching for hybrid SSM models |
| `--enable-nccl-comm-suspend` | bool | Free GPU memory by suspending idle NCCL comms |
| `--enable-scale-out` | bool | Register `/render`, `/derender`, `/inference/v1/generate` on `vllm serve` |

### Deprecated (removed upstream in vLLM 0.30.0)

- `--default-max-num-batched-tokens` — removed; use `--max-num-batched-tokens` or scheduler config overrides
- `--enable-bf16x3-router-gemm` — removed; BF16x3 router GEMM is now default-on for SM100

### Expanded Choices

- `--spec-method`: 39 → **40** (added `glm5_next_mtp`)
- `--moe-backend`: 20 → **22** (added `aiter_triton_mxfp4_bf16`, `rdna3`)
- `--linear-backend` (21) and `--quantization` (30) unchanged; Mamba backends/algorithms unchanged

### New Parsers

- **Tool call parsers** (49 → 51): `deepseek_v41`, `k2_horizon`
- **Reasoning parsers** (32 → 34): `deepseek_v41`, `k2_horizon`

## Breaking Changes & Deprecations (upstream vLLM)

- Scale-out endpoints (`/render`, `/derender`, `/inference/v1/generate`) are no longer registered on plain `vllm serve` unless `--enable-scale-out` is passed; `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` removed
- GPTQ group/dynamic activation ordering removed: `g_idx` ignored; related Marlin/GPTQ/CPU/RDNA3 kernels gone
- `all` Mamba cache mode deprecated — falls back to Model Runner V1
- `python -m vllm.entrypoints.grpc_server` deprecated — use `vllm serve --grpc`
- YaRN aligned with Transformers: derived `max_model_len` drops for some models (e.g. 131072 → 32768); `attn_factor`/`extrapolation_factor` ignored
- Attention implementations must explicitly declare DCP support; unsupported combos now fail at backend selection
- Default audio resampler switched from PyAV to torchaudio
- Env vars removed: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, `VLLM_MM_HASHER_ALGORITHM` (use config fields — already supported by vllm-cli), `VLLM_TEST_FORCE_FP8_MARLIN` era aliases
- New defaults: FlashInfer CuTeDSL NVFP4 W4A16 over Marlin (SM100/103), W4A4 NVFP4 (SM120/121), BF16x3 router GEMM (SM100), DeepEP v2 combine overlap, AITER custom AG/RS (ROCm)

## Dependencies

- Requires `vllm>=0.20.0,<0.31.0` (updated from `<0.30.0`)
- Python >= 3.10

## Migration Guide

### For Users Upgrading from v0.5.0.0

1. Profiles are forward-compatible; no profile migration required.
2. If you used `--enable-bf16x3-router-gemm`, remove it — the behavior is now the default on SM100.
3. If you relied on `/render`-style endpoints on `vllm serve`, add `--enable-scale-out`.
4. Check `max_model_len` for YaRN-rotated models after upgrading vLLM (some defaults shrink).

```bash
# Example: DeepSeek-V4.1-Flash style serving with new 0.30 capabilities
vllm-cli profile create dsv41 \
  --model DeepSeek-V4.1-Flash \
  --kv-cache-dtype auto \
  --spec-method dspark \
  --sparse-indexer-topk-backend flashinfer

# Elastic EP + DP tuning for large deployments
vllm-cli serve --profile dsv41 \
  --enable-scale-out \
  --dp-sync-interval 8 \
  --elastic-ep-max-dp-size 32
```

## Files Changed

- `src/vllm_cli/schemas/argument_schema.json` — v2.7.0, 329 args, last_synced v0.30.0
- `src/vllm_cli/config/cli_args_sync.py` — SUPPORTED_VLLM_VERSIONS + v0.30.0
- `src/vllm_cli/config/parser_sync.py` — v0.30.0 tool/reasoning parser mappings
- `pyproject.toml` — version 0.6.0.0, vllm<0.31.0
- `VERSION`, `CHANGELOG.md`, `README.md`, this file
