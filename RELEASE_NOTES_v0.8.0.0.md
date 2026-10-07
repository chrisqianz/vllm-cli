# vllm-cli v0.8.0.0 Release Notes (2026-10-07)

## Overview

vllm-cli v0.8.0.0 adds full support for **vLLM v0.31.0** (717 commits from 307 contributors, 96 new). Argument schema updated to **v2.8.0** with **338 arguments** (9 new, 2 deprecated) and synced against `vllm/engine/arg_utils.py` (EngineArgs) and `vllm/entrypoints/launchers/cli_args.py` (FrontendArgs) at tag v0.31.0.

This is a **major version bump** (0.7.0.1 → 0.8.0.0), matching the vLLM minor-version increment to 0.31.0.

## Highlights

### 🚀 Fast Restart: Preload Daemon & Engine Snapshots

`vllm preload` starts a weight-cache daemon that keeps post-quantized weights GPU-resident for instant engine restarts — now DP-aware, with MTP draft support and a `/health` endpoint (#56680, #57386, #57312, #58552). Experimental CRIU-based engine snapshots via `vllm snapshot create/restore` take single-node TP1 engines from cold boot to serving in seconds (#51360).

### ⚡ DeepSeek-V4.1-Flash Performance

FlashMLA mega attention with NVFP4 compressed KV is the new SM100 default (#56935); DeepGEMM sparse MQA logits (#56254), Mega-Gate (#56266), fused TP all-reduce + mHC + MoE finalize, Engram wkv TP-sharded host tables shared across DP replicas, vision-tower CUDA graphs, and SWA bounded replay (#56227) stack into a major throughput step for DeepSeek-V4.1-Flash serving.

### 🎯 Speculative Decoding on Model Runner V2

Draft-model speculative decoding lands on Model Runner V2 (#43091) with custom logits processors (#56497), the LiLiCorr drafter (#57934), DFlash async scheduling (#58065), and DSpark adaptive verification for Gemma4 (#57263).

### 🌐 Large-Scale Serving

MoonEP all2all backend (`--all2all-backend moonep`, #52101), PCP+DP (#57075), low-SM multimem reduce-scatter, DeepEPv2 + sequence parallelism, sharding-aware NCCL M2N weight transfer, and KV-offload back-pressure (#50045).

### 🧠 Scheduling Controls

New `--max-num-active-seqs` admission cap decouples scheduler RUNNING size from `max_num_seqs` (#56758); `--long-prefill-token-threshold-adaptive` floors the long-prefill threshold at a fair share of the token budget (#57951, #58459); KV-holding requests are scheduled first (#58947); a KV-connector + MTP deadlock was fixed (#57104).

### 🔒 Security Hardening

Per-request `mm_processor_kwargs` / `media_io_kwargs` are rejected unless the server is started with `--trust-request-mm-kwargs` (#58830); prefix-cache block hashes now carry source tags (#51899) and the active LoRA path (#59335).

### 🤖 New Models & Parsers

- **Granite 4.2** thinking support via new `granite_thinking_parser` (#55957)
- **DiffusionGemma** structured output (#57250); **MiMo V2 MXFP4**; **GLM-5.2-MXFP4** on ROCm; **Quark DeepSeek-V4.1-Flash-MXFP4** and **GLM-5.3-Flash MXFP4** checkpoints
- LoRA: Nemotron VL, ModernBert, VoyageQwen3, RoBERTa sequence classification, per-adapter `num_labels` (`--max-lora-cls-labels`)

### 🌍 API Additions

`--tool-strict-level` (#56268), `POST /release_kv_cache_memory` (#44890), DeepSeek-V4 FIM completion via suffix (#44229), Anthropic-style thinking blocks on `/v1/messages` (#58613), native Lark grammar support (#58321), per-request spec-decode metrics.

## Schema Changes (v2.7.0 → v2.8.0)

### New CLI Arguments (9)

| Argument | Type | Notes |
|---|---|---|
| `--aux-output-config` | JSON string | Auxiliary outputs config (routed-experts return, `max_bytes` LRU cap) |
| `--enable-mamba-shared-prefix-checkpoint` | bool | Mamba align checkpoint at shared-prefix junction; renamed from fine-grained variant (#57382) |
| `--log-level` | choice | CRITICAL / ERROR / WARNING / INFO / DEBUG / NOTSET — replaces `VLLM_LOGGING_LEVEL` |
| `--logging-config` | JSON string | Structured logging config, incl. `pylogging_config_file` |
| `--long-prefill-token-threshold-adaptive` | bool | Floor long-prefill threshold at a fair share of the token budget (#57951, #58459) |
| `--max-lora-cls-labels` | integer | Output size for LoRA classification heads |
| `--max-num-active-seqs` | integer | Scheduler RUNNING admission cap, decoupled from `max_num_seqs` (#56758) |
| `--sleep-preserve-parameter-names` | list | Parameter-name globs preserved across level-2 sleep |
| `--swa-bounded-replay` | bool (**default True**) | Keep sliding-window KV out of prefix caching; replay last window on hit (#56227) |

`--device-memory-utilization` is a new alias of `--gpu-memory-utilization` (#56547); it shares the same destination field, so vllm-cli tracks it under `gpu_memory_utilization` without a separate schema entry.

### Deprecated (upstream vLLM 0.31.0)

- `--enable-mamba-fine-grained-prefix-cache` — removed in vLLM 0.31.0, renamed to `--enable-mamba-shared-prefix-checkpoint` (#57382)
- `--log-config-file` — deprecated now, removal in vLLM v0.33.0; use `--logging-config` with `pylogging_config_file`

### Expanded Choices

- `--reasoning-parser`: 32 → **35** (backfilled `deepseek_v41`, `k2_horizon` from the 0.30 cycle; new `granite_thinking_parser`)
- `--tool-call-parser`: 49 → **51** (backfilled `deepseek_v41`, `k2_horizon`)

### New Parsers

- **Reasoning parsers** (34 → 35): `granite_thinking_parser` (Granite 4.2, #55957)

## Breaking Changes & Deprecations (upstream vLLM)

- Per-request multimodal kwargs (`mm_processor_kwargs`, `media_io_kwargs`) rejected unless the server runs with `--trust-request-mm-kwargs` (#58830)
- `tokenizer_mode="slow"` removed (#58545)
- Online `quantization="fp8"` is now shorthand for `fp8_per_tensor` (#53585)
- `--enable-mamba-fine-grained-prefix-cache` removed — use `--enable-mamba-shared-prefix-checkpoint` (#57382)
- AllSpark INT8 W8A16 removed (#58001)
- `--enforce-eager` now also disables JIT warmup (#58197)
- `VLLM_XPU_ENABLE_XPU_GRAPH` removed; XPU graphs are default-on (#51600)
- Comma-separated `--collect-detailed-traces` values removed (#55702)
- `VLLM_PLE_CPU_OFFLOAD` removed — use `--engram-config` (#57937)
- MRV1 + PP>1 + async scheduling + structured output is now rejected — use Model Runner V2 (#56250)
- SWA bounded replay is **on by default** for DeepSeek-V4.1 (except ROCm) (#56227, #57906)

## Dependencies

- Requires `vllm>=0.20.0,<0.32.0` (updated from `<0.31.0`)
- vLLM 0.31.0 runtime bumps you inherit: FlashInfer 0.7.0.post1, Transformers 5.17.0 (upper bound now enforced, #59614), XGrammar 0.2.7; `oss-harmony` replaces `openai-harmony`
- Python >= 3.10

## Migration Guide

### For Users Upgrading from v0.7.0.x

1. Profiles are forward-compatible; no profile migration required.
2. If any profile passes `--enable-mamba-fine-grained-prefix-cache`, rename it to `--enable-mamba-shared-prefix-checkpoint`.
3. If clients send per-request multimodal kwargs, start servers with `--trust-request-mm-kwargs` (or move those settings server-side).
4. Replace `--log-config-file` with `--logging-config` (JSON, `pylogging_config_file`) before vLLM 0.33 removes it.
5. For SWA models other than DeepSeek-V4.1, review `--swa-bounded-replay` (default True) and disable it if prefix-cache hit patterns regress.

```bash
# Example: DeepSeek-V4.1-Flash style serving with new 0.31 capabilities
vllm-cli profile create dsv41-flash \
  --model DeepSeek-V4.1-Flash \
  --max-num-active-seqs 256 \
  --long-prefill-token-threshold-adaptive \
  --log-level INFO

# Instant restarts after config tweaks (weights stay GPU-resident)
vllm-cli preload --model DeepSeek-V4.1-Flash
```

## Files Changed

- `src/vllm_cli/schemas/argument_schema.json` — v2.8.0, 338 args, last_synced v0.31.0
- `src/vllm_cli/config/cli_args_sync.py` — SUPPORTED_VLLM_VERSIONS + v0.31.0
- `src/vllm_cli/config/parser_sync.py` — granite_thinking_parser mapping
- `src/vllm_cli/config/command_import.py` — BOOLEAN_FLAGS + 3 new 0.31 flags (importer no longer swallows the following token)
- `pyproject.toml` — version 0.8.0.0, vllm<0.32.0
- `VERSION`, `CHANGELOG.md`, `README.md`, this file
