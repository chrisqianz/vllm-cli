# vllm-cli v0.5.0.0 Release Notes (2026-09-10)

## Overview

vllm-cli v0.5.0.0 adds full support for **vLLM v0.29.0** (594 commits from 277 contributors, 91 new!), headlined by **Model Runner V2 becoming the default for all models**, the Hy4-preview megamodel (Tencent's 770B/49B-active MoE with Gated DeepSeek Sparse Attention and native MTP), Qwen3.8-Flash-Next, major Kimi-K3/DeepSeek-V4 performance work, queue admission control, and per-request speculative decoding metrics.

This is a **major version bump** from 0.4.0.0 to 0.5.0.0, reflecting the vLLM major version upgrade from 0.28 to 0.29.

## Highlights

### 🏗️ Model Runner V2 Is Now the Default
- MRV2 is the **default runner for all models**, completing the rollout that began with pooling models
- CUDA graph memory profiling for KV cache auto-sizing
- Batch-sharded sampling cuts per-step logits memory by 1/TP (`--enable-batch-sharded-sampling`)
- Prompt embeds, `extract_hidden_states` speculation, padded FULL cudagraph dispatch
- ⚠️ **Model Runner V1 is now deprecated** — removal targeted for vLLM v0.32. vLLM still falls back to MRV1 for unsupported features (sequence parallelism, dual-batch overlap, elastic EP, custom logits processors, certain spec-decode methods)

### 🤖 New Models
- **Hy4-preview**: Tencent's 770B/49B-active MoE with Gated DeepSeek Sparse Attention and native MTP (`hy_v4_mtp`, `hy_v4` tool/reasoning parsers)
- **Qwen3.8-Flash-Next**: BF16/FP8/NVFP4 variants with MTP (`qwen4_exp_mtp`)
- **GraniteSWA / GraniteMoeSWA**
- **NemotronH_Omni_Reasoning_V3** with MTP for Nemotron VL models
- **Kimi K3 NVFP4 checkpoints**

### ⚡ Speculative Decoding
- **`--per-request-spec-decode-metrics`** (none/summary/detailed): acceptance stats in OpenAI API responses
- Adaptive verification extended to logprobs
- DFlash2 loading from speculators format
- PLaMo3 EAGLE-3/DFlash; Qwen3-Omni DSpark drafts
- SM100 sparse MLA for GLM-5.2 and DeepSeek V4 on SM90

### 🧠 Kimi-K3 & DeepSeek V4 Performance
- Fused MXFP4 top-k finalization in K3 latent tail (~5% E2E latency)
- K3 Mamba metadata prep in one Triton launch (6.6-7.6x kernel speedup)
- GEMM-RS extended to GEMM-AR; MLA gate merged into QKV-A projection
- K3 DCP with DSpark; DCP partial prefix cache hits; `--dcp-q-replicate`
- DSv4 shared experts fused into MegaMoE; opt-in FlashInfer `moe_ep` expert backend

### 💾 Mamba Prefix Caching
- Internal prefill checkpoints: **9%-25% TTFT improvement**
- **`--prefix-cache-retention-interval`** is now a CLI argument (default 0); dense retention auto-restored for hybrid models using EAGLE/MTP

### 🚦 Admission Control (New)
- **`--max-num-queued-reqs`** / **`--max-num-queued-tokens`**: queue admission control flags

### 🔁 RL Weight Sync
- New `sharded_rdt` P2P backend: each worker pulls only its TP/EP slice over NIXL or Ray Direct Transport
- Rank-local IPC weight updates; sparse checkpoint-coordinate updates

### 🔌 Quantization & Backends
- FlashInfer TRT-LLM MXFP8 linear; b12x FP4 MoE for SM120/SM121
- AutoRound block-wise FP8; Humming MoE with MXFP4 weights + block-FP8 activations
- Humming for compressed-tensors WNA16 MoE; weight-only NVFP4 via W4A16

### 🌐 API & Frontend
- `/v1/messages/render` (Anthropic Messages), `/cohere/v2/chat/render`
- **`--sse-keep-alive-interval`**: SSE keep-alive comments for idle streams
- Video embeds accepted by the Python frontend
- Rust frontend: `--generation-config vllm`, `truncate_prompt_tokens`, gRPC audio/video inputs

### 🛡️ Security
- `cache_salt` length bounded to 1024; oversized media rejected before full download
- `api_key` and `hf_token` redacted from startup logs

### ⚙️ New Defaults
- FlashInfer all-reduce enabled by default for TP CUDA groups (opt out: `VLLM_ALLREDUCE_USE_FLASHINFER=0`)
- Prefix-cache `NONE_HASH` deterministic by default (no more `PYTHONHASHSEED` pinning)

## Schema Changes (v2.5.0 → v2.6.0)

### New CLI Arguments (8)

| Argument | Type | Category | Notes |
|----------|------|----------|-------|
| `--per-request-spec-decode-metrics` | choice (none/summary/detailed) | monitoring | Acceptance stats in API responses |
| `--max-num-queued-reqs` | integer | scheduling | Queue admission control |
| `--max-num-queued-tokens` | integer | scheduling | Queue admission control |
| `--prefix-cache-retention-interval` | integer (default 0) | optimization | Mamba/SWA retention; env var deprecated |
| `--dcp-q-replicate` | boolean | parallelism | DCP query replication, default-on for GLM sparse attention |
| `--enable-batch-sharded-sampling` | boolean | performance | Logits memory cut by 1/TP (MRV2) |
| `--enable-trace-replay` | boolean | model | Deterministic decode replay (`trace_decode_token_ids`) |
| `--sse-keep-alive-interval` | integer (default 0) | api | SSE keep-alive for idle streams |

### Expanded Choices

**`--spec-method`** (37 → 39 values): added `hy_v4_mtp` (Hy4-preview native MTP), `qwen4_exp_mtp` (Qwen3.8-Flash-Next)

**`--moe-backend`** (17 → 20 values): added `b12x` (FP4 MoE for SM120/SM121), `flashinfer_moe_ep_mega_deep_gemm`, `flashinfer_moe_ep_mega_cutedsl` (opt-in FlashInfer moe_ep expert backend)

**`--linear-backend`** and **`--quantization`**: unchanged from v0.28.0 (21 / 30 methods)

### New Parsers

**Tool Call Parsers (49 total):** +`hy_v4` (Hy4-preview), +`granite-20b-fc` (back in the v0.29.0 registry)

**Reasoning Parsers (32 total):** +`hy_v4`

## Breaking Changes & Deprecations (upstream vLLM)
- **Ten deprecated architectures removed**: Arctic, Chameleon, Cheers, Fairseq2Llama, FireRedLID, GritLM, HCXVision, MPT, `RWForCausalLM`/`StableLMEpochForCausalLM` aliases, `PrithviGeoSpatialMAE`
- **FlexOlmo, Olmo3, Hunyuan V1/VL** migrated to the Transformers modeling backend
- **PyAV video decoder backend removed** — use OpenCV or Torchcodec
- `python -m vllm.entrypoints.openai.api_server` deprecated — use `vllm serve`
- `VLLM_TEST_FORCE_FP8_MARLIN` removed in favor of `--linear-backend` / `--moe-backend`
- `--attention-config.use_prefill_decode_attention` removed (dead)
- **Model Runner V1 deprecated** (removal targeted v0.32)

## Dependencies
- FlashInfer 0.6.18, huggingface-hub 1.28.0, tpu-inference v0.28.0, NIXL 1.3.2
- InstantTensor added to CUDA dependencies (loader not enabled by default)

## Migration Guide

### For Users Upgrading from v0.4.0.0

**Check these before starting servers:**
1. **Model Runner**: if you relied on MRV1-specific features (sequence parallelism, dual-batch overlap, elastic EP, custom logits processors), vLLM now auto-falls back to MRV1 — but plan migration, MRV1 removal is targeted for v0.32
2. **FP8 Marlin env var**: replace `VLLM_TEST_FORCE_FP8_MARLIN=1` with `--linear-backend marlin` / `--moe-backend marlin`
3. **PyAV**: if video inputs failed to decode, switch decoder to OpenCV or Torchcodec
4. **Hybrid + spec decode**: if EAGLE/MTP TTFT regressed, `--prefix-cache-retention-interval` handles dense retention automatically for hybrid models; set explicitly only to override

**New recommendations:**
```bash
# Admission control for high-traffic endpoints
vllm-cli serve --model my-model --max-num-queued-reqs 256 --max-num-queued-tokens 1000000

# Spec decode acceptance monitoring
vllm-cli serve --model my-model --spec-method eagle3 --per-request-spec-decode-metrics summary

# SM120/SM121 FP4 MoE
vllm-cli serve --model my-nvfp4-model --moe-backend b12x
```

## Files Changed
- `src/vllm_cli/schemas/argument_schema.json` — Schema v2.6.0: 320 arguments (8 new), 39 spec methods, 20 MoE backends, 49 tool parsers, 32 reasoning parsers
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.29.0 to supported versions
- `src/vllm_cli/config/parser_sync.py` — Added hy_v4 parser name mappings
- `VERSION` — 0.4.0.0 → 0.5.0.0
- `pyproject.toml` — Version bump, vLLM range update (0.20.0 - 0.29.0)
- `CHANGELOG.md` — Added v0.5.0.0 entry
- `README.md` — Added v0.5.0.0 What's New section
