# vllm-cli v0.4.0.0 Release Notes (2026-08-28)

## Overview

vllm-cli v0.4.0.0 adds full support for **vLLM v0.28.0** (584 commits from 270 contributors, 76 new!), including the Kimi-K3 performance stack (DCP, FlashKDA kernels, MegaMoE SiTU), DeepSeek-V4 sparse MLA end-to-end, DFlash2 speculative decoding, Model Runner V2 maturation (E/P/D disaggregation, weight offloading), tiered KV cache offloading with disk support, and new models (Muse Glimmer, Ling 3.0 Flash, Dots3 NOTE).

This is a **major version bump** from 0.3.0.0 to 0.4.0.0, reflecting the vLLM major version upgrade from 0.27 to 0.28.

## Highlights

### 🚀 Kimi-K3 Performance Push
The most significant optimization effort in this release:
- **Decode Context Parallel (DCP)** support
- **Fused FlashKDA** decode and prefill kernels (new `--kda-prefill-backend flashkda`)
- **SiTU activation** support for MegaMoE
- **GEMM-RS** for sequence parallelism
- **Adaptive speculative token budget**: ~60% better DSpark TTFT
- **Optional shared-expert sharding**: saves ~17 GiB per GPU
- **ROCm**: Kimi-K3 now runs on ROCm with the V2 model runner

### 🧠 DeepSeek V4
- **Sparse MLA** works end-to-end for plain decode, MTP, and DSpark speculative decoding
- **AMD Quark NVFP4** support
- **Reasoning-effort** prompts and mappings
- Sparse top-k metadata kernel optimizations
- ROCm enablement on gfx11 and gfx950

### ⚡ Speculative Decoding Advances
- **DFlash2**: Local convolution + candidate selector
- **DSpark**: Confidence-scheduled verification, top-k Markov projection
- **Async scheduling** auto-enabled for draft models
- Fused MTP trailing all-reduce with local-argmax draft tokens

### 🏗️ Model Runner V2 Maturation
- **E/P/D disaggregation** support
- **Weight offloading**
- **Multi-layer MTP** KV cache support
- **Encoder CUDA graphs**
- **Decoder token-wise pooling** + Transformers pooling models
- **Attention-free models**
- **`thinking_token_budget`** support

### 💾 Tiered KV Cache Offloading
- **Disk offloading** support (SimpleCPUOffloadConnector)
- **Out-of-tree secondary tier managers** via `module_path`
- Partial secondary-tier load results
- Tiering offloading metrics
- Canonical CPU layout for parallelism-agnostic offload

### 🆕 New Models
- **Muse Glimmer**: New model family with dedicated tool call + reasoning parsers
- **Ling 3.0 Flash**: BF16, MTP, and parser support + FP8 variant + hybrid MXFP4 routed experts
- **Dots3 NOTE**: Native multimodal support with MTP speculative decoding
- **Interns2mobius**
- **Qwen3.8**: Enabled on AMD ROCm

## Schema Changes

### New CLI Arguments (20)

| Argument | Type | Category | Notes |
|----------|------|----------|-------|
| `--mamba-ssu-algorithm` | choice | advanced | `auto`, `simple`, `vertical`, `horizontal` — SSU algorithm for FlashInfer Mamba backend |
| `--kda-prefill-backend` | choice | advanced | `auto`, `triton`, `flashkda` — Kimi K3 KDA prefill backend |
| `--fingerprint-mode` | choice | api | `full`, `hash`, `custom`, `none` |
| `--uvicorn-log-level` | choice | api | `critical`, `error`, `warning`, `info`, `debug`, `trace` |
| `--mm-hasher-algorithm` | choice | advanced | `blake3`, `sha256`, `sha512` |
| `--cohere-format` | string | api | Cohere chat format (default `cmd4`) |
| `--mm-processor-device` | string | advanced | Device for multimodal processor (default `auto`) |
| `--video-pruning-method` | string | advanced | Video frame pruning method |
| `--ec-manager-config` | JSON | advanced | Encoder cache manager configuration |
| `--fault-tolerance-config` | JSON | advanced | Fault tolerance configuration |
| `--mamba-config` | JSON | model | Mamba model configuration |
| `--cohere-is-reasoning-model` | boolean | api | Default `true` |
| `--enable-bf16x3-router-gemm` | boolean | performance | BF16x3 router GEMM |
| `--enable-fault-tolerance` | boolean | advanced | Fault tolerance |
| `--enable-moe-shared-loras` | boolean | lora | MoE shared LoRAs |
| `--max-num-scheduled-tokens` | integer | scheduling | Max tokens scheduled per step |
| `--mm-device-do-normalize` | boolean | advanced | On-device multimodal normalization |
| `--prefix-match-unit` | integer | optimization | Partial-tail prefix reuse unit |
| `--replayssm-buffer-len` | integer | memory | ReplaySSM buffer length |
| `--use-replayssm` | boolean | memory | Enable ReplaySSM |
| `--return-sampling-mask` | boolean | advanced | Return sampling mask |

### Expanded Choices

**`--spec-method`** (9 → 37 values):
- New top-level: `mlp_speculator`, `draft_model`, `suffix`, `custom_class`
- New Eagle variant: `extract_hidden_states`
- New MTP types: `dots3_note_mtp`, `bailing_hybrid_v3_mtp`
- Full MTP type list now included: `deepseek_mtp`, `mimo_mtp`, `mimo_v2_mtp`, `glm4_moe_mtp`, `glm4_moe_lite_mtp`, `glm_ocr_mtp`, `ernie_mtp`, `nemotron_h_mtp`, `exaone_moe_mtp`, `exaone4_5_mtp`, `qwen3_next_mtp`, `qwen3_5_mtp`, `longcat_flash_mtp`, `minimax_m3_mtp`, `bailing_hybrid_mtp`, `kimi_k3_mtp`, `pangu_ultra_moe_mtp`, `step3p5_mtp`, `hy_v3_mtp`, `gemma4_mtp`, `inkling_mtp`

**`--linear-backend`** (5 → 21 values):
Added: `b12x`, `deep_gemm`, `torch`, `machete`, `fbgemm`, `conch`, `exllama`, `emulation`, `xpu`, `xpu_woq`, `flashinfer_cudnn`, `flashinfer_b12x`, `aiter`

**`--quantization`** (17 → 30 methods):
Added: `auto_awq`, `auto_gptq`, `modelopt`, `modelopt_mixed`, `quark`, `moe_wna16`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`, `deepseek_v4_fp8`, `online`, `fp_quant`, `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `int8_per_channel_weight_only`, `nvfp4_per_token`, `mxfp8`
Removed: `bitsandbytes` (out-of-tree plugin), `auto-round`, `bitblas`, `gguf`, `nvfp4`

**`--mamba-backend`** (2 → 3): Added `cpu`

**Mamba cache dtypes**: Added `bfloat16` to `--mamba-cache-dtype` and `--mamba-ssm-cache-dtype`

### New Parsers

**Tool Call Parsers (3 new, 47 total):**
- `dots` — Dots3 NOTE
- `ling3` — Ling 3.0 Flash
- `muse_glimmer` — Muse Glimmer

**Reasoning Parsers (9 new, 2 removed, 31 total):**
- New: `cohere_command3`, `cohere_command4`, `glm45`, `holo2`, `inkling`, `ling3`, `mimo`, `minimax_m2_append_think`, `muse_glimmer`
- Removed: `cohere_command` (split into `cohere_command3`/`cohere_command4`), `identity` (no longer registered)

### Removed Arguments
- **`--calculate-kv-scales`**: Runtime KV scale calculation removed upstream
- **`--override-attention-dtype`**: Removed upstream
- **bitsandbytes quantization**: Migrated to out-of-tree plugin

### Deprecated Arguments (renamed upstream)
| Old | New |
|-----|-----|
| `--enable-stochastic-rounding` | `--enable-mamba-cache-stochastic-rounding` |
| `--stochastic-rounding-philox-rounds` | `--mamba-cache-philox-rounds` |
| `--policy` | `--scheduling-policy` |
| `--target-modules` | `--lora-target-modules` |
| `--cache-dtype` | `--kv-cache-dtype` |
| `--backend` | `--distributed-executor-backend` |

## New Defaults
- `max_num_batched_tokens`: 8192 → 16384
- Prefix caching enabled by default for Mamba models
- Blackwell CUDA graph capture default raised to 1024

## Hardware & Performance

### NVIDIA
- FlashInfer XQA decode on SM12x
- CuTeDSL fused query kernel on SM100
- Programmatic dependent launch for DSA decode kernels
- Native DSA decode path for MTP=3 on SM90
- GB10 fused-MoE FP8 tuning configs
- B12X dense linear backends

### AMD ROCm
- torch 2.12 / triton 3.7 stack bump
- AITER and FP8 inference on GFX120x
- DeepSeek-V4 on gfx11
- Optimized Triton sparse-MLA decode on gfx950
- FlyDSL decode-attention for 4-bit TurboQuant KV cache
- Fused Kimi-K3 KDA decode kernel

### Intel XPU
- torch linear backend with blockwise GEMM
- MXFP8 linear weights for INC DeepSeek V4
- Async-scheduling PP sampled-token broadcast
- XPU wheel in release pipeline

### CPU
- MLA backend (DeepSeek-V2/V3 on CPU)
- triton-cpu wheel
- GPTQ and AWQ on s390x
- Unquantized MoE for Power (VSX)

## Quantization
- **Online quantization**: MXFP4 support, weight scales shared across TP, NVFP4 expert packing precision
- **NVFP4**: Batch-invariant MoE via CUTLASS, KV 4-over-6 scale search, CuTeDSL MoE with SwiGLU-OAI and ReLU2
- **New kernels**: block-wise scaled_mm, DeepSeek-V4 AMD Quark NVFP4 with emulation kernel

## API & Frontend
- Request priority from HTTP header
- Session ID plumbing
- `count_reasoning_tokens` in streaming parser engine
- `content_parts` on `/inference/v1/generate`
- **Rust frontend**: Standalone renderer, gRPC multimodal image inference, explicit DP rank routing, RL lifecycle control, dynamic tools from developer messages
- **Anthropic API**: 4xx for client errors, `disable_parallel_tool_use` preserved, bounded stop sequences
- **Cohere**: Upstreamed parser fixes, vectorized binary embedding bit-packing
- **Structured output**: Request stop tokens masked until grammar terminates

## Security
- Fixed DoS via sample-rate forgery bypassing audio decode duration guard
- DeepStream classified as GPU backend with pixel limits enforced
- `_load_ov2_processor` guarded with `resolve_trust_remote_code`
- Documentation warns `--api-key` does not gate all endpoints

## Dependencies
- Transformers 5.15.0
- huggingface-hub 1.27.0
- fastsafetensors upgrade
- ROCm: torch 2.12, triton 3.7, torchaudio, torchvision
- Runtime image: Ubuntu 24.04, rdma-core > 44

## Migration Guide

### For Users Upgrading from v0.3.0.0

**Breaking changes:**
1. **bitsandbytes** quantization no longer works via `--quantization bitsandbytes` — install the out-of-tree plugin instead
2. **`--calculate-kv-scales`** and **`--override-attention-dtype`** are removed — use the replacement mechanisms
3. **`reasoning_content`** output removal is a breaking client change

**Renamed arguments** (old names still accepted with deprecation warnings):
```bash
# Old → New
--enable-stochastic-rounding → --enable-mamba-cache-stochastic-rounding
--stochastic-rounding-philox-rounds → --mamba-cache-philox-rounds
--policy → --scheduling-policy
--target-modules → --lora-target-modules
--cache-dtype → --kv-cache-dtype
--backend → --distributed-executor-backend
```

**Transformers upgrade**: vLLM 0.28.0 requires Transformers 5.15.0:
```bash
uv pip install --upgrade transformers
```

### New Features to Explore
```bash
# Kimi K3 with FlashKDA fused kernels
vllm-cli serve --model Kimi-K3 --kda-prefill-backend flashkda --enable-fault-tolerance

# Ling 3.0 Flash with MTP
vllm-cli serve --model Ling-3.0-Flash --spec-method ling3_mtp

# Muse Glimmer with dedicated parsers
vllm-cli serve --model Muse-Glimmer --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer

# Dots3 NOTE multimodal
vllm-cli serve --model Dots3-NOTE --tool-call-parser dots

# Tiered KV offloading with disk
vllm-cli serve --model DeepSeek-V4 --kv-offloading-backend simple --offload-backend disk

# Speculative decoding with DFlash2
vllm-cli serve --model my-model --speculative-config '{"method": "dflash", "num_speculative_tokens": 5}'
```

## Files Changed
- `src/vllm_cli/schemas/argument_schema.json` — Schema v2.5.0: 312 arguments, 20 new, 37 spec methods, 21 linear backends, 30 quantization methods, 47 tool parsers, 31 reasoning parsers
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.28.0 to supported versions
- `src/vllm_cli/config/parser_sync.py` — Added v0.28.0 parser name mappings
- `VERSION` — 0.3.0.0 → 0.4.0.0
- `pyproject.toml` — Version bump, vLLM range update (0.20.0 - 0.28.0)
- `CHANGELOG.md` — Added v0.4.0.0 entry
- `README.md` — Added v0.4.0.0 What's New section
