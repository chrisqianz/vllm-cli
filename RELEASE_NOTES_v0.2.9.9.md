# vllm-cli v0.2.9.9 Release Notes

## Overview

vllm-cli v0.2.9.9 adds full support for **vLLM v0.26.0**, including the Inkling model family, DeepSeek-V4 performance optimizations, Transformers 5.13.0 migration, Decode Context Parallel (DCP), PD disaggregation with NIXL, and expanded quantization support (Humming, NVFP4/MXFP4, INT2 XPU).

## Key Changes

### New Inkling Model Family
Full support stack for the Inkling model family:
- Base modeling with piecewise CUDA graph support
- Hopper FA4 relative attention
- MTP=1 speculative decoding
- LoRA support
- Standard ModelOpt NVFP4 quantization
- Dedicated tool call and reasoning parsers

### DeepSeek-V4 Performance Push
- Specialized routing kernel (2.94% E2E TPOT improvement)
- `fused_topk_bias` kernel (1.5-2x faster)
- Redundant repeat/copy removal (1.8% E2E TPOT)
- ROCm two-stage compressor for HCA prefill
- Sparse decode/prefill optimizations
- DSpark speculative decoding on AMD and XPU

### Transformers 5.13.0 Migration
More models migrated to the Transformers modeling backend:
- Olmo/Olmo2
- MistralLarge3 to AutoWeightsLoader
- HunyuanVL native transformers processor

### Decode Context Parallel (DCP)
- Hybrid attention support for DCP
- DCP + Eagle for Tokenspeed MLA backends

### PD Disaggregation
- NIXL pipeline-parallel prefill in push mode
- Encoder-cache connectors with CPU offloading

### Expanded Quantization
- Humming w[2-7]a[4,8] weight-only inference with compressed-tensors
- int4 quantization for emulation MoE backend
- INT2 XPU weight-only quant linear
- NVFP4/MXFP4: `nvfp4_per_token` online MoE quantization
- CuTe-DSL FlashInfer MXFP4 quantization

### Models Removed
- TeleChat (upstream removed)
- Persimmon and Fuyu (upstream removed)

## New Parsers

### Tool Call Parsers (13 new)
- `inkling` — Inkling model family
- `apertus` — Apertus
- `ernie45` — ERNIE 4.5
- `functiongemma` — FunctionGemma
- `gigachat3` — GigaChat 3
- `granite4` — Granite 4
- `hermes` — Hermes
- `hunyuan_a13b` — Hunyuan A13B
- `lfm2` — LFM2
- `olmo3` — OLMo 3
- `poolside_v1` — Poolside v1
- `pythonic` — Pythonic
- `xlam` — XLAM

### Reasoning Parsers (5 new)
- `inkling` — Inkling model family
- `ernie45` — ERNIE 4.5
- `hunyuan_a13b` — Hunyuan A13B
- `olmo3` — OLMO 3
- `poolside_v1` — Poolside v1

## Hardware & Performance Updates

### Blackwell
- FlashInfer fused all-reduce tuned for world_size=16 on GB300
- CuTeDSL/FA4-MLA warmup infrastructure
- B12x backend for non-gated MoEs

### AMD/ROCm
- Moved to torch 2.11 stable ABI
- AITER FlashAttention MLA prefill backend
- Fused shared-expert for GLM-4.5/6/7 and MiniMax-M3
- HybridW4A16 linear kernel

### Intel XPU
- INT2 weight-only quant linear
- DSpark speculative decoding

### CPU/RISC-V
- Accelerated unquantized MoE for AArch64
- Compressed-tensor w8a8 int8 MoE
- RVV path for W4A8 INT4 GEMM

## API & Frontend Updates
- Rust frontend: multimodal video and audio support
- OpenAI compatibility: `bad_words` in `/v1/completions`, `logprob_token_ids`, `include_reasoning` for non-Harmony models
- Endpoint plugins framework
- Deepstream video decoding backend
- Human-readable integers for more CLI args
- Grammar compilation failure handling (no engine crash)

## Schema Updates
- Schema version: 2.2.0 → 2.3.0 (pending sync)
- vLLM versions supported: 0.20.0 - 0.25.0 → 0.20.0 - 0.26.0
- 13 new tool call parsers
- 5 new reasoning parsers
- TeleChat, Persimmon, Fuyu parsers removed

## Migration Guide

### For Users Upgrading from v0.2.9.8
Your existing configurations will continue to work. The following models were removed upstream and are no longer supported:
- **TeleChat**: No longer available in vLLM v0.26.0
- **Persimmon**: No longer available in vLLM v0.26.0
- **Fuyu**: No longer available in vLLM v0.26.0

### New Features to Explore
```bash
# Install
pip install vllm-cli

# Sync CLI args from v0.26.0
vllm-cli recipes --sync-args --tag v0.26.0 --apply-args

# Serve with Inkling model
vllm-cli serve --model Inkling-3B --tool-call-parser inkling --reasoning-parser inkling

# Use new quantization
vllm-cli serve --model Qwen3-32B --quantization nvfp4
```

## Files Changed
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.26.0 to supported versions
- `src/vllm_cli/config/parser_sync.py` — Added 13 tool parsers, 5 reasoning parsers
- `VERSION` — 0.2.9.8 → 0.2.9.9
- `CHANGELOG.md` — Added v0.2.9.9 entry
- `pyproject.toml` — Version bump, vLLM version range update (0.20.0 - 0.26.0)
