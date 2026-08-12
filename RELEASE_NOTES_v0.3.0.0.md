# vllm-cli v0.3.0.0 Release Notes (2025-08-12)

## Overview

vllm-cli v0.3.0.0 adds full support for **vLLM v0.27.1** (and v0.27.0, 561 commits from 242 contributors), including the Kimi K3 full stack, Qwen3.5 text-only models, PyTorch 2.13.0 upgrade, FlashAttention 4 deepened integration, DeepSeek-V4 performance push, Model Runner V2 expansion, and expanded quantization (NVFP4, MXFP8, FP4 Qutlass).

This is a **major version bump** from 0.2.9.x to 0.3.0.0, reflecting the vLLM major version upgrade from 0.26 to 0.27.

## One-Click Command Import

New `vllm-cli import` command that parses raw `vllm serve` commands into profile configurations:

```bash
# Import a command directly
vllm-cli import 'vllm serve model --flag value --boolean-flag'

# With custom name
vllm-cli import 'vllm serve model --flag value' --name my-profile

# Preview without saving
vllm-cli import 'vllm serve model --flag value' --preview

# From file
vllm-cli import --file command.txt
```

Supports:
- Standard CLI flags (`--flag value`, `--boolean-flag`)
- Dot-notation config keys (`--speculative_config.method dflash`)
- Auto-generated profile names from model paths
- Type conversion (strings to int/float/boolean)
- Nested config dictionaries

## Key Changes

### Kimi K3 Full Stack Support
The most significant addition — Kimi K3 lands with complete support in a single release:
- Core model files and kernels
- Python and Rust frontends
- AttnRes kernels
- DeepGEMM support
- Compressed-tensors quantized checkpoints
- DSpark AR fusion
- Optional shared-expert sharding
- Dedicated `kimi_k3` tool call and reasoning parsers

### Qwen3.5 & More New Models
- **Qwen3.5**: Text-only dense and MoE models with EVS video token pruning
- **K-EXAONE-2.0-750B-A37B**: Large-scale Korean model support
- **VaultGemma**: Via Transformers modeling backend
- **jina-embeddings-v5-text-nano**: With EuroBERT encoder backbone

### PyTorch 2.13.0 Upgrade (Breaking Environment Change)
- PyTorch 2.13.0, torchvision 0.28.0, Triton 3.7.1
- XPU and CPU also upgraded to torch 2.13
- **Action required**: Ensure compatible CUDA/toolchain installation

### FlashAttention 4 Deepened
- FP8 KV cache support on SM100
- headdim-256 support
- New JIT warmup infrastructure
- Runner-owned Triton kernel warmup (eliminates first-request compilation stalls)

### DeepSeek-V4 Performance Push
- Sequence parallelism
- ~2x kernel improvement by skipping empty c128 launches
- 3.4% E2E TTFT from skipping unneeded topk/router
- 3.9% E2E TTFT from workspace reuse
- 1.88x kernel from removing redundant full kernel
- Adaptive topk width (1.0% E2E)
- 448 MiB GPU memory saved in PP buffer
- Compact MXFP4 indexer KV cache

### Model Runner V2 Expansion
- Encoder-only attention
- Sequence pooling for embedding/classification
- Encoder token classification
- Token embedding
- BGE-M3 pooling
- Multimodal on CPU
- Multi-layer MTP speculator

### Expanded Quantization
- **FP4 Qutlass**: Compressed-tensors integration
- **CuTeDSL MoE**: ReLU2 NVFP4
- **MXFP8 linear**: INC support
- **AutoRound W4A16**: MoE on XPU
- **MXFP4**: Linear/MoE on XPU
- **KV quant mode**: TurboQuant
- **ModelOpt FP8**: SM80 emulation
- **`--linear-backend`**: Honored for ModelOpt W4A16

### API & Frontend Updates
- **Cohere chat v2**: New API support
- **`cache_salt`**: Anthropic Messages API
- **Strict tool calling**: GPT-OSS Harmony constrained decoding
- **Unified Mistral parser**: Engine-based reasoning and tool calls
- **`stream_interval`**: Per-request sampling parameter
- **`--limit-mm-per-prompt`**: Multimodal limits
- **Rust frontend**: gRPC control plane, `vllm-bench` CLI integration

### Hardware & Performance
- **NVIDIA Rubin**: `sm_107` target with NVLink all-reduce
- **ROCm**: gfx1250 architecture enabled
- **NCCL 2.30.7**: DeepEPv2 support
- **Helion 1.4.0**: Latest Helion integration

### Models Removed
- **Plamo2**: Removed upstream in vLLM v0.27.0
- **Ouro**: Removed upstream in vLLM v0.27.0

### Arguments Removed
- **`max_num_partial_prefills`**: No longer supported in vLLM v0.27.0
- **`max_long_partial_prefills`**: No longer supported in vLLM v0.27.0

## New Parsers

### Tool Call Parsers (7 new)
- `kimi_k3` — Kimi K3 model family
- `mimo` — MiMo model
- `llama4_json` — Llama 4 JSON output
- `llama4_pythonic` — Llama 4 Pythonic output
- `cohere_command3` — Cohere Command R3
- `cohere_command4` — Cohere Command R4
- `glm45` — GLM 4.5

### Tool Call Parsers Changed
- `qwen3` replaced by `qwen3_coder` and `qwen3_xml` for more specific parser selection
- Removed: `rust`, `granite-20b-fc` (no longer in vLLM 0.27.0)

### Reasoning Parsers (1 new)
- `kimi_k3` — Kimi K3 model family

### Speculative Decoding (Fixed)
- `--speculative-config` is **NOT deprecated** (was incorrectly marked as removed in v0.22.0)
- Added `--spec-tokens` parameter (missing previously)
- Updated `--spec-method` choices: `ngram`, `ngram_gpu`, `medusa`, `eagle`, `eagle3`, `mtp`, `dflash`, `dspark`
- Clarified mutual exclusivity: `--spec-method`/`--spec-model`/`--spec-tokens` are convenience flags that populate `--speculative-config`

## Hardware & Performance Updates

### NVIDIA
- FlashAttention 4 deepened on SM100 (Blackwell)
- FP8 KV cache, headdim-256
- JIT warmup infrastructure
- `sm_107` target for Rubin

### AMD/ROCm
- gfx1250 architecture enabled
- NIXL and UCX upgrades
- AITER 0.1.19

### Intel XPU
- PyTorch 2.13 support
- AutoRound W4A16 MoE
- MXFP4 linear/MoE

## Dependencies
- **PyTorch**: 2.13.0 (was 2.12.x)
- **torchvision**: 0.28.0
- **Triton**: 3.7.1
- **Transformers**: 5.14.1 (was 5.13.0)
- **FlashInfer**: 0.6.16.post3 (was 0.6.15)
- **AITER**: 0.1.19 (was 0.1.16.post5)
- **NCCL**: 2.30.7

## Schema Updates
- Schema version: 2.3.0 → 2.4.0
- vLLM versions supported: 0.20.0 - 0.26.0 → 0.20.0 - 0.27.0
- 1 new tool call parser (`kimi_k3`)
- 1 new reasoning parser (`kimi_k3`)
- Plamo2, Ouro parsers removed
- `max_num_partial_prefills`, `max_long_partial_prefills` arguments removed

## Migration Guide

### For Users Upgrading from v0.2.9.9
Your existing configurations will continue to work. The following models were removed upstream and are no longer supported:
- **Plamo2**: No longer available in vLLM v0.27.0
- **Ouro**: No longer available in vLLM v0.27.0

### PyTorch 2.13.0 Upgrade
This is a breaking environment change. Make sure your CUDA toolchain is compatible:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### New Features to Explore
```bash
# Install
pip install vllm-cli

# Sync CLI args from v0.27.0
vllm-cli recipes --sync-args --tag v0.27.0 --apply-args

# Serve with Kimi K3
vllm-cli serve --model Kimi-K3 --tool-call-parser kimi_k3 --reasoning-parser kimi_k3

# Use NVFP4 quantization
vllm-cli serve --model Inkling-3B --quantization nvfp4

# Qwen3.5 MoE
vllm-cli serve --model Qwen3.5-32B-A3B-N32
```

## Files Changed
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.27.1, v0.27.0 to supported versions
- `src/vllm_cli/config/parser_sync.py` — Added kimi_k3 name mapping
- `src/vllm_cli/schemas/argument_schema.json` — Schema v2.4.0, kimi_k3 parser, nvfp4 quantization
- `VERSION` — 0.2.9.9 → 0.3.0.0
- `CHANGELOG.md` — Added v0.3.0.0 entry
- `pyproject.toml` — Version bump, vLLM version range update (0.20.0 - 0.27.0)
