# vllm-cli v0.2.9.8 Release Notes

## Overview

vllm-cli v0.2.9.8 adds full support for **vLLM v0.25.0**, including Model Runner V2 as the default for all dense models, PagedAttention removal, Transformers backend parity, new Streaming Parser Engine improvements, and universal speculative decoding.

## Key Changes

### Model Runner V2 Default
Model Runner V2 is now the standard execution path for all dense models, featuring:
- **EVS** (Embedding Value Scaling)
- **Realtime embeddings** support
- **Mamba hybrid prefix caching**
- **Multimodal-prefix bidirectional attention**
- **Dynamic speculative decoding** with full CUDA graphs
- Bounded memory for large-logprobs requests

### PagedAttention Removed
The legacy PagedAttention implementation has been completely deleted. V1/MRv2 backends are now the only execution path. If you were using `--attention-backend`, this argument is now deprecated.

### Transformers Backend Parity
The Transformers modeling backend now matches native vLLM performance with:
- FP8 MoE support
- CUDA graph + embed scaling fixes
- GPTBigCode/Starcoder2 and RoBERTa migration
- Tied-embedding `lm_head.bias` fix

### New Streaming Parser Engine
Unified tool-call/reasoning parsing framework with new parsers:
- **Kimi k2.5/k2.6/k2.7** parser
- **seed_oss** and **DeepSeek V4** ports
- **gpt-oss / Harmony** dedicated renderer

### Universal Speculative Decoding
Heterogeneous vocabulary support (Token Length Invariance) with:
- **DSpark** drafter with speculators checkpoint support
- **DFlash** backend selection, per-layer RMSNorm fusion, CPU support
- **Laguna XS.2.1** drafter
- Block verification for rejection sampling

### New Models
- LLaVA-OneVision-2
- Unlimited OCR with Triton R-SWA backend
- MOSS-Transcribe-Diarize
- openai/privacy-filter
- Hy3 with token-suffix and JSON Schema array support
- GLM-5 family (GLM-5 / DeepSeek-V3.2, GLM-5.2 FP32 gate)
- MiniMax-M3 with pipeline parallelism and NVFP4 support

## 14 New CLI Arguments

| Category | Arguments | Description |
|----------|-----------|-------------|
| Backend | `--backend` | Engine backend selection (mp, ray, debug) |
| Cache | `--cache-dtype` | KV cache data type |
| Monitoring | `--enable-per-request-metrics` | Per-request timing metrics on API responses |
| Monitoring | `--jit-monitor-mode` | JIT monitor mode for runtime performance |
| Monitoring | `--ray-workers-use-nsight` | Enable NVIDIA Nsight for Ray workers |
| Quantization | `--enable-stochastic-rounding` | Stochastic rounding for quantized operations |
| Quantization | `--quantization-config` | Path to quantization configuration file |
| Quantization | `--stochastic-rounding-philox-rounds` | Philox rounds for stochastic rounding |
| Multimodal | `--mm-ipc-gpu-memory-gb` | GPU memory for multimodal IPC shared memory |
| Scheduling | `--policy` | Scheduling policy override |
| Scheduling | `--ubatch-size` | Micro-batch size for compilation optimization |
| LoRA | `--target-modules` | Target modules for LoRA/adapter application |
| Debug | `--model-class-overrides` | Override model class for development |
| Debug | `--method` | Inference method override |

## 9 Deprecated Arguments (v0.25.0)

| Argument | Replacement | Reason |
|----------|-------------|--------|
| `--aggregate-engine-logging` | N/A | Consolidated into core logging |
| `--data-parallel-multi-port-external-lb` | `--data-parallel-external-lb` with hybrid_lb | Load balancer unified |
| `--default-max-num-seqs` | N/A | Sequence limits managed dynamically |
| `--fail-on-environ-validation` | N/A | Environment validation always enforced |
| `--limit-mm-per-prompt` | N/A | Multimodal limits handled internally |
| `--mamba-cache-philox-rounds` | `--stochastic-rounding-philox-rounds` | Unified stochastic rounding |
| `--shutdown-timeout` | N/A | Shutdown timeout managed internally |
| `--sliding-window` | `--attention-config` | Sliding window in attention config |
| (legacy v0.24.0 removals) | N/A | See v0.2.9.7 notes |

## New Parsers

### Tool Call Parsers
- `cohere_command` — Cohere Command series
- `deepseek_v3` — DeepSeek V3
- `minimax_m3` — MiniMax-M3
- `rust` — Rust frontend parser

### Reasoning Parsers
- `cohere_command` — Cohere Command series
- `identity` — Identity passthrough parser
- `minimax_m3` — MiniMax-M3

## Hardware & Performance Updates

### Blackwell
- FlashInfer fused all-reduce tuned for world_size=16 on GB300
- CuTeDSL/FA4-MLA warmup infrastructure
- B12x backend for non-gated MoEs

### AMD/ROCm
- Moved to torch 2.11 stable ABI
- AITER FlashAttention MLA prefill backend
- Fused shared-expert for GLM-4.5/6/7 and MiniMax-M3

### Intel XPU
- W8A8 FP8 linear kernel with multi-granularity quant
- Pipeline-parallel accuracy fix
- Uniform-batch CUDA graph for FA2

### CPU/RISC-V
- Accelerated unquantized MoE for AArch64
- Compressed-tensor w8a8 int8 MoE
- RVV path for W4A8 INT4 GEMM

## Schema Updates
- Schema version: 2.1.0 → 2.2.0
- Total arguments: 277 → 291
- Deprecated args: 105 → 113
- Last synced: v0.24.0 → v0.25.0
- vLLM versions supported: 0.20.0 - 0.24.0 → 0.20.0 - 0.25.0

## Migration Guide

### For Users Upgrading from v0.2.9.7
Your existing configurations will continue to work. The deprecated arguments remain in the schema for backward compatibility but are marked for removal.

1. **PagedAttention removal**: If you were using `--attention-backend`, the V1/MRv2 backends are now the default
2. **Stochastic rounding**: Replace `--mamba-cache-philox-rounds` with `--stochastic-rounding-philox-rounds`
3. **Sliding window**: Managed via `--attention-config` instead of `--sliding-window`

### For New Deployments
```bash
# Install
pip install vllm-cli

# Sync CLI args from v0.25.0
vllm-cli recipes --sync-args --tag v0.25.0 --apply-args

# Serve with new backend
vllm-cli serve --model Qwen3-32B --backend mp --tool-call-parser qwen3
```

## Files Changed
- `src/vllm_cli/schemas/argument_schema.json` — 14 new args, 9 deprecated
- `src/vllm_cli/schemas/default_profiles.json` — Version bump
- `src/vllm_cli/config/cli_args_sync.py` — Added v0.25.0 to supported versions
- `src/vllm_cli/config/parser_sync.py` — Updated parser name mappings
- `src/vllm_cli/system/dependencies.py` — Added doctor dependency checker
- `src/vllm_cli/system/__init__.py` — Exported doctor functions
- `VERSION` — 0.2.9.7 → 0.2.9.8
- `CHANGELOG.md` — Added v0.2.9.8 entry
- `pyproject.toml` — Version bump, vLLM version range update
- `README.md` — Updated changelog section

## Dependency Doctor (New in this release)
- **Command**: `vllm-cli doctor` (on-demand, not on every startup)
- **Purpose**: Detect missing system dependencies (FFmpeg, torchcodec libs)
- **Why**: vLLM 0.25.0+ introduces `torchcodec`, which requires FFmpeg shared libraries at runtime
- **Output**: Severity levels (critical/warning) with fix commands
- **Future-proof**: Extensible framework for detecting new C-extension dependencies