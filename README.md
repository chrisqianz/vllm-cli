# vLLM CLI

[![CI](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/ci.yml)
[![Release](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Chen-zexi/vllm-cli/actions/workflows/python-publish.yml)
[![PyPI version](https://badge.fury.io/py/vllm-cli.svg)](https://badge.fury.io/py/vllm-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Downloads](https://static.pepy.tech/badge/vllm-cli)](https://pepy.tech/projects/vllm-cli)

A command-line interface tool for serving Large Language Models using vLLM. Provides both interactive and command-line modes with features for configuration profiles, model management, and server monitoring.

![vLLM CLI Welcome Screen](asset/welcome-screen.png)
*Interactive terminal interface with GPU status and system overview*<br>
*Tip: You can customize the GPU stats bar in settings*

## Features

- **🎯 Interactive Mode** - Rich terminal interface with menu-driven navigation
- **⚡ Command-Line Mode** - Direct CLI commands for automation and scripting
- **🤖 Model Management** - Automatic discovery of local models with HuggingFace and Ollama support
- **🔧 Configuration Profiles** - Pre-configured and custom server profiles for different use cases
- **📊 Server Monitoring** - Real-time monitoring of active vLLM servers
- **🖥️ System Information** - GPU, memory, and CUDA compatibility checking
- **📝 Advanced Configuration** - Full control over vLLM parameters with validation
- **🔄 CLI Argument Sync** - Automatic synchronization with vLLM source code
- **📦 Official Recipes** - Import optimized profiles from vLLM community
- **📋 One-Click Import** - Parse raw `vllm serve` commands into profiles instantly

**Quick Links:** [📖 Docs](#documentation) | [🚀 Quick Start](#quick-start) | [📸 Screenshots](docs/screenshots.md) | [📘 Usage Guide](docs/usage-guide.md) | [❓ Troubleshooting](docs/troubleshooting.md) | [🗺️ Roadmap](docs/roadmap.md)

## What's New in v0.6.0.1

### 🐛 Multi-Model Proxy Fixed (no longer experimental)

- **Proxy menu fully working again**: translated menu options were compared against hard-coded English strings, so "配置新代理 / Configure new proxy" and all running-proxy management options silently did nothing (broken since i18n was introduced in v0.2.9.4). All handlers now match the translated labels.
- **Settings → Manage Proxy Configurations** no longer crashes (stale `ui.proxy_control` import fixed; broken since before v0.2.6).
- **Promoted from experimental**: menu label is now "多模型代理" / "Multi-Model Proxy" without the "(实验性)/(Exp)" marker.

## What's New in v0.6.0.0

### 🚀 vLLM 0.30.0 Full Support

Updated to support vLLM v0.30.0 (762 commits from 315 contributors). **Major version bump** from 0.5.0.0 to 0.6.0.0.

**vLLM 0.30 Highlights:**
- **New models**: DeepSeek-V4.1-Flash (KV entirely in MXFP8 via FlashMLA V4.1 on SM100, Mega-mHC, async Engram prefetch + DP sharding), DeepSeek-V4-Flash-Vision-Exp (ROCm + LoRA), GLM-5.3-Flash (EPLB, FlashKDA), K2-Horizon, Cohere Compass, Bailing V3 VL, Nanbeige4.2, DeepSeek-V4 CPU backend
- **Fast Start**: persistent per-GPU weight-cache daemon — engine restarts map weights over CUDA IPC with `--load-format ipc_cache` (FP4 checkpoints & multi-node TP supported)
- **Watermarking**: Gumbel-max watermarked generation/detection (keyed PRF, per-request opt-out, spec-decode compatible) via `--watermark-config`
- **HiSparse**: host-resident KV tier for sparse-MLA decode (`HiSparseConnector`) — spills KV to pinned host memory with a per-request GPU hot buffer
- **Model Runner V2**: dual-batch overlap (eager + FULL CUDA graphs), MTP/EAGLE3/DFlash/DSpark under pipeline parallelism, adaptive verification (online acceptance estimator), graph capture 12s→2s
- **New args**: `--engram-config`, `--watermark-config`, `--kda-decode-backend`, `--sparse-indexer-topk-backend`, `--dp-sync-interval`, `--elastic-ep-max-dp-size`, `--enable-mamba-fine-grained-prefix-cache`, `--enable-nccl-comm-suspend`, `--enable-scale-out`
- **Kimi K3 / Qwen3.8 perf**: KDA mixed-batch without gather/scatter (+5.2-7.7% E2E), FlashInfer KDA kernels, QSA prefill/decode indexer split, fused PLE kernels, FP8 indexer cache
- **Breaking**: scale-out endpoints now gated behind `--enable-scale-out`; GPTQ `g_idx` ordering removed; `all` Mamba cache mode deprecated (falls back to MRV1); YaRN alignment shrinks some `max_model_len` values; default audio resampler PyAV→torchaudio

**Schema v2.7.0:** 329 arguments (9 new), `--spec-method` 40 choices (+`glm5_next_mtp`), `--moe-backend` 22 choices (+`aiter_triton_mxfp4_bf16`, `rdna3`), 51 tool parsers (+`deepseek_v41`, `k2_horizon`), 34 reasoning parsers (+`deepseek_v41`, `k2_horizon`)

> See [Release Notes](RELEASE_NOTES_v0.6.0.0.md) for full details.

## What's New in v0.5.0.0

### 🚀 vLLM 0.29.0 Full Support

Updated to support vLLM v0.29.0 (594 commits from 277 contributors). **Major version bump** from 0.4.0.0 to 0.5.0.0.

**vLLM 0.29 Highlights:**
- **Model Runner V2 default for all models** — MRV1 deprecated, removal targeted v0.32; CUDA graph memory profiling, batch-sharded sampling (logits memory ÷ TP)
- **New models**: Hy4-preview (Tencent 770B/49B MoE, Gated DSA + native MTP), Qwen3.8-Flash-Next (BF16/FP8/NVFP4 + MTP), GraniteSWA/GraniteMoeSWA, NemotronH_Omni_Reasoning_V3, Kimi K3 NVFP4 checkpoints
- **Spec decode**: `--per-request-spec-decode-metrics` acceptance stats, DFlash2 from speculators format, adaptive verification with logprobs
- **Mamba prefix caching**: internal prefill checkpoints → 9%-25% TTFT gain; `--prefix-cache-retention-interval` (default 0)
- **Admission control**: `--max-num-queued-reqs` / `--max-num-queued-tokens`
- **Kimi K3 / DSv4 perf**: fused MXFP4 top-k finalization, 6.6-7.6x K3 Mamba metadata kernel, GEMM-AR, DCP+DSpark, `--dcp-q-replicate`
- **Quantization**: b12x FP4 MoE (SM120/121), FlashInfer TRT-LLM MXFP8 linear, AutoRound block FP8, Humming MXFP4/WNA16
- **API**: `/v1/messages/render`, `/cohere/v2/chat/render`, `--sse-keep-alive-interval`
- **Breaking**: 10 legacy architectures removed, PyAV decoder removed, `VLLM_TEST_FORCE_FP8_MARLIN` removed

**Schema v2.6.0:** 320 arguments (8 new), `--spec-method` 39 choices (+`hy_v4_mtp`, `qwen4_exp_mtp`), `--moe-backend` 20 choices (+`b12x`, +FlashInfer `moe_ep` mega backends), 49 tool parsers (+`hy_v4`, `granite-20b-fc`), 32 reasoning parsers (+`hy_v4`)

> See [Release Notes](RELEASE_NOTES_v0.5.0.0.md) for full details.

## What's New in v0.4.0.0

### 🚀 vLLM 0.28.0 Full Support

Updated to support vLLM v0.28.0 (584 commits from 270 contributors). **Major version bump** from 0.3.0.0 to 0.4.0.0.

**vLLM 0.28 Highlights:**
- **Kimi-K3 Performance Push**: DCP support, fused FlashKDA decode/prefill kernels, SiTU activation for MegaMoE, GEMM-RS sequence parallelism, adaptive speculative token budget (~60% better DSpark TTFT), optional shared-expert sharding (~17 GiB/GPU saved), ROCm V2 model runner
- **DeepSeek V4**: Sparse MLA end-to-end for decode/MTP/DSpark, AMD Quark NVFP4, reasoning-effort prompts, ROCm gfx11/gfx950
- **Speculative Decoding**: DFlash2 (local convolution + candidate selector), DSpark confidence-scheduled verification, async scheduling auto-enabled for draft models
- **Model Runner V2**: E/P/D disaggregation, weight offloading, multi-layer MTP KV cache, encoder CUDA graphs, `thinking_token_budget`
- **Tiered KV Offloading**: Disk offloading, out-of-tree secondary tier managers via `module_path`
- **New Models**: Muse Glimmer, Ling 3.0 Flash (BF16/MTP/FP8/MXFP4), Dots3 NOTE (multimodal), Interns2mobius
- **New Defaults**: `max_num_batched_tokens` 8192→16384, prefix caching on by default for Mamba
- **Breaking**: bitsandbytes moved to out-of-tree plugin, `calculate_kv_scales` and `override_attention_dtype` removed

**New Tool Call Parsers (3 additions, 47 total):**
`dots` (Dots3 NOTE), `ling3` (Ling 3.0 Flash), `muse_glimmer` (Muse Glimmer)

**New Reasoning Parsers (9 additions, 31 total):**
`cohere_command3`, `cohere_command4`, `glm45`, `holo2`, `inkling`, `ling3`, `mimo`, `minimax_m2_append_think`, `muse_glimmer`

**20 New CLI Arguments:**
- **Kimi K3**: `--kda-prefill-backend` (auto/triton/flashkda)
- **Mamba**: `--mamba-ssu-algorithm` (auto/simple/vertical/horizontal), `cpu` backend, `bfloat16` cache dtypes
- **Speculative**: `mlp_speculator`, `draft_model`, `suffix`, `custom_class`, `extract_hidden_states` spec methods
- **Fault Tolerance**: `--enable-fault-tolerance`, `--fault-tolerance-config`
- **Frontend**: `--cohere-format`, `--cohere-is-reasoning-model`, `--fingerprint-mode`, `--uvicorn-log-level`
- **Multimodal**: `--mm-device-do-normalize`, `--mm-hasher-algorithm`, `--mm-processor-device`, `--video-pruning-method`
- **Scheduling/Memory**: `--max-num-scheduled-tokens`, `--prefix-match-unit`, `--replayssm-buffer-len`, `--use-replayssm`
- **Expanded**: `--linear-backend` (21 backends), `--quantization` (30 methods)

> See [Release Notes](RELEASE_NOTES_v0.4.0.0.md) for full details.

## What's New in v0.3.0.0

### 🚀 vLLM 0.27.x Full Support

Updated to support vLLM v0.27.0/v0.27.1 (561 commits from 242 contributors). **Major version bump** from 0.2.9.x to 0.3.0.0.

**vLLM 0.27 Highlights:**
- **Kimi K3 Full Stack**: Core model files, kernels, Python/Rust frontends, AttnRes, DeepGEMM, compressed-tensors, DSpark AR fusion, shared-expert sharding
- **Qwen3.5**: Text-only dense and MoE models with EVS video token pruning
- **PyTorch 2.13.0**: Breaking environment upgrade with torchvision 0.28.0 and Triton 3.7.1
- **FlashAttention 4 Deepened**: FP8 KV cache, headdim-256, JIT warmup, runner-owned Triton kernel warmup
- **DeepSeek-V4 Performance**: Sequence parallelism, ~2x kernel improvement, adaptive topk width, compact MXFP4 indexer
- **Model Runner V2**: Expands to encoder-only, embedding/classification, multimodal on CPU
- **Expanded Quantization**: FP4 Qutlass, CuTeDSL MoE (NVFP4), MXFP8 linear (INC), AutoRound W4A16 MoE, MXFP4 (XPU), TurboQuant KV quant
- **New Models**: K-EXAONE-2.0-750B-A37B, VaultGemma, jina-embeddings-v5-text-nano
- **Hardware**: `sm_107` (NVIDIA Rubin), ROCm gfx1250, NCCL 2.30.7 (DeepEPv2)
- **Models Removed**: Plamo2, Ouro

**New Tool Call Parsers (7 additions):**
`kimi_k3`, `mimo`, `llama4_json`, `llama4_pythonic`, `cohere_command3`, `cohere_command4`, `glm45`

**New Reasoning Parsers (1 addition):**
`kimi_k3`

**Speculative Decoding Fixes:**
- `--speculative-config` deprecation status corrected (was incorrectly marked as removed)
- Added `--spec-tokens` parameter
- Updated `--spec-method` choices: `ngram`, `ngram_gpu`, `medusa`, `eagle`, `eagle3`, `mtp`, `dflash`, `dspark`
- Fixed dict-to-JSON serialization in CLI argument builder

**MoE Backend Expansion:**
- Expanded from 5 to 17 backends: `auto`, `triton`, `batched_triton`, `deep_gemm`, `deep_gemm_mega_moe`, `cutlass`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_cutedsl`, `flashinfer_b12x`, `marlin`, `humming`, `triton_unfused`, `aiter`, `flydsl`, `hpc`, `emulation`

### 📋 One-Click Command Import

New `vllm-cli import` command to parse raw `vllm serve` commands into profiles:

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

> See [Release Notes](RELEASE_NOTES_v0.3.0.0.md) for full details.

## What's New in v0.2.9.9

### 🚀 vLLM 0.26.0 Full Support

Updated to support vLLM v0.26.0 (411 commits from 212 contributors).

**vLLM 0.26 Highlights:**
- **Inkling Model Family**: Full support stack — base modeling, piecewise CUDA graphs, Hopper FA4 relative attention, MTP=1 speculative decoding, LoRA, and ModelOpt NVFP4 quantization
- **DeepSeek-V4 Performance Push**: Specialized routing kernel (2.94% E2E TPOT), `fused_topk_bias` (1.5-2x kernel), redundant repeat/copy removal (1.8% E2E TPOT)
- **Transformers 5.13.0**: Olmo/Olmo2, MistralLarge3 (AutoWeightsLoader), HunyuanVL native processor
- **Decode Context Parallel (DCP)**: Hybrid attention support, DCP + Eagle for Tokenspeed MLA
- **PD Disaggregation**: NIXL pipeline-parallel prefill in push mode
- **Expanded Quantization**: Humming w[2-7]a[4,8], NVFP4/MXFP4 (`nvfp4_per_token` online MoE), INT2 XPU weight-only
- **API & Frontend**: Rust frontend multimodal video/audio, OpenAI `bad_words`/`logprob_token_ids`/`include_reasoning`, endpoint plugins framework, deepstream video decoding
- **Models Removed**: TeleChat, Persimmon, Fuyu (removed upstream)

**New Tool Call Parsers (13 additions):**
`inkling`, `apertus`, `ernie45`, `functiongemma`, `gigachat3`, `granite4`, `hermes`, `hunyuan_a13b`, `lfm2`, `olmo3`, `poolside_v1`, `pythonic`, `xlam`

**New Reasoning Parsers (5 additions):**
`inkling`, `ernie45`, `hunyuan_a13b`, `olmo3`, `poolside_v1`

**Hardware & Performance:**
- **Blackwell**: FlashInfer fused all-reduce (world_size=16), CuTeDSL/FA4-MLA warmup, B12x backend
- **AMD/ROCm**: torch 2.11 stable ABI, AITER FlashAttention MLA, HybridW4A16 linear kernel
- **Intel XPU**: INT2 weight-only quant, DSpark speculative decoding
- **CPU/RISC-V**: Accelerated unquantized MoE (AArch64), w8a8 int8 MoE, RVV W4A8 INT4 GEMM

> See [Release Notes](RELEASE_NOTES_v0.2.9.9.md) for full details.

## What's New in v0.2.9.8

### 🚀 vLLM 0.25.0 Full Support

Updated to support vLLM v0.25.0. **291 CLI arguments** now supported (276 active + 15 new).

**vLLM 0.25 Highlights:**
- **Model Runner V2 Default**: Now the standard execution path for all dense models with EVS, realtime embeddings, Mamba hybrid prefix caching, multimodal-prefix bidirectional attention, and dynamic speculative decoding
- **PagedAttention Removed**: Legacy attention implementation deleted — V1/MRv2 backends are the only path
- **Transformers Backend Parity**: Now as fast as native vLLM with FP8 MoE support, CUDA graph fixes, and GPTBigCode/Starcoder2/RoBERTa migration
- **New Streaming Parser Engine**: Unified tool-call/reasoning parsing with Kimi k2.5/k2.6/k2.7, seed_oss, and DeepSeek V4 parsers
- **Universal Speculative Decoding**: Heterogeneous vocabulary support (TLI), new DSpark and DFlash drafters
- **New Models**: LLaVA-OneVision-2, Unlimited OCR, MOSS-Transcribe-Diarize, openai/privacy-filter, Hy3, GLM-5 family, MiniMax-M3
- **Hardware**: Blackwell/GB300 FlashInfer tuning, Helion kernels, AMD/ROCm torch 2.11 stable, Intel XPU W8A8 FP8, RISC-V/POWER support

**New CLI Arguments (14 additions):**
- **Backend**: `--backend` (mp, ray, debug engine backend selection)
- **Cache**: `--cache-dtype` (KV cache data type)
- **Monitoring**: `--enable-per-request-metrics`, `--jit-monitor-mode`, `--ray-workers-use-nsight`
- **Quantization**: `--enable-stochastic-rounding`, `--quantization-config`, `--stochastic-rounding-philox-rounds`
- **Multimodal**: `--mm-ipc-gpu-memory-gb`
- **Scheduling**: `--policy`, `--ubatch-size`
- **LoRA**: `--target-modules`
- **Debug**: `--model-class-overrides`, `--method`

**Deprecated (9 arguments):**
`--aggregate-engine-logging`, `--data-parallel-multi-port-external-lb`, `--default-max-num-seqs`, `--fail-on-environ-validation`, `--limit-mm-per-prompt`, `--mamba-cache-philox-rounds`, `--shutdown-timeout`, `--sliding-window`, plus legacy v0.24.0 removals

**New Parsers:**
- Tool call: `cohere_command`, `deepseek_v3`, `minimax_m3`, `rust`
- Reasoning: `cohere_command`, `identity`, `minimax_m3`

### 🔧 Dependency Doctor

New on-demand dependency health checker for diagnosing missing system libraries:

```bash
vllm-cli doctor
```

Detects:
- Missing FFmpeg shared libraries (required by torchcodec in vLLM 0.25.0+)
- torchcodec availability and LD_LIBRARY_PATH issues
- CUDA/PyTorch/vLLM installation status

Runs **only on-demand** (not on every startup), so it has zero impact on performance.

## What's New in v0.2.9.7

### 🚀 vLLM 0.24.0 Full Support

Updated to support vLLM v0.24.0. **276 CLI arguments** now supported (up from 216).

**vLLM 0.24 Highlights:**
- **Streaming Parser Engine**: Unified tool-call/reasoning parsing — parsers are built-in (no plugin files needed)
- **Model Runner V2**: Quantized models by default, GraniteMoE default, DFlash speculative decoding
- **DeepEP v2**: Integrated for expert parallelism
- **Rust Frontend**: API server migrated to Rust with new endpoints
- **Diffusion LLMs**: DiffusionGemma support with CPU path
- **Device Selection (BREAKING)**: `--device-ids` replaces `CUDA_VISIBLE_DEVICES`

**New CLI Arguments (60 additions):**
- **Device**: `--device-ids` (replaces CUDA_VISIBLE_DEVICES)
- **Data Parallel**: `--data-parallel-backend`, `--data-parallel-external-lb`, `--data-parallel-rank`
- **LoRA**: `--lora-dtype`, `--lora-target-modules`, `--fully-sharded-loras`
- **Multimodal**: `--mm-encoder-attn-backend`, `--mm-processor-cache-gb`, `--video-pruning-rate`
- **Model Loading**: `--safetensors-load-strategy`, `--model-weights`
- **Diffusion**: `--diffusion-config`
- **Streaming Parsers**: `glm47`, `glm51`, `glm52`, `nemotron_v3` added to reasoning_parser and tool_call_parser

**Deprecated (13 arguments):**
`--swap-space`, `--cpu-offload-space`, `--num-lookahead-slots`, `--num-speculative-tokens`, `--speculative-model`, `--rope-scaling`, `--rope-theta`, `--guided-decoding-backend`, `--generation-config-override`, `--limit-per-prompt`, `--tq-max-kv-splits-for-cuda-graph`, `--disable-async-output-proc`, `--disable-frontend-multiprocessing`

## What's New in v0.2.9.6

### 🚀 vLLM 0.22.0 / 0.22.1 Full Support

Updated to support vLLM v0.22.0 and v0.22.1. **213 CLI arguments** now supported (up from 161).

**vLLM 0.22 Highlights:**
- **DeepSeek V4 maturity**: Dedicated package with NVFP4 fused MoE, MTP speculative decoding
- **Model Runner V2**: Default for Qwen3 dense models with sleep-mode weight reload
- **Rust frontend**: Experimental Rust front-end with DP Supervisor
- **Batch invariance**: Cutlass FP8 for 28.9% E2E latency improvement
- **Multi-tier KV offloading**: Python filesystem tier, Mooncake disk offloading

**New CLI Arguments (52 additions):**
- **Linear backend**: `--linear-backend` (auto, flashinfer, cutlass, triton, native)
- **Speculative decoding**: `--spec-method` (ngram, eagle, medusa) + `--spec-model`
- **HF auth**: `--hf-token` for private models
- **DP Supervisor**: `--data-parallel-supervisor-port`, `--dp-supervisor-probe-*`
- **Distribution**: `--master-addr`, `--master-port`, `--nnodes`, `--node-rank`
- **FlashAttention**: `--enable-flash-late-interaction`
- **Model loading**: `--load-format`, `--download-dir`, `--convert`, `--code-revision`

**Enhanced CLI Args Sync Tool:**
- Version-specific sync: `--tag v0.22.1`
- Dataclass field extraction for v0.22+ architecture
- Kwargs unpacking detection

> **Note**: vLLM 0.22 restructured its CLI — 103 arguments moved from CLI flags to environment variables/config. All remain available in the schema for backward compatibility.

## What's New in v0.2.9.5

### 🚀 vLLM 0.20.0 Full Support

Updated to support vLLM v0.20.0 with 752 commits from 320 contributors. **161 CLI arguments** now supported (up from 83).

**TurboQuant 2-bit KV Cache:**
- `turboquant_k8v4`: K8V4 compression with 4× KV cache capacity
- `turboquant_4bit_nc`: 4-bit non-compressed KV cache
- `turboquant_k3v4_nc` / `turboquant_3bit_nc`: 3-bit KV cache variants
- New `--kv-cache-dtype-skip-layers` for fine-grained control
- Auto-skip first/last 2 layers for boundary protection

**New KV Cache Options:**
- `fp8_ds_mla`, `int8_per_token_head`, `fp8_per_token_head`, `nvfp4`
- `--kv-cache-memory-bytes` for fine-grain memory control
- `--kv-offloading-size` / `--kv-offloading-backend` for CPU offloading

**New Parallelism Features:**
- Decode/Prefill context parallelism (`--decode-context-parallel-size`, `--prefill-context-parallel-size`)
- Expert parallel load balancing (`--enable-eplb`, `--eplb-config`)
- Dynamic batch optimization (`--enable-dbo`)
- Elastic expert parallelism (`--enable-elastic-ep`)
- NUMA binding for GPU workers (`--numa-bind`)

**New Model Support:**
- DeepSeek V4, Hunyuan v3 (HYV3), Granite 4.1 Vision, EXAONE-4.5
- New reasoning parsers: `hyv3`, `mimo`
- New tool call parsers: `hyv3`, `mimo`

**New Quantization:**
- `humming` quantization kernel support

**PyTorch 2.11 + CUDA 13.0 Default:**
- vLLM now ships on PyTorch 2.11 with CUDA 13.0 as default
- Transformers v5 support
- Python 3.14 added to supported versions

See [Release Notes](RELEASE_NOTES_v0.2.9.5.md) for details.

## What's New in v0.2.9

### 🚀 Official vLLM Recipes Integration

Integrated configurations from official vLLM recipes with 8 new optimized profiles:

**New Models:**
- **DeepSeek-V3.1** - Thinking mode + tool calling
- **Qwen3-Next** - Hybrid attention + MTP support
- **GLM-4.5** - Tool calling + reasoning

**New Hardware:**
- **AMD MI300x/MI325** - ROCm optimizations
- **AMD MI355x** - Advanced Triton kernels

**20+ New Environment Variables** for fine-tuned control

**CLI Argument Sync:**
- Automate sync of vLLM CLI arguments from GitHub source
- Commands:
  - `vllm-cli recipes --sync-args` — Check for new/removed CLI args
  - `vllm-cli recipes --sync-args --apply-args` — Apply changes to local schema
  - `vllm-cli recipes --clean-deprecated` — Remove deprecated args from schema
- New CLI args, deprecated args tracking, and cleanup

See [Release Notes](RELEASE_NOTES_v0.2.9.md) for details.

## What's New in v0.2.8

### ✨ Enhanced Chat Template Configuration

Improved `--chat-template` parameter configuration with interactive file browsing:
- **Interactive File Browser** - Automatically finds `.jinja` template files
- **Smart Discovery** - Searches common locations (./examples/, ~/.cache/huggingface/)
- **File Validation** - Checks if files exist before using
- **Better Guidance** - Clear hints about when to use chat templates

**Why This Matters:** Some models (Qwen3, Llama3.1) require specific chat templates for proper tool calling. Now it's much easier to configure!

See [Release Notes](RELEASE_NOTES_v0.2.8.md) for details.

## What's New in v0.2.6

### 🎯 GPT-OSS-20B Support with Tool Calling

New profiles specifically optimized for OpenAI's GPT-OSS-20B model:
- **`gpt_oss_20b`** - Full performance (~25-30GB VRAM)
- **`gpt_oss_20b_low_memory`** - Memory-optimized (~13-15GB VRAM)

Both profiles include:
- ✓ Pre-configured tool calling with `openai` parser
- ✓ Automatic tool choice enabled
- ✓ MoE optimizations for reduced GPU load
- ✓ Ready to use out of the box

**New Documentation:**
- [GPT-OSS-20B Setup Guide](GPT_OSS_20B_SETUP.md)
- [Memory Optimization Guide](MEMORY_OPTIMIZATION.md)
- [Profile Comparison](GPT_OSS_20B_PROFILES_COMPARISON.md)
- [Environment Variables Reference](ENVIRONMENT_VARIABLES.md)

## What's New in v0.2.5

### Multi-Model Proxy Server (Experimental)

The Multi-Model Proxy is a new experimental feature that enables serving multiple LLMs through a single unified API endpoint. This feature is currently under active development and available for testing.

**What It Does:**
- **Single Endpoint** - All your models accessible through one API
- **Live Management** - Add or remove models without stopping the service
- **Dynamic GPU Management** - Efficient GPU resource distribution through vLLM's sleep/wake functionality
- **Interactive Setup** - User-friendly wizard guides you through configuration

**Note:** This is an experimental feature under active development. Your feedback helps us improve! Please share your experience through [GitHub Issues](https://github.com/Chen-zexi/vllm-cli/issues).

For complete documentation, see the [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## What's New in v0.2.4

### 🚀 Hardware-Optimized Profiles for GPT-OSS Models
New built-in profiles specifically optimized for serving GPT-OSS models on different GPU architectures:
- **`gpt_oss_ampere`** - Optimized for NVIDIA A100 GPUs
- **`gpt_oss_hopper`** - Optimized for NVIDIA H100/H200 GPUs
- **`gpt_oss_blackwell`** - Optimized for NVIDIA Blackwell GPUs

Based on official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) for maximum performance.

### ⚡ Shortcuts System
Save and quickly launch your favorite model + profile combinations:
```bash
vllm-cli serve --shortcut my-gpt-server
```

### 🦙 Full Ollama Integration
- Automatic discovery of Ollama models
- GGUF format support (experimental)
- System and user directory scanning

### 🔧 Enhanced Configuration
- **Environment Variables** - Universal and profile-specific environment variable management
- **GPU Selection** - Choose specific GPUs for model serving (`--device 0,1`)
- **Enhanced System Info** - vLLM feature detection with attention backend availability

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## Quick Start

### Important: vLLM Installation Notes
⚠️ **Binary Compatibility Warning**: vLLM contains pre-compiled CUDA kernels that must match your PyTorch version exactly. Installing mismatched versions will cause errors.

vLLM-CLI will not install vLLM or Pytorch by default.

### Installation

#### Option 1: Install vLLM seperately and then install vLLM CLI (Recommended)
```bash
# Install vLLM -- Skip this step if you have vllm installed in your environment
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
# Or specify a backend: uv pip install vllm --torch-backend=cu128

# Install vLLM CLI
uv pip install --upgrade vllm-cli
uv run vllm-cli

# If you are using conda:
# Activate the environment you have vllm installed in
pip install vllm-cli
vllm-cli
```
#### Option 2: Install vLLM CLI + vLLM

```bash
# Install vLLM CLI + vLLM
pip install vllm-cli[vllm]
vllm-cli
```

#### Option 3: Build from source (You still need to install vLLM seperately)
```bash
git clone https://github.com/Chen-zexi/vllm-cli.git
cd vllm-cli
pip install -e .
```

#### Option 4: For Isolated Installation (pipx/system packages)

⚠️ **Compatibility Note:** pipx creates isolated environments which may have compatibility issues with vLLM's CUDA dependencies. Consider using uv or conda (see above) for better PyTorch/CUDA compatibility.

```bash
# If you do not want to use virtual environment and want to install vLLM along with vLLM CLI
pipx install "vllm-cli[vllm]"

# If you want to install pre-release version
pipx install --pip-args="--pre" "vllm-cli[vllm]"
```

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- vLLM package installed
- For dependency issues, see [Troubleshooting Guide](docs/troubleshooting.md#dependency-conflicts)

### Basic Usage

```bash
# Interactive mode - menu-driven interface
vllm-cl
# Serve a model
vllm-cli serve --model openai/gpt-oss-20b

# Use a shortcut
vllm-cli serve --shortcut my-model
```

For detailed usage instructions, see the [📘 Usage Guide](docs/usage-guide.md) and [🌐 Multi-Model Proxy Guide](docs/multi-model-proxy.md).

## Configuration

### Built-in Profiles

vLLM CLI includes 40+ optimized profiles for different use cases:

**General Purpose:**
- `standard` - Minimal configuration with smart defaults
- `high_throughput` - Maximum performance configuration
- `low_memory` - Memory-constrained environments
- `moe_optimized` - Optimized for Mixture of Experts models

**Hardware-Specific (GPT-OSS):**
- `gpt_oss_ampere` - NVIDIA A100 GPUs
- `gpt_oss_hopper` - NVIDIA H100/H200 GPUs
- `gpt_oss_blackwell` - NVIDIA Blackwell GPUs
- `gpt_oss_20b` - OpenAI GPT-OSS-20B with tool calling (~25-30GB VRAM)
- `gpt_oss_20b_low_memory` - GPT-OSS-20B optimized for low memory (~13-15GB VRAM)

See [**📋 Profiles Guide**](docs/profiles.md) for detailed information.

### Configuration Files
- **Main Config**: `~/.config/vllm-cli/config.yaml`
- **User Profiles**: `~/.config/vllm-cli/user_profiles.json`
- **Shortcuts**: `~/.config/vllm-cli/shortcuts.json`


## Documentation

- [**📘 Usage Guide**](docs/usage-guide.md) - Complete usage instructions
- [**🌐 Multi-Model Proxy**](docs/multi-model-proxy.md) - Serve multiple models simultaneously
- [**📋 Profiles Guide**](docs/profiles.md) - Built-in profiles details
- [**❓ Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions
- [**📸 Screenshots**](docs/screenshots.md) - Visual feature overview
- [**🔍 Model Discovery**](docs/MODEL_DISCOVERY_QUICK_REF.md) - Model management guide
- [**🦙 Ollama Integration**](docs/ollama-integration.md) - Using Ollama models
- [**⚙️ Custom Models**](docs/custom-model-serving.md) - Serving custom models
- [**🗺️ Roadmap**](docs/roadmap.md) - Future development plans

## Integration with hf-model-tool

vLLM CLI uses [hf-model-tool](https://github.com/Chen-zexi/hf-model-tool) for model discovery:
- Comprehensive model scanning
- Ollama model support
- Shared configuration

## Development

### Project Structure
```
src/vllm_cli/
├── cli/           # CLI command handling
├── config/        # Configuration management (profiles, sync, import)
├── models/        # Model management
├── server/        # Server lifecycle
├── ui/            # Terminal interface
├── schemas/       # JSON schemas
└── validation/    # Configuration validation
```

### Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

MIT License - see [LICENSE](LICENSE) file for details.
