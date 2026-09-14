# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.5.0.0] - 2026-09-10

### Added
- **vLLM 0.29.0 Full Support**: Updated to support vLLM v0.29.0 (594 commits from 277 contributors)
- **Model Runner V2 Default**: MRV2 now default for all models; MRV1 deprecated (removal targeted v0.32)
- **Hy4-preview**: Tencent 770B/49B-active MoE with Gated DeepSeek Sparse Attention and native MTP (`hy_v4_mtp`, `hy_v4` parsers)
- **Qwen3.8-Flash-Next**: BF16/FP8/NVFP4 with MTP (`qwen4_exp_mtp`)
- **New CLI Arguments (8)**:
  - `--per-request-spec-decode-metrics` (none/summary/detailed): acceptance stats in OpenAI API responses
  - `--max-num-queued-reqs` / `--max-num-queued-tokens`: queue admission control
  - `--prefix-cache-retention-interval` (default 0): Mamba/SWA prefix cache retention; env var deprecated
  - `--dcp-q-replicate`: DCP query replication (default-on for GLM sparse attention)
  - `--enable-batch-sharded-sampling`: MRV2 batch-sharded sampling, logits memory cut by 1/TP
  - `--enable-trace-replay`: deterministic decode replay (`trace_decode_token_ids`)
  - `--sse-keep-alive-interval`: SSE keep-alive comments for idle streams
- **New Models**: GraniteSWA, GraniteMoeSWA, NemotronH_Omni_Reasoning_V3 (with MTP), Kimi K3 NVFP4 checkpoints
- **New Tool Call Parsers**: `hy_v4`, `granite-20b-fc` (back in registry) — 49 total
- **New Reasoning Parser**: `hy_v4` — 32 total
- **Mamba Prefix Caching**: internal prefill checkpoints, 9%-25% TTFT improvement
- **RL Weight Sync**: `sharded_rdt` P2P backend (NIXL/RDT slice pulls), rank-local IPC updates
- **New Quantization Backends**: FlashInfer TRT-LLM MXFP8 linear, b12x FP4 MoE (SM120/SM121), AutoRound block FP8, Humming MXFP4+blockFP8, Humming WNA16 MoE
- **API**: `/v1/messages/render` (Anthropic), `/cohere/v2/chat/render`, video embeds in Python frontend
- **Security**: `cache_salt` bounded to 1024; oversized media rejected pre-download; `api_key`/`hf_token` redacted from logs
- **New Defaults**: FlashInfer all-reduce on by default for TP CUDA groups; deterministic `NONE_HASH` for prefix caching

### Changed
- **Schema version**: 2.5.0 → 2.6.0 (320 arguments)
- **`--spec-method` choices**: 37 → 39 (added `hy_v4_mtp`, `qwen4_exp_mtp`)
- **`--moe-backend` choices**: 17 → 20 (added `b12x`, `flashinfer_moe_ep_mega_deep_gemm`, `flashinfer_moe_ep_mega_cutedsl`)
- **vLLM version range**: 0.20.0 - 0.28.0 → 0.20.0 - 0.29.0

### Removed (upstream vLLM 0.29.0)
- Ten deprecated architectures (Arctic, Chameleon, Cheers, Fairseq2Llama, FireRedLID, GritLM, HCXVision, MPT, RWForCausalLM/StableLMEpoch aliases, PrithviGeoSpatialMAE)
- PyAV video decoder backend (use OpenCV or Torchcodec)
- `VLLM_TEST_FORCE_FP8_MARLIN` (use `--linear-backend`/`--moe-backend`)
- `--attention-config.use_prefill_decode_attention` (dead)

## [v0.4.0.0] - 2026-08-28

### Added
- **vLLM 0.28.0 Full Support**: Updated to support vLLM v0.28.0 (584 commits from 270 contributors)
- **Kimi-K3 Performance Stack**: DCP support, fused FlashKDA decode/prefill kernels (`--kda-prefill-backend flashkda`), SiTU activation for MegaMoE, GEMM-RS sequence parallelism, adaptive speculative token budget, optional shared-expert sharding, ROCm V2 model runner
- **DeepSeek V4 Sparse MLA**: End-to-end for decode/MTP/DSpark, AMD Quark NVFP4, reasoning-effort prompts, ROCm gfx11/gfx950
- **Speculative Decoding Advances**: DFlash2 (local convolution + candidate selector), DSpark confidence-scheduled verification, async scheduling auto-enabled for draft models
- **New Tool Call Parsers (3)**: `dots` (Dots3 NOTE), `ling3` (Ling 3.0 Flash), `muse_glimmer` (Muse Glimmer) — 47 total
- **New Reasoning Parsers (9)**: `cohere_command3`, `cohere_command4`, `glm45`, `holo2`, `inkling`, `ling3`, `mimo`, `minimax_m2_append_think`, `muse_glimmer` — 31 total
- **New CLI Arguments (20)**:
  - **Speculative**: `--mlp_speculator`/`--draft-model`/`--suffix`/`--custom-class`/`--extract-hidden-states` spec methods, `dots3_note_mtp`, `bailing_hybrid_v3_mtp` MTP types
  - **Mamba**: `--mamba-config` (JSON), `--mamba-ssu-algorithm` (auto/simple/vertical/horizontal), `cpu` mamba backend, `bfloat16` SSM cache dtype
  - **KV Offload**: `--prefix-match-unit` (partial-tail prefix reuse)
  - **Fault Tolerance**: `--enable-fault-tolerance`, `--fault-tolerance-config`
  - **Frontend**: `--cohere-format`, `--cohere-is-reasoning-model`, `--fingerprint-mode`, `--uvicorn-log-level`
  - **Multimodal**: `--mm-device-do-normalize`, `--mm-hasher-algorithm`, `--mm-processor-device`, `--video-pruning-method`
  - **Scheduling**: `--max-num-scheduled-tokens`, `--replayssm-buffer-len`, `--use-replayssm`
  - **Other**: `--ec-manager-config`, `--enable-bf16x3-router-gemm`, `--enable-moe-shared-loras`, `--return-sampling-mask`
- **Expanded `--linear-backend`**: 21 backends (added `b12x`, `deep_gemm`, `torch`, `machete`, `fbgemm`, `conch`, `exllama`, `emulation`, `xpu`, `xpu_woq`)
- **Expanded `--quantization`**: 30 methods (added online shorthands `fp8_per_tensor`/`fp8_per_block`/`mxfp8`/`nvfp4_per_token`, `quark`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`, `deepseek_v4_fp8`, `modelopt_mixed`, `auto_*` variants)
- **New Models**: Muse Glimmer, Ling 3.0 Flash (BF16/MTP/FP8/MXFP4), Dots3 NOTE (multimodal), Interns2mobius, Qwen3.8 (ROCm)
- **Model Runner V2 Maturation**: E/P/D disaggregation, weight offloading, multi-layer MTP KV cache, encoder CUDA graphs, decoder token-wise pooling, `thinking_token_budget`
- **Tiered KV Cache Offloading**: Disk offloading, out-of-tree secondary tier managers via `module_path`, tiering metrics
- **New Defaults**: `max_num_batched_tokens` raised 8192→16384, prefix caching enabled by default for Mamba models

### Changed
- **Schema version**: 2.4.0 → 2.5.0 (312 arguments, 47 tool parsers, 31 reasoning parsers)
- **`--spec-method` choices**: Expanded to 37 values (added `mlp_speculator`, `draft_model`, `suffix`, `custom_class`, `extract_hidden_states`, all MTP types, `dots3_note_mtp`, `bailing_hybrid_v3_mtp`)
- **vLLM version range**: 0.20.0 - 0.27.0 → 0.20.0 - 0.28.0

### Removed
- **`--calculate-kv-scales`**: Runtime KV scale calculation removed upstream in v0.28.0
- **`--override-attention-dtype`**: Removed upstream in v0.28.0
- **`bitsandbytes` quantization**: Migrated to out-of-tree plugin in v0.28.0
- **Reasoning parsers**: `cohere_command` (split into `cohere_command3`/`cohere_command4`), `identity` (no longer registered)

### Deprecated
- **`--enable-stochastic-rounding`** → `--enable-mamba-cache-stochastic-rounding`
- **`--stochastic-rounding-philox-rounds`** → `--mamba-cache-philox-rounds`
- **`--policy`** → `--scheduling-policy`
- **`--target-modules`** → `--lora-target-modules`
- **`--cache-dtype`** → `--kv-cache-dtype`
- **`--backend`** → `--distributed-executor-backend`
- **`--method`**: Removed in v0.28.0

## [v0.3.0.0] - 2025-08-12

### Added
- **One-Click Command Import**: New `vllm-cli import` command to parse raw `vllm serve` commands into profiles
  - Supports standard CLI flags (`--flag value`, `--boolean-flag`)
  - Supports dot-notation config keys (`--speculative_config.method dflash`)
  - Auto-generates profile names from model paths
  - Preview mode with `--preview` flag
  - File input with `--file` option
- **vLLM 0.27.1 / 0.27.0 Full Support**: Updated to support vLLM v0.27.x (561 commits from 242 contributors in v0.27.0)
- **Kimi K3 Full Stack Support**: New model with dedicated tool call parser, reasoning parser, DSpark AR fusion, DeepGEMM, compressed-tensors quantized checkpoints, and optional shared-expert sharding
- **Qwen3.5 Support**: Text-only dense and MoE models with EVS video token pruning
- **New Models**: K-EXAONE-2.0-750B-A37B, VaultGemma (Transformers backend), jina-embeddings-v5-text-nano (EuroBERT encoder)
- **New Tool Call Parsers**: `kimi_k3`, `mimo`, `llama4_json`, `llama4_pythonic`, `cohere_command3`, `cohere_command4`, `glm45`
- **New Reasoning Parser**: `kimi_k3`
- **`--spec-tokens` parameter**: Added missing speculative decoding token count parameter
- **PyTorch 2.13.0**: Breaking environment upgrade with torchvision 0.28.0 and Triton 3.7.1
- **FlashAttention 4 Deepened**: FP8 KV cache support, headdim-256, JIT warmup infrastructure, runner-owned Triton kernel warmup
- **DeepSeek-V4 Performance**: Sequence parallelism, ~2x kernel improvement, 3.4% E2E TTFT, adaptive topk width, compact MXFP4 indexer KV cache
- **Model Runner V2**: Expands to non-generative workloads (encoder-only, embedding/classification, multimodal on CPU)
- **New Quantization**: FP4 Qutlass (compressed-tensors), CuTeDSL MoE for ReLU2 NVFP4, MXFP8 linear (INC), AutoRound W4A16 MoE, MXFP4 (XPU), KV quant mode (TurboQuant), ModelOpt FP8 (SM80), `--linear-backend` for ModelOpt W4A16
- **API Enhancements**: Cohere chat v2, `cache_salt` (Anthropic Messages), strict tool calling (GPT-OSS Harmony), unified Mistral parser, `stream_interval` per-request, `--limit-mm-per-prompt`
- **Hardware**: `sm_107` target (NVIDIA Rubin), ROCm gfx1250, NCCL 2.30.7 (DeepEPv2)
- **Rust Frontend**: gRPC control plane, `vllm-bench` integrated into `vllm` CLI
- **Schema v2.4**: Argument schema updated for v0.27.0 sync
- `cli_args_sync.py`: Added v0.27.1 and v0.27.0 to SUPPORTED_VLLM_VERSIONS
- `parser_sync.py`: Added kimi_k3, mimo, llama4_json, llama4_pythonic, cohere_command3/4 name mappings

### Removed
- **Plamo2 model**: Removed upstream in vLLM v0.27.0
- **Ouro model**: Removed upstream in vLLM v0.27.0
- **`max_num_partial_prefills` argument**: Removed in vLLM v0.27.0
- **`max_long_partial_prefills` argument**: Removed in vLLM v0.27.0
- **`qwen3` tool parser**: Replaced by `qwen3_coder` and `qwen3_xml` for more specific selection
- **`rust` tool parser**: No longer in vLLM 0.27.0
- **`granite-20b-fc` tool parser**: No longer in vLLM 0.27.0

### Changed
- **Dependency Range**: vLLM updated to `>=0.20.0,<0.28.0`
- **Version**: Major bump from 0.2.9.x to 0.3.0.0 (vLLM major version 0.26 → 0.27)
- **`--speculative-config`**: Fixed incorrect deprecation status (was marked as removed in v0.22.0, but is still active)
- **`--spec-method`**: Updated choices to include `ngram_gpu`, `eagle3`, `mtp`, `dflash`, `dspark`
- **`--spec-tokens`**: Added missing parameter

### Notes
- PyTorch 2.13.0 is a breaking environment change; ensure compatible CUDA/toolchain
- Transformers 5.14.1 is now the supported backend
- FlashInfer 0.6.16.post3 required for latest kernel optimizations
- Kimi K3 requires `--trust-remote-code` for initial loading
- Speculative decoding: `--spec-method`/`--spec-model`/`--spec-tokens` are convenience flags that populate `--speculative-config`; they are mutually exclusive for the same keys

## [v0.2.9.9] - 2026-07-27

### Added
- **vLLM 0.26.0 Full Support**: Updated to support vLLM v0.26.0 (411 commits from 212 contributors)
- **Inkling Model Family**: Full support including tool call parser, reasoning parser, speculative decoding, LoRA, and NVFP4 quantization
- **13 New Tool Call Parsers**: `inkling`, `apertus`, `ernie45`, `functiongemma`, `gigachat3`, `granite4`, `hermes`, `hunyuan_a13b`, `lfm2`, `olmo3`, `poolside_v1`, `pythonic`, `xlam`
- **5 New Reasoning Parsers**: `inkling`, `ernie45`, `hunyuan_a13b`, `olmo3`, `poolside_v1`
- **DeepSeek-V4 Performance**: Specialized routing kernel (2.94% TPOT), fused_topk_bias (1.5-2x), redundant repeat/copy removal (1.8% TPOT)
- **Decode Context Parallel (DCP)**: Hybrid attention support, DCP + Eagle for Tokenspeed MLA
- **PD Disaggregation**: NIXL pipeline-parallel prefill in push mode
- **Expanded Quantization**: Humming w[2-7]a[4,8], NVFP4/MXFP4 (nvfp4_per_token online MoE), INT2 XPU weight-only, CuTe-DSL FlashInfer MXFP4
- **Transformers 5.13.0**: Olmo/Olmo2, MistralLarge3 (AutoWeightsLoader), HunyuanVL native processor migration
- **Rust Frontend**: Multimodal video and audio support
- **OpenAI Compatibility**: `bad_words` in `/v1/completions`, `logprob_token_ids`, `include_reasoning` for non-Harmony models
- **Endpoint Plugins Framework**: Extensible endpoint architecture
- **Deepstream Video Decoding**: Hardware-accelerated video backend
- **Human-Readable CLI Args**: Integer formatting for more arguments
- **Grammar Compilation Safety**: Handle grammar compilation failures without crashing the engine
- **Schema v2.3**: Argument schema updated for v0.26.0 sync
- `cli_args_sync.py`: Added v0.26.0 to SUPPORTED_VLLM_VERSIONS
- `parser_sync.py`: Added 13 tool parsers and 5 reasoning parsers with name mappings

### Removed
- **TeleChat model**: Removed upstream in vLLM v0.26.0
- **Persimmon model**: Removed upstream in vLLM v0.26.0
- **Fuyu model**: Removed upstream in vLLM v0.26.0

### Changed
- **Dependency Range**: vLLM updated to `>=0.20.0,<0.27.0`

### Notes
- Transformers 5.13.0 is now the supported backend
- DCP and PD disaggregation enable new large-scale serving topologies
- Inkling model family has full support stack (modeling, CUDA graphs, FA4, LoRA, NVFP4)

## [v0.2.9.8] - 2026-07-13

### Added
- **vLLM 0.25.0 Full Support**: Updated to support vLLM v0.25.0
- **14 New CLI Arguments**: Schema expanded to 291 arguments
- **Backend Selection**: `--backend` (mp, ray, debug engine backend)
- **Cache**: `--cache-dtype` (KV cache data type)
- **Monitoring**: `--enable-per-request-metrics`, `--jit-monitor-mode`, `--ray-workers-use-nsight`
- **Quantization**: `--enable-stochastic-rounding`, `--quantization-config`, `--stochastic-rounding-philoX-rounds`
- **Multimodal**: `--mm-ipc-gpu-memory-gb`
- **Scheduling**: `--policy`, `--ubatch-size`
- **LoRA**: `--target-modules`
- **Debug**: `--model-class-overrides`, `--method`
- **New Tool Call Parsers**: `cohere_command`, `deepseek_v3`, `minimax_m3`, `rust`
- **New Reasoning Parsers**: `cohere_command`, `identity`, `minimax_m3`
- **Schema v2.2**: Argument schema updated to version 2.2.0 with v0.25.0 sync
- **Dependency Doctor**: On-demand dependency health checker for detecting missing system libraries (FFmpeg, torchcodec)
  - `vllm-cli doctor` command with severity levels and fix commands
  - `get_system_dependency_issues()` for programmatic dependency checking
  - Detects torchcodec/FFmpeg issues common with vLLM 0.25.0+

### Deprecated
- `--aggregate-engine-logging` (removed in vLLM v0.25.0)
- `--data-parallel-multi-port-external-lb` (use data_parallel_external_lb with hybrid_lb)
- `--default-max-num-seqs` (sequence limits managed dynamically)
- `--fail-on-environ-validation` (env validation always enforced)
- `--limit-mm-per-prompt` (multimodal limits handled internally)
- `--mamba-cache-philox-rounds` (use stochastic_rounding_philoX_rounds)
- `--shutdown-timeout` (managed internally)
- `--sliding-window` (managed by attention_config)

### Notes
- **PagedAttention Removed**: Legacy attention backend fully deleted in vLLM v0.25.0
- **Model Runner V2 Default**: Standard execution path for all dense models
- **Transformers Backend Parity**: Now matches native vLLM performance

## [v0.2.9.7] - 2026-06-30

### Added
- **vLLM 0.24.0 Full Support**: Updated to support vLLM v0.24.0
- **60 New CLI Arguments**: Schema expanded from 216 to 276 arguments
- **Device Selection (BREAKING)**: `--device-ids` replaces `CUDA_VISIBLE_DEVICES` — vLLM no longer sets CUDA_VISIBLE_DEVICES internally
- **Streaming Parser Engine**: Unified tool-call/reasoning parsing with new parsers: `qwen3`, `minimax_m2`, `glm47`, `glm51`, `glm52`, `nemotron_v3`
- **Data Parallel Enhancements**: `--data-parallel-backend`, `--data-parallel-external-lb`, `--data-parallel-hybrid-lb`, `--data-parallel-multi-port-external-lb`, `--data-parallel-rank`, `--disable-nccl-for-dp-synchronization`
- **DBO Token Thresholds**: `--dbo-decode-token-threshold`, `--dbo-prefill-token-threshold`
- **LoRA Enhancements**: `--lora-dtype`, `--lora-target-modules`, `--enable-mixed-moe-lora-format`, `--enable-tower-connector-lora`, `--specialize-active-lora`, `--fully-sharded-loras`
- **Multimodal Encoder**: `--mm-encoder-attn-backend`, `--mm-encoder-attn-dtype`, `--mm-encoder-fp8-scale-path`, `--mm-encoder-only`, `--mm-processor-cache-gb`, `--mm-tensor-ipc`, `--interleave-mm-strings`, `--limit-mm-per-prompt`, `--skip-mm-profiling`, `--video-pruning-rate`, `--enable-mm-embeds`
- **Model Loading**: `--safetensors-load-strategy`, `--safetensors-prefetch-block-size`, `--safetensors-prefetch-num-threads`, `--pt-load-map-location`, `--model-weights`
- **Diffusion Models**: `--diffusion-config` for DiffusionGemma support
- **KV Cache**: `--kv-sharing-fast-prefill`, `--num-gpu-blocks-override`, `--cp-kv-cache-interleave-size`, `--dcp-kv-cache-interleave-size`
- **NUMA Binding**: `--numa-bind-cpus`, `--numa-bind-nodes`
- **Scheduler**: `--prefill-schedule-interval`, `--sliding-window`, `--watermark`
- **Monitoring**: `--aggregate-engine-logging`, `--collect-detailed-traces`, `--jit-monitor-verbose`
- **Quantization**: `--allow-deprecated-quantization`
- **Misc**: `--enable-cumem-allocator`, `--fail-on-environ-validation`, `--mamba-cache-philox-rounds`, `--show-hidden-metrics-for-version`, `--shutdown-timeout`, `--tokenizer-revision`, `--use-fp64-gumbel`, `--worker-extension-cls`, `--default-max-num-seqs`, `--default-max-num-batched-tokens`, `--distributed-timeout-seconds`, `--cpu-distributed-timeout-seconds`
- **New Profiles**: `diffusion_gemma`, `deepseek_v4`
- **Schema v2.1**: Argument schema updated to version 2.1.0 with v0.24.0 sync

### Changed
- **Dependency Range**: vLLM updated to `>=0.20.0,<0.25.0`
- **13 CLI Arguments Deprecated**: `--swap-space`, `--cpu-offload-space`, `--num-lookahead-slots`, `--num-speculative-tokens`, `--speculative-model`, `--rope-scaling`, `--rope-theta`, `--guided-decoding-backend`, `--generation-config-override`, `--limit-per-prompt`, `--tq-max-kv-splits-for-cuda-graph`, `--disable-async-output-proc`, `--disable-frontend-multiprocessing`
- **reasoning_parser**: Added `glm47`, `glm51`, `glm52`, `nemotron_v3` streaming parsers
- **tool_call_parser**: Added `qwen3`, `minimax_m2`, `glm47`, `glm51`, `glm52`, `nemotron_v3` streaming parsers

### Notes
- vLLM 0.24.0 introduces Rust frontend as the main API server
- Model Runner V2 (MRv2) supports quantized models by default
- DeepEP v2 integrated for expert parallelism
- DiffusionGemma and other diffusion LLMs supported
- `--device-ids` is the new way to select GPUs (replaces CUDA_VISIBLE_DEVICES)

## [v0.2.9.6] - 2026-06-06

### Added
- **vLLM 0.22.0 / 0.22.1 Full Support**: Updated to support vLLM v0.22.0 and v0.22.1
- **52 New CLI Arguments**: Schema expanded from 161 to 213 arguments
- **Linear Backend Selection**: `--linear-backend` (auto, flashinfer, cutlass, triton, native)
- **Speculative Decoding**: `--spec-method` (ngram, eagle, medusa, none) and `--spec-model`
- **HF Token Support**: `--hf-token` for private model authentication
- **Data Parallel Supervisor**: `--data-parallel-supervisor-port`, `--dp-supervisor-probe-*` args
- **Distributed Training**: `--master-addr`, `--master-port`, `--nnodes`, `--node-rank`
- **FlashAttention Late Interaction**: `--enable-flash-late-interaction`
- **Model Loading**: `--load-format`, `--download-dir`, `--convert`, `--code-revision`, `--ignore-patterns`
- **API Enhancements**: `--allow-credentials`, `--disable-fastapi-docs`, `--enable-request-id-headers`, `--root-path`, `--middleware`, `--tool-server`, `--trust-request-chat-template`, `--uds`
- **SSL/TLS**: `--ssl-ca-certs`, `--ssl-cert-reqs`, `--ssl-ciphers`, `--enable-ssl-refresh`
- **Monitoring**: `--enable-server-load-tracking`, `--log-error-stack`, `--use-tqdm-on-load`
- **CLI Args Sync Tool**: Enhanced with version-specific sync (`--tag`), dataclass field extraction, and kwargs unpacking detection
- **Schema v2.0**: Argument schema updated to version 2.0.0 with deprecation tracking

### Changed
- **103 CLI Arguments Deprecated**: Parameters moved from CLI to environment variables/config in vLLM 0.22
- **Dependency Range**: vLLM updated to `>=0.20.0,<0.23.0`
- **Schema Version**: Updated from 1.0.1 to 2.0.0

### Deprecated
- **103 v0.20/v0.21 CLI args**: `tensor_parallel_size`, `gpu_memory_utilization`, `swap_space`, `max_num_seqs`, `enable_lora`, `enable_prefix_caching`, `kv_offloading_backend`, `kv_offloading_size`, `enable_chunked_prefill`, and 93 more (still available via env vars/config)

### Notes
- vLLM 0.22 restructured CLI arguments to use dataclass-based `EngineArgs` with kwargs unpacking
- Deprecated parameters remain in schema for backward compatibility
- Use `python -m vllm_cli.config.cli_args_sync --tag v0.22.1 --verbose` to check for updates

## [v0.2.9.5] - 2026-04-29

### Added
- **vLLM 0.20.0 Full Support**: Updated minimum vLLM dependency to `>=0.20.0`
- **TurboQuant 2-bit KV Cache**: Added `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc` KV cache options
- **New KV Cache Types**: `fp8_ds_mla`, `int8_per_token_head`, `fp8_per_token_head`, `nvfp4`
- **KV Cache Offloading**: `--kv-offloading-size` and `--kv-offloading-backend` (native/lmcache)
- **KV Cache Memory Control**: `--kv-cache-memory-bytes` for fine-grain memory management
- **KV Cache Skip Layers**: `--kv-cache-dtype-skip-layers` for layer-level quantization control
- **Prefix Caching Hash Algorithm**: `--prefix-caching-hash-algo` (sha256, sha256_cbor, xxhash, xxhash_cbor)
- **Decode/Prefill Context Parallelism**: `--decode-context-parallel-size` and `--prefill-context-parallel-size`
- **Expert Parallel Load Balancing**: `--enable-eplb` and `--eplb-config`
- **Dynamic Batch Optimization**: `--enable-dbo`
- **Elastic Expert Parallelism**: `--enable-elastic-ep`
- **NUMA Binding**: `--numa-bind` for GPU workers
- **Attention Backend Selection**: `--attention-backend` with FlashAttention 4, Triton, ROCm backends
- **TurboQuant CUDA Graph**: `--tq-max-kv-splits-for-cuda-graph`
- **MoE Backend Selection**: `--moe-backend` (auto, triton, cutlass, trtllm, aiter)
- **Optimization/Performance Modes**: `--optimization-level` (O0-O3) and `--performance-mode` (latency/throughput)
- **Mamba Config**: `--mamba-backend`, `--mamba-cache-mode`, `--mamba-cache-dtype`, `--mamba-ssm-cache-dtype`, `--mamba-block-size`
- **New Reasoning Parsers**: `hyv3` (Hunyuan v3), `mimo`
- **New Tool Call Parsers**: `hyv3`, `mimo`
- **Humming Quantization**: Added `humming` to quantization choices
- **Model Implementation**: `--model-impl` (transformers/vllm)
- **GDN Prefill Backend**: `--gdn-prefill-backend` (flashinfer/triton)
- **Enhanced Monitoring**: `--kv-cache-metrics`, `--cudagraph-metrics`, `--enable-mfu-metrics`, `--enable-layerwise-nvtx-tracing`
- **Scheduling Policy**: `--scheduling-policy` (fcfs/utilization)
- **Offload Backend**: `--offload-backend` (uva/prefetch) with granular offload controls
- **vLLM IR**: `--ir-op-priority` for operator priority configuration
- **FlashInfer Autotune**: `--enable-flashinfer-autotune`
- **78 New CLI Arguments** total, bringing schema to 161 arguments

### Changed
- **PyTorch 2.11 + CUDA 13.0**: vLLM now defaults to PyTorch 2.11 with CUDA 13.0
- **Transformers v5**: Updated to support HuggingFace transformers>=5
- **Python 3.14**: Added to supported Python versions
- **CUDAGraph Memory Profiling**: Now enabled by default
- **Async Scheduling**: Default OFF for pooling models

### Deprecated
- **Petit NVFP4**: Removed from vLLM 0.20
- **LLM.reward**: Deprecated, use `LLM.encode` instead

## [v0.2.9.3] - 2026-03-23

### Added
- **ModelOpt Quantization Support**: Added support for NVIDIA ModelOpt quantization methods
  - `modelopt_fp4`: NVIDIA ModelOpt NVFP4 quantization for maximum memory efficiency
  - `modelopt_mxfp8`: NVIDIA ModelOpt MXFP8 mixed-precision quantization for balanced performance
- **New Optional Dependency**: `nvidia-modelopt>=0.5.0` - Install with `pip install vllm-cli[modelopt]`

### Fixed
- **i18n Submenu Translation Fix**: Fixed issue where Chinese menu text would revert to English when entering submenus
  - All submenu screens now properly display translated text when a language is selected
  - Updated `settings.py`, `profiles.py`, and `shortcuts.py` to accept and use i18n_manager

### Technical Details
- Updated quantization choices in `argument_schema.json`, `validation/schema.py`, `cli/parser.py`
- Added quantization suggestions in `errors/recovery.py` and `ui/system_info.py`
- Added fallback list entries in `system/dependencies.py`
- Added i18n translations for new quantization methods in English and Chinese

### Updated Files (10 + 3 i18n fix)
- `schemas/argument_schema.json`
- `validation/schema.py`
- `cli/parser.py`
- `ui/custom_config.py`
- `errors/recovery.py`
- `ui/system_info.py`
- `system/dependencies.py`
- `pyproject.toml` (added modelopt optional dependency)
- `i18n/translations/en.json`
- `i18n/translations/zh.json`
- `ui/settings.py` (i18n fix)
- `ui/profiles.py` (i18n fix)
- `ui/shortcuts.py` (i18n fix)

## [v0.2.9.1] - 2026-03-07

### Added
- **CLI Argument Synchronization**: Automatic sync of vLLM CLI arguments from GitHub source using direct HTTP fetch
- **New CLI Commands**:
  - `vllm-cli recipes --sync-args` - Preview pending CLI argument changes
  - `vllm-cli recipes --sync-args --apply-args` - Apply changes to local schema
- **Interactive UI**: "Sync CLI Args" entry in Profile Management menu

### Enhanced
- **Argument Schema**: 91+ CLI arguments with comprehensive metadata
- **Change Tracking**: Automatic identification of new and deprecated arguments
- **Backward Compatibility**: Deprecated arguments preserved when detected

### New Modules
- `src/vllm_cli/config/cli_args_sync.py` - Core sync implementation

### Technical Details
- Uses `urllib.request` for GitHub source fetching (no external dependencies)
- Supports fallback to `subprocess` for curl if available
- Regex patterns for argument extraction from Python source code
- Automatic argument name normalization (--arg-name → arg_name)
- Dry-run mode for preview before applying changes

### Usage Examples
```bash
# Preview pending changes (dry-run)
vllm-cli recipes --sync-args

# Apply changes to local schema
vllm-cli recipes --sync-args --apply-args

# Via UI menu
# Profile Management → Sync CLI Args

# Test with Python API
from src.vllm_cli.config.cli_args_sync import sync_cli_args
result = sync_cli_args(dry_run=True, verbose=True)
```

### Test Results
- **Remote args**: 91 unique arguments fetched successfully
- **Local args**: 77 existing arguments in schema
- **New arguments**: 56 detected
- **Removed/deprecated**: 42 detected
- **Success rate**: 100% fetch success
- `src/vllm_cli/config/cli_args_sync.py` - Core sync implementation
- Supports: `urllib.request` and `subprocess` for cross-platform compatibility
- No additional dependencies required

### Profile System
- Added "Sync CLI Args" entry in Profile Management menu
- "Sync Recipes" entry remains in Profile Management menu

### Documentation
- Updated `ENVIRONMENT_VARIABLES.md` with AMD ROCm section
- Added model-specific recommendations
- Added quick reference by GPU architecture
- Created `RELEASE_NOTES_v0.2.9.md`

### Technical Details
- Profile structure unchanged (backward compatible)
- Environment variables automatically applied by profiles
- No breaking changes to existing functionality

## [v0.2.8] - 2025-01-15

### Enhanced
- **Chat Template Configuration**: Significantly improved `--chat-template` parameter configuration
  - **Elevated Importance**: Changed from `low` to `medium` importance for better visibility
  - **Enhanced Description**: Added clear explanation that it's for `.jinja` template files
  - **Helpful Hints**: Added guidance about when to use (Qwen3, Llama3.1, etc.) with example paths
  - **Interactive File Browser**: New file browsing feature for easier template selection
    - Automatically searches for `.jinja` files in common locations (./examples/, cwd, ~/.cache/huggingface/)
    - Shows up to 20 found templates for selection
    - Displays relative paths for better readability
  - **File Validation**: Checks if specified file exists and warns if not found
  - **Three Input Methods**: Browse files, enter path manually, or keep current value
  - **Better User Experience**: Clear guidance about chat templates and their purpose

### Technical Details
- Updated `argument_schema.json`: Enhanced `chat_template` parameter definition
- Updated `custom_config.py`: Added `select_chat_template_file()` logic in string type handling
- File discovery uses `Path.rglob()` for recursive search with depth limiting
- Supports both absolute and relative paths with smart path resolution

### Use Cases
- Required for tool calling with models like Qwen3, Llama3.1
- Enables proper function calling support
- Custom conversation formatting
- Reference: https://docs.vllm.ai/en/latest/features/tool_calling/?h=qwen3#required-function-calling

## [v0.2.7] - 2025-01-15

### Fixed
- **enable_auto_tool_choice Configuration Bug**: Fixed issue where `enable_auto_tool_choice` could not be configured when creating custom profiles
  - Problem: Dependency check was too strict - it skipped the option even when `tool_call_parser` was explicitly set to `null`
  - Solution: Changed dependency logic to only skip if the dependency key is completely absent from config
  - Impact: Users can now properly configure `enable_auto_tool_choice` in custom configurations and profiles
  - Related parameters: `tool_call_parser`, `enable_auto_tool_choice`

### Technical Details
- Updated `configure_argument()` in `src/vllm_cli/ui/custom_config.py`
- Changed from `if not config.get(dependency)` to `if dependency not in config`
- This allows options with `depends_on` to be shown when dependency is set to any value (including `None`/`null`)

## [v3.0.0] - 2025-01-XX

### Added
- **🌍 Multi-Language Support (i18n)**
  - Complete internationalization system with English and Chinese support
  - First-launch language selection interface
  - Runtime language switching from Settings menu
  - All UI text translated (menus, messages, prompts, errors)
  - Configuration parameter descriptions and hints in both languages
  - Translation caching for optimal performance
  
- **New Components**
  - `I18nManager`: Core internationalization manager
    - Translation loading and caching
    - Language switching with persistence
    - Parameter interpolation support
  - `LanguageSelector`: Beautiful language selection UI
    - Bilingual prompts
    - Persistent preference storage
  - Translation files: `en.json` and `zh.json`
    - Structured, maintainable format
    - Complete coverage of all UI text

- **Configuration Enhancements**
  - New `language` field in config.yaml
  - New `language_set` field to track user preference
  - Automatic config migration from v2.x

### Changed
- **Updated UI Modules**
  - `__main__.py`: Integrated language selection flow
  - `menu.py`: Full i18n support with dynamic menu translation
  - `settings.py`: Added Language settings option
  - `welcome.py`: Translated welcome screen
  - `ConfigManager`: Added language preference management methods

### Documentation
- Added `docs/v3.0_i18n_guide.md`: Complete i18n user and developer guide
- Added `RELEASE_NOTES_v3.0.0.md`: Detailed release notes

### Notes
- **Backward Compatibility**: Fully compatible with v2.x configurations
- **No Breaking Changes**: All existing features work as before
- **Automatic Migration**: First launch prompts for language selection

## [v0.2.6] - 2025-11-13

### Added
- **GPT-OSS-20B Profiles**: Two new profiles optimized for OpenAI's GPT-OSS-20B model
  - `gpt_oss_20b`: Full performance configuration (~25-30GB VRAM)
    - Pre-configured with `tool_call_parser: openai` and `enable_auto_tool_choice: true`
    - Includes OpenAI content format for proper tool calling support
    - Includes `VLLM_USE_FLASHINFER_MXFP4_MOE` environment variable to reduce GPU load
  - `gpt_oss_20b_low_memory`: Memory-optimized configuration (~13-15GB VRAM)
    - FP8 KV cache, reduced context length (4K), CPU offloading
    - Suitable for consumer GPUs (RTX 3090, 4090) and multi-model deployments
    - Maintains good performance with 50%+ memory savings
- **Tool Call Parser Support**: Added `--tool-call-parser` parameter for function calling support
  - Added `openai` parser (recommended for GPT-OSS models)
  - Support for 12+ parsers: openai, hermes, mistral, internlm, llama3_json, granite, granite-20b-fc, jamba, deepseek_v3, deepseek_v31, glm45, hunyuan_a13b, kimi_k2
- **MoE Environment Variable**: Added `VLLM_USE_FLASHINFER_MXFP4_MOE` to environment variable presets
  - Reduces GPU load for MoE models
  - Improves efficiency and thermal management
- **Comprehensive Documentation**:
  - `MEMORY_OPTIMIZATION.md`: Complete guide for optimizing VRAM usage
  - `GPT_OSS_20B_PROFILES_COMPARISON.md`: Detailed comparison of GPT-OSS profiles
  - `ENVIRONMENT_VARIABLES.md`: Reference guide for all vLLM environment variables
  - `TROUBLESHOOTING_TOOL_PARSER.md`: Tool parser troubleshooting guide
  - `test_tool_parsers.py`: Automated testing script for tool calling functionality

### Fixed
- **GPT-OSS Tool Choice Configuration**: Resolved `--enable-auto-tool-choice` errors
  - Fixed TypeError: "Error: --enable-auto-tool-choice requires --tool-call-parser"
  - Corrected `tool_call_parser` to use `openai` parser per official vLLM documentation
  - Fixed `chat_template_content_format` to use correct values: auto, openai, string
- **Tool Parser Compatibility**: Added troubleshooting guide for parser API incompatibilities

### Changed
- Updated all GPT-OSS profiles to use `openai` parser instead of `hermes` (per official documentation)
- Improved tool parser selection hints and documentation

## [v0.2.5] - 2025-08-25

### Added
- **Multi-Model Proxy Server (Experimental)**: Enabling multiple LLMs through a single unified API endpoint
  - Single OpenAI-compatible endpoint for all models
  - Request routing based on model name
  - Save and reuse proxy configurations
- **Dynamic Model Management**: Add or remove models at runtime without restarting the proxy
  - Live model registration and unregistration
  - Pre-registration with verification lifecycle
  - Graceful handling of model failures without affecting other models
  - Model state tracking (pending, running, sleeping, stopped)
- **Model Sleep/Wake for GPU Memory Management**: Efficient GPU resource distribution
  - Sleep Level 1: CPU offload for faster wake-up
  - Sleep Level 2: Full memory discard for maximum savings
  - Real-time memory usage tracking and reporting
  - Models maintain their ports while sleeping
- **Test Coverage**: Added comprehensive tests for multi-model proxy and model registry

### Changed
- Improved error handling with detailed logs when PyTorch is not installed
- Better server cleanup and process management

### Fixed
- UI navigation improvements and minor display fixes

## [v0.2.4] - 2025-08-20

### Added
- **Hardware-Optimized Profiles for GPT-OSS Models**: New built-in profiles optimized for different GPU architectures
  - `gpt_oss_ampere`: Optimized for NVIDIA A100 GPUs
  - `gpt_oss_hopper`: Optimized for NVIDIA H100/H200 GPUs
  - `gpt_oss_blackwell`: Optimized for NVIDIA Blackwell (B100/B200) GPUs
  - Based on official [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- **Shortcuts System**: Save and quickly launch model + profile combinations
  - Quick launch from CLI: `vllm-cli serve --shortcut NAME`
  - Manage shortcuts through interactive mode or CLI commands
  - Import/export shortcuts for sharing configurations
- **Ollama Model Support**: Full integration with Ollama-downloaded models
  - Automatic discovery in user (`~/.ollama`) and system (`/usr/share/ollama`) directories
  - GGUF format detection and experimental serving support
- **Environment Variable Management**: Two-tier system for complete control
  - Universal environment variables for all servers
  - Profile-specific environment variables (override universal)
  - Clear indication of environment sources when launching
- **GPU Selection**: Select specific GPUs for model serving
  - CLI: `--device 0,1` to use specific GPUs
  - Interactive UI for GPU selection in advanced settings
  - Automatic tensor_parallel_size adjustment
- **Enhanced System Information**: vLLM built-in feature detection
  - Detailed attention backend availability (Flash Attention 2/3, xFormers)
  - Feature compatibility checking per backend
- **Server Cleanup Control**: Configure server behavior on CLI exit
- **Extended vLLM Arguments**: Added 16+ new arguments for v1 engine
  - Performance, optimization, API, configuration, and monitoring options

### Changed
- Enhanced Quick Serve menu shows last configuration and saved shortcuts
- Model field excluded from profiles for model-agnostic templates
- Model cache refresh properly respects TTL settings (>60s)
- Environment variables available in Custom Configuration menu

### Fixed
- Fixed manual cache refresh functionality
- Fixed profile creation inconsistency between menus
- Fixed UI consistency issues with prompt formatting


## [v0.2.3] - 2025-08-17

### Fixed
- **Critical**: Fixed missing built-in profiles when installing from PyPI - JSON schema files are now properly included in the package distribution

## [v0.2.2] - 2025-08-17

### Added
- **Model Manifest Support**: Introduced `models_manifest.json` for mapping custom models in vLLM CLI native way (see [custom-model-serving.md](docs/custom-model-serving.md) for more details)
- **Documentation**: Added [custom-model-serving.md](docs/custom-model-serving.md) for custom model serving guide

### Fixed
- Serving models from custom directories now works as expected
- Fixed some UI issues


## [0.2.1] - 2025-08-17

### Fixed
- **Critical**: Fixed package installation issue - setuptools now correctly includes all sub-packages

## [0.2.0] - 2025-08-17

### Added
- **LoRA Adapter Support**: Serve models with LoRA adapters - select base model and multiple LoRA adapters for serving
- **Enhanced Model List Display**: Comprehensive model listing showing HuggingFace models, LoRA adapters, and datasets with size information
- **Model Directory Management**: Configure and manage custom model directories for automatic model discovery
- **Model Caching**: Performance optimization through intelligent caching with TTL for model listings
- **Improved Model Discovery**: Integration with hf-model-tool for comprehensive model detection with fallback mechanisms
- **HuggingFace Token Support**: Authentication support for accessing gated models with automatic token validation
- **Profile Management Enhancements**:
  - View/Edit profiles in unified interface with detailed configuration display
  - Direct editing of built-in profiles with user overrides
  - Reset customized built-in profiles to defaults

### Changed
- Refactored model management system with new `models/` package structure
- Enhanced error handling with comprehensive error recovery strategies
- Improved configuration validation framework with type checking and schemas
- Updated low_memory profile to use FP8 quantization instead of bitsandbytes

### Fixed
- Better handling of model metadata extraction
- Improved error messages for better user experience

## [0.1.1] - 2025-08-15

### Added
- Display complete log viewer when server startup fails
- Enhanced error handling and recovery options

### Fixed
- Small UI fixes for better terminal display
- Improved error messages clarity

## [0.1.0] - 2025-08-14

### Added
- **Interactive Mode**: Rich terminal interface with menu-driven navigation
- **Command-Line Mode**: Direct CLI commands for automation and scripting
- **Model Management**: Automatic discovery and management of local models
- **Remote Model Support**: Serve models directly from HuggingFace Hub without pre-downloading
- **Configuration Profiles**: Pre-configured server profiles (standard, moe_optimized, high_throughput, low_memory)
- **Custom Profiles**: User-defined configuration profiles support
- **Server Monitoring**: Real-time monitoring of active vLLM servers with GPU utilization
- **System Information**: GPU, memory, and CUDA compatibility checking
- **Quick Serve**: Auto-reuse last successful configuration
- **Process Management**: Global server registry with automatic cleanup on exit
- **Schema-Driven Configuration**: JSON schemas for validation of vLLM arguments
- **ASCII Fallback**: Environment detection for terminal compatibility

### Dependencies
- vLLM
- PyTorch with CUDA support
- hf-model-tool for model discovery
- Rich for terminal UI
- Inquirer for interactive prompts
- psutil for system monitoring
- PyYAML for configuration parsing

[0.2.2]: https://github.com/Chen-zexi/vllm-cli/compare/0.2.1...v0.2.2
[0.2.1]: https://github.com/Chen-zexi/vllm-cli/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/Chen-zexi/vllm-cli/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/Chen-zexi/vllm-cli/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/Chen-zexi/vllm-cli/releases/tag/0.1.0
