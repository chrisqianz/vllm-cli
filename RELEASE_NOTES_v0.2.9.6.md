# v0.2.9.6 Release Notes

**Release Date**: 2026-06-06

## What's New

### vLLM 0.22.0 / 0.22.1 Full Support

This release updates vLLM CLI to fully support **vLLM v0.22.0** and **v0.22.1**, representing a significant evolution in the vLLM ecosystem. The CLI argument schema has been updated to version 2.0 with **213 total arguments** (up from 161).

#### vLLM 0.22.0 Highlights (459 commits from 230 contributors)

**DeepSeek V4 Maturity**
- Dedicated `vllm/models/deepseek_v4/` package with NVFP4 fused MoE support
- Full + piecewise CUDA graph support
- MTP speculative decoding
- Fused kernels: MegaMoE, MHC, Q-norm, indexer, sparse MLA

**Model Runner V2 Advances**
- MRv2 is now default for Qwen3 dense models
- Sleep-mode weight reload, `update_config`, and shared KV-cache layers
- Auto-fallback to MRv1 for unsupported features

**Experimental Rust Frontend**
- New Rust front-end integration with DP Supervisor for data-parallel serving

**Batch Invariance Performance**
- Cutlass FP8 support delivering **28.9% end-to-end latency improvement**
- Compile-mode support on SM80, NVFP4 Cutlass linear path

**Multi-Tier KV Cache Offloading**
- New framework with Python filesystem secondary tier
- DSv4 support and Mooncake disk offloading

#### New CLI Arguments (52 additions)

**API & Frontend:**
- `--allow-credentials` — Allow credentials in CORS requests
- `--disable-access-log-for-endpoints` — Disable access logging for specific endpoints
- `--disable-fastapi-docs` — Disable FastAPI documentation endpoints
- `--disable-uvicorn-access-log` — Disable uvicorn access logging
- `--enable-force-include-usage` — Force include usage statistics
- `--enable-log-deltas` — Enable logging of response deltas
- `--enable-log-outputs` — Enable logging of model outputs
- `--enable-offline-docs` — Enable offline documentation
- `--enable-request-id-headers` — Include request ID in response headers
- `--enable-server-load-tracking` — Enable server load tracking metrics
- `--enable-tokenizer-info-endpoint` — Expose tokenizer information endpoint
- `--exclude-tools-when-tool-choice-none` — Exclude tools when tool_choice is none
- `--fingerprint-value` — Custom fingerprint for request tracking
- `--h11-max-header-count` — Maximum HTTP headers (h11 protocol)
- `--h11-max-incomplete-event-size` — Maximum incomplete event size
- `--middleware` — ASGI middleware for the server
- `--return-tokens-as-token-ids` — Return tokens as IDs instead of strings
- `--root-path` — Root path prefix for API endpoints
- `--ssl-ca-certs` — SSL CA certificates file path
- `--ssl-cert-reqs` — SSL certificate requirements
- `--ssl-ciphers` — SSL ciphers to use
- `--enable-ssl-refresh` — Automatic SSL certificate refresh
- `--tokens-only` — Return only tokens without metadata
- `--tool-server` — URL of the tool server
- `--trust-request-chat-template` — Trust client-provided chat template
- `--uds` — Unix domain socket path
- `--default-chat-template-kwargs` — Default kwargs for chat template

**Model & Loading:**
- `--code-revision` — Revision of model implementation code
- `--convert` — Convert model format on load
- `--default-mm-loras` — Default multimodal LoRA modules
- `--download-dir` — Directory to download and cache models
- `--hf-token` — Hugging Face authentication token
- `--ignore-patterns` — Patterns to ignore when loading model files
- `--load-format` — Format for loading model weights
- `--model-loader-extra-config` — Extra model loading configuration
- `--runner` — Runner mode: llm, worker, or scraper
- `--mamba-config` — Mamba model configuration

**Parallelism & Distribution:**
- `--data-parallel-supervisor-port` — Port for DP supervisor communication
- `--dp-supervisor-probe-failure-threshold` — DP probe failure threshold
- `--dp-supervisor-probe-interval-s` — DP probe interval (seconds)
- `--dp-supervisor-probe-timeout-s` — DP probe timeout (seconds)
- `--master-addr` — Master node address for distributed training
- `--master-port` — Master node port for distributed training
- `--nnodes` — Number of nodes in the cluster
- `--node-rank` — Rank of this node in the cluster
- `--worker-cls` — Custom worker class for distributed execution

**Performance & Speculative Decoding:**
- `--enable-flash-late-interaction` — FlashAttention late interaction
- `--linear-backend` — Backend for linear layer operations
- `--spec-method` — Speculative decoding method
- `--spec-model` — Speculative decoding draft model

**Monitoring:**
- `--log-error-stack` — Include stack traces in error responses
- `--use-tqdm-on-load` — Show progress bar during model loading

#### Arguments Removed from CLI in v0.22 (103 parameters)

vLLM 0.22 restructured its CLI surface, moving many parameters to environment variables and configuration objects. The following arguments are no longer exposed as CLI flags but remain available as `EngineArgs` dataclass fields:

**Parallelism:** `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `decode_context_parallel_size`, `prefill_context_parallel_size`, `all2all_backend`, `dcp_comm_backend`, `distributed_executor_backend`, `enable_expert_parallel`, `enable_elastic_ep`, `expert_placement_strategy`

**Memory & KV Cache:** `gpu_memory_utilization`, `swap_space`, `kv_cache_memory_bytes`, `kv_offloading_size`, `kv_offloading_backend`, `kv_cache_dtype_skip_layers`, `kv_cache_metrics`, `kv_cache_metrics_sample`, `prefix_caching_hash_algo`, `enable_prefix_caching`, `cpu_offload_space`

**Scheduling:** `max_num_seqs`, `max_num_batched_tokens`, `max_num_partial_prefills`, `max_long_partial_prefills`, `enable_chunked_prefill`, `scheduling_policy`, `scheduler_cls`, `scheduler_reserve_full_isl`, `async_scheduling`

**Performance:** `optimization_level`, `cudagraph_capture_sizes`, `max_cudagraph_capture_size`, `cudagraph_metrics`, `enable_flashinfer_autotune`, `override_attention_dtype`, `attention_backend`, `attention_config`, `disable_cascade_attn`, `gdn_prefill_backend`

**LoRA:** `enable_lora`, `max_parallel_loading_workers`

**Speculative Decoding:** `speculative_config`, `speculative_model`, `num_speculative_tokens`, `num_lookahead_slots`

**Offloading:** `offload_backend`, `offload_group_size`, `offload_num_in_group`, `offload_prefetch_step`, `offload_params`, `cpu_offload_params`, `kv_transfer_config`, `ec_transfer_config`, `weight_transfer_config`

**Monitoring:** `enable_log_requests`, `enable_logging_iteration_details`, `enable_layerwise_nvtx_tracing`, `enable_mfu_metrics`, `enable_mm_processor_stats`, `disable_log_stats`, `otlp_traces_endpoint`

**Model:** `enable_prompt_embeds`, `enable_return_routed_experts`, `enable_sleep_mode`, `skip_tokenizer_init`, `served_model_name`, `language_model_only`, `rope_scaling`, `rope_theta`, `trust_remote_code`, `generation_config`, `generation_config_override`, `logits_processors`, `headless`

**Mamba:** `mamba_cache_dtype`, `mamba_cache_mode`, `mamba_ssm_cache_dtype`, `enable_mamba_cache_stochastic_rounding`

**Other:** `allowed_local_media_path`, `allowed_media_domains`, `async_scheduling`, `calculate_kv_scales`, `compilation_config`, `data_parallel_address`, `data_parallel_rpc_port`, `data_parallel_size_local`, `data_parallel_start_rank`, `disable_async_output_proc`, `disable_chunked_mm_input`, `disable_custom_all_reduce`, `disable_frontend_multiprocessing`, `disable_hybrid_kv_cache_manager`, `disable_sliding_window`, `enable_ep_weight_filter`, `enable_mamba_cache_stochastic_rounding`, `io_processor_plugin`, `long_prefill_token_threshold`, `mamba_block_size`, `mm_encoder_tp_mode`, `mm_processor_cache_type`, `mm_processor_kwargs`, `renderer_num_workers`, `stream_interval`, `structured_outputs_config`, `tq_max_kv_splits_for_cuda_graph`, `reasoning_parser`, `reasoning_parser_plugin`

> **Note:** All deprecated parameters remain in the schema for backward compatibility and can be set via environment variables or configuration files.

#### vLLM 0.22.1 Patch Release (8 commits from 6 contributors)

- **Mellum v2**: New model support for JetBrains' Mellum v2 MoE code-generation model
- **DeepSeek-V4**: Fixed CUTLASS `fmin` compatibility issue
- **AMD Zen CPUs**: zentorch-accelerated quantized linear inference
- **Ray DP**: Fixed multi-node data-parallel serving hang
- **Docker**: Fixed NIXL KV-connector wheel installs

### CLI Args Sync Tool Improvements

The CLI args sync tool has been enhanced to support:
- **Version-specific sync**: Fetch arguments from specific vLLM version tags
- **Dataclass field extraction**: Properly handles v0.22+ dataclass-based argument definitions
- **Kwargs unpacking detection**: Extracts arguments from `**kwargs["name"]` patterns

```bash
# Sync against specific version
python -m vllm_cli.config.cli_args_sync --tag v0.22.1 --verbose

# Sync against main branch (default)
python -m vllm_cli.config.cli_args_sync --verbose
```

### Dependency Updates

- **vLLM**: Supported range updated to `>=0.20.0,<0.23.0`
- **Schema version**: Updated to 2.0.0
- **Total arguments**: 213 (161 existing + 52 new)

## Usage

### Using vLLM 0.22 Features

```bash
# Use linear backend selection
vllm-cli serve YOUR_MODEL --linear-backend cutlass

# Enable speculative decoding
vllm-cli serve YOUR_MODEL --spec-method eagle --spec-model /path/to/draft/model

# Use HF token for private models
vllm-cli serve YOUR_MODEL --hf-token YOUR_TOKEN

# Data parallel supervisor
vllm-cli serve YOUR_MODEL --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --master-port 23456
```

### Sync CLI Args Against Specific Version

```bash
# Check changes for v0.22.1
python -m vllm_cli.config.cli_args_sync --tag v0.22.1 --verbose

# Apply changes to local schema
python -m vllm_cli.config.cli_args_sync --tag v0.22.1 --apply --verbose
```

## Migration Notes

- Minimum vLLM version remains **0.20.0**
- vLLM 0.22.0 and 0.22.1 are fully supported
- 103 CLI arguments from v0.20/v0.21 are deprecated in v0.22 (moved to env vars/config)
- Use the sync tool to stay updated with vLLM CLI changes
- Deprecated parameters are still available through environment variables
