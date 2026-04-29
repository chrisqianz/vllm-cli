# v0.2.9.5 Release Notes

**Release Date**: 2026-04-29

## What's New

### vLLM 0.20.0 Full Support

This release updates vLLM CLI to fully support **vLLM v0.20.0**, a major release featuring 752 commits from 320 contributors. The CLI argument schema has been expanded from 83 to **161 arguments**, covering all new features in vLLM 0.20.

#### TurboQuant 2-bit KV Cache Compression

The headline feature of vLLM 0.20 is TurboQuant — a new attention backend delivering 2-bit KV cache compression with **4× capacity**:

- **`turboquant_k8v4`**: K8V4 compression for maximum KV cache capacity
- **`turboquant_4bit_nc`**: 4-bit non-compressed KV cache
- **`turboquant_k3v4_nc`**: K3V4 compression variant
- **`turboquant_3bit_nc`**: 3-bit KV cache for extreme memory savings
- **Auto-skip boundary layers**: First/last 2 layers automatically skipped for protection
- **FA3/FA4 prefill support**: Full FlashAttention 3/4 compatibility

New KV cache types added:
- `fp8_ds_mla` — FP8 for DeepSeek MLA models
- `int8_per_token_head` — Per-token-head INT8 quantization
- `fp8_per_token_head` — Per-token-head FP8 quantization
- `nvfp4` — NVIDIA FP4 KV cache

#### New CLI Arguments (78 additions)

**KV Cache & Memory:**
- `--kv-cache-memory-bytes` — Fine-grain KV cache memory per GPU
- `--kv-offloading-size` — CPU offloading buffer size in GiB
- `--kv-offloading-backend` — `native` or `lmcache`
- `--kv-cache-dtype-skip-layers` — Layer-level quantization skip patterns
- `--prefix-caching-hash-algo` — `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`
- `--tq-max-kv-splits-for-cuda-graph` — TurboQuant CUDA graph split count

**Parallelism:**
- `--decode-context-parallel-size` — Decode context parallelism
- `--prefill-context-parallel-size` — Prefill context parallelism
- `--enable-eplb` / `--eplb-config` — Expert parallel load balancing
- `--enable-dbo` — Dynamic batch optimization
- `--enable-elastic-ep` — Elastic expert parallelism
- `--numa-bind` — NUMA binding for GPU workers
- `--all2all-backend` — All-to-all communication backend
- `--enable-ep-weight-filter` — Expert weight filter
- `--expert-placement-strategy` — Expert placement strategy

**Performance:**
- `--optimization-level` — O0 through O3 optimization levels
- `--performance-mode` — `latency` or `throughput` mode
- `--scheduling-policy` — `fcfs` or `utilization`
- `--moe-backend` — `auto`, `triton`, `cutlass`, `trtllm`, `aiter`
- `--attention-backend` — `flash_attn`, `triton_attn`, `rocm_attn`, etc.

**Mamba Support:**
- `--mamba-backend` — `triton` or `flashinfer`
- `--mamba-cache-mode` — `none`, `all`, `align`
- `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype`
- `--mamba-block-size` — Cache block size (multiple of 8)
- `--enable-mamba-cache-stochastic-rounding`

**Offload:**
- `--offload-backend` — `uva` or `prefetch`
- `--cpu-offload-params`, `--offload-group-size`, `--offload-num-in-group`
- `--offload-prefetch-step`, `--offload-params`

**Monitoring:**
- `--kv-cache-metrics` / `--kv-cache-metrics-sample`
- `--cudagraph-metrics`
- `--enable-mfu-metrics` — Model FLOPs Utilization metrics
- `--enable-layerwise-nvtx-tracing` — Nsight Systems profiling
- `--enable-logging-iteration-details`
- `--enable-mm-processor-stats`

**Advanced:**
- `--model-impl` — `transformers` or `vllm` native implementation
- `--logprobs-mode` — `normal` or `fast`
- `--gdn-prefill-backend` — `flashinfer` or `triton`
- `--ir-op-priority` — vLLM IR operator priority
- `--enable-flashinfer-autotune`
- `--skip-tokenizer-init` — For embedding-only models
- `--enable-prompt-embeds` — Pre-computed embeddings input
- `--override-attention-dtype`
- `--disable-cascade-attn`
- `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size`
- `--stream-interval`
- `--disable-hybrid-kv-cache-manager`
- `--scheduler-reserve-full-isl`
- `--max-long-partial-prefills`
- `--disable-chunked-mm-input`
- `--scheduler-cls`
- `--allowed-local-media-path` / `--allowed-media-domains`
- `--config-format`
- `--pooler-config`
- `--logits-processors`
- `--io-processor-plugin`
- `--renderer-num-workers`
- `--hf-config-path`
- `--kv-transfer-config` / `--kv-events-config` / `--ec-transfer-config`
- `--reasoning-config` / `--structured-outputs-config`
- `--attention-config` / `--kernel-config` / `--profiler-config`
- `--weight-transfer-config` / `--additional_config`
- `--enable-return-routed-experts`
- `--disable-frontend-multiprocessing`

#### New Model Parsers

- **Reasoning parsers**: Added `hyv3` (Hunyuan v3) and `mimo`
- **Tool call parsers**: Added `hyv3` and `mimo`

#### New Quantization

- **`humming`**: Added humming quantization kernel support

#### Dependency Updates

- **vLLM**: Minimum version updated to `>=0.20.0`
- **PyTorch 2.11**: vLLM now ships on PyTorch 2.11 (CUDA and XPU)
- **CUDA 13.0**: Default CUDA wheel switched to CUDA 13.0
- **Transformers v5**: Full support for HuggingFace transformers>=5
- **Python 3.14**: Added to supported versions
- **FlashInfer**: Bumped to 0.6.8
- **DeepGEMM**: Integrated into the vLLM wheel via CMake

#### Breaking Changes (from vLLM 0.20)

1. **PyTorch 2.11 + CUDA 13.0(.2) default** — Environment dependency change
2. **Transformers v5** is the supported baseline
3. **Metrics rework**: `vllm:prompt_tokens_recomputed` removed
4. **Pooler config rename**: `logit_bias`/`logit_scale` → `logit_mean`/`logit_sigma`
5. **Async scheduling default OFF** for pooling models
6. **CUDAGraph memory profiling ON by default**
7. **Petit NVFP4 removed**

## Usage

### Using TurboQuant 2-bit KV Cache

```bash
vllm-cli serve YOUR_MODEL --kv-cache-dtype turboquant_k8v4
```

### Using KV Cache Offloading

```bash
vllm-cli serve YOUR_MODEL --kv-offloading-size 16 --kv-offloading-backend native
```

### Using Expert Parallel Load Balancing

```bash
vllm-cli serve YOUR_MODEL --enable-eplb --expert-placement-strategy balanced
```

## Migration Notes

- Minimum vLLM version is now **0.20.0**
- If using CUDA 12.9, install with `--torch-backend=cu129`
- Petit NVFP4 quantization is no longer available
- `LLM.reward` API is deprecated, use `LLM.encode` instead
