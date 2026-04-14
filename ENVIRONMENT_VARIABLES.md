# vLLM Environment Variables Reference

## MoE Model Optimization

These environment variables are particularly important for Mixture of Experts (MoE) models like GPT-OSS.

### VLLM_USE_FLASHINFER_MXFP4_MOE

**Recommended for: GPT-OSS models**

```bash
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
```

**Purpose:** Enables FlashInfer MXFP4 optimization for MoE models, significantly reducing GPU memory usage and computational load.

**Benefits:**
- Reduces GPU utilization from 100% to more manageable levels
- Improves inference efficiency
- Allows running larger models or handling more concurrent requests
- Better thermal management

**When to use:**
- Running GPT-OSS or other MoE models
- GPU is running at full capacity
- Need to reduce power consumption
- Want to improve throughput

### VLLM_USE_FLASHINFER_MOE_MXFP4_BF16

```bash
export VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
```

**Purpose:** Enables MoE optimization with BF16 precision, specifically designed for GPT-OSS models.

**Benefits:**
- Better numerical stability than FP16
- Optimized for modern GPUs (A100, H100, Blackwell)
- Maintains model quality while improving performance

### VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8

```bash
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
```

**Purpose:** Enables MoE optimization with FP8 precision for maximum efficiency.

**Benefits:**
- Maximum memory savings
- Highest throughput
- Best for inference workloads where slight quality trade-off is acceptable

### VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING

```bash
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
```

**Purpose:** Enables fused MoE activation chunking to reduce memory peaks.

**Benefits:**
- Reduces peak memory usage during MoE layer computation
- Allows running larger batch sizes
- Improves memory efficiency

### VLLM_FUSED_MOE_CHUNK_SIZE

```bash
export VLLM_FUSED_MOE_CHUNK_SIZE=65536
```

**Purpose:** Sets the chunk size for fused MoE operations.

**Tuning:**
- Larger values: Better throughput, higher memory usage
- Smaller values: Lower memory usage, potentially lower throughput
- Default is usually optimal for most cases

## GPU/Hardware Optimization

### VLLM_USE_TRITON_FLASH_ATTN

```bash
export VLLM_USE_TRITON_FLASH_ATTN=1
```

**Purpose:** Enables Triton-based flash attention implementation.

**When to use:**
- General purpose optimization
- Compatible with most GPU architectures

### VLLM_ATTENTION_BACKEND

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
```

**Purpose:** Specifies the attention backend to use.

**Options:**
- `TRITON_ATTN_VLLM_V1` - Recommended for A100 GPUs
- `FLASH_ATTN` - Standard flash attention
- `XFORMERS` - xFormers backend

### TensorRT-LLM Attention (Blackwell GPUs)

```bash
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1
```

**Purpose:** Enables TensorRT-LLM optimizations for NVIDIA Blackwell GPUs (B100/B200).

**Benefits:**
- Maximum performance on Blackwell architecture
- Optimized kernel implementations
- Better memory efficiency

### VLLM_FLASHINFER_FORCE_TENSOR_CORES

```bash
export VLLM_FLASHINFER_FORCE_TENSOR_CORES=1
```

**Purpose:** Forces the use of tensor cores for FlashInfer operations.

**Benefits:**
- Better performance on GPUs with tensor cores
- Optimized for A100, H100, and newer GPUs

## Memory & Performance

### VLLM_CPU_KVCACHE_SPACE

```bash
export VLLM_CPU_KVCACHE_SPACE=4
```

**Purpose:** Allocates CPU memory (in GiB) for KV cache offloading.

**When to use:**
- GPU memory is limited
- Running large models
- Need to handle longer contexts

### VLLM_ALLOW_LONG_MAX_MODEL_LEN

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

**Purpose:** Allows setting very long context lengths beyond default limits.

**When to use:**
- Need to process very long documents
- Have sufficient memory available

## Logging & Monitoring

### VLLM_LOGGING_LEVEL

```bash
export VLLM_LOGGING_LEVEL=INFO
```

**Options:** DEBUG, INFO, WARNING, ERROR

**Purpose:** Controls the verbosity of vLLM logs.

### VLLM_LOG_STATS_INTERVAL

```bash
export VLLM_LOG_STATS_INTERVAL=10
```

**Purpose:** Sets the interval (in seconds) for logging statistics.

## CUDA Configuration

### CUDA_VISIBLE_DEVICES

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**Purpose:** Specifies which GPUs are visible to the application.

**Examples:**
- `0` - Use only GPU 0
- `0,1` - Use GPUs 0 and 1
- `1,3` - Use GPUs 1 and 3

### PYTHONUNBUFFERED

```bash
export PYTHONUNBUFFERED=1
```

**Purpose:** Disables Python output buffering for real-time log viewing.

## How to Use in vLLM CLI

### Method 1: Profile Configuration

Environment variables can be configured in profiles:

1. Start vLLM CLI: `vllm-cli`
2. Go to Settings → Manage Profiles
3. Edit or create a profile
4. Select "Edit environment variables"
5. Choose from presets or add custom variables

### Method 2: Universal Environment Variables

Apply variables to all servers:

1. Start vLLM CLI: `vllm-cli`
2. Go to Settings → Universal Environment Variables
3. Configure variables that apply to all servers

### Method 3: Manual Export

Set variables before starting vLLM:

```bash
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
vllm-cli
```

## Recommended Combinations

### For GPT-OSS Models (Optimal)

```bash
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
```

### For A100 GPUs

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
```

### For H100/H200 GPUs

```bash
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
export VLLM_FLASHINFER_FORCE_TENSOR_CORES=1
```

### For Blackwell GPUs (B100/B200)

```bash
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
```

## Troubleshooting

### High GPU Utilization

If GPU is constantly at 100%:
1. Enable `VLLM_USE_FLASHINFER_MXFP4_MOE=1`
2. Enable `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`
3. Consider using FP8 precision with `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1`

### Out of Memory Errors

1. Enable `VLLM_CPU_KVCACHE_SPACE=4` (or higher)
2. Enable `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`
3. Reduce `gpu_memory_utilization` in config
4. Use quantization (FP8, AWQ, GPTQ)

### Slow Inference

1. Enable appropriate attention backend for your GPU
2. Enable `VLLM_FLASHINFER_FORCE_TENSOR_CORES=1`
3. Use `VLLM_USE_FLASHINFER_MXFP4_MOE=1` for MoE models
4. Increase batch size if memory allows

## References

- [vLLM Environment Variables Documentation](https://docs.vllm.ai/en/latest/serving/env_vars.html)
- [vLLM GPT-OSS Guide](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html)
- [FlashInfer Documentation](https://docs.flashinfer.ai/)


## AMD ROCm Optimization (v0.2.9+)

### For AMD MI300x/MI325 GPUs

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
```

**Purpose:** Enables AMD-specific optimizations for MI300x and MI325 series GPUs.

**Benefits:**
- Optimized attention mechanisms for AMD architecture
- Better performance on ROCm platform
- Improved memory efficiency

### For AMD MI355x GPUs

```bash
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1
```

**Purpose:** Advanced optimizations for MI355x series with Triton kernels.

**Benefits:**
- Fused operations for better performance
- Optimized GEMM operations
- Better scaling for large models

## Advanced MoE Optimizations (v0.2.9+)

### VLLM_USE_FLASHINFER_MOE_FP8

```bash
export VLLM_USE_FLASHINFER_MOE_FP8=1
```

**Purpose:** Enables FP8 precision for MoE layers using FlashInfer.

**When to use:**
- Qwen3-Next FP8 models
- SM90/SM100 GPUs (H100, H200, B200)
- Maximum throughput requirements

### VLLM_FLASHINFER_MOE_BACKEND

```bash
export VLLM_FLASHINFER_MOE_BACKEND=latency
```

**Purpose:** Selects the FlashInfer MoE backend optimization strategy.

**Options:**
- `latency`: Optimized for low latency
- `throughput`: Optimized for high throughput

### VLLM_USE_DEEP_GEMM

```bash
export VLLM_USE_DEEP_GEMM=0
```

**Purpose:** Controls the use of DeepGEMM optimizations globally.

**When to disable (=0):**
- Using FP8 FlashInfer MoE
- SM90/SM100 GPUs with FlashInfer
- Compatibility issues with certain models

### VLLM_MOE_USE_DEEP_GEMM

```bash
export VLLM_MOE_USE_DEEP_GEMM=0
```

**Purpose:** Controls DeepGEMM specifically for MoE expert layers.

**When to disable (=0):**
- MiniMax-M2 and similar MoE models
- Better performance with Cutlass BlockScaled GroupedGemm for MoE
- Keep DeepGEMM enabled for linear layers while disabling for MoE

**Note:** This provides better performance than fully enabling or disabling DeepGEMM.

## Attention Backend Selection (v0.2.9+)

### VLLM_ATTENTION_BACKEND

```bash
# For AMD GPUs
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# For NVIDIA with MLA (Multi-head Latent Attention)
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA
```

**Purpose:** Explicitly selects the attention backend.

**Options:**
- `FLASH_ATTN`: FlashAttention (general purpose)
- `TRITON_ATTN_VLLM_V1`: Triton attention (A100 optimized)
- `CUTLASS_MLA`: CUTLASS Multi-head Latent Attention
- `FLASHINFER`: FlashInfer backend

### VLLM_USE_TRTLLM_ATTENTION

```bash
export VLLM_USE_TRTLLM_ATTENTION=0
```

**Purpose:** Controls TensorRT-LLM attention usage.

**When to disable (=0):**
- Using FlashInfer on SM90/SM100
- Compatibility with certain attention patterns
- Debugging attention issues

## Video Processing (v0.2.9+)

### VLLM_VIDEO_LOADER_BACKEND

```bash
export VLLM_VIDEO_LOADER_BACKEND=opencv
```

**Purpose:** Selects the backend for video loading in multimodal models.

**Options:**
- `opencv`: OpenCV-based video loading
- Other backends may be available depending on vLLM version

## Configuration Tuning (v0.2.9+)

### VLLM_TUNED_CONFIG_FOLDER

```bash
export VLLM_TUNED_CONFIG_FOLDER=/path/to/tuned/configs
```

**Purpose:** Specifies directory containing tuned MoE kernel configurations.

**Usage:**
1. Run `benchmark_moe` to generate tuned configs
2. Set this variable to the output directory
3. vLLM will automatically load optimized configs for your hardware

**Benefits:**
- Hardware-specific optimizations
- Better performance than default configs
- Reduced tuning time on subsequent runs

## Model-Specific Recommendations (v0.2.9+)

### DeepSeek-V3.1

```bash
# No special environment variables required
# Use profile: deepseek_v31
```

### Qwen3-Next

```bash
# Standard configuration
# Use profile: qwen3_next

# For FP8 models on SM90/SM100
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_TRTLLM_ATTENTION=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Use profile: qwen3_next_fp8
```

### GLM-4.5 / GLM-4.5-Air

```bash
# No special environment variables required
# Use profile: glm_45 or glm_45_air_fp8
```

## Quick Reference by GPU Architecture

### NVIDIA A100 (Ampere)

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
```

### NVIDIA H100/H200 (Hopper)

```bash
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
export VLLM_FLASHINFER_FORCE_TENSOR_CORES=1
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
```

### NVIDIA B100/B200 (Blackwell)

```bash
export VLLM_USE_TRTLLM_ATTENTION=1
export VLLM_USE_TRTLLM_DECODE_ATTENTION=1
export VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1
export VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1
export VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1
```

### AMD MI300x/MI325

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
```

### AMD MI355x

```bash
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export VLLM_USE_AITER_TRITON_GEMM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TRITON_HIP_PRESHUFFLE_SCALES=1
```

## Version History

- **v0.2.9**: Added AMD ROCm optimizations, advanced MoE options, video processing, model-specific recommendations
- **v0.2.8**: Enhanced chat template configuration
- **v0.2.7**: Fixed enable_auto_tool_choice dependency
- **v0.2.6**: Added GPT-OSS-20B profiles and MoE optimizations
- **v0.2.5**: Multi-model proxy support
- **v0.2.4**: Hardware-optimized profiles

## See Also

- [Profiles Guide](docs/profiles.md) - Pre-configured profiles using these variables
- [Chat Template Guide](CHAT_TEMPLATE_GUIDE.md) - Chat template configuration
- [Memory Optimization Guide](MEMORY_OPTIMIZATION.md) - Memory management strategies
- [vLLM Official Documentation](https://docs.vllm.ai/)
