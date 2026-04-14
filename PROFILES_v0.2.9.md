# vLLM CLI v0.2.9 - Complete Profiles List

## 📊 Profile Overview

**Total Profiles**: 25 (9 existing + 16 new)

---

## 🆕 New Profiles (v0.2.9)

### DeepSeek Family (2 profiles)

#### 1. deepseek_v31
- **Model**: DeepSeek-V3.1
- **Hardware**: 8xH200/H20
- **Features**: Thinking mode, tool calling
- **Key Config**: `enable_expert_parallel`, `tool_call_parser=deepseek_v31`

#### 2. deepseek_v32
- **Model**: DeepSeek-V3.2-Exp
- **Hardware**: 8xH200/B200
- **Features**: Sparse attention, DP+EP
- **Key Config**: `data_parallel_size=8`, `enable_expert_parallel`
- **Environment**: `VLLM_USE_DEEP_GEMM=0`

### Qwen Family (5 profiles)

#### 3. qwen3_next
- **Model**: Qwen3-Next-80B
- **Hardware**: 4xH200/A100
- **Features**: Hybrid attention, sparse MoE
- **Key Config**: `tensor_parallel_size=4`

#### 4. qwen3_next_mtp
- **Model**: Qwen3-Next-80B
- **Hardware**: 4xH200/A100
- **Features**: Multi-Token Prediction
- **Key Config**: `speculative_config` with MTP

#### 5. qwen3_next_fp8
- **Model**: Qwen3-Next-80B-FP8
- **Hardware**: SM90/SM100 (H100, H200, B200)
- **Features**: FP8 optimization
- **Environment**: 5 FP8-specific variables

#### 6. qwen3_coder
- **Model**: Qwen3-Coder-480B
- **Hardware**: 8xH200
- **Features**: Tool calling, coding
- **Key Config**: `max_model_len=32000`, `tool_call_parser=qwen3_coder`

#### 7. qwen3_coder_fp8
- **Model**: Qwen3-Coder-480B-FP8
- **Hardware**: 8xH200
- **Features**: FP8, DP+EP
- **Environment**: `VLLM_USE_DEEP_GEMM=1`

### GLM Family (2 profiles)

#### 8. glm_45
- **Model**: GLM-4.5
- **Hardware**: 8 GPUs
- **Features**: Tool calling, reasoning
- **Key Config**: `max_model_len=65536`, `tool_call_parser=glm45`

#### 9. glm_45_air_fp8
- **Model**: GLM-4.5-Air-FP8
- **Hardware**: 8 GPUs
- **Features**: Low cost, FP8
- **Key Config**: Same as glm_45

### MiniMax Family (2 profiles)

#### 10. minimax_m2
- **Model**: MiniMax-M2
- **Hardware**: 4xH200/A100
- **Features**: Tool calling, reasoning
- **Key Config**: `tensor_parallel_size=4`, `tool_call_parser=minimax_m2`
- **Environment**: `VLLM_MOE_USE_DEEP_GEMM=0`

#### 11. minimax_m2_dp
- **Model**: MiniMax-M2
- **Hardware**: >4 GPUs
- **Features**: DP+EP for scaling
- **Key Config**: `data_parallel_size=8`, `enable_expert_parallel`
- **Environment**: `VLLM_MOE_USE_DEEP_GEMM=0`

### Kimi Family (3 profiles)

#### 12. kimi_k2
- **Model**: Kimi-K2-Instruct
- **Hardware**: 16xH800
- **Features**: Tool calling, TP+PP
- **Key Config**: `tensor_parallel_size=8`, `pipeline_parallel_size=2`

#### 13. kimi_k2_dp
- **Model**: Kimi-K2-Instruct
- **Hardware**: 16xH800
- **Features**: DP+EP
- **Key Config**: `data_parallel_size=16`, `enable_expert_parallel`

#### 14. kimi_k2_thinking
- **Model**: Kimi-K2-Thinking
- **Hardware**: 8xH200
- **Features**: Deep reasoning
- **Key Config**: `tensor_parallel_size=8`

### AMD GPU (2 profiles)

#### 15. amd_mi300
- **Model**: Any
- **Hardware**: AMD MI300x/MI325
- **Features**: ROCm optimizations
- **Environment**: 3 AITER variables

#### 16. amd_mi355
- **Model**: Any
- **Hardware**: AMD MI355x
- **Features**: Advanced Triton kernels
- **Environment**: 7 optimization variables

---

## 📋 Existing Profiles (v0.2.8 and earlier)

### General Purpose (4 profiles)

#### 17. standard
- **Description**: Minimal configuration with auto-detected settings
- **Use Case**: Default, general purpose

#### 18. high_throughput
- **Description**: Optimized for maximum request throughput
- **Key Config**: `gpu_memory_utilization=0.95`, `enable_chunked_prefill`

#### 19. low_memory
- **Description**: Minimizes memory usage
- **Key Config**: `max_model_len=4096`, `gpu_memory_utilization=0.70`

#### 20. moe_optimized
- **Description**: Optimized for MoE models
- **Key Config**: `enable_expert_parallel`

### GPT-OSS Family (5 profiles)

#### 21. gpt_oss_ampere
- **Hardware**: NVIDIA A100
- **Features**: Ampere optimizations

#### 22. gpt_oss_hopper
- **Hardware**: NVIDIA H100/H200
- **Features**: Hopper optimizations

#### 23. gpt_oss_blackwell
- **Hardware**: NVIDIA B100/B200
- **Features**: Blackwell optimizations

#### 24. gpt_oss_20b
- **Model**: GPT-OSS-20B
- **Features**: Tool calling, full performance

#### 25. gpt_oss_20b_low_memory
- **Model**: GPT-OSS-20B
- **Features**: Memory-optimized (~13GB VRAM)

---

## 📊 Profile Statistics

### By Model Family
- **GPT-OSS**: 5 profiles
- **DeepSeek**: 2 profiles
- **Qwen**: 5 profiles
- **GLM**: 2 profiles
- **MiniMax**: 2 profiles
- **Kimi**: 3 profiles
- **General**: 4 profiles
- **AMD**: 2 profiles

### By Hardware
- **NVIDIA Ampere (A100)**: 3 profiles
- **NVIDIA Hopper (H100/H200)**: 8 profiles
- **NVIDIA Blackwell (B200)**: 3 profiles
- **AMD MI300x/MI325**: 1 profile
- **AMD MI355x**: 1 profile
- **General/Multi-GPU**: 9 profiles

### By Feature
- **Tool Calling**: 12 profiles
- **MoE Optimized**: 10 profiles
- **FP8 Optimized**: 6 profiles
- **Thinking/Reasoning**: 5 profiles
- **DP+EP**: 6 profiles
- **Low Memory**: 2 profiles

---

## 🎯 Quick Selection Guide

### By Use Case

**Coding & Development**
- qwen3_coder / qwen3_coder_fp8
- minimax_m2

**Reasoning & Thinking**
- deepseek_v31 / deepseek_v32
- kimi_k2_thinking
- glm_45

**Tool Calling**
- minimax_m2
- kimi_k2
- qwen3_coder
- gpt_oss_20b

**High Throughput**
- high_throughput
- qwen3_next_fp8
- glm_45_air_fp8

**Low Memory**
- low_memory
- gpt_oss_20b_low_memory

### By GPU Count

**Single GPU**
- standard
- low_memory

**4 GPUs**
- qwen3_next (all variants)
- minimax_m2

**8 GPUs**
- deepseek_v31 / deepseek_v32
- qwen3_coder (all variants)
- glm_45 (all variants)
- kimi_k2_thinking
- gpt_oss_* (all variants)

**16 GPUs**
- kimi_k2 / kimi_k2_dp

**Multi-Node**
- Any profile with DP+EP support

### By Hardware Platform

**NVIDIA A100**
- gpt_oss_ampere
- qwen3_next
- minimax_m2

**NVIDIA H100/H200**
- gpt_oss_hopper
- All DeepSeek profiles
- All Qwen profiles
- All GLM profiles
- All Kimi profiles

**NVIDIA B200**
- gpt_oss_blackwell
- deepseek_v32
- qwen3_next_fp8

**AMD MI300x/MI325**
- amd_mi300

**AMD MI355x**
- amd_mi355

---

## 💡 Usage Examples

### Basic Usage
```bash
vllm-cli serve --model <model-name> --profile <profile-name>
```

### Examples by Profile

```bash
# DeepSeek-V3.1
vllm-cli serve --model deepseek-ai/DeepSeek-V3.1 --profile deepseek_v31

# Qwen3-Next with MTP
vllm-cli serve --model Qwen/Qwen3-Next-80B-A3B-Instruct --profile qwen3_next_mtp

# MiniMax-M2
vllm-cli serve --model MiniMaxAI/MiniMax-M2 --profile minimax_m2

# Kimi-K2 with DP+EP
vllm-cli serve --model moonshotai/Kimi-K2-Instruct --profile kimi_k2_dp

# Qwen3-Coder FP8
vllm-cli serve --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 --profile qwen3_coder_fp8

# AMD MI300x
vllm-cli serve --model <your-model> --profile amd_mi300
```

---

## 📚 Related Documentation

- [RELEASE_NOTES_v0.2.9.md](RELEASE_NOTES_v0.2.9.md) - Full release notes
- [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) - Environment variables guide
- [default_profiles.json](src/vllm_cli/schemas/default_profiles.json) - Profile definitions

---

**Version**: v0.2.9  
**Last Updated**: 2025-01-15  
**Total Profiles**: 25
