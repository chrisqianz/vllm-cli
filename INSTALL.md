# vLLM CLI v0.2.9 安装说明

## 快速开始

```bash
# 1. 解压
tar -xzf vllm-cli-0.2.9.tar.gz
cd vllm-cli-0.2.9

# 2. 安装依赖（如需要）
pip install -r requirements.txt

# 3. 运行
./vllm-cli
```

## 新功能 (v0.2.9)

✨ **官方vLLM Recipes集成**
- 27个新profiles
- 7个新参数
- 20+环境变量
- AMD GPU支持
- 多节点部署

详见: RELEASE_NOTES_v0.2.9.md

## 支持的模型

- DeepSeek (V3.1, V3.2, R1)
- Qwen (Qwen3-Next, Qwen3-Coder, Qwen2.5-VL, Qwen3-VL)
- GLM (4.5, 4.5-Air, 4.5V)
- MiniMax (M2)
- Kimi (K2, K2-Thinking, Linear)
- ERNIE (4.5-21B, 4.5-300B)
- InternVL, Nemotron, Ring-1T
- GPT-OSS (20B, 120B)

## 文档

- PROFILES_v0.2.9.md - 完整profiles列表
- ENVIRONMENT_VARIABLES.md - 环境变量指南
- README.md - 完整文档
