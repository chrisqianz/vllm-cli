# Chat Template Examples

This directory contains example chat templates for various models.

## What are Chat Templates?

Chat templates are Jinja2 files that define how conversations are formatted for language models. They're essential for:
- Tool calling / Function calling
- Proper conversation formatting
- Model-specific requirements

## Available Templates

To use these templates with vLLM CLI, you can:

1. **Download from vLLM repository**:
   ```bash
   # Qwen3 tool calling
   curl -o examples/tool_chat_template_qwen3.jinja \
     https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_qwen3.jinja
   
   # Llama 3.1 JSON mode
   curl -o examples/tool_chat_template_llama3.1_json.jinja \
     https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_llama3.1_json.jinja
   
   # Mistral tool calling
   curl -o examples/tool_chat_template_mistral.jinja \
     https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_mistral.jinja
   ```

2. **Use the interactive file browser** (v0.2.8+):
   ```bash
   vllm-cli
   # Select: Serve with Custom Config
   # Navigate to: API > chat_template
   # Choose: Browse for .jinja files
   ```

## Template Usage

### Qwen3 Models
```bash
vllm-cli serve --model Qwen/Qwen2.5-7B-Instruct \
  --chat-template examples/tool_chat_template_qwen3.jinja \
  --tool-call-parser qwen3 \
  --enable-auto-tool-choice
```

### Llama 3.1 Models
```bash
vllm-cli serve --model meta-llama/Llama-3.1-8B-Instruct \
  --chat-template examples/tool_chat_template_llama3.1_json.jinja \
  --tool-call-parser llama3_json
```

### Mistral Models
```bash
vllm-cli serve --model mistralai/Mistral-7B-Instruct-v0.3 \
  --chat-template examples/tool_chat_template_mistral.jinja \
  --tool-call-parser mistral
```

## More Information

- [Chat Template Guide](../CHAT_TEMPLATE_GUIDE.md)
- [vLLM Tool Calling Documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [vLLM Examples](https://github.com/vllm-project/vllm/tree/main/examples)

## Creating Custom Templates

See the [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template) for information on creating custom chat templates.
