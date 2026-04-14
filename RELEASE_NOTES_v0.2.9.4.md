# v0.2.9.4 Release Notes

**Release Date**: 2026-04-12

## What's New

### Auto-Sync Parser Choices

Added automatic synchronization for `reasoning_parser` and `tool_call_parser` choices from vLLM GitHub source. This ensures the CLI always has the latest parser options when new models are supported.

#### Features

- **Automatic Parser Discovery**: Fetches available parsers from vLLM GitHub repositories:
  - `vllm/tool_parsers/` - Tool call parsers
  - `vllm/reasoning/` - Reasoning parsers

- **CLI Integration**: New sync commands:
  ```bash
  # View available parser updates
  vllm-cli recipes --sync-parsers
  
  # Apply parser updates to local schema
  vllm-cli recipes --sync-parsers --apply-parsers
  ```

- **New Parsers Added**:
  - Reasoning Parser: `gemma4`, `deepseek_v3`, `ernie45`, `identity`, `kimi_k2`, `nemotron_v3`, `olmo3`
  - Tool Call Parser: `gemma4`, `deepseek_v32`, `ernie45`, `functiongemma`, `gigachat3`, `granite4`, `olmo3`, `phi4mini`, `pythonic`, `qwen3xml`, `seed_oss`, `step3`, `step3p5`, `xlam`

For more information, see [vLLM Reasoning Models](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

---

## Bug Fixes

### i18n Submenu Translation Fix

Fixed an issue where Chinese menu text would revert to English when entering submenus (Settings > Profiles, Settings > Shortcuts, etc.). All submenu screens now properly display translated text when a language is selected.

---

## Dependencies

### Updated Dependencies

| Package | Old Version | New Version |
|---------|-------------|-------------|
| `vllm` | `>=0.10.0` | `>=0.18.0` |
| `httpx` | `>=0.24.0,<1.0.0` | `>=0.28.0,<1.0.0` |
| `fastapi` | `>=0.100.0` | `>=0.135.0` |
| `uvicorn` | `>=0.23.0` | `>=0.41.0` |
| `requests` | `>=2.28.0` | `>=2.32.0` |

### New Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `nvidia-modelopt` | `>=0.5.0` | ModelOpt quantization support (optional) |

### New Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `annotated-types` | `>=0.7.0` | Pydantic v2 compatibility |
| `pydantic-settings` | `>=2.0` | Settings management |
| `typer` | `>=0.9.0` | CLI interface |
| `python-dotenv` | `>=1.0.0` | Environment variables |
| `certifi` | `>=2024.0.0` | SSL certificates |
| `numpy` | `>=1.20.0` | Numerical computing |

---

## Technical Details

### Updated Files

| File | Change |
|------|--------|
| `config/parser_sync.py` | NEW - Parser sync module |
| `cli/parser.py` | Added `--sync-parsers` and `--apply-parsers` options |
| `cli/handlers.py` | Added parser sync handler |
| `schemas/argument_schema.json` | Added parser choices from vLLM GitHub |
| `validation/schema.py` | Synced quantization choices |
| `cli/parser.py` | Added new quantization options to CLI |
| `ui/custom_config.py` | Added interactive config options |
| `errors/recovery.py` | Added quantization suggestions |
| `ui/system_info.py` | Added quantization use cases |
| `system/dependencies.py` | Added fallback list entries |
| `pyproject.toml` | Added `modelopt` optional dependency |
| `i18n/translations/en.json` | Added ModelOpt descriptions |
| `i18n/translations/zh.json` | Added ModelOpt descriptions |
| `ui/settings.py` | Fixed i18n parameter passing |
| `ui/profiles.py` | Added i18n support |
| `ui/shortcuts.py` | Added i18n support |

### Quantization Methods Available

The following quantization methods are now supported:

| Method | Description |
|--------|-------------|
| `awq` | AutoAWQ (Activation-aware Weight Quantization) |
| `awq_marlin` | AutoAWQ with Marlin kernel |
| `auto-round` | Auto round quantization |
| `bitsandbytes` | 8-bit/4-bit quantization |
| `bitblas` | BitBLAS quantization |
| `gguf` | GGML universal format |
| `gptq` | GPT Quantization |
| `gptq_marlin` | GPTQ with Marlin kernel |
| `fp8` | 8-bit floating point |
| `fbgemm_fp8` | FBGEMM FP8 |
| `compressed-tensors` | INT4/INT8 compressed |
| `experts_int8` | Expert quantization |
| `modelopt_fp4` | NVIDIA ModelOpt NVFP4 |
| `modelopt_mxfp8` | NVIDIA ModelOpt MXFP8 |

---

## Upgrade Notes

### For Users

- **Python 3.10+ required**: This release requires Python 3.10 or higher (previously 3.9+)
- **No breaking changes**: This release is fully backward compatible with Python 3.10+
- **Optional ModelOpt support**: Install with `pip install vllm-cli[modelopt]` to enable ModelOpt quantization
- **Language preference preserved**: Chinese users will now see translated text in all submenus
- **Parser choices auto-sync**: Run `vllm-cli recipes --sync-parsers --apply-parsers` to update parser choices

### For Developers

- Quantization validation schema has been updated - ensure your code handles the new options
- The `i18n_manager` parameter is now accepted by all submenu functions in `settings.py`

---

## Known Issues

None.

---

## Next Steps

- Check out the [vLLM documentation](https://docs.vllm.ai/) for more information about quantization
- Report issues at: https://github.com/Chen-zexi/vllm-cli/issues
