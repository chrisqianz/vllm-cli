# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
