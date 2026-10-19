# vllm-cli v0.9.0.0 Release Notes (2026-10-19)

## Overview

vllm-cli v0.9.0.0 adds full support for **vLLM v0.32.0** (814 commits from 363 contributors, 121 new). Argument schema updated to **v2.9.0** with **342 arguments** (4 new, 1 deprecated) and synced against `vllm/engine/arg_utils.py` (EngineArgs) and `vllm/entrypoints/launchers/cli_args.py` (FrontendArgs) at tag v0.32.0.

This is a **major version bump** (0.8.0.0 → 0.9.0.0), matching the vLLM minor-version increment to 0.32.0.

## Highlights

### 🧭 Generic `hf` Parsers & Structured Decisions

The new `hf` tool and reasoning parser (`--tool-call-parser hf`, `--reasoning-parser hf`) reads the checkpoint's `response_template`, so models that ship one need no model-specific parser — in both the Python and Rust frontends (#58604, #58602, #59005). The new `POST /v1/systemone` endpoint answers `choice`, `noul` and `score` questions over a structured state and is served by default (#59299, #60677, #60696), including `Cloudflare/clef-omni` (#61064), `LiquidAI/d1-3B` (#60676) and DiffusionGemma (#60466).

### 🗄️ Low-Precision KV Cache Everywhere

NVFP4 KV cache now runs on SM8x and SM12x (#46963), SM90 (#60620), with XQA decode on SM12x (#60452) and MiniMax-M3 sparse attention on SM100 (#59300). GLM-5.x defaults to an FP8 KV cache on SM100 (halved KV memory, +2.3–5.5% throughput, #60140, #60281); `fp8_ds_mla` on SM90 gives GLM-5.3-Flash up to +31% output throughput (#59246). ROCm gains the 4-bit `ultraquant_4bit` KV cache with ~2× the density of FP8 (#57057).

### 💤 Deeper Sleep Mode (RL weight swapping)

Sleep now releases workspace scratch (up to 4.5 GiB/GPU, #59156) and KV-init state (~4.6 GiB/GPU, #59158); new `--sleep-mode-offload-cudagraph` backs CUDA graph memory up to CPU and restores it in place on wake (#59160, #59523). `sleep(level=2)` keeps FP8 KV scales (#57163), `--sleep-preserve-parameter-names 'draft.*'` protects draft models (#61046), and weight-transfer communicators are released without stopping inference (#61060).

### 🚀 Faster Startup & Restarts

`vllm preload --preload-autotune` keeps the FlashInfer autotune table warm across restarts (#60085) — pairs with vllm-cli's Weight Cache Preload menu. `vllm download-kernels` installs FlashInfer precompiled kernels (#58765); the shared-memory tensor arena broadcasts large multimodal tensors to workers with one copy (#51207); tokenizers load from the local snapshot (#60825); and with `--api-server-count` above 1 each connection goes to the least-loaded API worker (+43–85% req/s on short requests, #60723). Fast Start now works with pipeline parallelism (#55477), and daemon-held weights count against `gpu_memory_utilization` (#57298).

### 🧠 Model Runner V2 & Speculative Decoding

GPU n-gram speculator with variable-length drafts (#40704), Elastic EP with drain-then-commit scaling (#53934), PP with prefill context parallelism for MLA models (#59139), a stock `torch.compile` mode (#59079), and MRV2 is now the default on ROCm (#56360).

### 🤖 New Models

**EmbeddingGemma 2** omni-modal embeddings (#60254), **MiniCPM-V 4.7** (#58674, with dedicated `minicpmv` reasoning parser), **Nemotron 3.5 ASR** (#59827), PLaMo 3 reasoning/tool parsers (#56575), zerank-2 reranker score template (#60415). DeepSeek-V4 gains decoder-side SWA bounded replay (#58132) with `compilation_config.decoder_replay_trim_threshold` (#59532); GLM-5.3 sparse-MLA index conversions are reused across layers (3.5–3.9× on that kernel, #59464).

### 🔒 Security

Three GHSA fixes — chat-template resource exhaustion (#50300), tokenizer `max_token_id` off-by-one (#59491), P2P ZMQ socket exhaustion (#51504). CORS no longer allows every origin by default (#61009); untrusted images are limited to raster formats (#60022); shared-memory multimodal cache handles are HMAC-signed (#59357); JSON schemas deeper than 64 levels are rejected (#60036).

## Schema Changes (v2.8.0 → v2.9.0)

### New CLI Arguments (4)

| Argument | Type | Notes |
|---|---|---|
| `--custom-histogram-buckets` | JSON string | Per-family Prometheus histogram bucket boundaries (`request_latency`, `time_to_first_token`, `inter_token_latency`, …, #48867) |
| `--enable-hc-sp` | bool | HyperConnection sequence parallelism — Qwen4Exp on CUDA with TP>1, PP=1; ignored at TP=1 (#56322) |
| `--enable-shm-tensor-arena` | bool | Zero-copy shared-memory arena for large CPU→worker tensors (multimodal pixel values; node-local readers, #51207) |
| `--sleep-mode-offload-cudagraph` | bool | Back up CUDA graph memory to CPU during sleep; needs `--enable-sleep-mode` + cumem backend + CUDA graphs (#59160, #59523) |

`--enable-shm-tensor-arena` is described as default-on in the upstream release notes, but the `ParallelConfig` field default in the v0.32.0 source is still `False`; the schema mirrors the source.

### Deprecated (upstream vLLM 0.32.0)

- `--disable-cascade-attn` — cascade attention removed (#61126); the flag still parses but only warns, and is removed from the vLLM CLI in v0.33.0

### Choices Changes

- `--tool-call-parser`: 52 → **55** (added `hf` #58604, `plamo3` #56575, `qwen3_json` #60136; note `qwen3_xml` is now an alias of `qwen3_coder` XML in the Rust frontend)
- `--reasoning-parser`: 36 → **39** (added `hf` #58602, `minicpmv` #58674, `plamo3`)
- `--kv-cache-dtype`: 13 → **19** (added `ultraquant_4bit` #57057; backfilled `float16`, `bfloat16`, `nvfp4_ds_mla`, `int4_per_token_head`, `nvfp4_4over6` — all live in source)
- `--moe-backend`: 22 → **23** (added `flashinfer_moe_ep_cutedsl` #54049, `aiter_mega_moe` #59685; dropped stale `flashinfer_moe_ep_mega_cutedsl`)
- `--sparse-indexer-topk-backend`: added `aiter` (ROCm)
- `--gdn-prefill-backend` (Gated DeltaNet): added `cutedsl` (CUDA) and `aiter_flydsl` (ROCm, #57560)
- `--mamba-cache-mode`: `all` **removed** — `--mamba-cache-mode all` now fails validation (#58997)
- `--logging-config` gained a `formatter: "json"` sub-field via `LoggingConfig.formatter` (#58739) — structured JSON logs, no new flag needed

### Schema Hygiene

- `--all2all-backend` and `--gdn-prefill-backend` carried stale "removed upstream" deprecation markers although both flags are live; both un-deprecated and their choice lists refreshed (`--all2all-backend` now lists all 13 CLI-selectable backends incl. `moonep`; `passthrough` is intentionally excluded — vLLM binds it from `--moe-backend`)
- `parser_sync.py`: removed a duplicate `llama4_pythonic` entry that shadowed the correct identity mapping

## Breaking Changes & Deprecations (upstream vLLM)

- **Python 3.10 dropped** — vLLM v0.32.0 requires Python 3.11–3.14 (#60402). Rebuild your venvs with 3.11+
- Default CORS origins changed from `*` to none — browser clients need `--allowed-origins` (#61009)
- `transformers>=5.16.1,<5.20.0` and `tokenizers>=0.23.2` required (#57387, #59796)
- torch.compile sequence-parallelism / AsyncTP passes removed (`PassConfig.enable_sp`, `fuse_gemm_comms`, `sp_min_token_num`); model-level SP is unaffected (#60517)
- Cascade attention removed; `--disable-cascade-attn` warns only (#61126)
- `mamba_cache_mode="all"` removed (#58997)
- `max_tokens` is an upper bound, no longer reserved from the context window — long prompts are no longer rejected (#57035)
- Responses API reasoning stream events renamed to `response.content_part.added/done` with `part.type="reasoning_text"` (#59652)
- Previously deprecated APIs removed: `wait_for_inflight_requests`, `AsyncLLM.pause_generation()`, `IOProcessor.output_to_response`, `vllm.scripts.main()`, `vllm bench throughput --dataset` (#60923); `sonnet` bench dataset (#60129); LoRA tensorizer (#60024); PyO3 `vllm._rust_tool_parser` bridge (#59744)
- `return_mm_kwargs` deprecated in favor of `mm_payload` (#61427); sleep API invalid levels now return 400 (#52864)
- Rust frontend: `qwen3_xml` now parses Qwen3-Coder XML (alias of `qwen3_coder`) — use the new `qwen3_json` for the JSON format (#60136)
- Watermarking: an omitted request field now means enabled; only `false` opts out (#56801)
- New defaults: shared-memory tensor arena on (#51207); GLM-5.x FP8 KV on SM100 with `--kv-cache-dtype auto` (#60140, #60281); `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1` (#57991); Humming preferred over Marlin on SM90 (#56997); `sparse_mla_prefill_workspace_factor` 2 above 256K `max_model_len` (#61023); MRV2 default on ROCm (#56360); `VLLM_ROCM_AITER_MLA_DCP_VERIFY=auto` (#59965); `dtype=auto` → bfloat16 on PowerPC (#58528); MXFP4 W4A4 falls back to W4A16 with a warning when no kernel exists (#60079)
- Stricter startup validation: incompatible `--prefix-match-unit` (#58021), `max_num_scheduled_tokens=0` (#60714), HiSparse with `cudagraph_mode=FULL` (#59688), negative `--device-ids` (#49053)

## Dependencies

- Requires `vllm>=0.20.0,<0.33.0` (updated from `<0.32.0`)
- vLLM 0.32.0 runtime bumps you inherit: `transformers>=5.16.1,<5.20.0`, `tokenizers>=0.23.2`, NIXL 1.5.0, `humming-kernels==0.1.17`, AITER 0.1.24.post1 (ROCm), `aiohttp>=3.14.3`, `pillow>=12.3.0`
- vllm-cli itself still runs on Python >= 3.10, but the vLLM 0.32.0 runtime requires Python 3.11+ — create new venvs with 3.11–3.14

## Migration Guide

### For Users Upgrading from v0.8.0.x

1. Profiles are forward-compatible; no profile migration required.
2. **Rebuild venvs with Python 3.11+** (`uv venv --python 3.11 && uv pip install -e .`) — vLLM 0.32.0 has no 3.10 wheels.
3. Browser clients: add `--allowed-origins https://your.origin` to server profiles (CORS default is now deny-all).
4. Replace any `--mamba-cache-mode all` in profiles with `align` (recommended) or `none`.
5. Remove `--disable-cascade-attn` from profiles — it warns and will be rejected once vLLM v0.33 removes the flag.
6. If you used `qwen3_xml` for JSON tool parsing on the Rust frontend, switch to `qwen3_json`.
7. If your `--compilation-config` JSON sets `pass_config.enable_sp` / `fuse_gemm_comms` / `sp_min_token_num`, remove those keys (passes deleted, #60517).

```bash
# Example: multimodal serving with new 0.32 capabilities
vllm-cli profile create qwen-vl-fast \
  --model Qwen3-VL-8B-Instruct \
  --enable-shm-tensor-arena \
  --custom-histogram-buckets '{"request_latency": [0.05, 0.1, 0.5, 1, 5]}' \
  --tool-call-parser hf

# Deeper sleep for RL weight swapping
vllm-cli serve mymodel --enable-sleep-mode --sleep-mode-offload-cudagraph
```

## Files Changed

- `src/vllm_cli/schemas/argument_schema.json` — v2.9.0, 342 args, last_synced v0.32.0
- `src/vllm_cli/config/cli_args_sync.py` — SUPPORTED_VLLM_VERSIONS + v0.32.0 (17 versions)
- `src/vllm_cli/config/parser_sync.py` — v0.32.0 tool/reasoning parser mappings (hf, plamo3, qwen3_json, minicpmv); duplicate `llama4_pythonic` key fixed
- `src/vllm_cli/config/command_import.py` — BOOLEAN_FLAGS + 3 new 0.32 flags (importer no longer swallows the following token)
- `pyproject.toml` — version 0.9.0.0, vllm<0.33.0
- `VERSION`, `CHANGELOG.md`, `README.md`, this file
