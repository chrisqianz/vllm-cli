# vllm-cli v0.10.0.0 Release Notes (2026-11-04)

## Overview

vllm-cli v0.10.0.0 adds full support for **vLLM v0.33.0** (927 commits from 389 contributors, 122 new). Argument schema updated to **v2.10.0** with **346 arguments** (4 new, 2 deprecated) and synced against `vllm/engine/arg_utils.py` (EngineArgs) and `vllm/entrypoints/common/cli_args.py` (FrontendArgs, moved out of `launchers/` this release) at tag v0.33.0.

This is a **major version bump** (0.9.0.0 → 0.10.0.0), matching the vLLM minor-version increment to 0.33.0.

## Highlights

### 🎯 Speculative Decoding Gets Faster

The new XPress refiner for block-diffusion drafters such as DFlash and DSpark (#54448) and FLy, an opt-in lossy verification policy worth 1.04–1.31× decode throughput on MI355X and B300 (#53987), headline the release. DFlash/DSpark drafts no longer reserve slots in the token budget — DeepSeek-V4.1-Flash + DSpark on GB300 went from 1373 to 3390 output tok/s in the upstream benchmark (#59468) — and `draft_lm_head_quantization` adds +18% decode at concurrency 1 (#59973). Lookahead block hashes keep the last block on EAGLE/MTP prefix-cache hits, on by default (#61589). Medusa and the MLP speculator were removed (#61701).

### 🔩 ROCm: MonoKernel Decode & ROCM_OCTAVE KV

`--monokernel` (new in vllm-cli) runs eligible decode layers as a single persistent kernel: 2.32× P50 interactivity at concurrency 1 for DeepSeek-V4.1-Flash on MI300X/MI325X (#60731), every layer on MI355X (#61082), and −16.6% TPOT for MiniMax-M3-MXFP4 (#62712, #62994). The `ROCM_OCTAVE` 3–4 bit KV cache family (`octave_k3v4`, `octave_k3v3`, `octave_k3v3_compact`) replaces `turboquant_*` and holds ~4.6× the KV tokens of fp16 on RX 7900 XTX (#59774). The default ROCm image moves to ROCm 10.0, with 7.2 kept as `-rocm72` (#58761), and gfx1250 gains gpt-oss-120b, DeepSeek-R1/V4-Flash and MiniMax-M3 (#58800).

### 🧩 New Models & Parsers

Reflection Beam-1.0 (501B MoE, 23B active) with its DFlash2 drafter (#62074), Yandex AliceAI-Foundation-80B-A3B (#57963), Recursive Flint (#62938), Muse Spark Open (#61879), Step-5 Preview (#61727, #63147) and NVIDIA Alpamayo 2 Super (#62957) arrive, along with `beam`, `muse`/`musespark1x` and `step5` tool and reasoning parsers (#62052, #62207) — all now selectable in vllm-cli menus and one-click import.

### 🩺 Operability: Watchdog & Faster Boots

A new engine/worker watchdog dumps stack traces from stuck processes; configure it with `--watchdog-config '{"timeout":300,"check_interval":10,"dump_dir":"..."}'` (#55700). Concurrent local worker startup saves ~50 s at TP8 (#60028), the shared Triton JIT compile pool cuts cold warmup 9–28% (#61422), and Fast Start now supports expert parallelism (#55490) — which pairs with vllm-cli's Weight Cache Preload menu. The Rust frontend gained broad parity: no more HTTP 500 on long conversations (#62374) and unsupported flags now fail at startup (#62431).

### 💤 Pluggable Sleep Mode

`--sleep-mode-backend` (#63138) decouples sleep/wake from the built-in `cumem` CuMemAllocator so plugins can register CUDA checkpoint, CRIU, or durable-snapshot backends. Sleep mode also works with the NIXL pull connector (#59635) and Mooncake over RDMA (#59625).

### 🧠 Reasoning-Effort Compatibility

`--reasoning-effort-map` (#61094) rewrites a request's reasoning effort (Chat Completions, Responses, Anthropic Messages APIs) to values the model's chat template actually knows — e.g. `{"medium": "low", "xhigh": "max"}` — instead of silently falling back. The OpenAI-compatible `/v1/decisions` endpoint is on by default for supported generation models (#60465, #61550).

## Schema Changes (v2.9.0 → v2.10.0)

### New CLI Arguments (4)

| Argument | Type | Notes |
|---|---|---|
| `--monokernel` | boolean | MonoKernel persistent-kernel decode (ROCm; #60731, #61082) |
| `--sleep-mode-backend` | string | Sleep mechanism, default `cumem`; plugin-registered backends (#63138) |
| `--watchdog-config` | string (JSON) | `{"timeout","check_interval","dump_dir"}` stack-dump watchdog (#55700) |
| `--reasoning-effort-map` | string (JSON) | Effort-level rewrite map for chat templates (#61094) |

### Deprecated (upstream vLLM 0.33.0)

- `--mamba-cache-mode` — no effect since vLLM 0.33.0 (#60043); still parses, marked deprecated in schema
- `--ir-op-priority` — deprecated, removal planned in vLLM v0.34.0 (#63371)

### Choices Changes

- `--kv-cache-dtype`: +`octave_k3v4`, `octave_k3v3`, `octave_k3v3_compact` (ROCM_OCTAVE, #59774); −4 `turboquant_*` values
- `--tool-call-parser` / `--reasoning-parser`: +`beam`, `muse`, `musespark1x`, `step5` (58 / 42 with sentinel)
- `--moe-backend`: +`aiter_triton_mxfp4_mxfp4` (#62731)
- `--gdn-prefill-backend`: +`sycl` (XPU, #56835)
- `--tokenizer-mode`: synced to source — +`alpamayo2_super` (#62957) and backfilled `hf`, `deepseek_v32`, `deepseek_v4`, `inkling`, `kimi_k3`; removed dead `custom`/`slow`

### Schema Hygiene

- `--tokenizer-mode` no longer advertises `custom`/`slow`, already removed upstream in vLLM 0.31 (#58545)
- One-click command import handles `--monokernel` without swallowing the following token

## Breaking Changes & Deprecations (upstream vLLM)

- **Medusa & MLP speculator removed** (#61701); legacy Ray compiled-graph executor removed (#61712, #62228)
- **`--mamba-cache-mode` deprecated** — no effect (#60043)
- **Requests during paused generation return HTTP 503** (#51488)
- **NIXL connector requires NIXL ≥ 1.5** (#61032)
- **LM Format Enforcer & `--ir-op-priority` deprecated** (#63423, #63371)
- **CUDA 12.9 wheels/images dropped** — CUDA 13.0 only (#62575); CUDA wheel now libtorch ABI-stable
- **PyTorch upgraded to 2.14.0** with torchvision 0.29.0 and Triton 3.8.0 (#52183)
- **ROCm default moves to 10.0** (7.2 via `-rocm72`) (#58761)

## Dependencies

- `vllm>=0.20.0,<0.34.0` (was `<0.33.0`)
- vLLM 0.33 requires **Python 3.11+** (3.10 was dropped upstream in 0.32)
- vLLM 0.33 wheels target **CUDA 13.0** (driver compat for 12.8+) and **ROCm 10.0**

## Migration Guide

### For Users Upgrading from v0.9.0.x

1. Stop running servers, then upgrade the checkout: `git pull` (or download `vllm-cli-0.10.0.0.zip`)
2. Recreate the venv with Python 3.11+ if needed: `uv venv --python 3.11 && uv pip install -e .`
3. `--mamba-cache-mode <value>` in old profiles is now a no-op upstream — vllm-cli marks it deprecated; remove it from profiles to silence confusion
4. If you relied on Medusa/MLP speculative decoding, migrate to EAGLE/MTP/DFlash-style methods (`--speculative-model`, `--speculative-method`) — Medusa no longer exists in vLLM 0.33
5. Replace saved `--kv-cache-dtype turboquant_*` settings with the `octave_*` family (ROCm) or an fp8/nvfp4 variant
6. To adopt the watchdog: set `--watchdog-config` to JSON like `{"timeout":300,"check_interval":10,"dump_dir":"/tmp/vllm-stacks"}` (empty `dump_dir` disables)
7. Try the new parsers on supported checkpoints: `--tool-call-parser step5`, `--reasoning-parser muse`, etc.
8. On ROCm, use the `vllm/vllm:v0.33.0-rocm` (ROCm 10.0) image or wheel index, and try `--monokernel` with DeepSeek-V4.1-Flash / MiniMax-M3

## Files Changed

- `src/vllm_cli/schemas/argument_schema.json` — v2.10.0: 346 args, +4 new, 2 deprecated, choices refreshed
- `src/vllm_cli/config/cli_args_sync.py` — `SUPPORTED_VLLM_VERSIONS` + v0.33.0 (18 entries)
- `src/vllm_cli/config/parser_sync.py` — tool/reasoning name maps +`beam`, `muse`, `musespark1x`, `step5`
- `src/vllm_cli/config/command_import.py` — `BOOLEAN_FLAGS` + `--monokernel`
- `pyproject.toml` — version 0.10.0.0, `vllm>=0.20.0,<0.34.0`
- `VERSION`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES_v0.10.0.0.md`
