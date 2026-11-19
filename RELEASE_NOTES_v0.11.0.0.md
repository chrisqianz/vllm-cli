# vllm-cli v0.11.0.0 Release Notes (2026-11-19)

## Overview

vllm-cli v0.11.0.0 adds **full support for vLLM v0.34.0** (1,061 commits from 412 contributors, 120 new). The argument schema is synced against `vllm/engine/arg_utils.py`, `vllm/entrypoints/common/cli_args.py` and the vLLM config dataclasses at the v0.34.0 tag.

- Schema: **v2.10.0 → v2.11.0** — 352 arguments (6 new), 19 supported vLLM versions (0.20.0 – 0.34.0)
- Dependency range: `vllm>=0.20.0,<0.35.0` (was `<0.34.0`)
- This is the routine upstream-sync release; no vllm-cli feature changes.

## Highlights (vLLM 0.34.0)

- **Extensible KV cache, on by default** (`--enable-extensible-kv-cache`, #56492): the KV cache is sized from memory actually free after warmup instead of a fixed `gpu_memory_utilization` slice. MiniMax-M3 on H200 goes from 433k to 672k cached tokens. vLLM now recommends *removing* `gpu_memory_utilization` from most configs; it defaults to 1.0 and remains useful for GPU sharing or when the feature is disabled. Trade-off: TPOT can rise 6–18% on some H200 models at concurrency 128 — opt out with `--no-enable-extensible-kv-cache` or pin `--gpu-memory-utilization 0.9`.
- **Model Runner V1 removed** (#63373): MRV2 is the only GPU runner. `VLLM_USE_V2_MODEL_RUNNER=0` fails at startup; custom logits processors must port to `vllm.v1.worker.gpu.sample.logits_processor`.
- **Performance**: up to 1.95× decode (Step5 on GB200), 1.6× throughput (GLM-5.3-Flash), 1.24–1.40× faster long-prompt TTFT (GLM-5.3); breakable CUDA graphs on Qwen3.5 cut decode ITL 3–8.8% (#51012); token-sharded sparse-MLA prefill above 1024 prefill tokens (#63956).
- **Speculative decoding on MRV2**: suffix decoding with a GPU speculator, no Arctic Inference needed (#60339); declarative custom-speculator configs + `register_speculative_method` plugin API (#63205); CUDA graphs for MRV2 draft models (#60989); dual-batch overlap with spec decode and DCP (#61464).
- **Serving & operations**: CRIU + `cuda-checkpoint` checkpoint/restore via `/checkpoint/prepare|status|restore` in both frontends (#62570); fault-tolerant DP+EP scale-down without restart (#46370); opt-in cache-aware admission ordering (#54625).
- **AMD ROCm**: release images/wheels are ROCm 10.0 only (#61939); MoRI EPv2 via `--all2all-backend mori` (#61361); HIP GDN prefill backend for gfx1151 (#56307); TRITON_ATTN default on RDNA3/RDNA4 (#60418); opt-in W4A16 dense prefill cache cuts TTFT 22–26% on Strix Halo.
- **10 new models**: Mistral Large 4 (1T total / 52B active, FP8 + NVFP4, #64534), MiniMax-M3.1-Flash (#64111), Nemotron-3.5-Transcribe (#64282), Ming3-ASR (#63585), and more.

## Schema Changes (v2.10.0 → v2.11.0)

### New arguments (6, marked `new_in: v0.34.0`)

| Argument | Type | Category | Default | Notes |
|---|---|---|---|---|
| `--cache-aware-admission-window` | integer | performance | 0 | Steps window for prefix-hit-ratio-ordered admission; 0 disables (#54625) |
| `--cache-aware-admission-threshold` | float | performance | 0.5 | Hit-ratio threshold (0–1) for earlier admission; needs window > 0 |
| `--max-num-inflight-kv-loads` | integer | performance | — | Async KV-cache loads in flight while batch is full (KV-connector tuning) |
| `--checkpoint-prepare-on-startup` | boolean | optimization | false | Pre-arm CRIU + cuda-checkpoint allocator; requires `--enable-sleep-mode` (#62570) |
| `--enable-extensible-kv-cache` | boolean | optimization | true | **On by default in vLLM 0.34.0** — consider removing `gpu_memory_utilization` (#56492) |
| `--shuffle-safetensors-files` | boolean | model | false | Shuffle shard read order to spread I/O across ranks |

`--headless-client-handshake-address` is registered with `help=SUPPRESS` (internal) and intentionally has no schema entry, as before.

### Expanded choices

- `--tool-call-parser`: 58 → **62** — `+granite_toolcall_parser`, `+ripple`, `+zaya`, `+zaya_xml`; `musespark1x` renamed upstream to `musespark`
- `--reasoning-parser`: 42 → **44** — `+ripple`, `+zaya`; `musespark1x` → `musespark`
- `--tokenizer-mode`: `+glm5_next`
- `--all2all-backend`: `+mori` (MoRI EPv2, #61361)
- `--gdn-prefill-backend`: `+hip` (gfx1151, #56307)

### No new deprecations this cycle

No flags were newly deprecated this cycle; total deprecated count unchanged at 121. (`--ir-op-priority` still parses in 0.34 but remains marked for removal — vLLM deferred the #63371 removal.)

## Breaking changes in vLLM v0.34.0 (upstream)

- **Default `--block-size` is 64** (was 16) (#63231). Pass `--block-size 16` to keep old behavior; ROCm paged attention still uses 16.
- **Model Runner V1 removed** (#63373). Removed with it: the legacy logits-processor API, the `custom_class` speculative method, `use_heterogeneous_vocab`, `disable_padded_drafter_batch`, `parallel_drafting` outside DFlash/DSpark, PP with `external_launcher`, `NGramPerReqLogitsProcessor`, and x86 CPUs without Triton.
- **`gpu_memory_utilization` defaults to 1.0** with extensible KV cache on; vLLM recommends leaving it unset for most deployments (#56492).
- **ROCm 7.2 artifacts dropped** — ROCm 10.0 wheels/images only (#61939).
- **`huggingface_hub >= 2.0.0` required.**
- Best-effort NUMA binding on by default (#61258); `VLLM_COMPILE_ON_ONE_RANK` default (#63247).

## Dependencies

- `vllm>=0.20.0,<0.35.0`
- Python 3.11+ (unchanged; vLLM dropped 3.10 in 0.32)

## Migration guide

1. `cd ~/vllm-cli-0.11.0.0 && uv venv && uv pip install -e .`
2. In profiles that pin `gpu_memory_utilization`: for vLLM ≥ 0.34 deployments, consider deleting the key to get maximum KV cache capacity; keep it only when sharing a GPU or when disabling extensible KV cache.
3. If you rely on 16-token blocks (e.g. certain KV connectors), set `--block-size 16` explicitly.
4. If you use custom logits processors or the `custom_class` speculative method, port them to the MRV2 interfaces before upgrading.
5. ROCm users: switch to ROCm 10.0 wheels/images.
6. Try `--cache-aware-admission-window 32` (with `--cache-aware-admission-threshold`) on prefix-heavy workloads, and `--checkpoint-prepare-on-startup` if you use sleep mode with CRIU flows.
7. Import a real command to verify coverage: `vllm-cli import --command "vllm <model> --enable-extensible-kv-cache --all2all-backend mori ..."`

## Files Changed

- `src/vllm_cli/schemas/argument_schema.json` — 2.11.0: 6 new entries, parser/tokenizer/all2all/gdn choices, meta sync
- `src/vllm_cli/config/cli_args_sync.py` — `SUPPORTED_VLLM_VERSIONS` prepend `v0.34.0` (19 entries)
- `src/vllm_cli/config/parser_sync.py` — tool/reasoning name maps for 0.34 parsers; `musespark1x` removed
- `src/vllm_cli/config/command_import.py` — `BOOLEAN_FLAGS` for the 3 new valueless flags; unknown-flag value parsing guarded
- `pyproject.toml`, `VERSION`, `CHANGELOG.md`, `README.md`, `RELEASE_NOTES_v0.11.0.0.md`
