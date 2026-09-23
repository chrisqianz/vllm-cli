# vllm-cli v0.7.0.1 — Release Notes

**Release date:** 2026-09-23
**Type:** Patch release (i18n gaps found in field testing)
**vLLM support:** unchanged — vLLM 0.20.0 → 0.30.0 (schema 2.7.0, 329 arguments)

---

## 🐛 Fixed

### "Back" option was never translated
`unified_prompt` appended a hard-coded `← Back` entry to every menu. It now renders
`← 返回` in Chinese (`messages.back`), while still returning the internal `BACK`
sentinel to callers — no caller changes required.

### Model Management submenu was fully English
`handle_model_management` builds its options through a legacy local translator bound
to `menu.model_management.*` keys, which never existed in the translation files
(pre-v0.7.0.0 blind spot: the rollout only audited `tr()` call sites, not the older
`t()` closures). The whole submenu therefore fell back to English. All six keys, plus
`menu.profiles.title` and `settings.env_vars`, were added to `en.json`/`zh.json`:

| Key | en | zh |
|---|---|---|
| `menu.model_management.title` | Model Management | 模型管理 |
| `menu.model_management.open_tool` | Open Model Management Tool | 打开模型管理工具 |
| `menu.model_management.list_models` | List All Models | 列出所有模型 |
| `menu.model_management.manage_assets` | Manage Assets | 管理模型资产 |
| `menu.model_management.view_details` | View Model Details | 查看模型详情 |
| `menu.model_management.refresh_cache` | Refresh Model Cache | 刷新模型缓存 |
| `menu.profiles.title` | Configuration Profiles | 配置文件 |
| `settings.env_vars` | environment variable(s) | 个环境变量 |

## 🛡️ New audit closing the blind spot

A repo-wide AST scan (`t()` **and** `tr()` call sites × zh.json/en.json key sets)
now runs as part of the release checks. Current result: **0 keys referenced by code
but missing from the translation files**. This makes the class of "silently
untranslated menu" bugs structurally detectable instead of user-discoverable.

## Verification

- zh/en functional dispatch suite: 9/9
- v0.6.0.1 proxy regression suite: 5/5
- Rendered-menu probe: choices now show `['A选项', 'B选项', '← 返回']` and the back
  selection still resolves to the internal `BACK` action.
