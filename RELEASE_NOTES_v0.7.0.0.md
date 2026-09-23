# vllm-cli v0.7.0.0 — Release Notes

**Release date:** 2026-09-23
**Type:** Feature release (full Chinese localization)
**vLLM support:** unchanged — vLLM 0.20.0 → 0.30.0 (schema 2.7.0, 329 arguments)

---

## 🀄 Full Chinese Localization

Before this release, only ~12% of user-facing strings actually went through the i18n layer —
the main menu, settings labels, and part of the welcome screen were translated, while roughly
700 strings inside the model manager, profiles, custom configurations, model directories,
shortcuts, server monitor, log viewer, proxy wizard/monitor, recipes sync, system info and CLI
handlers were hard-coded English. Selecting 中文 changed the storefront but not the product.

v0.7.0.0 localizes the entire interactive surface.

### What changed

| Metric | Before | After |
|---|---|---|
| Translation keys (en) | 420 | 1,873 |
| Visible-string coverage (UI layer) | 12% | 91% |
| Visible-string coverage (incl. CLI messages) | ~12% | ~99% |

The residual hard-coded strings are deliberate: dynamic data (model names, paths, profile
names, server IDs) is shown verbatim and must not be translated.

### Modules now fully localized

`ui/settings` · `ui/model_manager` · `ui/shortcuts` · `ui/profiles` · `ui/model_directories` ·
`ui/custom_config` · `ui/system_info` · `ui/hf_integration` · `ui/welcome` · `ui/server_control` ·
`ui/server_monitor` · `ui/log_viewer` · `ui/recipes_sync` · `ui/proxy/control` ·
`ui/proxy/menu` · `ui/proxy/monitor` · `cli/handlers`

## 🧱 New i18n architecture

- **`tr(key, english_default, **kwargs)`** — process-wide lookup via a manager singleton
  (registered automatically when `I18nManager` is constructed). When no manager exists or a key
  is missing, the embedded English default is returned, so CLI/headless execution and any future
  untranslated path behave exactly like before. Dynamic values are passed as format arguments,
  never interpolated into keys.
- **`prompt_choice(key, message, [(value, label), …])`** — option prompts now display translated
  labels but return **stable ASCII action keys**. A menu option can no longer die because its
  displayed label drifted from the string it is compared against — the exact bug class that
  silently killed the Multi-Model Proxy menu in every locale from v0.2.9.4 to v0.6.0.0
  (fixed surgically in v0.6.0.1; now structurally impossible in migrated menus).
  31 option prompts across profiles, model manager, proxy and shortcuts were migrated.

## 🐛 Fixed along the way

- The running-proxy action menu re-looped forever when the prompt was cancelled
  (Ctrl+C / EOF returns `None`, which matched no branch). It now exits like **Back**.

## 🛡️ Anti-regression machinery (new with this release)

- **Static checks** (run in CI-style, exit non-zero on violation):
  1. no f-string/concatenation may appear inside a `tr()` key or its format-value args;
  2. no function that displays a translated option list may compare the selection against a
     bare English literal unless the value is a declared `prompt_choice` action key.
- **Functional suite** (9 cases) exercises zh and en dispatch through the real menu code with a
  scripted prompt: proxy wizard entry (zh & en), running-proxy monitor/refresh actions,
  Settings → Manage Proxy Configurations, `prompt_choice` label→key mapping under zh,
  tr() fallback without a manager, and en/zh key parity — plus the v0.6.0.1 proxy regression
  suite (5 cases). All 14 pass.

## Compatibility

- No config, profile, or schema format changes. Existing `~/.vllm-cli*` files are untouched.
- English behavior is byte-identical where translations resolve to the same English text;
  `en.json` gained the same 1,453 keys (English values equal the former hard-coded strings).
