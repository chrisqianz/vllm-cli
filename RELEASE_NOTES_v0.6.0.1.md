# vllm-cli v0.6.0.1 — Release Notes

**Release date:** 2026-09-23
**Type:** Patch release (bug fixes)
**vLLM support:** unchanged — vLLM 0.20.0 → 0.30.0 (schema 2.7.0, 329 arguments)

---

## 🐛 Fixed

### Multi-Model Proxy menu was non-functional in every locale

Since i18n was introduced in **v0.2.9.4**, the proxy screens rendered their options through
translation files while the handlers compared the returned selection against hard-coded
English strings. The comparison never matched, so selecting an option silently did nothing:

- **Main proxy menu** — `配置新代理` (and even the English `Configure New Proxy` from
  `en.json`, which differs in letter case from the compared `"Configure new proxy"`)
  never entered the configuration wizard. This is the direct cause of the reported
  "多模型代理 doesn't work" symptom: the entry point itself dead-ends.
- **Running-proxy screen** — in any non-English locale, all five options
  (`监控代理日志`, `监控模型日志`, `管理模型`, `刷新模型注册表`, `停止所有服务器`)
  were dead branches; only `返回 (BACK)` worked, which made the screen look frozen.

All option comparisons now use the exact translated labels produced for display.

### Settings → Manage Proxy Configurations crashed

`ui/settings.py` still imported from `vllm_cli.ui.proxy_control`, a module that was
relocated to `vllm_cli/ui/proxy/control.py`. Opening the screen raised
`ModuleNotFoundError` (observed in real user logs as far back as 2025-11). Import fixed.

### Language context propagation

`Return to Proxy Monitoring` from the main menu now passes the i18n manager into the
proxy management screens, so they no longer fall back to English mid-session.

## ✨ Changed

- **Multi-Model Proxy promoted from experimental**: the menu label drops the
  `(实验性)` / `(Exp)` marker — now simply `多模型代理` / `Multi-Model Proxy`.

## ✅ Verification

Functional tests were run with the real `zh.json` translations driving simulated
menu selections (locale = zh):

1. Main proxy menu `配置新代理` → configuration wizard entered ✔
2. Running-proxy `刷新模型注册表` → registry refresh executed ✔
3. Model management `添加新模型` → add-model handler executed ✔
4. Settings `管理代理配置` → opens without `ModuleNotFoundError` ✔
5. Experimental marker removed from zh/en/fallback labels ✔

## 📦 Upgrade

```bash
# standalone directory, create your own venv
cd ~/vllm-cli-0.6.0.1
uv venv && source .venv/bin/activate
uv pip install -e .
```

No schema, profile, or configuration migrations are required; existing saved
profiles and proxy configurations are untouched.
