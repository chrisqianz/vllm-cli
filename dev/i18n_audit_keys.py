#!/usr/bin/env python3
"""Find every t()/tr() key used in code that is MISSING from zh.json."""
import ast, json, os, sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "vllm_cli")
TR = os.path.join(SRC, "i18n/translations")

def flat(o, p=""):
    r = {}
    for k, v in o.items():
        kk = f"{p}.{k}" if p else k
        if isinstance(v, str): r[kk] = v
        else: r.update(flat(v, kk))
    return r

zh = flat(json.load(open(os.path.join(TR, "zh.json"))))
en = flat(json.load(open(os.path.join(TR, "en.json"))))

missing = {}  # key -> (file, default_str)
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if not f.endswith(".py"): continue
        path = os.path.join(root, f)
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name not in ("t", "tr"): continue
                if not node.args: continue
                a0 = node.args[0]
                if not (isinstance(a0, ast.Constant) and isinstance(a0.value, str)): continue
                key = a0.value
                # skip non-i18n t-like calls (dict.get etc. are attribute-different; accept noise)
                if "." not in key and len(key.split()) > 3: continue
                default = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    default = node.args[1].value
                if key not in zh:
                    if key not in missing:
                        missing[key] = (os.path.relpath(path, SRC), default)

print(f"keys used in code but MISSING from zh.json: {len(missing)}")
byfile = {}
for k, (f, d) in sorted(missing.items()):
    byfile.setdefault(f, []).append((k, d))
for f, ks in sorted(byfile.items(), key=lambda x: -len(x[1])):
    print(f"\n== {f} ({len(ks)}) ==")
    for k, d in ks[:40]:
        dd = (str(d)[:70].replace(chr(10)," ") if d is not None else "?")
        print(f"  {k}  |  {dd}")
# also keys missing from en
miss_en = [k for k in missing if k not in en]
print(f"\n(also missing from en.json: {len(miss_en)})")
