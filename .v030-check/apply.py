import json, re, ast

ROOT = '/Users/chrisqian/Development/vllm-cli-0.2.9/release/vllm-cli-0.2.9.4'
SP = ROOT + '/src/vllm_cli/schemas/argument_schema.json'
s = json.load(open(SP))
a = s['arguments']

tpl_bool = a['enable_batch_sharded_sampling']
tpl_int = a['prefix_cache_retention_interval']
tpl_ch = a['mamba_ssu_algorithm']
tpl_str = a['mamba_config']

def mk_bool(flag, desc, hint):
    e = dict(tpl_bool); e.pop('new_in', None)
    e.update({'description': desc, 'default': False, 'cli_flag': flag, 'new_in': 'v0.30.0', 'hint': hint})
    return e

def mk_int(flag, desc, dflt, hint):
    e = dict(tpl_int); e.pop('new_in', None)
    e.update({'description': desc, 'default': dflt, 'cli_flag': flag, 'new_in': 'v0.30.0', 'hint': hint})
    return e

def mk_choice(flag, desc, choices, hint):
    e = dict(tpl_ch); e.pop('new_in', None)
    e.update({'description': desc, 'default': None, 'cli_flag': flag, 'choices': choices, 'new_in': 'v0.30.0', 'hint': hint})
    return e

new_args = {
    'engram_config': dict(tpl_str, description='Engram N-gram lookup configuration (JSON string)', cli_flag='--engram-config', importance='low', added_in='v0.30.0'),
    'watermark_config': dict(tpl_str, description='KV cache watermark policy configuration (JSON string)', cli_flag='--watermark-config', importance='low', added_in='v0.30.0'),
    'dp_sync_interval': mk_int('--dp-sync-interval', 'Data parallel step sync interval (vLLM 0.30.0+)', 16, 'Steps between DP rank syncs (vLLM 0.30.0+)'),
    'elastic_ep_max_dp_size': mk_int('--elastic-ep-max-dp-size', 'Maximum data-parallel size for elastic EP scaling (vLLM 0.30.0+)', None, 'Used with elastic EP scaling (vLLM 0.30.0+)'),
    'enable_mamba_fine_grained_prefix_cache': mk_bool('--enable-mamba-fine-grained-prefix-cache', 'Enable fine-grained prefix caching for Mamba/hybrid models (vLLM 0.30.0+)', 'Fine-grained prefix cache for hybrid SSM models (vLLM 0.30.0+)'),
    'enable_nccl_comm_suspend': mk_bool('--enable-nccl-comm-suspend', 'Enable NCCL communicator suspend/resume support (vLLM 0.30.0+)', 'Suspend NCCL comms to save GPU memory when idle (vLLM 0.30.0+)'),
    'enable_scale_out': mk_bool('--enable-scale-out', 'Enable scale-out serving mode (vLLM 0.30.0+)', 'Large-scale serving expansion mode (vLLM 0.30.0+)'),
    'kda_decode_backend': mk_choice('--kda-decode-backend', 'KDA decode kernel backend (vLLM 0.30.0+)', ['auto', 'native', 'flashinfer', 'triton', None], 'KDA (Kimi Delta Attention) decode backend (vLLM 0.30.0+)'),
    'sparse_indexer_topk_backend': mk_choice('--sparse-indexer-topk-backend', 'Sparse indexer top-k backend (vLLM 0.30.0+)', ['auto', 'deep_select', 'cooperative', 'persistent', 'per_row', 'flashinfer', 'torch', None], 'DSA/NSA indexer top-k kernel backend (vLLM 0.30.0+)'),
}
assert all(k not in a for k in new_args), 'some new args already exist'
a.update(new_args)

def add_choices(label, new):
    ch = a[label]['choices']
    tail_null = ch[-1] is None
    core = sorted(set(c for c in ch if c is not None) | set(new))
    a[label]['choices'] = core + ([None] if tail_null else [])

add_choices('moe_backend', ['aiter_triton_mxfp4_bf16', 'rdna3'])
a['moe_backend']['hint'] = 'v0.30.0: added aiter_triton_mxfp4_bf16, rdna3. v0.29.0: added b12x, flashinfer_moe_ep_mega_*'
add_choices('spec_method', ['glm5_next_mtp'])
a['spec_method']['description'] = "Speculative decoding method. Convenience flag for --speculative-config['method']. Includes v0.30.0 glm5_next_mtp."

dep_info = {
    'default_max_num_batched_tokens': 'Removed in vLLM v0.30.0; use --max-num-batched-tokens or scheduler config overrides instead.',
    'enable_bf16x3_router_gemm': 'Removed in vLLM v0.30.0.',
}
for n, why in dep_info.items():
    a[n]['deprecated'] = True
    a[n]['deprecated_in'] = 'v0.30.0'
    a[n]['deprecation_reason'] = why

dep_count = sum(1 for v in a.values() if v.get('deprecated'))
s.update({
    'version': '2.7.0',
    'description': 'vLLM server argument definitions and metadata (supports v0.20.0 - v0.30.0)',
    'last_synced_version': 'v0.30.0',
    'deprecated_count': dep_count,
    'new_in_v030_count': len(new_args),
})
if '0.30.0' not in s['vllm_versions_supported']:
    s['vllm_versions_supported'].append('0.30.0')
json.dump(s, open(SP, 'w'), indent=2, ensure_ascii=False)
print(f"schema: {s['version']}  args: {len(a)}  deprecated: {dep_count}  moe: {len([c for c in a['moe_backend']['choices'] if c])}  spec: {len([c for c in a['spec_method']['choices'] if c])}")

# cli_args_sync.py
p = ROOT + '/src/vllm_cli/config/cli_args_sync.py'
src = open(p).read()
assert '"v0.30.0"' not in src
src = src.replace('SUPPORTED_VLLM_VERSIONS = [\n', 'SUPPORTED_VLLM_VERSIONS = [\n    "v0.30.0",\n', 1)
open(p, 'w').write(src)
print('cli_args_sync: v0.30.0 added')

# parser_sync.py
p = ROOT + '/src/vllm_cli/config/parser_sync.py'
src = open(p).read()
add30 = '    # v0.30.0 additions\n    "deepseek_v41": "deepseek_v41",\n    "k2_horizon": "k2_horizon",\n'
tool_anchor = '    # v0.29.0 additions\n    "hy_v4": "hy_v4",\n    "internlm2": "internlm",\n'
reason_anchor = '    # v0.29.0 additions\n    "hy_v4": "hy_v4",\n}'
assert tool_anchor in src and reason_anchor in src
src = src.replace(tool_anchor, tool_anchor + add30, 1)
src = src.replace(reason_anchor, '    # v0.29.0 additions\n    "hy_v4": "hy_v4",\n' + add30.rstrip('\n') + '\n}', 1)
open(p, 'w').write(src)
print('parser_sync: v0.30.0 mappings added (tool+reasoning)')

# VERSION + pyproject
open(ROOT + '/VERSION', 'w').write('0.6.0.0\n')
pp = open(ROOT + '/pyproject.toml').read()
assert 'version = "0.5.0.0"' in pp and 'vllm>=0.20.0,<0.30.0' in pp
pp = pp.replace('version = "0.5.0.0"', 'version = "0.6.0.0"').replace('vllm>=0.20.0,<0.30.0', 'vllm>=0.20.0,<0.31.0')
open(ROOT + '/pyproject.toml', 'w').write(pp)
print('VERSION -> 0.6.0.0; pyproject 0.6.0.0 + vllm<0.31.0')

ast.parse(open(ROOT + '/src/vllm_cli/config/parser_sync.py').read())
ast.parse(open(ROOT + '/src/vllm_cli/config/cli_args_sync.py').read())
print('py syntax OK')

# release digest for docs
body = open(ROOT + '/.v030-check/release_body.md').read()
def section(title):
    m = re.search(r'^## ' + title + r'\n(.*?)(?=^## |\Z)', body, re.S | re.M)
    return m.group(1).strip() if m else ''
for t in ['Highlights', 'Model Support', 'Breaking Changes & Deprecations']:
    print('\n===== %s =====\n%s' % (t, section(t)[:3200]))
