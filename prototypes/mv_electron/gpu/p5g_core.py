"""P5f-1 NumPy-only source, identity and bitwise gate contracts."""
import json
import re
import numpy as np
from . import p5b_core as core

# Retain the gate1 writer's SOURCES contract without fingerprinting tools.
SOURCES = ('p5g_kernel.h', 'p5g_cpu.cpp', 'p5g_gpu.cu')
KERNEL_SOURCES = (*core.FROZEN, 'p5b_wrap.h', *SOURCES)
STATES = ('P', 'C', 'E', 'H', 'D')
TRACE_COLUMNS = ('from', 'to', 'steps', 'calls', 'top', 'charge', 'energy',
                 'reason', 'hard_count', 'brems_count')
KERNELS = ('p5g_run_kernel', 'p5g_per_history_kernel', 'p5g_audit_kernel', *core.KERNELS[2:])
STATE_SPEC = dict(states=STATES, order=STATES[:4], block=128, ownership='lane private by pointer',
                  masks='all snapshots before transitions; full warp including D',
                  stack=64, coarse=150, fine=600, trace_columns=TRACE_COLUMNS,
                  cache='reset at P charged pop; retained at C to C',
                  rng='Philox seed/history/calls', output='once after H')


def source(mapped=False):
    """Flatten read-only helpers and new transitions; optionally retain #line mapping."""
    core.fingerprints()
    names = ('p5a0_common.h', 'p5_rng.h', 'p5_kernel.h', 'p5b_wrap.h', 'p5g_kernel.h')
    chunks = []
    for name in names:
        text = (core.HERE/name).read_text()
        # Keep line count when stripping local includes.
        text = re.sub(r'^\s*#include "(?:'+'|'.join(map(re.escape, names))+r')"[^\n]*$',
                      '', text, flags=re.MULTILINE)
        chunks.append((f'#line 1 "{name}"\n' if mapped else '// '+name+'\n')+text+'\n')
    return (core.HERE/'p5g_gpu.cu').read_text().replace('//@@COMMON@@', ''.join(chunks))


def identity(manifest, environment, launches, options=core.OPTIONS):
    return dict(source_sha256=core.sha(source().encode()),
        sources={k:core.file_sha(core.HERE/k) for k in KERNEL_SOURCES},
        entries=list(KERNELS)+['p5g_run_kernel_fault'], options=list(options),
        state=json.loads(json.dumps(STATE_SPEC)), launches=launches, environment=environment,
        bundle_sha256={k:v['sha256'] for k,v in manifest['files'].items()},
        seed=core.SEED, per_batch=100000, batches=100, nspec=20)


def exact(actual, expected, keys=None):
    try:
        keys = tuple(expected) if keys is None else keys
        checks = {k:bool(np.isfinite(actual[k]).all() and np.isfinite(expected[k]).all()
                         and core.bits(actual[k], expected[k])) for k in keys}
        return core.verdict(all(checks.values()), fields=checks)
    except (KeyError, ValueError, TypeError, AttributeError, core.Stop):
        return core.verdict(False)


def trace_check(trace, meta, stored=False):
    used, overflow = map(int, meta)
    if (trace.ndim != 2 or trace.shape[1] != 10 or overflow != 0 or used > len(trace)
        or used < 3 or not np.isfinite(trace[:used]).all()):
        return core.verdict(False)
    v = trace[:used]
    pairs = v[:, :2].astype(int)
    allowed = {(-1,0),(0,0),(0,1),(0,3),(1,1),(1,2),(1,3),
               (2,0),(2,3),(3,4),(5,4)}
    ok = (all(tuple(x) in allowed for x in pairs)
          and np.count_nonzero(pairs[:,0] == -1) == 1
          and np.count_nonzero(pairs[:,0] == 3) == 1
          and np.count_nonzero(pairs[:,0] == 5) == int(stored)
          and np.all(np.diff(v[:,2]) >= 0) and np.all(np.diff(v[:,3]) >= 0)
          and tuple(pairs[0]) == (-1,0)
          and tuple(pairs[-1]) == ((5,4) if stored else (3,4)))
    for i in range(1, len(pairs)):
        if pairs[i,0] != 5: ok &= pairs[i,0] == pairs[i-1,1]
    return core.verdict(bool(ok), used=used)


def require_gate1(path):
    r = json.loads(path.read_text())
    names=KERNEL_SOURCES
    hashes={k:core.file_sha(core.HERE/k) for k in names}
    if (r.get('status') != 'pass' or r.get('batches') != 100 or r.get('sources') != hashes
        or r.get('outputs') != list(core.RUN_ORDER[:-1])
        or set(r.get('checks',{})) != {str(b) for b in range(100)}
        or any(v.get('status')!='pass' or v.get('fields')!={k:True for k in core.RUN_ORDER[:-1]}
               for v in r.get('checks',{}).values())):
        raise core.Stop('gate 1 missing, failed or source changed; do not proceed to GPU')
    return r


def weighted_map(records):
    """SASS rows are primary; source-line means are ratios of sums, never row means."""
    groups = {}
    for r in records:
        for kind in ('all', 'fp64'):
            if kind == 'fp64' and not re.search(r'\bD(?:ADD|MUL|FMA|DIV|SQRT|SET|SETP|MNMX|RCP)\b',r['instruction']):
                continue
            key = (r['function'], r['source'], r['line'], kind)
            v = groups.setdefault(key, [0.,0.,0.])
            for i,k in enumerate(('warp','thread','predicated')): v[i] += r[k]
    return [dict(function=k[0],source=k[1],line=k[2],kind=k[3],warp=v[0],thread=v[1],
                 predicated=v[2],lanes=v[1]/v[0] if v[0] else None,
                 predicated_lanes=v[2]/v[0] if v[0] else None) for k,v in groups.items()]


def trace_events(v):
    """Evidence from the recorded transition itself, including resumed parent identity."""
    pairs=v[:,:2].astype(int);reason=v[:,7].astype(int);seen=set()
    if np.any((pairs[:,0]==1)&(pairs[:,1]==2)&((reason&4)>0)):seen.add('flight')
    if np.any((pairs[:,0]==1)&(pairs[:,1]==2)&((reason&8)>0)&(v[:,5]==1)):seen.add('escape')
    if np.any((pairs[:,0]==2)&(pairs[:,1]==0)&(v[:,5]==1)&(v[:,6]==0)&((reason&6)==0)):seen.add('rest')
    for i in range(1,len(v)):
        if tuple(pairs[i])!=(1,2):continue
        kind='brems' if reason[i]&2 else ('hard' if v[i,8]>v[i-1,8] else None)
        if kind is None:continue
        seen.add(kind)
        # A suspension pushes parent then child. When the parent is popped again,
        # its kinetic energy is unchanged and top is two below its suspension top.
        later=v[i+1:]
        resumed=((later[:,0]==0)&(later[:,1]==1)&(later[:,4]==v[i,4]-2)&(later[:,6]==v[i,6]))
        if np.any(resumed):seen.add(kind+'_resume');seen.add('reentry')
    return seen
