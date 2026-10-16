"""P5i source and generated-data identity. Tools are deliberately not fingerprinted."""
import json
import re
from pathlib import Path
from . import p5b_core as core, p5g_core as g
from . import p5i as tables
SOURCES = ('p5i_logs.h', 'p5i_kernel.h', 'p5i_cpu.cpp', 'p5i_gpu.cu')
KERNEL_SOURCES = (*(n for n in core.FROZEN if not n.endswith('.py')), 'p5b_wrap.h', *g.SOURCES, *SOURCES)
KERNELS = ('p5i_run_kernel', 'p5i_per_history_kernel', 'p5i_audit_kernel',
           *core.KERNELS[2:], 'p5i_gate0_kernel', 'p5i_initialize_kernel', 'p5i_diagnostic_kernel', 'p5i_initialization_diagnostic_kernel')
exact = g.exact


def source(mapped=False):
    core.fingerprints()
    names = ('p5a0_common.h', 'p5_rng.h', 'p5_kernel.h', 'p5b_wrap.h',
             'p5i_logs.h', 'p5i_kernel.h')
    chunks = []
    for name in names:
        text = (core.HERE/name).read_text()
        text = re.sub(r'^\s*#include "(?:'+'|'.join(map(re.escape, names))+r')"[^\n]*$',
                      '', text, flags=re.MULTILINE)
        chunks.append((f'#line 1 "{name}"\n' if mapped else '// '+name+'\n')+text+'\n')
    return (core.HERE/'p5i_gpu.cu').read_text().replace('//@@COMMON@@', ''.join(chunks))


def identity(data, environment=None, launches=None, options=core.OPTIONS+('-DP5I_CHECK=0',)):
    arrays = tables.layout(data)
    return dict(source_sha256=core.sha(source().encode()),
                sources={k:core.file_sha(core.HERE/k) for k in KERNEL_SOURCES},
                generated=core.describe(arrays), inputs=core.describe(data),
                options=list(options), cpu_options=['-O2','-ffp-contract=off','-std=c++11',
                    '-shared','-fPIC','-DP5_RNG_PHILOX','-DP5_TRANSPORT_FAULT=0',
                    '-DP5_STACK_CAP=64'], check_defines=['-DP5I_CHECK=0','-DP5I_CHECK=1'],
                environment=environment, launches=launches,
                seed=core.SEED, per_batch=100000, batches=100,nspec=20)


def counters(values, require_coverage=True):
    import numpy as np
    v=np.asarray(values).reshape(9,2)
    return core.verdict(bool(not np.any(v[:,1]) and
        (not require_coverage or np.all(v[:,0]>0))),
        sites={name:dict(calls=int(row[0]),sentinels=int(row[1]))
            for name,row in zip(tables.SITES,v)})


def require_gate1(path, data):
    r = json.loads(Path(path).read_text())
    if (r.get('status') != 'pass' or r.get('identity') != identity(data)
        or r.get('batches') != 100 or set(r.get('checks', {})) != {str(i) for i in range(100)}):
        raise core.Stop('current P5i 100-batch gate1 required before GPU')
    for v in r['checks'].values():
        for key in ('ordinary','checked','counters'):
            core.require(v.get(key,core.verdict(False)), 'gate1 '+key)
        for key in ('ordinary','checked'):
            if v[key].get('fields') != {k:True for k in core.RUN_ORDER[:-1]}:
                raise core.Stop('incomplete gate1 outputs')
    return r
