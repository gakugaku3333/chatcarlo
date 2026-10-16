"""P5j source and generated-data identity. Tools are deliberately not fingerprinted."""
import json
import re
from pathlib import Path
from . import p5b_core as core, p5g_core as g, p5i_core as i
from . import p5j as tables
SOURCES = ('p5j_logs.h', 'p5j_terms.h', 'p5j_kernel.h', 'p5j_cpu.cpp', 'p5j_gpu.cu')
KERNEL_SOURCES = (*(n for n in core.FROZEN if not n.endswith('.py')), 'p5b_wrap.h', *g.SOURCES, *i.SOURCES, *SOURCES)
KERNELS = ('p5j_run_kernel', 'p5j_per_history_kernel', 'p5j_audit_kernel',
           *core.KERNELS[2:], 'p5j_gate0_kernel', 'p5j_initialize_kernel', 'p5j_diagnostic_kernel', 'p5j_initialization_diagnostic_kernel', 'p5j_contract_kernel')
exact = g.exact


def source(mapped=False):
    core.fingerprints()
    names = ('p5a0_common.h', 'p5_rng.h', 'p5_kernel.h', 'p5b_wrap.h',
             'p5i_logs.h', 'p5i_kernel.h', 'p5j_logs.h', 'p5j_terms.h', 'p5j_kernel.h')
    chunks = []
    for name in names:
        text = (core.HERE/name).read_text()
        text = re.sub(r'^\s*#include "(?:'+'|'.join(map(re.escape, names))+r')"[^\n]*$',
                      '', text, flags=re.MULTILINE)
        chunks.append((f'#line 1 "{name}"\n' if mapped else '// '+name+'\n')+text+'\n')
    return (core.HERE/'p5j_gpu.cu').read_text().replace('//@@COMMON@@', ''.join(chunks))


def identity(data, environment=None, launches=None, options=core.OPTIONS+('-DP5J_CHECK=0',), generated=None):
    arrays = tables.layout(data)
    return dict(source_sha256=core.sha(source().encode()),
                sources={k:core.file_sha(core.HERE/k) for k in KERNEL_SOURCES},
                generated=generated, layout=core.describe(arrays), inputs=core.describe(data),
                options=list(options), cpu_options=['-O2','-ffp-contract=off','-std=c++11',
                    '-shared','-fPIC','-DP5_RNG_PHILOX','-DP5_TRANSPORT_FAULT=0',
                    '-DP5_STACK_CAP=64'], check_defines=['-DP5J_CHECK=0','-DP5J_CHECK=1'],
                environment=environment, launches=launches,
                seed=core.SEED, per_batch=100000, batches=100,nspec=20)


TERM_SITES=('A','B_log','B_log1p','C_constant','C_fallback',
            'D_rad_reuse','D_rad_fallback','D_u_reuse','D_u_fallback',
            'D_density_reuse','D_density_fallback')


def counters(values,require_coverage=True,gate0=False):
    import numpy as np
    v=np.asarray(values).reshape(20,2)
    required=list(range(11)) if gate0 else [0,1,2,3,5,7,9]
    good=not np.any(v[:,1]) and (not require_coverage or all(v[9+j,0]>0 for j in required))
    return core.verdict(bool(good),sites={n:int(v[9+j,0]) for j,n in enumerate(TERM_SITES)})


def require_gate0(report,checked):
    if (report.get('status')!='pass' or report.get('randoms')!=1000000
        or set(report.get('checks',{}))!={'0','1','2','3','4','5','8','contract'}):
        raise core.Stop('complete current gate0 required')
    for value in report['checks'].values():core.require(value,'gate0 output')
    core.require(report.get('counters',core.verdict(False)),'gate0 counters')
    if checked and (set(report['counters'].get('sites',{}))!=set(TERM_SITES)
                    or not all(v>0 for v in report['counters']['sites'].values())):
        raise core.Stop('gate0 fallback coverage required')


def require_gate1(path,data):
    r=json.loads(Path(path).read_text());ident=r.get('identity',{})
    if (r.get('status')!='pass' or ident!=identity(data,generated=ident.get('generated'))
        or not ident.get('generated') or r.get('batches')!=100
        or set(r.get('checks',{}))!={str(i) for i in range(100)}):
        raise core.Stop('current P5j 100-batch gate1 required before GPU')
    for label in ('ordinary','checked'):
        g0=r.get('gate0',{}).get(label,{})
        require_gate0(g0,label=='checked')
        if g0.get('identity')!=ident:
            raise core.Stop('current CPU gate0 required')
    for v in r['checks'].values():
        for key in ('ordinary','checked','counters'):
            core.require(v.get(key,core.verdict(False)),'gate1 '+key)
        for key in ('ordinary','checked'):
            if v[key].get('fields')!={k:True for k in core.RUN_ORDER[:-1]}:
                raise core.Stop('incomplete gate1 outputs')
    return r
