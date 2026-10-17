"""P5k contracts. Gate identities exclude launch/measurement Python tools."""
import json
import re
from pathlib import Path
from . import p5b_core as core, p5j_core as j
VARIANTS=('F','Q1','Q4','Q8','Q1-512')
SOURCES=('p5k_driver.h','p5k_cpu.cpp','p5k_gpu.cu')
FROZEN={'p5_cpu.cpp': '11fd05ed68100db0b3bc02231c659f86b72b7403ae17fc88085d91791e9a7f81', 'p5_kernel.h': '39d82e04f5c969c92c1ee35d026241119364134fecf1db801ff5d0a4a44bd549', 'p5_rng.h': '6c3257588ee4be2d0881a05f5d0869b65b942f68fd596a9f0ea981bcaaa9760f', 'p5a0_common.h': '8c209cfe016241bbbb5d586830dfe5a1ddff648f8f75d3b431dd2ae6d61e70c2', 'p5b_wrap.h': '249492b5fac400e819c26ce71e271c1365381b2de82739962b93c52899912edc', 'p5g_kernel.h': 'd9d13e496028efc97021620ddfdf4b22e400804aae8eb70344c3bb03c7026126', 'p5g_cpu.cpp': '3f6a420b27ea57c7c2ca3de990c123ef4ece9a6a8e3d23ab0cb9b9627bc2a60b', 'p5g_gpu.cu': 'de9a30a3017389ff83df80bcc62024c143f2bbaa74043cfafe89a5534a11eba2', 'p5i_logs.h': 'ca6e1f6ac016a30bf7579d88df62d3da2319f2cb45bff2eeacb7ef1843025e7b', 'p5i_kernel.h': '18dc844f2eae05dccda7fabac912ce3701dcc5aee08262ac12b8a0f94a209c0a', 'p5i_cpu.cpp': '534bfff6516c48ea938e79c281c879d7b5acc8bdd61d4efa7964ae9267cface5', 'p5i_gpu.cu': 'ee308a504fa924d7c1c0328578bafa41579f91d6543089f467b2734288247385', 'p5j_logs.h': '8e04b1c4be2b82d29e2e5d6dddd3b684f14b99c32454bf8e617c23adff4d7c12', 'p5j_terms.h': '2e7800cbd39a2dcffe3bfc53f7992859e10485551789bca994928573ff424e82', 'p5j_kernel.h': '7644fecf498030eca653e9cabd5317269e6c572bd1e08d9e09c1d02d22aabd38', 'p5j_cpu.cpp': 'd76326f75049c83905657fc276728ba13f22966ce61e023804148bf82134b58b', 'p5j_gpu.cu': '6a5a11603465f9707f8e6c81f75776544d1ebb47b35bdd91dc556ac1008ea52f'}


def frozen():
    core.fingerprints()
    if any(core.file_sha(core.HERE/name)!=digest for name,digest in FROZEN.items()):
        raise core.Stop('frozen P5j/dependency source changed')


def source(mapped=False):
    frozen()
    text=j.source(mapped)
    header=(core.HERE/'p5k_driver.h').read_text()
    header=re.sub(r'^#include "p5[^\n]+\n','',header,flags=re.M)
    return text+'\n'+('#line 1 "p5k_driver.h"\n' if mapped else '')+header+'\n'+(core.HERE/'p5k_gpu.cu').read_text()


def identity(data,generated=None):
    frozen()
    return dict(sources={n:core.file_sha(core.HERE/n) for n in (*j.KERNEL_SOURCES,*SOURCES)},
        inputs=core.describe(data),generated=generated,nspec=20,seed=core.SEED,
        variants=list(VARIANTS),options=list(core.OPTIONS),checks=[0,1],
        cpu_options=['-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC',
                     '-DP5_RNG_PHILOX','-DP5_TRANSPORT_FAULT=0','-DP5_STACK_CAP=64'])


def require_gate1(path,data):
    r=json.loads(Path(path).read_text());ident=r.get('identity',{})
    if (r.get('status')!='pass' or r.get('batches')!=100 or
        ident!=identity(data,ident.get('generated')) or not ident.get('generated') or
        set(r.get('checks',{}))!={str(b) for b in range(100)}):
        raise core.Stop('current complete P5k 100-batch gate1 required')
    for label in ('ordinary','checked'):
        gate=r.get('gate0',{}).get(label,{})
        for key in ('management','transport'):core.require(gate.get(key,core.verdict(False)),'gate1 gate0 '+key)
    for b in r['checks'].values():
        if set(b)!={'ordinary','checked'}:raise core.Stop('gate1 modes incomplete')
        for mode in b.values():
            if set(mode.get('variants',{}))!=set(VARIANTS):raise core.Stop('gate1 variants incomplete')
            for v in mode['variants'].values():
                for label in ('histories','saved','counters'):core.require(v[label],'gate1 '+label)
    return r


def exact(a,b):
    from . import p5g_core
    return p5g_core.exact(a,b,core.HISTORY_ORDER)


def traces(a,am,b,bm):
    import numpy as np
    good=(a.ndim==3 and b.ndim==3 and a.shape[2]==10 and b.shape[2]==10
          and am.shape==(len(a),2) and bm.shape==(len(b),2) and core.bits(am,bm)
          and not np.any(am[:,1]) and np.all(am[:,0]>=3)
          and np.all(am[:,0]<=a.shape[1]) and np.all(bm[:,0]<=b.shape[1]))
    if good:
        good=all(core.bits(a[i,:int(m[0])].copy(),b[i,:int(m[0])].copy()) for i,m in enumerate(am))
    return core.verdict(bool(good),overflow=int(np.sum(am[:,1])))
