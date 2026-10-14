"""P5a0 toolchain trial. No transport; no physics imports in colab/compare.

Random123 definition and KAT source (verified at implementation time):
https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150
No downloads or package installation at runtime. Work products are retained.
"""
import argparse
import ctypes as ct
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R123 = '9545ff6413f258be2f04c1d319d99aaef7521150'
MATH_NAMES = ['log', 'exp', 'sqrt', 'sin', 'cos', 'pow', 'atan2', 'acos', 'arithmetic']
LOCAL = {'T1': True, 'T2': True, 'T3': True, 'T4': True, 'T5a': True,
         'T5b': False, 'T6': True, 'T7': True, 'T8': True}
GPU = {'G0': True, 'G1': True, 'G2a': True, 'G2b': False, 'G3a': True,
       'G3b': False, 'G4': True, 'G5': True, 'G7': True, 'G6': False}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def digest(a):
    return sha(np.ascontiguousarray(a).tobytes())


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def diff(ref, actual):
    ref, actual = np.asarray(ref), np.asarray(actual)
    if ref.shape != actual.shape:
        raise ValueError('comparison shape mismatch')
    finite = np.isfinite(ref) & np.isfinite(actual)
    delta = np.abs(ref.astype(float)-actual.astype(float))
    nz = finite & (ref != 0)
    # Bitwise mismatch includes signed zero.
    a = np.ascontiguousarray(ref).view(np.uint8).reshape(-1, ref.dtype.itemsize)
    b = np.ascontiguousarray(actual).view(np.uint8).reshape(-1, actual.dtype.itemsize)
    return dict(count=int(ref.size), mismatches=int(np.count_nonzero(np.any(a != b, axis=1))),
                max_abs=float(delta[finite].max(initial=0)),
                max_rel=float((delta[nz]/np.abs(ref[nz])).max(initial=0)),
                nonfinite=int(np.count_nonzero(~finite)))


def finite_report(value):
    if isinstance(value, dict):
        return value.get('nonfinite', 0) == 0 and all(finite_report(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_report(v) for v in value)
    return True


def record(result, key, ok=None, **data):
    required = result['checks'][key]['required']
    status = 'report' if ok is None else ('pass' if ok else 'fail')
    if not finite_report(data):
        status = 'fail'
    result['checks'][key] = dict(required=required, status=status, **data)
    print(key, status, flush=True)
    if status == 'fail':
        raise TrialStop(key + ' failed')


class TrialStop(Exception):
    pass


def verdict(checks):
    return 'fail' if any(c['status'] == 'fail' or (c['required'] and c['status'] != 'pass')
                         for c in checks.values()) else 'pass'


def philox_python(triples):
    """Independent vectorized NumPy integer implementation, no C/Numba."""
    t = np.asarray(triples, dtype=np.uint64)
    mask = np.uint64(0xffffffff)
    c = np.stack((t[:, 1] & mask, t[:, 1] >> 32, t[:, 2] & mask, t[:, 2] >> 32), axis=1)
    keys = np.stack((t[:, 0] & mask, t[:, 0] >> 32), axis=1)
    for r in range(10):
        prod = c[:, [0, 2]] * np.array([0xD2511F53, 0xCD9E8D57], dtype=np.uint64)
        c = np.stack(((prod[:, 1] >> 32) ^ c[:, 1] ^ keys[:, 0], prod[:, 1] & mask,
                      (prod[:, 0] >> 32) ^ c[:, 3] ^ keys[:, 1], prod[:, 0] & mask), axis=1)
        keys = (keys + np.array([0x9E3779B9, 0xBB67AE85], dtype=np.uint64)) & mask
    words = c.astype(np.uint32)
    u = ((words[:, 0] >> 5).astype(float)*67108864.0 + (words[:, 1] >> 6))/9007199254740992.0
    return words, u


def rng_inputs():
    t = np.empty((1_000_000, 3), dtype=np.uint64)
    t[:, 0] = 20291902
    t[:, 1] = np.repeat(np.arange(1000, dtype=np.uint64), 1000)
    t[:, 2] = np.tile(np.arange(1000, dtype=np.uint64), 1000)
    boundary = [2**32-1, 2**32, 2**32+1, 0x123456789abcdef0, 2**64-1]
    extra = [[s, h, c] for s in boundary for h in boundary for c in boundary]
    return np.concatenate((t, np.array(extra, dtype=np.uint64)))


def known_answers():
    # Random123 tests/kat_vectors at R123: philox4x32 10, all three rows.
    inputs = np.array([[0, 0, 0], [2**64-1]*3,
                       [0x299f31d0a4093822, 0x85a308d3243f6a88, 0x0370734413198a2e]], dtype=np.uint64)
    expected = np.array([[0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
                         [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
                         [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]], dtype=np.uint32)
    return inputs, expected


def toy_python(n=1000, start=0):
    # Precompute independent Philox at every possible draw for each history.
    # Bounded generation 8 implies <=511 draws, no lookup beyond that bound.
    t = np.empty((n*511, 3), dtype=np.uint64)
    t[:, 0] = 20291901
    t[:, 1] = np.repeat(np.arange(start, start+n, dtype=np.uint64), 511)
    t[:, 2] = np.tile(np.arange(511, dtype=np.uint64), n)
    randoms = philox_python(t)[1].reshape(n, 511)
    meta = np.zeros((n, 4), dtype=np.uint64)
    bins = np.zeros((n, 50), dtype=np.float64)
    for h in range(n):
        stack = [(list(range(1, 24)), 0)]
        steps, depth = 0, 1
        while stack:
            state, generation = stack.pop()
            total = sum((j+1)*value for j, value in enumerate(state))
            bins[h, total % 50] += (total % 1024+1)/1024
            u = randoms[h, steps]
            steps += 1
            children = (0 if u < .05 else 1 if u < .15 else 2) if generation < 8 else 0
            for k in range(children):
                child = [state[j]+(k+1)*(j+1)+state[(j+1) % 23] for j in range(23)]
                stack.append((child, generation+1))
                depth = max(depth, len(stack))
        meta[h] = steps, steps, depth, 0
    return meta, bins


def workdir(path):
    p = Path(path).resolve()
    if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'.p5a0_work'):
        raise ValueError('work products must be outside repository or in .p5a0_work')
    p.mkdir(parents=True, exist_ok=True)
    return p


class CPU:
    def __init__(self, work, compiler='clang++', fault=0, cap=16):
        work = workdir(work)
        exe = shutil.which(compiler)
        if exe is None:
            raise RuntimeError(compiler+' not available')
        flags = ['-O2', '-ffp-contract=off', '-std=c++11', '-shared', '-fPIC',
                 f'-DP5_FAULT={fault}', f'-DP5_CAP={cap}']
        identity = sha((HERE/'p5a0_cpu.cpp').read_bytes()+(HERE/'p5a0_common.h').read_bytes()
                       + (compiler+str(flags)).encode())[:16]
        output = work / ('p5a0_'+identity+('.dylib' if sys.platform == 'darwin' else '.so'))
        begin = time.perf_counter()
        subprocess.run([exe, *flags, str(HERE/'p5a0_cpu.cpp'), '-o', str(output)], check=True, capture_output=True)
        self.build_seconds = time.perf_counter()-begin
        self.flags = flags
        self.compiler = subprocess.check_output([exe, '--version'], text=True).splitlines()[0]
        self.lib = ct.CDLL(str(output))
        d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        u = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        w = np.ctypeslib.ndpointer(dtype=np.uint32, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        signatures = {'p5_rng': [u, ct.c_int, w, d], 'p5_math': [d, ct.c_int, d],
                      'p5_quant': [d, ct.c_int, d, d, ct.c_int, d, d, ct.c_int, ct.c_double, ct.c_int, d, i, d],
                      'p5_toy': [ct.c_uint64, ct.c_uint64, ct.c_int, ct.c_int, u, d],
                      'p5_mt': [ct.c_uint32, ct.c_int, d]}
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = None

    def rng(self, triples):
        t = np.ascontiguousarray(triples, dtype=np.uint64)
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError('rng triples shape')
        w, u = np.empty((len(t), 4), np.uint32), np.empty(len(t))
        self.lib.p5_rng(t, len(t), w, u)
        return w, u

    def quant(self, b, disabled=False):
        t = np.ascontiguousarray(b['T_eval'], dtype=np.float64)
        if not np.isfinite(t).all() or np.any(t > 20):
            raise ValueError('energy must be finite and <=20 MeV')
        e, v, le, er = [np.ascontiguousarray(b[k], dtype=np.float64) for k in ('energies', 'values', 'LOGET', 'ER')]
        if e.ndim != 1 or v.shape != (len(e), 5) or le.ndim != 1 or er.shape != le.shape or len(e)<2 or len(le)<2:
            raise ValueError('table shape')
        out, idx, raw = np.empty((len(t), 4)), np.empty(len(t), np.int32), np.empty((len(t), 12))
        self.lib.p5_quant(t, len(t), e, v, len(e), le, er, len(le), b['CUT'].item(), int(disabled), out, idx, raw)
        return out, idx, raw

    def maths(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 5:
            raise ValueError('math input shape')
        out = np.empty((len(x), 9))
        self.lib.p5_math(x, len(x), out)
        return out

    def toy(self, n, start=0, shared=False):
        meta = np.empty((n, 4), np.uint64)
        out = np.zeros(50 if shared else (n, 50))
        self.lib.p5_toy(20291901, start, n, int(shared), meta, out)
        return meta, out


def bundle(out):
    from prototypes.mv_electron.bremsstrahlung_p3r import tables, CUT
    from prototypes.mv_electron.physics import LOGET, ER
    workdir(Path(out).parent)
    e, v = tables()
    t = np.zeros((1_000_000, 3), np.uint64)
    t[:, 0] = 20291900
    t[:, 2] = np.arange(len(t), dtype=np.uint64)
    u = philox_python(t)[1]
    points = [CUT+(20-CUT)*u*u, e, np.nextafter(e, 0), np.nextafter(e, np.inf),
              np.array([CUT, CUT/2, np.nextafter(CUT, 0)])]
    te = np.concatenate(points)
    te = te[(te > 0) & (te <= 20)]
    x = np.empty((1_000_000, 5))
    for j in range(5):
        t[:, 1] = j+1
        x[:, j] = philox_python(t)[1]
    if np.any(x <= 0) or np.any(x >= 1):
        raise ValueError('fixed math input not in (0,1)')
    arrays = dict(energies=e, values=v, LOGET=LOGET, ER=ER, CUT=np.array(CUT), T_eval=te, math_inputs=x)
    arrays = {k: np.ascontiguousarray(a, dtype=np.float64) for k, a in arrays.items()}
    meta = dict(code_commit=commit(), random123_commit=R123,
                arrays={k: digest(a) for k, a in arrays.items()})
    np.savez(out, **arrays, metadata=np.array(json.dumps(meta)))
    print(json.dumps(dict(bundle_sha256=sha(Path(out).read_bytes()), shapes={k:list(a.shape) for k,a in arrays.items()})))


def load_bundle(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        b = {k: z[k] for k in meta['arrays']}
    for k, a in b.items():
        if digest(a) != meta['arrays'][k] or a.dtype != np.float64 or not a.flags.c_contiguous or not np.isfinite(a).all():
            raise ValueError('bundle array integrity: '+k)
    return b, meta


def samples(a):
    return dict(sha256=digest(a), shape=list(a.shape), first_10000=a[:10000].tolist())


def cpu_artifacts(cpu, b):
    rng = cpu.rng(rng_inputs())
    toy = cpu.toy(1000)
    q = [cpu.quant(b, disabled)[0] for disabled in (False, True)]
    math = cpu.maths(b['math_inputs'])
    return dict(philox=[digest(a) for a in rng], toy=[digest(a) for a in toy],
                quantities=[samples(a) for a in q], maths=samples(math)), (rng, toy, q, math)


def local_checks(r, cpu, b, work, outputs):
    from numba import njit
    from prototypes.mv_electron.bremsstrahlung_p3r import quantities
    rng, toy, q, math = outputs
    inp, expected = known_answers()
    actual = cpu.rng(inp)[0]
    record(r, 'T1', same(expected, actual), words=diff(expected, actual))
    ref = philox_python(rng_inputs())
    record(r, 'T2', all(same(a, c) for a,c in zip(ref,rng)),
           words=diff(ref[0],rng[0]), uniform=diff(ref[1],rng[1]))

    @njit
    def mt():
        np.random.seed(12345)
        return np.random.random(2_000_000)
    actual = np.empty(2_000_000)
    cpu.lib.p5_mt(12345, len(actual), actual)
    ref = mt()
    record(r, 'T3', same(ref, actual), values=diff(ref,actual))

    @njit
    def pq(t, e, v, disabled):
        out = np.empty((len(t),4))
        for j in range(len(t)):
            out[j] = quantities(t[j],e,v,disabled)
        return out
    qdiff, equal = [], True
    for disabled in (False, True):
        ref = pq(b['T_eval'], b['energies'], b['values'], disabled)
        actual, idx, raw = cpu.quant(b, disabled)
        expected_idx = np.clip(np.searchsorted(b['energies'], b['T_eval'])-1, 0, len(b['energies'])-2).astype(np.int32)
        expected_idx[(b['T_eval'] <= b['CUT'].item()) | disabled] = -1
        equal &= same(ref, actual) and same(expected_idx, idx)
        qdiff.append(dict(disabled=disabled, outputs=[diff(ref[:,j],actual[:,j]) for j in range(4)], indices=diff(expected_idx,idx)))
    rejected = False
    try:
        cpu.quant(dict(b, T_eval=np.array([np.nextafter(20.,np.inf)])))
    except ValueError:
        rejected = True
    record(r, 'T4', equal and rejected, cases=qdiff, above_20_rejected=rejected)

    @njit
    def pm(x):
        out=np.empty((len(x),9))
        for i in range(len(x)):
            u,u2,a,bb,c=x[i]
            out[i,0]=np.log(u);out[i,1]=np.exp(-20*u);out[i,2]=np.sqrt(u)
            out[i,3]=np.sin(6.283185307179586*u);out[i,4]=np.cos(6.283185307179586*u)
            out[i,5]=(u+.1)**1.3;out[i,6]=np.arctan2(u-.5,u2);out[i,7]=np.arccos(2*u-1)
            out[i,8]=a*bb+c*u-bb/(1.+a)
        return out
    ref=pm(b['math_inputs'])
    record(r, 'T5a', same(ref[:,[2,8]],math[:,[2,8]]),
           sqrt=diff(ref[:,2],math[:,2]), arithmetic=diff(ref[:,8],math[:,8]))
    record(r, 'T5b', **{name:diff(ref[:,j],math[:,j]) for j,name in enumerate(MATH_NAMES) if j not in (2,8)})
    reference=toy_python()
    aggregate=cpu.toy(1000,shared=True)
    overflow=CPU(work,cap=4).toy(1000)
    details=toy_details(reference,toy)
    ok=all(same(a,c) for a,c in zip(reference,toy)) and np.all(toy[0][:,3]==0)
    ok &= toy[0][:,2].max()>=8 and same(toy[1].sum(axis=0),aggregate[1])
    ok &= np.count_nonzero(overflow[0][:,3]==1)>0
    record(r,'T6',bool(ok), **details, max_depth=int(toy[0][:,2].max()),
           aggregate=diff(toy[1].sum(axis=0),aggregate[1]), overflow_count=int(np.count_nonzero(overflow[0][:,3]==1)))
    faults=[]
    for fault in (1,2,3):
        actual=CPU(work,fault=fault).toy(1000)
        faults.append(fault_details(toy,actual,fault))
    record(r,'T7',all(f['detected'] for f in faults),faults=faults)
    parts=[cpu.toy(400),cpu.toy(600,start=400)]
    joined=tuple(np.concatenate([p[j] for p in parts]) for j in range(2))
    record(r,'T8',all(same(a,c) for a,c in zip(toy,joined)),**toy_details(toy,joined))


def toy_details(ref, actual):
    return dict(metadata=diff(ref[0],actual[0]), bins=diff(ref[1],actual[1]),
                status_counts={str(k):int(v) for k,v in zip(*np.unique(actual[0][:,3],return_counts=True))})


def fault_details(ref, actual, fault):
    d=toy_details(ref,actual)
    d['fault']=fault
    d['total']=diff(ref[1].sum(axis=1),actual[1].sum(axis=1))
    d['draw_counts']=diff(ref[0][:,0],actual[0][:,0])
    d['executed']=True
    d['detected']=bool(np.all(actual[0][:,3]==0) and
        (d['bins']['mismatches']>0 if fault==1 else
         d['bins']['mismatches']>0 and d['total']['mismatches']>0 if fault==2 else
         d['draw_counts']['mismatches']>0))
    return d


class GPUBackend:
    def __init__(self, cp, fault=0, options=('--fmad=false',)):
        self.cp=cp
        self.options=options
        header=(HERE/'p5a0_common.h').read_text()
        source=(HERE/'p5a0_gpu.cu').read_text()
        if source.count('//@@COMMON@@')!=1:
            raise ValueError('common header marker')
        self.source=f'#define P5_FAULT {fault}\n'+source.replace('//@@COMMON@@',header)
        self.kernels={}
        begin=time.perf_counter()
        for name in ('p5_rng','p5_quant','p5_math','p5_toy'):
            kernel=cp.RawKernel(self.source,name,options=options)
            kernel.compile()
            self.kernels[name]=kernel
        self.compile_seconds=time.perf_counter()-begin

    def launch(self,name,n,args,block=256):
        self.kernels[name](((n+block-1)//block,),(block,),args)

    def rng(self,t):
        cp=self.cp;t=cp.asarray(t,dtype=cp.uint64)
        w=cp.empty((len(t),4),dtype=cp.uint32);u=cp.empty(len(t),dtype=cp.float64)
        self.launch('p5_rng',len(t),(t,np.int32(len(t)),w,u))
        return cp.asnumpy(w),cp.asnumpy(u)

    def quant(self,b,disabled=False):
        cp=self.cp
        t,e,v,le,er=[cp.asarray(b[k],dtype=cp.float64,order='C') for k in ('T_eval','energies','values','LOGET','ER')]
        out=cp.empty((len(t),4),cp.float64);idx=cp.empty(len(t),cp.int32);raw=cp.empty((len(t),12),cp.float64)
        args=(t,np.int32(len(t)),e,v,np.int32(len(e)),le,er,np.int32(len(le)),np.float64(b['CUT'].item()),np.int32(disabled),out,idx,raw)
        self.launch('p5_quant',len(t),args)
        return tuple(cp.asnumpy(a) for a in (out,idx,raw))

    def maths(self,x):
        cp=self.cp;x=cp.asarray(x,dtype=cp.float64,order='C');out=cp.empty((len(x),9),cp.float64)
        self.launch('p5_math',len(x),(x,np.int32(len(x)),out))
        return cp.asnumpy(out)

    def toy(self,n,start=0,shared=False,block=256):
        cp=self.cp;meta=cp.empty((n,4),cp.uint64);out=cp.zeros(50 if shared else (n,50),cp.float64)
        self.launch('p5_toy',n,(np.uint64(20291901),np.uint64(start),np.int32(n),np.int32(shared),meta,out),block)
        return cp.asnumpy(meta),cp.asnumpy(out)


def gpu_checks(r,cpu,b,outputs):
    try:
        import cupy as cp
    except ImportError:
        print('GPUなし: CuPy がありません。G 項目は not_run。',flush=True)
        r['gpu_unavailable']='CuPy not installed'
        return
    r['environment']['cupy']=cp.__version__
    r['environment']['cuda_runtime']=cp.cuda.runtime.runtimeGetVersion()
    r['environment']['cuda_driver']=cp.cuda.runtime.driverGetVersion()
    r['environment']['device']=str(cp.cuda.runtime.getDeviceProperties(0)['name'])
    r['gpu_options']=['--fmad=false']
    try:
        gpu=GPUBackend(cp)
    except Exception as exc:
        diagnostic={}
        try:
            GPUBackend(cp,options=())
            diagnostic['default_compile']='pass'
        except Exception as other:
            diagnostic['default_compile']='fail'
            diagnostic['default_error']=str(other)
        record(r,'G0',False,error=str(exc),diagnostic=diagnostic)
        return
    record(r,'G0',True,compile_seconds=gpu.compile_seconds,common_sha256=sha((HERE/'p5a0_common.h').read_bytes()))
    rng,toy,q,math=outputs
    actual=gpu.rng(rng_inputs())
    record(r,'G1',all(same(a,c) for a,c in zip(rng,actual)),words=diff(rng[0],actual[0]),uniform=diff(rng[1],actual[1]))
    readouts=[];qd=[];ok=True
    for disabled in (False,True):
        ref=cpu.quant(b,disabled);actual=gpu.quant(b,disabled)
        ok &= same(ref[1],actual[1]) and same(ref[2],actual[2])
        readouts.append(dict(disabled=disabled,indices=diff(ref[1],actual[1]),raw=diff(ref[2],actual[2])))
        qd.append(dict(disabled=disabled,outputs=[diff(ref[0][:,j],actual[0][:,j]) for j in range(4)]))
    record(r,'G2a',bool(ok),cases=readouts)
    record(r,'G2b',cases=qd)
    actual=gpu.maths(b['math_inputs'])
    record(r,'G3a',same(math[:,[2,8]],actual[:,[2,8]]),sqrt=diff(math[:,2],actual[:,2]),arithmetic=diff(math[:,8],actual[:,8]))
    normal=GPUBackend(cp,options=()).maths(b['math_inputs'])
    record(r,'G3b',fmad_false={name:diff(math[:,j],actual[:,j]) for j,name in enumerate(MATH_NAMES) if j not in (2,8)},
           default_options=[],default={name:diff(math[:,j],normal[:,j]) for j,name in enumerate(MATH_NAMES)})
    totals=np.zeros(50);atomics=np.zeros(50);mismatches=0;bad=0;maxdepth=0
    for start in range(0,1_000_000,10_000):
        ref=cpu.toy(10_000,start);a=gpu.toy(10_000,start);agg=gpu.toy(10_000,start,shared=True)
        mismatches+=diff(ref[0],a[0])['mismatches']+diff(ref[1],a[1])['mismatches']
        mismatches+=diff(a[0],agg[0])['mismatches']
        totals+=a[1].sum(axis=0);atomics+=agg[1]
        bad+=int(np.count_nonzero(a[0][:,3]))
        maxdepth=max(maxdepth,int(a[0][:,2].max()))
    record(r,'G4',mismatches==0 and bad==0 and same(totals,atomics),histories=1_000_000,
           mismatches=mismatches,abnormal_histories=bad,max_depth=maxdepth,aggregate=diff(totals,atomics))
    faults=[]
    for fault in (1,2,3):
        faults.append(fault_details(toy,GPUBackend(cp,fault=fault).toy(1000),fault))
    record(r,'G5',all(f['detected'] for f in faults),faults=faults)
    one=gpu.toy(10_000,block=64)
    split=[gpu.toy(4000,block=64),gpu.toy(6000,start=4000,block=64)]
    joined=tuple(np.concatenate([p[j] for p in split]) for j in range(2))
    large=gpu.toy(10_000,block=256)
    record(r,'G7',all(same(a,c) and same(a,d) for a,c,d in zip(one,joined,large)),
           split=toy_details(one,joined),blocks=toy_details(one,large))
    gpu_timing(r,cpu,gpu)


def gpu_timing(r,cpu,gpu):
    cp=gpu.cp;n=1_000_000
    # Compile and allocation outside timing; initialization and transfer timed separately.
    before=time.perf_counter();meta=cp.empty((n,4),cp.uint64);out=cp.empty(50,cp.float64)
    cp.cuda.Stream.null.synchronize();allocation=time.perf_counter()-before
    def launch(start):
        gpu.launch('p5_toy',n,(np.uint64(20291901),np.uint64(start),np.int32(n),np.int32(1),meta,out))
    out.fill(0);launch(0);cp.cuda.Stream.null.synchronize()
    measured=[];initialization=[];transfer=[]
    for _ in range(3):
        cp.cuda.Stream.null.synchronize();t=time.perf_counter();out.fill(0);cp.cuda.Stream.null.synchronize()
        initialization.append(time.perf_counter()-t)
        t=time.perf_counter()
        for start in range(0,10_000_000,n):
            launch(start)
        cp.cuda.Stream.null.synchronize();measured.append(time.perf_counter()-t)
        t=time.perf_counter();host=cp.asnumpy(out);cp.cuda.Stream.null.synchronize();transfer.append(time.perf_counter()-t)
        if not np.isfinite(host).all():
            record(r,'G6',False,nonfinite=int(np.count_nonzero(~np.isfinite(host))))
    # CPU uses the very same B entry, allocation and initialization separated.
    before=time.perf_counter();cm=np.empty((n,4),np.uint64);co=np.empty(50)
    cpu_allocation=time.perf_counter()-before
    before=time.perf_counter();co.fill(0);cpu_init=time.perf_counter()-before
    before=time.perf_counter()
    for start in range(0,10_000_000,n):
        cpu.lib.p5_toy(20291901,start,n,1,cm,co)
    cpu_seconds=time.perf_counter()-before
    record(r,'G6',histories=10_000_000,path='B',chunk=1_000_000,cpu_seconds=cpu_seconds,
           gpu_seconds=measured,gpu_initialization_seconds=initialization,gpu_transfer_seconds=transfer,
           cpu_initialization_seconds=cpu_init,cpu_allocation_seconds=cpu_allocation,
           gpu_allocation_seconds=allocation,cpu_compile_seconds=cpu.build_seconds,gpu_compile_seconds=gpu.compile_seconds,
           nonfinite=int(np.count_nonzero(~np.isfinite(co))),
           caveat='Toy histories only; not a transport speed estimate.')


def run(mode,bundle_path,work,out):
    required=LOCAL if mode=='local' else GPU
    r=dict(mode=mode,checks={k:dict(status='not_run',required=v) for k,v in required.items()},
           code_commit=commit(),random123_commit=R123,bundle_sha256=sha(Path(bundle_path).read_bytes()),
           common_sha256=sha((HERE/'p5a0_common.h').read_bytes()),
           source_sha256={p.name:sha(p.read_bytes()) for p in HERE.iterdir() if p.suffix in ('.py','.h','.cpp','.cu')},
           environment=dict(python=platform.python_version(),numpy=np.__version__,system=platform.system(),machine=platform.machine()))
    try:
        b,meta=load_bundle(bundle_path);r['bundle_metadata']=meta
        cpu=CPU(work,compiler='clang++' if mode=='local' else 'g++')
        r['cpu_flags']=cpu.flags;r['environment']['compiler']=cpu.compiler
        r['cpu_outputs'],outputs=cpu_artifacts(cpu,b)
        if mode=='local':
            local_checks(r,cpu,b,work,outputs)
        else:
            gpu_checks(r,cpu,b,outputs)
    except TrialStop as exc:
        r['stop_reason']=str(exc)
    except Exception as exc:
        r['execution_error']=type(exc).__name__+': '+str(exc)
        for key,c in r['checks'].items():
            if c['status']=='not_run':
                c['status']='fail';c['error']=r['execution_error'];break
    r['physics_imported']='prototypes.mv_electron.physics' in sys.modules
    r['verdict']=verdict(r['checks'])
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r[k] for k in ('mode','verdict','physics_imported')}))
    return r


def compare(local_path,colab_path):
    a,b=[json.loads(Path(p).read_text()) for p in (local_path,colab_path)]
    checks={'X1a':dict(status='not_run',required=True),'X1b':dict(status='not_run',required=False)}
    r=dict(checks=checks)
    for key in ('bundle_sha256','code_commit','common_sha256','source_sha256'):
        if a.get(key)!=b.get(key) or key not in a:
            return dict(r,verdict='fail',comparison='比較不能',reason=key+' mismatch')
    try:
        aa,bb=a['cpu_outputs'],b['cpu_outputs']
        record(r,'X1a',aa['philox']==bb['philox'] and aa['toy']==bb['toy'],
               philox_hash_equal=aa['philox']==bb['philox'],toy_hash_equal=aa['toy']==bb['toy'])
        data=[]
        for x,y in zip(aa['quantities']+[aa['maths']],bb['quantities']+[bb['maths']]):
            data.append(dict(full_sha_equal=x['sha256']==y['sha256'],
                             sample=diff(np.array(x['first_10000']),np.array(y['first_10000']))))
        record(r,'X1b',arrays=data,sample_points=10000)
    except TrialStop:
        pass
    r['verdict']=verdict(checks)
    r['p5a_ready']=r['verdict']=='pass' and a.get('mode')=='local' and b.get('mode')=='colab' and a.get('verdict')=='pass' and b.get('verdict')=='pass'
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('bundle');p.add_argument('--out',required=True)
    for mode in ('local','colab'):
        p=sub.add_parser(mode)
        for opt in ('bundle','work','out'):
            p.add_argument('--'+opt,required=True)
    p=sub.add_parser('compare');p.add_argument('local');p.add_argument('colab')
    args=parser.parse_args()
    if args.mode=='bundle':
        bundle(args.out)
    elif args.mode=='compare':
        result=compare(args.local,args.colab);print(json.dumps(result,indent=2));return int(result['verdict']!='pass')
    else:
        result=run(args.mode,args.bundle,args.work,args.out)
        return int(result['verdict']!='pass')
    return 0


if __name__=='__main__':
    sys.exit(main())
