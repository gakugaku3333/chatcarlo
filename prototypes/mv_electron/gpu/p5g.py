"""P5f-1 local CPU driver and frozen 100-batch gate. Retains all artifacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import p5b_core as core, p5d_core as d, p5g_core as g
HERE=core.HERE


class CPU(d.CPU):
    def __init__(self, work, data, cap=64, baseline=False):
        core.fingerprints()
        core.check_inputs(data)
        if core.sha(core.source().encode()) != d.SOURCE_SHA:
            raise core.Stop('source fingerprint mismatch')
        exe = shutil.which('clang++')
        if exe is None:
            raise core.Stop('compiler unavailable: '+'clang++')
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        self.flags = d.FLAGS[:-1]+[f'-DP5_STACK_CAP={cap}']
        sources = {k: core.file_sha(core.HERE/k) for k in
                   (*core.FROZEN, 'p5g_kernel.h', 'p5g_cpu.cpp')}
        self.compiler = subprocess.check_output([exe, '--version'], text=True).splitlines()[0]
        self.identity = core.sha(json.dumps(dict(compiler=self.compiler, flags=self.flags,
                                                sources=sources), sort_keys=True).encode())
        library = work/('p5g_'+('baseline_' if baseline else '')+self.identity[:16]+('.dylib' if sys.platform == 'darwin' else '.so'))
        build = subprocess.run([exe, *self.flags, str(core.HERE/('p5_cpu.cpp' if baseline else 'p5g_cpu.cpp')), '-o', str(library)],
                               capture_output=True, text=True)
        (work/('build_'+self.identity[:16]+'.log')).write_text(build.stdout+build.stderr)
        build.check_returncode()
        self.data, self.tables = data, self.make_tables(data)
        self.lib = ct.CDLL(str(library))
        dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.lib.p5a_run.argtypes = [ct.POINTER(d.Tables), ct.c_uint64, ct.c_uint64,
            ct.c_int, ct.c_int, ct.c_int, dp, dp, dp, dp, i, i, i, i, dp, i, dp, ct.c_int, i]
        self.lib.p5a_run.restype = None

        if not baseline:
            self.lib.p5g_trace.argtypes = [ct.POINTER(d.Tables),ct.c_uint64,ct.c_uint64,
                                          ct.c_int,np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),ct.c_int,np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")]
            self.lib.p5g_trace.restype = None

    def trace(self, seed=core.SEED, h=0, maxsteps=core.MAXSTEPS, capacity=100000):
        v=np.zeros((capacity,10)); meta=np.zeros(2,np.int64)
        self.lib.p5g_trace(ct.byref(self.tables),seed,h,maxsteps,v,capacity,meta)
        return v[:min(capacity,int(meta[0]))],meta


def _gate_init(work,data):
    global _gate_cpu
    _gate_cpu=CPU(Path(work)/str(os.getpid()),data)


def _gate_batch(b):
    return b,_gate_cpu.run(100000,core.SEED,start=100000*b)


def gate1(work, saved):
    from . import p5b
    work,saved=Path(work),Path(saved)
    data,config=p5b.load_local_inputs(saved)
    cpu=CPU(work/'build',data)
    if cpu.compiler != config['compiler']: raise core.Stop('saved compiler mismatch')
    expected=[f'batch_{i:03}.npz' for i in range(100)]
    if sorted(p.name for p in saved.glob('batch*.npz')) != expected:
        raise core.Stop('saved 100 batches incomplete')
    report=dict(status='incomplete',batches=100,outputs=list(core.RUN_ORDER[:-1]),
        sources={k:core.file_sha(HERE/k) for k in (*core.FROZEN,'p5b_wrap.h',*g.SOURCES)},
        compiler=cpu.compiler,flags=cpu.flags,checks={})
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),
                                 initializer=_gate_init,initargs=(work/'build',data)) as pool:
            for b,actual in pool.map(_gate_batch,range(100)):
                core.save_npz(work/f'batch_{b:03}.npz',actual)
                ref=p5b.saved_batch(saved,b,config)
                check=g.exact(actual,ref,d.OUTPUTS)
                report['checks'][str(b)]=check
                core.write_json(work/'gate1.json',report)
                core.require(check,f'gate1 batch {b}')
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally: core.write_json(work/'gate1.json',report)
    return report


def unusual(work, data):
    """Observe (a)-(d), compare against frozen CPU, and retain full transition records."""
    work=Path(work); checks={}; evidence=set()
    for cap,maxsteps,label,n in ((64,2,'photon_limit',100),(64,1,'charged_limit',100),
                                 (3,core.MAXSTEPS,'stack',100),(64,core.MAXSTEPS,'ordinary',2000)):
        fixture=dict(data,edges=np.array([.015,.016] if label=='photon_limit' else [9.99,10.]),cum=np.array([1.]))
        new,old=CPU(work/'build',fixture,cap=cap),CPU(work/'build',fixture,cap=cap,baseline=True)
        a=new.run(n,core.SEED,maxsteps=maxsteps);b=old.run(n,core.SEED,maxsteps=maxsteps)
        core.save_npz(work/(label+'_new.npz'),a);core.save_npz(work/(label+'_frozen.npz'),b)
        checks[label]=g.exact(a,b,d.OUTPUTS);core.require(checks[label],label)
        # Each history has its own trace, preserving actual particle and RNG order.
        for h in range(n):
            v,meta=new.trace(h=h,maxsteps=maxsteps)
            core.require(g.trace_check(v,meta),'transition record')
            pairs=v[:,:2].astype(int); reasons=v[:,7].astype(int)
            seen=set()
            if np.any((pairs[:,0]==0)&(pairs[:,1]==3)&((reasons&1)>0)):seen.add('photon_limit' if label=='photon_limit' else 'photon_failure')
            if np.any((pairs[:,0]==1)&(pairs[:,1]==3)&((reasons&1)>0)):seen.add('charged_limit' if label=='charged_limit' else 'charged_failure')
            if cap==3 and a['counts'][6]>0:seen.add('stack')
            seen.update(g.trace_events(v))
            if seen-evidence:
                core.save_npz(work/f'{label}_trace_{h:04}.npz',dict(trace=v,meta=meta))
                evidence.update(seen)
    required={'photon_limit','charged_limit','stack','hard','brems','reentry','flight','rest','escape','hard_resume','brems_resume'}
    result=core.verdict(required<=evidence,checks=checks,evidence=sorted(evidence),missing=sorted(required-evidence))
    core.write_json(work/'unusual.json',result);core.require(result,'unusual transitions')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['gate1','unusual'])
    p.add_argument('--work',type=Path,default=Path('~/.cache/chatcarlo/p5/p5g').expanduser())
    p.add_argument('--saved',type=Path,default=Path('~/.cache/chatcarlo/p5/p5a/philox').expanduser())
    a=p.parse_args()
    if a.command=='gate1':gate1(a.work,a.saved)
    else:
        from . import p5b
        data,_=p5b.load_local_inputs(a.saved);unusual(a.work/'unusual',data)


if __name__=='__main__': main()
