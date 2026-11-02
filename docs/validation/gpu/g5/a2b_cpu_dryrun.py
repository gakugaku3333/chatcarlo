"""A2b pre-T4 CPU dry run: the Colab paths with the CPU twin standing in for the CUDA class.

Run before every T4 attempt (A2b 2026-11-02: five Colab-only defects were found one T4 run at a time).
  gates      seed_t4_checks.run (B0-B2) end to end; V1c/V2b history loops shortened to 2 chunks
  production seed_colab.worker with the full per-field count: 2 batches, simulated VM loss, resume to 3
GPU-only behaviour (atomicAdd targets, NVRTC compile, device memory) is NOT covered.
The production mode needs an approved new-entry directory (seed_entry preview/approve) as ROOT.
"""
import argparse, ctypes as ct, json, subprocess, sys, time, types
from pathlib import Path
import numpy as np
from unittest.mock import patch
from chatcarlo.gpu import seed_core, seed_cuda, seed_gates, seed_bundle, therapy, physics_sets, seed_t4_checks
HERE=Path(seed_core.__file__).parent;CPP=Path(__file__).with_name('a2b_cpu_fixed.cpp');WORK=None

def build(fault):
    lib=WORK/f'dry_f{fault}.dylib'
    if not lib.exists():
        flags=['-O2','-ffp-contract=off','-std=c++14','-shared','-fPIC','-DP5_RNG_PHILOX','-DP8G_STACK_CAP=18','-DT2S_CHECK=0','-DT2_SEED=1',f'-DT2_SEED_FAULT={fault}',f'-DG5_SET_ID={physics_sets.declaration(therapy.NAME)["id"]}',f'-DG5_SET_TAG={physics_sets.declaration(therapy.NAME)["tag"]}ULL','-I',str(HERE)]
        subprocess.run(['clang++',*flags,str(CPP),'-o',str(lib)],check=True)
    return lib
real_compile=seed_core.compile_cpu
def compile_cpu(work,fault=0):
    lib=ct.CDLL(str(build(fault)))
    with patch.object(seed_core.ct,'CDLL',lambda p:lib),patch.object(seed_core.subprocess,'run',lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='',stderr='')):
        return real_compile(WORK/f'work{fault}',fault)
cpns=types.SimpleNamespace(asnumpy=np.asarray,zeros=np.zeros,asarray=np.asarray)
class Shim:
    def __init__(self,cp,materials,*,constants,strength,fault=0,**kw):
        self.e=seed_core.CPU(WORK/'work',materials,constants=constants,strength=strength,fault=fault,**kw)
        self.cp=cpns;self.g=self.e.g;self.grids=self.e.grids;self.values=self.e.values;self.adds=None
        self.args=self.e.args;self.geometry=ct.byref(self.e.g);lib=self.e.lib;lib.t2_seed_fixed.restype=None;self.observed=self.e.observed
        self.module=types.SimpleNamespace(get_function=lambda name:(lambda grid,block,args:lib.t2_seed_fixed(*[a.ctypes.data_as(ct.c_void_p) for a in args[:3]],args[3],args[4].ctypes.data_as(ct.c_void_p),ct.c_uint64(int(args[5])),args[6].ctypes.data_as(ct.c_void_p))))
    def upload(self):
        if self.adds is not None:self.e.adds=self.adds
        seed_core.bind_grids(self.e.g,self.grids,self.values,self.e.adds)
    def clear(self):self.e.clear()
    def run(self,*a,**k):return self.e.run(*a,**k)
    def probe(self,*a,**k):return self.e.probe(*a,**k)
    def boundary(self,*a):return self.e.boundary(*a)
    def source(self,*a):return self.e.source(*a)
    def score(self,*a,**k):return self.e.score(*a,**k)
def short(*a):
    r=range(*a)  # only the history-chunk loops: V1c range(0,200000,1000), V2b range(100)
    return range(r.start,r.start+2*r.step,r.step) if (r.step==1000 or (r.start,r.stop,r.step)==(0,100,1)) else r

def gates(bundle):
    m,inp,_=seed_bundle.load(bundle)
    with patch.object(seed_core,'compile_cpu',compile_cpu),patch.object(seed_cuda,'GPU',Shim),patch.object(seed_t4_checks,'GPU',Shim),patch.object(seed_gates,'range',short,create=True):
        report=seed_t4_checks.run(cpns,m,inp,WORK/'checks')
    return {k:(v.get('status') if isinstance(v,dict) else v) for k,v in report.items()}

def production(root):
    sys.modules['cupy']=cpns
    from chatcarlo.gpu import seed_colab,seed_run
    report=json.loads((WORK/'checks/gates.json').read_text());n=json.loads((root/'condition.json').read_text())['per_batch_per_field']
    class VMLost(Exception):pass
    out=[]
    for stop in (2,3):
        def run(self,count,seed,start=0,**k):
            if start>=stop*n and k.get('field',0)==3:raise VMLost(stop)
            return self.e.run(count,seed,start=start,**k)
        with patch.object(seed_core,'compile_cpu',compile_cpu),patch.object(seed_cuda,'GPU',Shim),patch.object(seed_t4_checks,'run',lambda *a,**k:report),patch.object(Shim,'run',run):
            try:seed_colab.worker(root)
            except VMLost:pass
        out.append(seed_run.preflight(root/'results',json.loads((root/'condition.json').read_text())).completed)
    if out!=[[0,1],[0,1,2]]:raise AssertionError('loss/resume: '+str(out))
    return dict(completed_after_loss=out[0],completed_after_resume=out[1])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('gates','production'));p.add_argument('--bundle',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--root',type=Path)
    a=p.parse_args();WORK=a.work;WORK.mkdir(parents=True,exist_ok=True);t=time.time()
    print(json.dumps(gates(a.bundle) if a.mode=='gates' else production(a.root)),round(time.time()-t),'s')
