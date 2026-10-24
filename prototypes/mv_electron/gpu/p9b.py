"""P9b local gates, export and analysis. No writes to the frozen/cache inputs."""
import argparse
import hashlib
import ctypes as ct
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from . import p9b_core as core
from .p9b_core import Geometry,D,I,DP,IP
from . import p9a as reference
from .. import physics,p9a_data

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'docs/validation/mv/p9b'
WORK=HERE.parents[2]/'runs/p9b'
U=np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')

class CPU:
    def __init__(self,work,materials,**kw):
        self.g,self.arrays,self.grid=core.geometry(**kw)
        if len(materials)<=int(self.arrays['ids'].max()):raise ValueError('material IDs')
        self.materials=materials;self.kw=kw;self.data,self.meta=core.core.pack(materials)
        self.constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        compiler=shutil.which('clang++')
        if compiler is None:raise RuntimeError('clang++ unavailable')
        self.flags=[*core.core.CPU_FLAGS,'-DP5_RNG_PHILOX','-DP8G_STACK_CAP=18','-DP8G2_FAULT=0','-DP8G2_CHECK=1']
        lib=work/(hashlib.sha256(json.dumps(core.fingerprint(),sort_keys=True).encode()).hexdigest()+('.dylib' if sys.platform=='darwin' else '.so'))
        if not lib.exists():
            proc=subprocess.run([compiler,*self.flags,str(HERE/'p9b_cpu.cpp'),'-o',str(lib)],text=True,capture_output=True)
            (work/'build.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError(proc.stderr)
        self.lib=ct.CDLL(str(lib))
        signatures=dict(p9b_injected=[D,I,D,ct.POINTER(Geometry),D,ct.c_uint64,D,U],p9b_score_probe=[ct.POINTER(Geometry),ct.c_int,ct.c_int,D,D,ct.c_double,D,D,U,I,D],p9b_init=[D,I,D,ct.c_int],p9b_histories=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,DP,D,I,DP,I,D,U,I],p9b_probe=[D,I,D,ct.POINTER(Geometry),D,D,ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_int,D])
        for name,args in signatures.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=None
        self.lib.p9b_geometry_size.restype=ct.c_int
        if self.lib.p9b_geometry_size()!=ct.sizeof(Geometry):raise RuntimeError('geometry ABI')
        self.lib.p9b_init(self.data,self.meta,self.constants,len(materials))

    def histories(self,n,seed,start=0,injected=None,maxsteps=10000000,field_history=False):
        if type(n) is not int or not 0<n<=200000 or not 0<=seed<2**64 or not 0<=start<start+n<2**64 or maxsteps<0:raise ValueError('history range')
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        g=self.g;d=np.zeros((n,g.nb+g.nc+g.nf+37));i=np.zeros((n,79+g.nb),np.int64)
        field=np.zeros((n,3420)) if field_history else None;sites=np.zeros((n,6),np.int64);ledger=np.zeros((n,6));adds=np.zeros(3420,np.uint64);lateral=np.zeros(n,np.int64)
        self.lib.p9b_histories(self.data,self.meta,self.constants,ct.byref(g),seed,start,n,maxsteps,DP() if x is None else x.ctypes.data_as(DP),d,i,DP() if field is None else field.ctypes.data_as(DP),sites,ledger,adds,lateral)
        self.last=dict(d=d,i=i,field=field,sites=sites,ledger=ledger,additions=adds,lateral=lateral)
        return core.core.unpack(d,i,g.nb,g.nc,g.nf)

    def injected_probe(self,state,seed=42):
        out=np.zeros(15);adds=np.zeros(3420,np.uint64)
        self.lib.p9b_injected(self.data,self.meta,self.constants,ct.byref(self.g),np.ascontiguousarray(state,float),seed,out,adds)
        return out

    def score_probe(self,p0,p1,energy,region,site):
        if site not in range(1,7) or not 0<=region<self.g.nl+self.g.ns:raise ValueError('deposit probe')
        field=np.zeros(3420);depth=np.zeros(self.g.nb);adds=np.zeros(3420,np.uint64);sites=np.zeros(6,np.int64);ledger=np.zeros(6)
        self.lib.p9b_score_probe(ct.byref(self.g),region,site,np.ascontiguousarray(p0,float),np.ascontiguousarray(p1,float),energy,field,depth,adds,sites,ledger)
        return dict(field=field,depth=depth,additions=adds,sites=sites,ledger=ledger)

    def probe(self,states,directions=None,mode=0,seed=core.core.SEED2,start=0):
        states=np.ascontiguousarray(states,float).reshape(-1,23);n=len(states)
        directions=np.ascontiguousarray(states[:,3:6] if directions is None else directions,float).reshape(n,3)
        if mode not in (0,1) or not np.isfinite(states).all() or not np.isfinite(directions).all():raise ValueError('probe inputs')
        out=np.zeros((n,74));self.lib.p9b_probe(self.data,self.meta,self.constants,ct.byref(self.g),states,directions,mode,seed,start,n,out);return out


def local(work=WORK):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    materials,identity=p9a_data.load_bundles()
    from ..run_p3a import load_spectrum
    e,_,c=load_spectrum();kw=dict(layers=reference.reference.LAYERS,densities=identity['densities'],edges=e,cum=c)
    old=reference.CPU(work/'reference',materials,**kw);new=CPU(work/'build',materials,**kw)
    t=time.perf_counter();expected=old.histories(2000,core.core.SEED2,reference=False);old_s=time.perf_counter()-t
    t=time.perf_counter();actual=new.histories(2000,core.core.SEED2);new_s=time.perf_counter()-t
    core.core.exact_histories(actual,expected,'G0 all used record fields')
    # Direct raw bit comparison includes signed zero and all used fields.
    from .p9b_cases import g3
    result=dict(G0=dict(status='pass',histories=2000,fields=list(actual),reference_us=old_s*1e6/2000,p9b_us=new_s*1e6/2000),G3=g3(work/'g3',materials,identity['densities']),gpu='pending',sources=core.fingerprint())
    (work/'local.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('local','export','analyse'));p.add_argument('--work',type=Path,default=WORK);a=p.parse_args()
    if a.command=='local':result=local(a.work)
    elif a.command=='export':
        from .p9b_io import export
        result=export(a.work)
    else:
        from .p9b_io import analyse
        result=analyse(a.work)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
