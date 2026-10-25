"""P9c CPU entry and explicit local/export commands. Input caches are read-only."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import p9c_core as core
from .p9c_core import D,I,DP,IP,Geometry
from .. import physics
U=np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
GP=ct.POINTER(Geometry)

def compile_cpu(work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(core.fingerprint(),sort_keys=True).encode()).hexdigest()
    path=work/(digest+('.dylib' if sys.platform=='darwin' else '.so'))
    if not path.exists():
        flags=[*core.core.CPU_FLAGS,'-DP5_RNG_PHILOX','-DP8G_STACK_CAP=18','-DP8G2_CHECK=0']
        result=subprocess.run([shutil.which('clang++') or 'clang++',*flags,str(core.HERE/'p9c_cpu.cpp'),'-o',str(path)],capture_output=True,text=True)
        (work/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
    lib=ct.CDLL(str(path))
    signatures=dict(p9c_init=[D,I,D,ct.c_int],p9c_histories=[D,I,D,GP,ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,DP,D,I,D,I,I,DP],p9c_boundary_probe=[GP,D,ct.c_int,D],p9c_score_probe=[GP,D,D,ct.c_double,ct.c_int],p9c_source_probe=[GP,D,D,D])
    for name,args in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=None
    lib.p9c_geometry_size.restype=ct.c_int
    if lib.p9c_geometry_size()!=ct.sizeof(Geometry):raise RuntimeError('P9c ABI mismatch')
    return lib

def bind_grids(g,grids,values,adds=None,device=False):
    for j,grid in enumerate(grids):
        a=values[j];g.grids[j].values=ct.cast(int(a.data.ptr),DP) if device else a.ctypes.data_as(DP)
        g.grids[j].adds=ct.POINTER(ct.c_ulonglong)() if adds is None else (ct.cast(int(adds[j].data.ptr),ct.POINTER(ct.c_ulonglong)) if device else adds[j].ctypes.data_as(ct.POINTER(ct.c_ulonglong)))

def field_seed(seed,field):
    if type(seed) is not int or not 0<=seed<2**61 or type(field) is not int or not 0<=field<8:raise ValueError('seed/field')
    return seed*8+field  # injective (seed, field) key; Philox counter is history ID

def check_result(h,ledger,counts):
    if not core.core.history_health(h):raise AssertionError('P9c history health')
    if np.any(abs(ledger.sum(1)-h['dep'])>1e-12*h['sources']):raise AssertionError('P9c ledger')

class CPU:
    def __init__(self,work,materials=None,**kw):
        self.lib=compile_cpu(work);self.g,self.arrays,self.grids=core.geometry(**kw)
        self.values=[np.zeros(g.shape) for g in self.grids];self.adds=[np.zeros(g.shape,np.uint64) for g in self.grids]
        bind_grids(self.g,self.grids,self.values,self.adds)
        self.args=None
        if materials is not None:
            if len(materials)<=self.arrays['ids'].max():raise ValueError('material IDs')
            data,meta=core.core.pack(materials);constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
            self.args=(data,meta,constants);self.lib.p9c_init(*self.args,len(materials))
    def clear(self):
        for x in (*self.values,*self.adds):x.fill(0)
    def boundary(self,states):
        a=np.ascontiguousarray(states,float).reshape(-1,23);out=np.zeros((len(a),6));self.lib.p9c_boundary_probe(ct.byref(self.g),a,len(a),out);return out
    def source(self,s,d):
        out=np.zeros(5);self.lib.p9c_source_probe(ct.byref(self.g),np.ascontiguousarray(s,float),np.ascontiguousarray(d,float),out);return out
    def score(self,p0,p1,loss=1.,site=3,repeats=1):
        if site not in range(1,7):raise ValueError('site')
        for _ in range(repeats):self.lib.p9c_score_probe(ct.byref(self.g),np.ascontiguousarray(p0,float),np.ascontiguousarray(p1,float),loss,site)
        return self.values
    def run(self,n,seed,start=0,field=0,maxsteps=10000000,injected=None,regions=False):
        if type(n) is not int or not 0<n<=200000 or type(start) is not int or not 0<=start<start+n<2**64:raise ValueError('history range')
        if self.args is None:raise ValueError('materials required')
        self.clear();d=np.zeros((n,37));i=np.zeros((n,79),np.int64);ledger=np.zeros((n,6));sites=np.zeros((n,6),np.int64);counts=np.zeros((n,2),np.int64)
        r=np.zeros((n,3420)) if regions else None
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        self.lib.p9c_histories(*self.args,ct.byref(self.g),field_seed(seed,field),start,n,maxsteps,DP() if x is None else x.ctypes.data_as(DP),d,i,ledger,sites,counts,DP() if r is None else r.ctypes.data_as(DP))
        h=core.core.unpack(d,i,0,0,0);check_result(h,ledger,counts)
        if not self.g.beam and np.any(counts[:,1]):raise AssertionError('rectangle body miss: investigate source geometry')
        if abs(self.values[1].sum()-h['dep'].sum())>1e-9*h['sources'].sum():raise AssertionError('coarse grid sum / ledger')
        return dict(histories=h,ledger=ledger,sites=sites,source_counts=counts,regions=r,grids=self.values,additions=self.adds)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('local','export'));p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    from .p9c_io import export
    from .p9c_checks import local
    print(json.dumps(local(a.work) if a.command=='local' else export(a.work),indent=2))
if __name__=='__main__':main()
