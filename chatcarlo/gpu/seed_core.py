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
from . import seed_geometry as core
from . import therapy,physics_sets,seed_checks
from .seed_geometry import D,I,DP,IP,Geometry
from .therapy import HERE
U=np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
GP=ct.POINTER(Geometry)

def compile_cpu(work,fault=0,_test_legacy_physics=False):
    work=Path(work);therapy.space(work);work.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(json.dumps(dict(sources=source_hashes(),fault=fault,legacy=_test_legacy_physics),sort_keys=True).encode()).hexdigest()
    path=work/(digest+('.dylib' if sys.platform=='darwin' else '.so'))
    if not path.exists():
        flags=[f'-DG5_TEST_LEGACY_ELECTRON={int(_test_legacy_physics)}','-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_PHILOX','-DP8G_STACK_CAP=18','-DT2S_CHECK=0','-DT2_SEED=1',f'-DT2_SEED_FAULT={fault}',f'-DG5_SET_ID={physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON)["id"]}',f'-DG5_SET_TAG={physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON)["tag"]}ULL']
        compiler=shutil.which('clang++') or shutil.which('g++')
        if compiler is None:raise RuntimeError('clang++ or g++ is required for CPU twin')
        result=subprocess.run([compiler,*flags,str(HERE/'t2_seed_cpu.cpp'),'-o',str(path)],capture_output=True,text=True)
        (work/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
    lib=ct.CDLL(str(path))
    signatures=dict(t2_seed_init=[D,I,D,ct.c_int],t2_seed_histories=[D,I,D,GP,ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,DP,D,I,D,I,I,DP],t2_seed_boundary_probe=[GP,D,ct.c_int,D],t2_seed_score_probe=[GP,D,D,ct.c_double,ct.c_int],t2_seed_source_probe=[GP,D,D,D])
    for name,args in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=None
    lib.t2_seed_probe.argtypes=[D,I,D,GP,D,D,ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_int,D]
    lib.t2_seed_geometry_size.restype=ct.c_int
    if lib.t2_seed_geometry_size()!=ct.sizeof(Geometry):raise RuntimeError('P9c ABI mismatch')
    return lib

def bind_grids(g,grids,values,adds=None,device=False):
    for j,grid in enumerate(grids):
        a=values[j];g.grids[j].values=ct.cast(int(a.data.ptr),DP) if device else a.ctypes.data_as(DP)
        g.grids[j].adds=ct.POINTER(ct.c_ulonglong)() if adds is None else (ct.cast(int(adds[j].data.ptr),ct.POINTER(ct.c_ulonglong)) if device else adds[j].ctypes.data_as(ct.POINTER(ct.c_ulonglong)))

def field_seed(seed,field):
    if type(seed) is not int or not 0<=seed<2**61 or type(field) is not int or not 0<=field<8:raise ValueError('seed/field')
    return seed*8+field  # injective (seed, field) key; Philox counter is history ID

def check_result(h,ledger,counts):
    if not seed_checks.history_health(h):raise AssertionError('P9c history health')
    if np.any(abs(ledger.sum(1)-h['dep'])>1e-12*h['sources']):raise AssertionError('P9c ledger')

class CPU:
    def __init__(self,work,materials=None,*,constants=None,strength=None,fault=0,_test_legacy_physics=False,**kw):
        self.lib=compile_cpu(work,fault,_test_legacy_physics);self.g,self.arrays,self.grids=core.geometry(**kw)
        self.values=[np.zeros(g.shape) for g in self.grids];self.adds=[np.zeros(g.shape,np.uint64) for g in self.grids]
        bind_grids(self.g,self.grids,self.values,self.adds)
        self.args=None
        self.lib.t2_seed_set_id.restype=ct.c_int;self.lib.t2_seed_set_tag.restype=ct.c_uint64
        self.observed=physics_sets.observation(therapy.NAME,self.lib.t2_seed_set_id(),self.lib.t2_seed_set_tag())
        physics_sets.check(self.observed,physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON))
        if materials is not None:
            if len(materials)<=self.arrays['ids'].max():raise ValueError('material IDs')
            data,meta=therapy.pack(materials)
            if constants is None or strength is None:raise ValueError('sealed constants/strength required')
            self.arrays['strength']=np.ascontiguousarray(strength,float);self.g.strength=self.arrays['strength'].ctypes.data_as(DP)
            constants=np.ascontiguousarray(constants,float)
            self.args=(data,meta,constants);self.lib.t2_seed_init(*self.args,len(materials))
    def clear(self):
        for x in (*self.values,*self.adds):x.fill(0)
    def boundary(self,states):
        a=np.ascontiguousarray(states,float).reshape(-1,23);out=np.zeros((len(a),6));self.lib.t2_seed_boundary_probe(ct.byref(self.g),a,len(a),out);return out
    def source(self,s,d):
        out=np.zeros(5);self.lib.t2_seed_source_probe(ct.byref(self.g),np.ascontiguousarray(s,float),np.ascontiguousarray(d,float),out);return out
    def score(self,p0,p1,loss=1.,site=3,repeats=1):
        if site not in range(1,7):raise ValueError('site')
        for _ in range(repeats):self.lib.t2_seed_score_probe(ct.byref(self.g),np.ascontiguousarray(p0,float),np.ascontiguousarray(p1,float),loss,site)
        return self.values
    def run(self,n,seed,start=0,field=0,maxsteps=10000000,injected=None,regions=False):
        if type(n) is not int or not 0<n<=200000 or type(start) is not int or not 0<=start<start+n<2**64:raise ValueError('history range')
        if self.args is None:raise ValueError('materials required')
        self.clear();d=np.zeros((n,37));i=np.zeros((n,79),np.int64);ledger=np.zeros((n,6));sites=np.zeros((n,6),np.int64);counts=np.zeros((n,4),np.int64)
        r=np.zeros((n,3435)) if regions else None  # 3420 region scores + 15 field sums per history
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        self._invoke((*self.args,ct.byref(self.g),field_seed(seed,field),start,n,maxsteps,DP() if x is None else x.ctypes.data_as(DP),d,i,ledger,sites,counts,DP() if r is None else r.ctypes.data_as(DP)))
        h=seed_checks.raw(d,i,ledger,counts,start);check_result(h,ledger,counts)
        if not self.g.beam and np.any(counts[:,1]):raise AssertionError('rectangle body miss: investigate source geometry')
        if abs(self.values[1].sum()-h['dep'].sum())>1e-9*h['sources'].sum():raise AssertionError('coarse grid sum / ledger')
        return dict(d=d,i=i,observed=self.last_observation,seconds=0.,histories=h,ledger=ledger,sites=sites,source_counts=counts,regions=None if r is None else r[:,:3420],grids=self.values,additions=self.adds)

    def _invoke(self,args):
        g=ct.cast(args[3],GP).contents
        arrays={}
        for k,t in therapy.Geometry._fields_[:8]:
            a=self.arrays[k];arrays[k]=np.ctypeslib.as_array(getattr(g,k),shape=(a.size,)).reshape(a.shape).copy()
        arrays['seeds']=np.ctypeslib.as_array(g.seeds,shape=(g.ns*5,)).reshape(-1,5).copy()
        arrays['parents']=np.ctypeslib.as_array(g.parents,shape=(g.ns,)).copy()
        declaration=physics_sets.observation(therapy.NAME,self.lib.t2_seed_set_id(),self.lib.t2_seed_set_tag())
        self.last_observation=observation(g,arrays,args[:3],args[4]//8,args[5],args[6],args[4]%8,declaration)
        self.lib.t2_seed_histories(*args)


    def probe(self,states,directions=None,mode=0,seed=9502101,start=0):
        a=np.ascontiguousarray(states,float).reshape(-1,23);directions=np.ascontiguousarray(a[:,3:6] if directions is None else directions,float)
        out=np.zeros((len(a),74));self.lib.t2_seed_probe(*self.args,ct.byref(self.g),a,directions,mode,seed,start,len(a),out);return out


    def observe(self,seed,start,n,field):
        arrays={}
        for k,t in therapy.Geometry._fields_[:8]:
            a=self.arrays[k]
            arrays[k]=np.ctypeslib.as_array(getattr(self.g,k),shape=(a.size,)).reshape(a.shape).copy()
        arrays['seeds']=np.ctypeslib.as_array(self.g.seeds,shape=(self.g.ns*5,)).reshape(-1,5).copy()
        arrays['parents']=np.ctypeslib.as_array(self.g.parents,shape=(self.g.ns,)).copy()
        return observation(self.g,arrays,self.args,seed,start,n,field,self.observed)


def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.suffix in ('.h','.cpp','.cu','.py','.json')}

def cuda_source():
    def expand(name,seen):
        out=[]
        source=(HERE/name).read_text()
        if name=='t2_kernel.h':source='#include "t2_seed_kernel.h"'
        for line in source.splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child == "photon_physics.h" or child not in seen:seen.add(child);out.append(expand(child,seen))
            else:out.append(line)
        return '\n'.join(out)
    return expand('t2_seed_gpu.cu',set())

def observation(g,arrays,args,seed,start,n,field,declared):
    # Callers pass arrays reconstructed from the final native ABI pointers.
    from .catalog import digest
    return dict(physics_set=declared,seed=field_seed(seed,field),start=int(start),count=int(n),
        material_data={str(mid)+'.'+str(j):digest(args[0][row[4*j]:row[4*j]+row[4*j+1]]) for mid,row in enumerate(args[1]) for j in range(33)},material_photons={str(mid):photon_digest(args[0],row) for mid,row in enumerate(args[1])},material_meta=digest(args[1]),constants=digest(args[2]),
        geometry={k:digest(v) for k,v in arrays.items()},
        scalars={k:getattr(g,k) for k in ('nl','ns','nspec','outer','beam','angle','binding','fluorescence','axis','energy','field_half','ssd','scoring_enabled')},
        radii=list(g.radii),half=list(g.half),scoring=hashlib.sha256(bytes(g.scoring)).hexdigest(),
        grids=[dict(lo=list(x.lo),step=list(x.step),shape=list(x.n)) for x in g.grids])

def unpack(d,i):
    nb=nc=nf=0
    sizes=[nb,nc,nf,6,3,10,2,1,1,14];names=['histories','coarse','fine','totals','maxima','brems','p8_max','sources','dep','diagnostic']
    r={};off=0
    for k,s in zip(names,sizes):r[k]=np.ascontiguousarray(d[:,off:off+s]);off+=s
    r['diagnostic']=r['diagnostic'].reshape(-1,2,7)
    off=0
    for k,s in zip(('violations','counts','steps','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches','calls','diagnostic_count','maxdepth','nspec_index','additions'),(3,16,1,5,18,18,8,6,1,1,1,1,nb)):
        r[k]=np.ascontiguousarray(i[:,off:off+s]);off+=s
        if s==1:r[k]=r[k][:,0]
    r['dep']=r['dep'][:,0];r['sources']=r['sources'][:,0]
    return r


def photon_digest(data,row):
    from .catalog import digest
    start=int(row[133])+37;p=data[start:];tail=int(p[1]);size=tail+1+3*int(p[0])+4*int(p[tail])
    return digest(p[:size])
