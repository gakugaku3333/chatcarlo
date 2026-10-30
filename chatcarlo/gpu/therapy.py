"""G5 A2 CPU twin. No prototype imports, tables or spectrum values in runtime."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import physics_sets, catalog

HERE=Path(__file__).resolve().parent
NAME='therapy-electron-v1'
D=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS')
U=np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
DP=ct.POINTER(ct.c_double);IP=ct.POINTER(ct.c_longlong)


def space(path, projected=0):
    path=Path(path)
    while not path.exists():path=path.parent
    if shutil.disk_usage(path).free-projected<3*1024**3:
        raise RuntimeError('A2 disk space would fall below 3 GiB')


def lengths(m):
    n=len(m.rates) if m.rates.shape[1]==3 else len(m.gs_T)
    return [len(m.brems_E),3*len(m.brems_E),n-1,2*n,len(m.gs_T)-1,*([len(m.pair_E)]*3),
            len(m.stop_T),len(m.stop_T),len(m.photo_E),len(m.photo_E),len(m.photo_E)-1,len(m.photo_E)-1,len(m.photo_E),*([len(m.z)]*3)]


def pack(materials):
    chunks=[];rows=[];offset=0
    for m in materials:
        nz=len(m.z)
        shapes=dict(bindings=(nz,5),pm=(nz,5,4),shell_nj=(nz,),yields=(nz,4),
                    lines=(nz,4,m.lines.shape[2],4),line_counts=(nz,4),
                    gs_y=(len(m.gs_T),len(m.gs_K),len(m.gs_u)),brems_V=(len(m.brems_E),5),
                    dl=(6,6),thresholds=(2,))
        if not 1<=nz<=30 or any(np.shape(getattr(m,k))!=shape for k,shape in shapes.items()):raise ValueError('material shape')
        for k in ('ne','excitation','c','bpar','delcm','delpos'):
            if np.shape(getattr(m,k))!=():raise ValueError('scalar '+k)
        for k in ('stop_delta','stop_rad','log_stop_T'):
            if np.shape(getattr(m,k))!=np.shape(m.stop_T):raise ValueError('stopping shape')
        if len(m.stop_collision) not in (0,len(m.stop_T)):raise ValueError('collision shape')
        if m.rates.ndim!=2 or m.rates.shape[1] not in (2,3):raise ValueError('rates shape')
        for k in ('stop_T','gs_T','gs_K','gs_u','brems_E'):
            v=np.asarray(getattr(m,k))
            if v.ndim!=1 or len(v)<2 or np.any(np.diff(v)<=0):raise ValueError('grid '+k)
        for x,y in (('photo_E','photo'),('pair_E','pair_n'),('pair_E','pair_e')):
            if np.shape(getattr(m,x))!=np.shape(getattr(m,y)):raise ValueError('photon shape')
        row=[]
        for name in catalog.P8_DEFINITIONS:
            a=np.asarray(getattr(m,name),np.float64)
            if not np.isfinite(a).all():raise ValueError('nonfinite '+name)
            shape=a.shape
            row.extend((offset,a.size,shape[0] if shape else 1,
                        shape[2] if name=='lines' else shape[1] if len(shape)>1 else 1))
            chunks.append(a.ravel());offset+=a.size
        rows.append(row)
    meta=np.zeros((len(rows),136),np.int64);meta[:,:132]=rows
    for mid,m in enumerate(materials):
        size=sum(lengths(m));meta[mid,132:135]=[offset,offset+size,mid]
        chunks.extend((np.full(size,np.nan),np.zeros(37)));offset+=size+37
    if len(materials)>1:meta[0,135]=meta[1,132]+sum(lengths(materials[1])[:17])
    return np.ascontiguousarray(np.concatenate(chunks)),meta


class Geometry(ct.Structure):
    _fields_=[(k,DP) for k in ('faces','rho','bins','strength','edges','cum')]+[(k,IP) for k in ('ids','owner')]+[(k,ct.c_int) for k in ('nl','nb','nc','nf','nspec','binding','fluorescence')]+[('energy',ct.c_double)]


class FieldGeometry(ct.Structure):
    _fields_=[('lo',ct.c_double*45),('hi',ct.c_double*45),('origins',ct.c_double*15),('widths',ct.c_double*15),('axes',ct.c_int*15),('sizes',ct.c_int*15),('offsets',ct.c_int*15)]


class CPU:
    def __init__(self,work,materials,arrays,*,cap=18,fault=0,field_geometry=None,
                 half=(0.,0.),ssd=100.,physics_declaration=None,quantity='absorbed-dose'):
        physics_sets.require_quantity(NAME,quantity)
        self.physics_set=physics_sets.declaration(NAME)
        if physics_declaration is not None:physics_sets.check(physics_declaration,self.physics_set)
        if not 1<=cap<=4096 or fault not in (0,1,2,3,4,5,6,7,8,11,12,13,14,15,16):raise ValueError('build configuration')
        self.data,self.meta=pack(materials);self.arrays=arrays
        self.constants=np.ascontiguousarray(arrays['constants'],float)
        self.field_geometry=field_geometry;self.half=half;self.ssd=ssd
        if not np.isfinite([*half,ssd]).all() or len(half)!=2 or min(half)<0 or ssd<=0 or (half[0]==0)!=(half[1]==0):raise ValueError('field source')
        nl,nb,nc,nf,nspec=map(int,arrays['sizes'])
        shapes=dict(faces=(nl+1,),rho=(nl,),bins=(nb+1,),strength=(17,),edges=(nspec+1 if nspec else 0,),cum=(nspec,),ids=(nl,),owner=(nb,),constants=(6,))
        for k,shape in shapes.items():
            if arrays[k].shape!=shape or not np.isfinite(arrays[k]).all():raise ValueError('geometry '+k)
        if not 1<=nl<=16 or nspec>64 or np.any(arrays['rho']<=0) or np.any(np.diff(arrays['faces'])<=0) or np.any(np.diff(arrays['bins'])<=0):raise ValueError('geometry range')
        if np.any(arrays['ids']<0) or np.any(arrays['ids']>=len(materials)) or np.any(arrays['owner']<0) or np.any(arrays['owner']>=nl):raise ValueError('material/layer ID')
        if nspec and (arrays['cum'][-1]!=1 or arrays['cum'][0]<=0 or np.any(np.diff(arrays['cum'])<=0) or arrays['edges'][0]<0 or arrays['edges'][-1]>10 or np.any(np.diff(arrays['edges'])<=0)):raise ValueError('spectrum')
        if not 0<nb<=300 or not 0<nc<=150 or not 0<nf<=600:raise ValueError('scoring capacity')
        self.g=Geometry(*[arrays[k].ctypes.data_as(t) for k,t in Geometry._fields_[:8]],nl,nb,nc,nf,nspec,1,1,1.)
        work=Path(work);space(work);work.mkdir(parents=True,exist_ok=True)
        flags=['-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_PHILOX',
               '-DP8G2_CHECK=1',f'-DP8G_STACK_CAP={cap}',f'-DT2_FAULT={fault}',
               f'-DT2_FIELD={int(field_geometry is not None)}',
               f'-DG5_SET_ID={self.physics_set["id"]}',f'-DG5_SET_TAG={self.physics_set["tag"]}ULL']
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('t2_*')) if p.suffix in ('.h','.cpp')}
        sources.update({n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('g5_base.h','physics_set.h')})
        self.build=dict(sources=sources,flags=flags)
        self.identity=hashlib.sha256(json.dumps(self.build,sort_keys=True).encode()).hexdigest()
        lib=work/(self.identity+('.dylib' if sys.platform=='darwin' else '.so'))
        if not lib.exists():
            result=subprocess.run(['clang++',*flags,str(HERE/'t2_cpu.cpp'),'-o',str(lib)],text=True,capture_output=True)
            (work/(self.identity+'.log')).write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(result.stderr)
        self.lib=ct.CDLL(str(lib))
        self.lib.t2_set_id.restype=ct.c_int;self.lib.t2_set_tag.restype=ct.c_uint64
        self.observed=dict(name=NAME,id=self.lib.t2_set_id(),tag=self.lib.t2_set_tag())
        physics_sets.check(self.observed,self.physics_set)
        self.lib.p8g2_init.argtypes=[D,I,D,ct.c_int]
        self.lib.p8g2_histories.argtypes=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int,ct.c_int,DP,ct.POINTER(FieldGeometry),ct.c_double,ct.c_double,ct.c_double,D,D,U,D,D,I,I,U,D,U]
        self.lib.p8g2_init(self.data,self.meta,self.constants,len(materials))

    def histories(self,n,seed,start=0,slots=1,maxsteps=10000000,injected=None):
        if not isinstance(n,int) or not 0<n<=200000 or not 1<=slots<=10240 or not 0<=seed<2**64 or not 0<=start<start+n<2**64 or maxsteps<0:raise ValueError('history range')
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        nb,nc,nf=self.g.nb,self.g.nc,self.g.nf;dw=nb+nc+nf+37;iw=79+nb
        # This is the small-interval verification API, never a production history tensor.
        d=np.zeros((n,dw));i=np.zeros((n,iw),np.int64);visits=np.zeros(n,np.int64);hits=np.zeros(80,np.uint64)
        groups=np.zeros((100,dw));gi=np.zeros((100,iw),np.uint64)
        fs=np.zeros(3420);fl=np.zeros(15);fa=np.zeros(3420,np.uint64);sources=np.zeros((n,6))
        self.lib.p8g2_histories(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,maxsteps,slots,0,
            DP() if x is None else x.ctypes.data_as(DP),None if self.field_geometry is None else ct.byref(self.field_geometry),*self.half,self.ssd,fs,fl,fa,sources,d,i,visits,hits,groups,gi)
        return dict(d=d,i=i,visits=visits,hits=hits,groups=groups,group_ints=gi,field=fs,ledger=fl,field_additions=fa,sources=sources)


def cuda_source():
    """NVRTC source; includes only package-owned headers."""
    def expand(name,seen):
        out=[]
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child not in seen:
                    seen.add(child);out.append(expand(child,seen))
            else:out.append(line)
        return '\n'.join(out)
    return expand('t2_gpu.cu',set())
