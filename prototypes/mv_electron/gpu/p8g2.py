"""P8g-2 state-machine CPU gate and Claude-only export/gate1 entry."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import p8g as frozen,p8g2_core as core
from .. import physics

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/'docs/validation/mv/p8g2'
CACHE=Path.home()/'.cache/chatcarlo/p8g2'
D,I,IP,DP=frozen.D,frozen.I,frozen.IP,frozen.DP
U=np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
write_json,save,load_sealed=frozen.write_json,frozen.save,frozen.load_sealed
inputs,geometry,Geometry,Output=frozen.inputs,frozen.geometry,frozen.Geometry,frozen.Output

class CPU:
    run=frozen.CPU.run
    def __init__(self,work,materials,layers=None,densities=None,edges=None,cum=None,rng='philox',fault=0,cap=18,check=True,binding=True,fluorescence=True,energy=1.):
        from ..run_p8e import LAYERS
        if rng!='philox' or not 1<=cap<=4096 or fault not in range(7):raise ValueError('build configuration')
        self.rng,self.cap,self.fault=rng,cap,fault;self.materials=materials
        self.data,self.meta=core.pack(materials)
        self.constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
        self.g,self.arrays,self.grid=geometry(LAYERS if layers is None else layers,[1.,.26,1.85,11.35] if densities is None else densities,edges,cum,binding,fluorescence,energy)
        work=Path(work);work.mkdir(parents=True,exist_ok=True);compiler=shutil.which('clang++')
        if compiler is None:raise RuntimeError('clang++ unavailable')
        flags=[*core.CPU_FLAGS,'-DP5_RNG_PHILOX',f'-DP8G_STACK_CAP={cap}',f'-DP8G2_FAULT={fault}',f'-DP8G2_CHECK={int(check)}']
        self.build=dict(sources=core.fingerprint(),flags=flags,capacity=cap,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0])
        self.identity=hashlib.sha256(json.dumps(self.build,sort_keys=True).encode()).hexdigest()
        lib=work/(self.identity+('.dylib' if sys.platform=='darwin' else '.so'))
        result=subprocess.run([compiler,*flags,str(HERE/'p8g2_cpu.cpp'),'-o',str(lib)],capture_output=True,text=True)
        (work/(self.identity+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        self.lib=ct.CDLL(str(lib))
        for name,args in dict(p8g2_init=[D,I,D,ct.c_int],p8g2_histories=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.c_int,ct.c_int,DP,D,I,I,U,D,U],p8g_run=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.POINTER(Output),DP,IP,DP,DP,DP,IP]).items():
            getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=None
        self.lib.p8g2_init(self.data,self.meta,self.constants,len(materials))

    def histories(self,n,seed,start=0,slots=1,maxsteps=10000000,reference=False,injected=None,details=False):
        if not isinstance(n,int) or not 0<n<=200000 or not 1<=slots<=10240 or not 0<=seed<2**64 or not 0<=start<start+n<2**64 or maxsteps<0:raise ValueError('history range')
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        nb,nc,nf=self.g.nb,self.g.nc,self.g.nf;dw=nb+nc+nf+37;iw=79+nb
        d=np.zeros((n,dw));i=np.zeros((n,iw),np.int64);visits=np.zeros(n,np.int64);hits=np.zeros(80,np.uint64);groups=np.zeros((100,dw));gi=np.zeros((100,iw),np.uint64)
        self.lib.p8g2_histories(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,maxsteps,slots,int(reference),DP() if x is None else x.ctypes.data_as(DP),d,i,visits,hits,groups,gi)
        h=core.unpack(d,i,nb,nc,nf);ag=core.reduce_records(groups,gi,nb,nc,nf,n,self.grid['mass_g_cm2']);ag.update(_groups_d=groups,_groups_i=gi)
        if details:return h,visits,hits,ag
        core.check_visits(visits);return h


def run_identity(identity,edges,cum):
    return dict(frozen.run_identity(identity,edges,cum),sources=core.fingerprint(),capacity=core.CAPACITY,launches=[list(x) for x in core.LAUNCHES],cuda_options=list(core.CUDA_OPTIONS),cpu_flags=list(core.CPU_FLAGS))


def gate1(work,reference):
    from .p8f import require_gate
    work=Path(work);reference=Path(reference);manifest=frozen.verify_reference(reference)
    m,i,e,c=inputs();identity=run_identity(i,e,c);require_gate(OUT,'gate0',identity)
    cpu=CPU(work/'build',m,densities=i['densities'],edges=e,cum=c,cap=18)
    expected,_=load_sealed(reference/'p8f_ref_gate2.npz',dict(identity=manifest['identity'],n=20000,seed=core.SEED2,start=0))
    actual=cpu.histories(20000,core.SEED2);core.exact_histories(actual,expected,'1a saved history')
    save(work/'p8g2_ref_gate2.npz',actual,dict(identity=identity,n=20000,start=0,seed=core.SEED2))
    for b in (0,50,99):
        expected,_=load_sealed(reference/f'p8f_ref_batch_{b:03}.npz',dict(identity=manifest['identity'],n=100000,seed=core.SEED3,start=b*100000,batch=b))
        actual=cpu.run(100000,core.SEED3,b*100000)
        if actual.keys()!=expected.keys():raise AssertionError('saved aggregate keys')
        for key in expected:
            if key!='seconds':core.bitwise(actual[key],expected[key],f'1a batch {b} {key}')
    import pytest
    code=pytest.main([str(HERE.parent/'tests/test_p8g2_gates.py'),'-q','-k','fault'])
    if code:raise AssertionError('1b fault replay')
    write_json(work/'gate1.json',dict(status='pass',identity=identity,reference_identity=manifest['identity'],capacity=18))


def export(work,reference):
    import zipfile
    from .p8f import require_gate
    work=Path(work);reference=Path(reference);manifest=frozen.verify_reference(reference)
    _,i,e,c=inputs();identity=run_identity(i,e,c);require_gate(OUT,'gate0',identity);gate=require_gate(work,'gate1',identity)
    if gate['reference_identity']!=manifest['identity']:raise ValueError('reference identity')
    dest=work/'export';dest.mkdir(parents=True,exist_ok=True);files=[]
    for name in (*manifest['files'],'p8f_manifest.json'):
        shutil.copyfile(reference/name,dest/name);files.append(name)
    for name in ('p8g2_ref_gate2.npz','p8g2_ref_gate2.json'):
        shutil.copyfile(work/name,dest/name);files.append(name)
    load_sealed(dest/'p8g2_ref_gate2.npz',dict(identity=identity,n=20000,start=0,seed=core.SEED2))
    with zipfile.ZipFile(dest/'p8g2_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        names=set(core.fingerprint())|{f.name for f in HERE.glob('*.py')}|{f.name for f in HERE.glob('p5*') if f.suffix in ('.h','.cpp','.cu')}
        for name in sorted(names):z.write(HERE/name,'prototypes/mv_electron/gpu/'+name)
        for parent in ('prototypes','prototypes/mv_electron','prototypes/mv_electron/gpu'):z.writestr(parent+'/__init__.py','')
    files.append('p8g2_source.zip')
    write_json(dest/'p8g2_manifest.json',dict(status='pass',identity=identity,reference_identity=manifest['identity'],capacity=18,files={n:core.sha(dest/n) for n in sorted(files)}));print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('local','gate1','export'));parser.add_argument('--work',type=Path,default=CACHE);parser.add_argument('--reference',type=Path,default=Path.home()/'.cache/chatcarlo/p8f_prof/p8f/export');args=parser.parse_args()
    if args.command=='local':
        import pytest
        code=pytest.main([str(HERE.parent/'tests'),'-q','-k','p8g2'])
        if code==0:
            _,i,e,c=inputs();write_json(OUT/'gate0.json',dict(status='pass',identity=run_identity(i,e,c),gates=['0a','0b','0c','0d','0e'],deferred=['gate1','gate2']))
        raise SystemExit(code)
    globals()[args.command](args.work,args.reference)

if __name__=='__main__':main()
