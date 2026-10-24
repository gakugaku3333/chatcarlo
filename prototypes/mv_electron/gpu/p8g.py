"""P8g A1 CPU build, local gate and sealed Claude hand-off."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from . import p8g_core as core
from . import p8f as frozen
from .p5a import DP
from .. import physics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=Path.home()/'.cache/chatcarlo/p8g1'
BUNDLES=Path.home()/'.cache/chatcarlo/p8e/materials'
PRODUCTION=Path.home()/'.cache/chatcarlo/p8e/proto/production'
OUT=ROOT/'docs/validation/mv/p8g1'
IP=ct.POINTER(ct.c_longlong)
D=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
I=np.ctypeslib.ndpointer(np.int64,flags='C_CONTIGUOUS')

class Geometry(ct.Structure):
    _fields_=[(k,DP) for k in ('faces','rho','bins','strength','edges','cum')]+[(k,IP) for k in ('ids','owner')]+[(k,ct.c_int) for k in ('nl','nb','nc','nf','nspec','binding','fluorescence')]+[('energy',ct.c_double)]

class Output(ct.Structure):
    _fields_=[(k,DP) for k in ('coarse','fine','totals','maxima','brems','p8_batches','p8_max')]+[(k,IP) for k in ('violations','counts','steps','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches')]+[('maxdepth',ct.c_int)]

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def save(path,result,receipt):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending.npz');np.savez(tmp,**result);tmp.replace(path)
    write_json(path.with_suffix('.json'),dict(receipt,sha256=core.sha(path)))

def load_sealed(path,expected=None):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text())
    if core.sha(path)!=receipt['sha256']:raise ValueError('artifact hash changed: '+str(path))
    if expected and any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('receipt differs: '+str(path))
    with np.load(path,allow_pickle=False) as f:result={k:f[k].copy() for k in f.files}
    return result,receipt

def inputs():
    from ..photon_transport_p8e import load_bundles
    from ..run_p3a import load_spectrum,SPECTRUM
    materials,identity=load_bundles(BUNDLES)
    edges,_,cum=load_spectrum()
    config=json.loads((PRODUCTION/'config.json').read_text())
    fp=config['fingerprint']
    if fp['bundle_sha256']!=core.sha(BUNDLES/'bundles.json') or fp['inputs']!=identity['files'] or fp['spectrum_sha256']!=core.sha(SPECTRUM):
        raise ValueError('saved production input identity changed')
    if fp['source_hashes']!=identity['source_hashes']:raise ValueError('saved P8e sources changed')
    return materials,identity,edges,cum

def geometry(layers,densities,edges=None,cum=None,binding=True,fluorescence=True,energy=1.):
    from ..run_p8e import layout
    from ..photon_transport_p8e import BOUNDARY_STRENGTH_EDGES
    grid=layout(layers,densities)
    faces=np.r_[0.,np.cumsum([t for _,t,_ in layers])]
    ids=np.array([m-1 for m,_,_ in layers],np.int64)
    rho=np.array([densities[i] for i in ids],float)
    arrays=dict(faces=faces,rho=rho,bins=grid['edges_cm'],strength=BOUNDARY_STRENGTH_EDGES,
        edges=np.empty(0) if edges is None else edges,cum=np.empty(0) if cum is None else cum,
        ids=ids,owner=grid['bin_layer'].astype(np.int64))
    arrays={k:np.ascontiguousarray(v) for k,v in arrays.items()}
    nb=len(grid['mass_g_cm2']);nc=round(faces[-1]/.2);nf=round(faces[-1]/.05)
    if nb>300 or nc>150 or nf>600:raise ValueError('scoring capacity')
    if (not np.isfinite(energy) or energy<=0 or energy>10 or
        (edges is None)!=(cum is None)):raise ValueError('source')
    if cum is not None and (len(cum)<1 or len(cum)>64 or len(edges)!=len(cum)+1 or cum[-1]!=1 or cum[0]<=0 or np.any(np.diff(cum)<=0) or np.any(np.diff(edges)<=0) or edges[0]<0 or edges[-1]>10 or not np.isfinite(edges).all() or not np.isfinite(cum).all()):raise ValueError('spectrum')
    g=Geometry(*[arrays[k].ctypes.data_as(t) for k,t in Geometry._fields_[:8]],len(ids),nb,nc,nf,len(arrays['cum']),int(binding),int(fluorescence),energy)
    return g,arrays,grid

class CPU:
    def __init__(self,work,materials,layers=None,densities=None,edges=None,cum=None,rng='mt',fault=0,cap=4096,check=False,binding=True,fluorescence=True,energy=1.):
        from ..run_p8e import LAYERS
        if rng not in ('mt','philox') or not 1<=cap<=4096:raise ValueError('build configuration')
        self.rng,self.cap,self.fault=rng,cap,fault
        self.materials=materials;self.data,self.meta=core.pack(materials)
        self.constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
        self.g,self.arrays,self.grid=geometry(LAYERS if layers is None else layers,[1.,.26,1.85,11.35] if densities is None else densities,edges,cum,binding,fluorescence,energy)
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        compiler=shutil.which('clang++')
        if compiler is None:raise RuntimeError('clang++ unavailable')
        flags=[*core.CPU_FLAGS,'-DP5_RNG_'+rng.upper(),f'-DP8G_FAULT={fault}',f'-DP8G_CHECK={int(check)}',f'-DP8G_STACK_CAP={cap}']
        identity=dict(source_hashes=core.fingerprint(),flags=flags,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0])
        import hashlib
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        lib=work/(self.identity+('.dylib' if sys.platform=='darwin' else '.so'))
        build=subprocess.run([compiler,*flags,str(HERE/'p8g_cpu.cpp'),'-o',str(lib)],capture_output=True,text=True)
        (work/(self.identity+'.log')).write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(lib))
        signatures=dict(p8g_probe=[ct.c_int,D,I,D,ct.c_int,D,ct.c_int,ct.c_int,ct.c_uint64,D,I,D],
            p8g_fixed_transitions=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,D,I],
            p8g_fixed=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,D,ct.POINTER(Output),D,I,D],
            p8g_histories=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,D,I],
            p8g_score_probe=[ct.POINTER(Geometry),D,ct.c_int,D,D],
            p8g_run=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.POINTER(Output),DP,IP,DP,DP,DP,IP])
        signatures.update(p8g_init=[D,I,D,ct.c_int],p8g_values=[D,I,D,ct.c_int,D])
        for name,args in signatures.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=None

        self.lib.p8g_init(self.data,self.meta,self.constants,len(materials))

    def probe(self,which,x,mid=0,seed=core.SEED0):
        x=np.ascontiguousarray(x,dtype=float)
        if x.ndim==1:x=x[:,None]
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('probe input')
        out=np.zeros((len(x),128));state=np.zeros(626,np.int64);next=np.zeros(16)
        self.lib.p8g_probe(which,self.data,self.meta,self.constants,mid,x,len(x),x.shape[1],seed,out,state,next)
        return out,state,next

    def score(self,x):
        x=np.ascontiguousarray(x,float);out=np.zeros((len(x),self.g.nb));bound=np.zeros((len(x),2))
        self.lib.p8g_score_probe(ct.byref(self.g),x,len(x),out,bound)
        return out,bound

    def run(self,n,seed,start=0,per_history=False,maxsteps=10000000):
        if not isinstance(n,int) or n<=0 or n>20000000 or seed<0 or seed>=2**64 or start<0 or start+n>=2**64:raise ValueError('history range')
        if self.rng=='mt' and (seed>=2**32 or start):raise ValueError('MT uses one sequential stream from history zero')
        r={k:np.zeros(s,dtype=d) for k,s,d in (
            ('coarse',(100,self.g.nc),float),('fine',(100,self.g.nf),float),('totals',6,float),('maxima',3,float),('brems',10,float),('p8_batches',(100,self.g.nb),float),('p8_max',2,float),
            ('violations',3,np.int64),('counts',16,np.int64),('steps',100,np.int64),('boundary_counts',5,np.int64),('boundary_hist',18,np.int64),('boundary_omitted_hist',18,np.int64),('fluorescence_counts',8,np.int64),('branches',6,np.int64))}
        o=Output(*[r[k].ctypes.data_as(t) for k,t in Output._fields_[:-1]],0)
        extras={k:np.zeros(s,dtype=d) for k,s,d in [('histories',(n,self.g.nb),float),('calls',n,np.int64),('sources',n,float),('dep',n,float),('diagnostic',(n,2,7),float),('diagnostic_count',n,np.int64)]} if per_history else {}
        pointers=[extras[k].ctypes.data_as(t) if per_history else t() for k,t in zip(('histories','calls','sources','dep','diagnostic','diagnostic_count'),(DP,IP,DP,DP,DP,IP))]
        begin=time.perf_counter();self.lib.p8g_run(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,maxsteps,ct.byref(o),*pointers)
        r.update(extras);r['seconds']=np.array(time.perf_counter()-begin);r['maxdepth']=np.array(o.maxdepth,np.int64)
        r['dose']=r['p8_batches'].sum(axis=0)/n/self.grid['mass_g_cm2']
        return r


    def fixed(self,state,seed):
        r={k:np.zeros(s,dtype=d) for k,s,d in (
            ('coarse',(100,self.g.nc),float),('fine',(100,self.g.nf),float),('totals',6,float),('maxima',3,float),('brems',10,float),('p8_batches',(100,self.g.nb),float),('p8_max',2,float),
            ('violations',3,np.int64),('counts',16,np.int64),('steps',100,np.int64),('boundary_counts',5,np.int64),('boundary_hist',18,np.int64),('boundary_omitted_hist',18,np.int64),('fluorescence_counts',8,np.int64),('branches',6,np.int64))}
        o=Output(*[r[k].ctypes.data_as(t) for k,t in Output._fields_[:-1]],0)
        events=np.zeros((16,10));mt=np.zeros(627,np.int64);next=np.zeros(16)
        self.lib.p8g_fixed(self.data,self.meta,self.constants,ct.byref(self.g),seed,np.ascontiguousarray(state),ct.byref(o),events,mt,next)
        r['events']=events[:min(16,mt[626])];r['mt']=mt;r['next']=next
        transitions=np.zeros((16,5),np.int64)
        self.lib.p8g_fixed_transitions(self.data,self.meta,self.constants,ct.byref(self.g),seed,np.ascontiguousarray(state),transitions)
        r['transitions']=transitions;r['maxdepth']=np.array(o.maxdepth,np.int64)
        return r

    def histories(self,n,seed,start=0):
        if self.rng!='philox':raise ValueError('per-history references use Philox')
        dw=self.g.nb+self.g.nc+self.g.nf+37;iw=79+self.g.nb
        d=np.zeros((n,dw));i=np.zeros((n,iw),np.int64)
        self.lib.p8g_histories(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,d,i)
        return core.unpack(d,i,self.g.nb,self.g.nc,self.g.nf)


    def counters(self):
        return np.array([self.data[q[133]:q[133]+37].view(np.uint64).copy() for q in self.meta])

    def values(self,mid):
        q=self.meta[mid];out=np.full((q[133]-q[132],5),np.nan)
        self.lib.p8g_values(self.data,self.meta,self.constants,mid,out)
        return self.data[q[132]:q[133]].copy(),out


def run_identity(identity,edges,cum):
    return dict(frozen.run_identity(identity,edges,cum),sources=core.fingerprint())


def verify_reference(reference):
    reference=Path(reference);manifest=json.loads((reference/'p8f_manifest.json').read_text())
    if manifest['status']!='pass' or manifest['identity']['sources']!=core.frozen.fingerprint():raise ValueError('P8f source identity changed')
    for name,digest in manifest['files'].items():
        if core.sha(reference/name)!=digest:raise ValueError('P8f export hash '+name)
    return manifest


def gate1(work,reference):
    work=Path(work);manifest=verify_reference(reference);reference=Path(reference)
    m,i,e,c=inputs();identity=run_identity(i,e,c);frozen.require_gate(OUT,'gate0',identity)
    cap=manifest['capacity'];kw=dict(densities=i['densities'],edges=e,cum=c,rng='philox',cap=cap)
    cpu=CPU(work/'build',m,**kw);old=frozen.CPU(work/'p8f',m,**kw)
    for n,start in ((20000,0),(2000,7000)):
        actual=cpu.histories(n,core.SEED2,start);core.exact_histories(actual,old.histories(n,core.SEED2,start),'1a')
        if start==0:
            expected,_=frozen.load_sealed(reference/'p8f_ref_gate2.npz',dict(identity=manifest['identity'],n=n,seed=core.SEED2,start=0))
            core.exact_histories(actual,expected,'1b saved history')
            save(work/'p8g_ref_gate2.npz',actual,dict(identity=identity,n=n,start=0,seed=core.SEED2))
    for b in (0,50,99):
        expected,_=frozen.load_sealed(reference/f'p8f_ref_batch_{b:03}.npz',dict(identity=manifest['identity'],n=100000,seed=core.SEED3,start=b*100000,batch=b))
        actual=cpu.run(100000,core.SEED3,b*100000)
        if actual.keys()!=expected.keys():raise AssertionError('saved aggregate keys')
        for key in expected:
            if key!='seconds':core.bitwise(actual[key],expected[key],f'1b batch {b} {key}')
    # f2 is a successful build; demonstrate a transport mismatch, not a hash failure.
    bad=CPU(work/'f2',m,fault=2,check=True,**kw)
    try:core.exact_histories(bad.histories(20000,core.SEED2),cpu.histories(20000,core.SEED2),'f2 gate1')
    except AssertionError as error:fault_rejection=str(error)
    else:raise AssertionError('f2 not rejected by gate1')
    write_json(work/'gate1.json',dict(status='pass',identity=identity,reference_identity=manifest['identity'],capacity=cap,f2_rejected=fault_rejection))


def export(work,reference):
    import zipfile
    work=Path(work);manifest=verify_reference(reference);reference=Path(reference)
    _,i,e,c=inputs();identity=run_identity(i,e,c);frozen.require_gate(OUT,'gate0',identity)
    gate=frozen.require_gate(work,'gate1',identity)
    if gate['reference_identity']!=manifest['identity']:raise ValueError('reference identity')
    dest=work/'export';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    # Reuse sealed P8f bundle and original references verbatim, no source edits.
    for name in manifest['files']:
        shutil.copyfile(reference/name,dest/name);files.append(name)
    shutil.copyfile(reference/'p8f_manifest.json',dest/'p8f_manifest.json');files.append('p8f_manifest.json')
    for name in ('p8g_ref_gate2.npz','p8g_ref_gate2.json'):
        shutil.copyfile(work/name,dest/name);files.append(name)
    frozen.load_sealed(dest/'p8g_ref_gate2.npz',dict(identity=identity,n=20000,start=0,seed=core.SEED2))
    with zipfile.ZipFile(dest/'p8g_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(core.fingerprint())|{'p8g_colab.py','p8g.py','p8f_colab.py'}|{f.name for f in HERE.glob('p5*') if f.suffix in ('.py','.h','.cpp','.cu')}):
            z.write(HERE/name,'prototypes/mv_electron/gpu/'+name)
        for parent in ('prototypes/','prototypes/mv_electron/','prototypes/mv_electron/gpu/'):z.writestr(parent+'__init__.py','')
    files.append('p8g_source.zip')
    write_json(dest/'p8g_manifest.json',dict(status='pass',identity=identity,reference_identity=manifest['identity'],capacity=gate['capacity'],files={n:core.sha(dest/n) for n in sorted(files)}))
    print(dest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('local','gate1','export'))
    parser.add_argument('--work',type=Path,default=CACHE)
    parser.add_argument('--reference',type=Path,default=Path.home()/'.cache/chatcarlo/p8f_prof/p8f/export')
    args=parser.parse_args()
    if args.command=='local':
        import pytest
        code=pytest.main([str(HERE.parent/'tests/test_p8g_gates.py'),str(HERE.parent/'tests/test_p8g_contracts.py'),'-q','-s'])
        if code==0:
            _,i,e,c=inputs();write_json(OUT/'gate0.json',dict(status='pass',identity=run_identity(i,e,c),gates=['0a','0b','0c','0d','0e'],deferred=['gate1','gate2']))
        raise SystemExit(code)
    globals()[args.command](args.work,args.reference)

if __name__=='__main__':main()
