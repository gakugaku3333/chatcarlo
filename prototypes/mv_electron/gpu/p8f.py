"""P8f CPU build, deterministic gates, sealed references and hand-off commands."""
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
from . import p8f_core as core
from .p5a import Tables, DP
from .. import physics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CACHE=Path.home()/'.cache/chatcarlo/p8f'
BUNDLES=Path.home()/'.cache/chatcarlo/p8e/materials'
PRODUCTION=Path.home()/'.cache/chatcarlo/p8e/proto/production'
OUT=ROOT/'docs/validation/mv/p8f'
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
    def __init__(self,work,materials,layers=None,densities=None,edges=None,cum=None,rng='mt',fault=0,cap=4096,binding=True,fluorescence=True,energy=1.):
        from ..run_p8e import LAYERS
        if rng not in ('mt','philox') or not 1<=cap<=4096:raise ValueError('build configuration')
        self.rng,self.cap,self.fault=rng,cap,fault
        self.data,self.meta=core.pack(materials)
        self.constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
        self.g,self.arrays,self.grid=geometry(LAYERS if layers is None else layers,[1.,.26,1.85,11.35] if densities is None else densities,edges,cum,binding,fluorescence,energy)
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        compiler=shutil.which('clang++')
        if compiler is None:raise RuntimeError('clang++ unavailable')
        flags=[*core.CPU_FLAGS,'-DP5_RNG_'+rng.upper(),f'-DP8F_FAULT={fault}',f'-DP8F_STACK_CAP={cap}']
        identity=dict(source_hashes=core.fingerprint(),flags=flags,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0])
        import hashlib
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        lib=work/(self.identity+('.dylib' if sys.platform=='darwin' else '.so'))
        build=subprocess.run([compiler,*flags,str(HERE/'p8f_cpu.cpp'),'-o',str(lib)],capture_output=True,text=True)
        (work/(self.identity+'.log')).write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(lib))
        signatures=dict(p8f_probe=[ct.c_int,D,I,D,ct.c_int,D,ct.c_int,ct.c_int,ct.c_uint64,D,I,D],
            p8f_fixed_transitions=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,D,I],
            p8f_fixed=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,D,ct.POINTER(Output),D,I,D],
            p8f_histories=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,D,I],
            p8f_score_probe=[ct.POINTER(Geometry),D,ct.c_int,D,D],
            p8f_run=[D,I,D,ct.POINTER(Geometry),ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,ct.POINTER(Output),DP,IP,DP,DP,DP,IP])
        for name,args in signatures.items():getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=None

    def probe(self,which,x,mid=0,seed=core.SEED0):
        x=np.ascontiguousarray(x,dtype=float)
        if x.ndim==1:x=x[:,None]
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('probe input')
        out=np.zeros((len(x),128));state=np.zeros(626,np.int64);next=np.zeros(16)
        self.lib.p8f_probe(which,self.data,self.meta,self.constants,mid,x,len(x),x.shape[1],seed,out,state,next)
        return out,state,next

    def score(self,x):
        x=np.ascontiguousarray(x,float);out=np.zeros((len(x),self.g.nb));bound=np.zeros((len(x),2))
        self.lib.p8f_score_probe(ct.byref(self.g),x,len(x),out,bound)
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
        begin=time.perf_counter();self.lib.p8f_run(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,maxsteps,ct.byref(o),*pointers)
        r.update(extras);r['seconds']=np.array(time.perf_counter()-begin);r['maxdepth']=np.array(o.maxdepth,np.int64)
        r['dose']=r['p8_batches'].sum(axis=0)/n/self.grid['mass_g_cm2']
        return r


    def fixed(self,state,seed):
        r={k:np.zeros(s,dtype=d) for k,s,d in (
            ('coarse',(100,self.g.nc),float),('fine',(100,self.g.nf),float),('totals',6,float),('maxima',3,float),('brems',10,float),('p8_batches',(100,self.g.nb),float),('p8_max',2,float),
            ('violations',3,np.int64),('counts',16,np.int64),('steps',100,np.int64),('boundary_counts',5,np.int64),('boundary_hist',18,np.int64),('boundary_omitted_hist',18,np.int64),('fluorescence_counts',8,np.int64),('branches',6,np.int64))}
        o=Output(*[r[k].ctypes.data_as(t) for k,t in Output._fields_[:-1]],0)
        events=np.zeros((16,10));mt=np.zeros(627,np.int64);next=np.zeros(16)
        self.lib.p8f_fixed(self.data,self.meta,self.constants,ct.byref(self.g),seed,np.ascontiguousarray(state),ct.byref(o),events,mt,next)
        r['events']=events[:min(16,mt[626])];r['mt']=mt;r['next']=next
        transitions=np.zeros((16,5),np.int64)
        self.lib.p8f_fixed_transitions(self.data,self.meta,self.constants,ct.byref(self.g),seed,np.ascontiguousarray(state),transitions)
        r['transitions']=transitions;r['maxdepth']=np.array(o.maxdepth,np.int64)
        return r

    def histories(self,n,seed,start=0):
        if self.rng!='philox':raise ValueError('per-history references use Philox')
        dw=self.g.nb+self.g.nc+self.g.nf+37;iw=79+self.g.nb
        d=np.zeros((n,dw));i=np.zeros((n,iw),np.int64)
        self.lib.p8f_histories(self.data,self.meta,self.constants,ct.byref(self.g),seed,start,n,d,i)
        return core.unpack(d,i,self.g.nb,self.g.nc,self.g.nf)


def run_identity(identity,edges,cum):
    from ..run_p3a import SPECTRUM
    return dict(sources=core.fingerprint(),bundle_sha256=core.sha(BUNDLES/'bundles.json'),
                inputs=identity['files'],spectrum_sha256=core.sha(SPECTRUM))

def require_gate(work,name,identity):
    receipt=json.loads((Path(work)/(name+'.json')).read_text())
    if receipt.get('status')!='pass' or receipt.get('identity')!=identity:raise ValueError('current '+name+' pass required')
    return receipt

def stack(work):
    from .. import photon_transport_p8e as k
    from ..run_p8e import LAYERS
    m,i,e,c=inputs();identity=run_identity(i,e,c)
    require_gate(OUT,'gate0',identity)
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    before=time.perf_counter()
    # run_coupled only validates/assembles arguments, then calls _run_coupled;
    # maxdepth is its raw return, not inferred from saved KEYS.
    result=k.run_coupled(m,LAYERS,1_000_000,core.SEED0,densities=i['densities'],spectrum_edges=e,spectrum_cum=c)
    numba_seconds=time.perf_counter()-before
    if not k.healthy(result):raise AssertionError('0f Numba health')
    maximum=int(result['maxdepth']);cap=max(16,2*maximum)
    if cap>4096:raise ValueError('stack capacity exceeds tested maximum')
    cpu=CPU(work/'build',m,densities=i['densities'],edges=e,cum=c,cap=cap)
    good=cpu.run(10000,core.SEED0)
    if not k.healthy(good):raise AssertionError('0f normal build health')
    bad=CPU(work/'overflow',m,densities=i['densities'],edges=e,cum=c,cap=1).run(100,core.SEED0)
    if not bad['counts'][6] or k.healthy(bad):raise AssertionError('overflow not rejected')
    projected=float(good['seconds'])/10000*10_000_000
    report=dict(status='pass',identity=identity,N=1_000_000,seed=core.SEED0,maxdepth=maximum,capacity=cap,
        overflow_count=int(bad['counts'][6]),numba_seconds=numba_seconds,
        cpp_us_per_history=float(good['seconds'])*100,reference_seconds=projected,
        reference_schedule='prepare shared references with P8F_BATCH_RANGE=0:1, then two processes with 1:50 and 50:100' if projected>10800 else 'one process, resumable outer batches')
    write_json(work/'stack.json',report);print(json.dumps(report,indent=2))


def capacity(work,identity):
    row=require_gate(work,'stack',identity)
    cap=row['capacity']
    if cap<16 or cap<2*row['maxdepth']:raise ValueError('invalid measured capacity')
    return cap


def gate1(work):
    from .. import photon_transport_p8e as k
    from ..run_p8e import LAYERS,_verify_prototype
    from ..batch_runner_p3u import batch_seeds
    work=Path(work);m,i,e,c=inputs();identity=run_identity(i,e,c)
    require_gate(OUT,'gate0',identity);cap=capacity(work,identity)
    report=json.loads((ROOT/'docs/validation/mv/p8e/proto_production.json').read_text())
    config=json.loads((PRODUCTION/'config.json').read_text());_verify_prototype(PRODUCTION,report,config['params'])
    seeds=batch_seeds(820003,100);cpu=CPU(work/'build',m,densities=i['densities'],edges=e,cum=c,cap=cap)
    rows=[]
    for b in (0,50,99):
        with np.load(PRODUCTION/f'batch_{b:03}.npz') as f:expected={key:f[key].copy() for key in core.KEYS}
        actual=cpu.run(85646,seeds[b])
        for key in core.KEYS:core.bitwise(actual[key],expected[key],f'1a batch {b} {key}')
        if not k.healthy(actual):raise AssertionError('1a health')
        branches=dict(photon_boundary=int(actual['branches'][0]),photoelectric=int(actual['counts'][9]),
            fluorescence_emission=int(actual['fluorescence_counts'][0]),fluorescence_escape=int(actual['fluorescence_counts'][1]),
            material_boundary=int(actual['boundary_counts'][0]),omitted=int(actual['boundary_counts'][1]),reflection=int(actual['boundary_counts'][4]))
        rows.append(dict(batch=b,seed=seeds[b],branches=branches))
    for key in rows[0]['branches']:
        if sum(row['branches'][key] for row in rows)==0:raise AssertionError('1a branch not exercised: '+key)
    # 1b uses the legacy analytic model, not production ESTAR water.
    legacy=k.legacy_bundle();water=CPU(work/'water',[legacy],layers=[(1,30.,.1)],densities=[1.]*4,binding=False,fluorescence=False,cap=cap)
    expected=k.run_coupled([legacy],[(1,30.,.1)],10000,2026102082,binding_enabled=False,fluorescence_enabled=False)
    actual=water.run(10000,2026102082)
    for key in core.KEYS+('coarse','fine'):
        core.bitwise(actual[key],expected[key],'1b '+key)
    # Independent new Numba stream, saved once; every fault must reach transport.
    ref=k.run_coupled(m,LAYERS,20000,core.SEED0,densities=i['densities'],spectrum_edges=e,spectrum_cum=c)
    save(work/'gate1_reference.npz',{key:ref[key] for key in core.KEYS},dict(identity=identity,n=20000,seed=core.SEED0,start=0))
    normal=cpu.run(20000,core.SEED0)
    for key in core.KEYS:core.bitwise(normal[key],ref[key],'1c normal '+key)
    faults={}
    for fault in (1,2,3,4):
        bad=CPU(work/f'fault{fault}',m,densities=i['densities'],edges=e,cum=c,fault=fault,cap=cap).run(20000,core.SEED0)
        differing=[key for key in core.KEYS if bad[key].tobytes()!=ref[key].tobytes()]
        if not differing:raise AssertionError(f'1c fault {fault} undetected')
        faults[str(fault)]=differing
    write_json(work/'gate1.json',dict(status='pass',identity=identity,batches=rows,water='all outputs byte identical',faults=faults))


def reference(work):
    from .. import photon_transport_p8e as k
    work=Path(work);m,i,e,c=inputs();identity=run_identity(i,e,c)
    require_gate(work,'gate1',identity);cap=capacity(work,identity)
    cpu=CPU(work/'philox'/str(__import__('os').getpid()),m,densities=i['densities'],edges=e,cum=c,rng='philox',cap=cap)
    fixed=work/'reference/fixed.npz'
    if not fixed.exists():
        probes,states=core.fixed_inputs();values={'states':states}
        for j,probe in enumerate(probes):
            out,state,next=cpu.probe(probe['which'],[probe['x']],probe['mid'],core.SEED2)
            values[f'probe_{j}_out']=out;values[f'probe_{j}_state']=state;values[f'probe_{j}_next']=next
        old_nspec=cpu.g.nspec;cpu.g.nspec=0
        for j,a in enumerate(states):
            cpu.g.energy=a[6]
            for key,value in cpu.fixed(a,core.SEED2).items():values[f'case_{j}_{key}']=value
        cpu.g.nspec=old_nspec;cpu.g.energy=1.
        save(fixed,values,dict(identity=identity,seed=core.SEED2,n=len(states),start=0))
    else:load_sealed(fixed,dict(identity=identity))
    path=work/'reference/gate2.npz'
    expected=dict(identity=identity,n=20000,seed=core.SEED2,start=0)
    if path.exists():load_sealed(path,expected)
    else:
        result=cpu.histories(20000,core.SEED2)
        if not core.history_health(result):raise AssertionError('CPU gate2 reference health')
        core.check_local_bins(result['histories'],result['dep']);save(path,result,expected)
    # An outer-batch subset can be selected for a second process. At most two.
    begin,end=map(int,__import__('os').environ.get('P8F_BATCH_RANGE','0:100').split(':'))
    if not 0<=begin<end<=100:raise ValueError('P8F_BATCH_RANGE')
    for b in range(begin,end):
        path=work/f'reference/batch_{b:03}.npz';expected=dict(identity=identity,n=100000,seed=core.SEED3,start=100000*b,batch=b)
        if path.exists():load_sealed(path,expected);continue
        result=cpu.run(100000,core.SEED3,start=b*100000)
        if not k.healthy(result):raise AssertionError('CPU gate3 reference health')
        save(path,result,expected);print('reference batch',b,flush=True)


def export(work):
    import shutil
    work=Path(work);m,i,e,c=inputs();identity=run_identity(i,e,c)
    require_gate(work,'gate1',identity);cap=capacity(work,identity)
    dest=work/'export';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    def copy(src,name=None):
        target=dest/(src.name if name is None else name);shutil.copyfile(src,target);files.append(target.name)
    for name in i['files']:copy(BUNDLES/name)
    copy(BUNDLES/'bundles.json')
    save(dest/'p8f_spectrum.npz',dict(edges=e,cum=c),dict(identity=identity));files+=['p8f_spectrum.npz','p8f_spectrum.json']
    for src in sorted((work/'reference').glob('*.npz')):
        load_sealed(src,dict(identity=identity));copy(src,'p8f_ref_'+src.name);copy(src.with_suffix('.json'),'p8f_ref_'+src.with_suffix('.json').name)
    expected_refs={'p8f_ref_fixed.npz','p8f_ref_gate2.npz',*[f'p8f_ref_batch_{b:03}.npz' for b in range(100)]}
    if not expected_refs.issubset(files):raise ValueError('all CPU references required')
    g,arrays,grid=geometry(__import__('prototypes.mv_electron.run_p8e',fromlist=['LAYERS']).LAYERS,i['densities'],e,c)
    values=dict(arrays,**grid,constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA]))
    values['sizes']=np.array([g.nl,g.nb,g.nc,g.nf,g.nspec],np.int64)
    save(dest/'p8f_inputs.npz',values,dict(identity=identity));files+=['p8f_inputs.npz','p8f_inputs.json']
    from .p5a import load_inputs
    np.savez(dest/'p8f_water_baseline.npz',**load_inputs(full=True));files.append('p8f_water_baseline.npz')
    # The source snapshot excludes Mohan values and includes only immutable
    # dependencies plus these P8f launch tools. Drivers are excluded from identity.
    import zipfile
    with zipfile.ZipFile(dest/'p8f_source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(core.fingerprint())|{'p8f_colab.py'}|{f.name for f in HERE.glob('p5*') if f.suffix in ('.py','.h','.cpp','.cu')}):
            z.write(HERE/name,'prototypes/mv_electron/gpu/'+name)
        for name in ('__init__.py',):z.writestr('prototypes/'+name,'');z.writestr('prototypes/mv_electron/'+name,'');z.writestr('prototypes/mv_electron/gpu/'+name,'')
    files.append('p8f_source.zip')
    copy(HERE/'p8f_colab.ipynb')
    manifest=dict(status='pass',identity=identity,capacity=cap,densities=i['densities'],files={name:core.sha(dest/name) for name in sorted(files)},
                  reference_seconds=json.loads((work/'stack.json').read_text())['reference_seconds'])
    write_json(dest/'p8f_manifest.json',manifest)
    print('Copy every file below to MyDrive/viveMonte/p8f, wait until all appear, then T4 / Run all:')
    print('\n'.join(sorted(files)+['p8f_manifest.json']))


def analyse(work):
    from .. import run_p8e as old
    from .. import photon_transport_p8e as k
    work=Path(work);m,i,e,c=inputs();identity=run_identity(i,e,c)
    require_gate(work,'gate1',identity)
    require_gate(work/'results','gate2',identity);require_gate(work/'results','gate3',identity)
    completed=require_gate(work/'results','complete',identity)
    if completed['n']!=20000000 or completed['seed']!=core.SEED_PRODUCTION or completed['nbatch']!=100:raise ValueError('production completion range')
    grid=old.layout(old.LAYERS,i['densities'])
    # Frozen reference readers verify both seals, N/seed, geometry and bundles.
    egs,egs_report=old.load_egs_result(ROOT/'docs/validation/mv/p8e',Path.home()/'.cache/chatcarlo/p8e/egs',grid,i['densities'])
    report=json.loads((ROOT/'docs/validation/mv/p8e/proto_production.json').read_text());config=json.loads((PRODUCTION/'config.json').read_text())
    old._verify_prototype(PRODUCTION,report,config['params'])
    if report['N']!=8564600 or report['seed']!=820003:raise ValueError('MT reference range')
    mt=old._read_result(ROOT/'docs/validation/mv/p8e/proto_production.npz',report,8564600,820003,grid,i['densities'])
    doses=[];boundary=[];fluor=[];seconds=[]
    for b in range(100):
        r,receipt=load_sealed(work/f'results/production/batch_{b:03}.npz',dict(identity=identity,n=200000,seed=core.SEED_PRODUCTION,start=200000*b,batch=b))
        if not k.healthy(r):raise AssertionError('unhealthy production; no judgement')
        doses.append(r['dose']);boundary.append(r['boundary_counts']);fluor.append(r['fluorescence_counts']);seconds.append(float(r['seconds']))
    gpu=np.asarray(doses)
    def ratio(a,b):
        return [float(x/y) if y else None for x,y in zip(a,b)]
    result=dict(A=core.judge(gpu,egs,grid,dref=core.DREF),B=core.judge(gpu,mt,grid,dref=core.DREF),
        B_description='CPU to GPU port as a whole, including MT to Philox',identity=identity,N=20000000,reference_N={'A':egs_report['N'],'B':report['N']},
        report_only=dict(boundary_counts=np.sum(boundary,axis=0).tolist(),fluorescence_counts=np.sum(fluor,axis=0).tolist(),
            boundary_ratio=ratio(np.sum(boundary,axis=0),report['boundary_count']),
            fluorescence_ratio=ratio(np.sum(fluor,axis=0),report['fluorescence_counts']),
            boundary_per_history_ratio=ratio(np.sum(boundary,axis=0)*report['N']/20000000,report['boundary_count']),
            fluorescence_per_history_ratio=ratio(np.sum(fluor,axis=0)*report['N']/20000000,report['fluorescence_counts']),
            buildup_0_3=gpu[:,:30].mean(axis=0).tolist(),lead_bins=gpu[:,-10:].mean(axis=0).tolist(),buildup_difference_over_Dref=((gpu.mean(0)[:30]-egs.mean(0)[:30])/core.DREF).tolist(),lead_difference_over_Dref=((gpu.mean(0)[-10:]-egs.mean(0)[-10:])/core.DREF).tolist(),seconds=sum(seconds)))
    result['status']='pass' if all(v=='pass' for j in ('A','B') for v in result[j]['status']) else 'fail' if any(v=='fail' for j in ('A','B') for v in result[j]['status']) else 'insufficient-statistics'
    write_json(OUT/'judgement.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('local','stack','gate1','reference','export','analyse'))
    parser.add_argument('--work',type=Path,default=CACHE)
    args=parser.parse_args()
    if args.command=='local':
        # Gate 0a-e builds and reports under the repository; never writes ~/.cache.
        import pytest
        code=pytest.main([str(HERE.parent/'tests/test_p8f_gates.py'),str(HERE.parent/'tests/test_p8f_contracts.py'),'-q','-s'])
        if code==0:
            _,i,e,c=inputs();write_json(OUT/'gate0.json',dict(status='pass',identity=run_identity(i,e,c),gates=['0a','0b','0c','0d','0e'],deferred=['0f']))
        raise SystemExit(code)
    globals()[args.command](args.work)

if __name__=='__main__':main()
