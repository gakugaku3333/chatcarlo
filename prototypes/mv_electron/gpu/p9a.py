"""P9a variable-material driver using the unchanged P8g-2 transport."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from . import p8g as frozen, p8g2_core as core
from . import p8g3_core as storage_helpers
from .. import physics, p9a as reference, p9a_data as data
HERE = Path(__file__).resolve().parent
D,I,IP,DP = frozen.D,frozen.I,frozen.IP,frozen.DP
U = np.ctypeslib.ndpointer(np.uint64,flags='C_CONTIGUOUS')
Geometry, Output = frozen.Geometry, frozen.Output
write_json = storage_helpers.write_json
GPU_SEED = 920101
PILOT_SEED = 920102
CALL_N = 200000

def geometry(layers,densities,edges=None,cum=None,binding=True,fluorescence=True,energy=1.):
    from ..p9a import layout
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
    if nb>220 or nc>108 or nf>430:raise ValueError('scoring capacity')
    if (not np.isfinite(energy) or energy<=0 or energy>10 or
        (edges is None)!=(cum is None)):raise ValueError('source')
    if cum is not None and (len(cum)<1 or len(cum)>64 or len(edges)!=len(cum)+1 or cum[-1]!=1 or cum[0]<=0 or np.any(np.diff(cum)<=0) or np.any(np.diff(edges)<=0) or edges[0]<0 or edges[-1]>10 or not np.isfinite(edges).all() or not np.isfinite(cum).all()):raise ValueError('spectrum')
    g=Geometry(*[arrays[k].ctypes.data_as(t) for k,t in Geometry._fields_[:8]],len(ids),nb,nc,nf,len(arrays['cum']),int(binding),int(fluorescence),energy)
    return g,arrays,grid

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


from .p9a_spec import shapes_from_sizes


def input_record(materials, bundle_identity, edges, cum, layers, n=CALL_N,
                 seed=GPU_SEED, mode='gpu', call_n=CALL_N):
    from .p9a_store import spec
    if type(call_n) is not int or not 100 <= call_n <= CALL_N or call_n % 100:
        raise ValueError('call history count')
    if type(n) is not int or n < call_n or n % call_n or n > 240000000:
        raise ValueError('GPU histories must be call multiples within cap')
    g, arrays, grid = geometry(layers, bundle_identity['densities'], edges, cum)
    sizes = [g.nl,g.nb,g.nc,g.nf,g.nspec]
    arrays.update(sizes=np.array(sizes,np.int64), mass_g_cm2=grid['mass_g_cm2'],
                  constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA]))
    record = dict(materials=bundle_identity['names'], bundles=bundle_identity,
                  densities=bundle_identity['densities'], layers=[list(x) for x in layers],
                  sizes=sizes, grid={k:v.tolist() for k,v in grid.items()},
                  fine_grid={k:v.tolist() for k,v in reference.fine_layout(layers,bundle_identity['densities']).items()},
                  sources=core.fingerprint(), capacity=18, launch=[40,256],
                  spectrum=dict(edges=np.asarray(edges).tolist(),cum=np.asarray(cum).tolist()),
                  schedule=spec(n//call_n,call_n,mode,seed,shapes_from_sizes(sizes,mode)))
    if len(materials) != len(record['materials']):
        raise ValueError('material order/count')
    record['arrays']={k:v.tolist() for k,v in arrays.items()}
    return record, arrays


def kernel_check():
    passed = json.loads((reference.ROOT/'docs/validation/mv/p8g2/t4/gate2.json').read_text())
    if passed['status'] != 'pass' or passed['identity']['sources'] != core.fingerprint():
        raise ValueError('kernel differs from passed P8g-2; stop')
    return dict(status='pass', sources=core.fingerprint())


def component_checks(probe, material, photon):
    from ..photon_transport_p8e import material_coefficients, shell_distribution
    edges = np.unique(photon['edges'])
    energies = np.unique(np.r_[photon['E'],photon['check_E'],
        np.nextafter(edges,0),edges,np.nextafter(edges,np.inf)])
    got = probe.probe(2,energies,mid=4)[0][:,:4]
    expected = np.array([material_coefficients(float(t),material) for t in energies])
    if not np.allclose(got,expected,rtol=1e-12,atol=0):
        raise AssertionError('seed photon coefficient gate')
    shell_rows=[]
    for energy in (.01,.03,.1,1.,10.):
        expected = shell_distribution(float(energy),material)
        local = float(expected.sum() == 0)
        distribution = probe.probe(6,[energy],mid=4)[0][0,:expected.size].reshape(expected.shape)
        if not np.allclose(distribution,expected,rtol=1e-12,atol=1e-14):
            raise AssertionError('seed shell distribution')
        draws=probe.probe(7,np.full(10000,energy),mid=4)[0][:,:2].astype(int)
        for i in range(expected.shape[0]):
            for j in range(5):
                f=np.mean((draws[:,0]==i)&(draws[:,1]==j));p=expected[i,j]
                if abs(f-p)>4*np.sqrt(p*(1-p)/10000)+1/10000:
                    raise AssertionError('seed shell selection')
        shell_rows.append(dict(energy=energy,local=float(local)))
    relax=[]
    for z in (47,22):
        idx=np.flatnonzero(material.z==z)
        if len(idx)!=1:raise ValueError('bundle Z identity')
        idx=int(idx[0]);binding=material.bindings[idx,0]
        rows=probe.probe(9,np.tile([idx,0,1.],(10000,1)),mid=4)[0]
        # probe records transported photons and the terminal local energy.
        photons=rows[:,3:35:4]
        residual=np.max(abs(rows[:,2]+photons.sum(axis=1)-binding))
        if residual>1e-12:raise AssertionError('seed relaxation energy')
        emitted=np.mean(np.any(photons>=.01,axis=1))
        y=material.yields[idx,0] if z==47 else 0.
        if abs(emitted-y)>4*np.sqrt(y*(1-y)/10000):
            raise AssertionError('seed K fluorescence yield')
        relax.append(dict(Z=z,observed=float(emitted),expected=float(y),max_residual_MeV=float(residual)))
    return dict(status='pass',coefficient_points=len(energies),shells=shell_rows,relaxation=relax)


def local(work, cache=data.CACHE, out=reference.OUT):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    kernel=kernel_check()
    if data.read(Path(out)/'check-data.json')['status'] != 'pass':
        raise ValueError('material data checks must pass before transport gates')
    m,b=data.load_bundles(cache,out)
    from ..run_p3a import load_spectrum
    e,_,c=load_spectrum()
    kw=dict(layers=reference.p8e.LAYERS,densities=b['densities'],edges=e,cum=c)
    old=CPU(work/'build',m[:4],**kw)
    new=CPU(work/'build',m,**kw)
    if old.identity!=new.identity:raise AssertionError('regression binary differs')
    for n,start in ((20000,0),(2000,7000)):
        core.exact_histories(new.histories(n,core.SEED2,start),old.histories(n,core.SEED2,start),'old four materials')
    # Probe wrapper's geometry is irrelevant to these material-only functions.
    probe=frozen.CPU(work/'probe',m,layers=reference.p8e.LAYERS,
                     densities=b['densities'][:4],rng='philox',cap=18)
    checks=component_checks(probe,m[4],data.read(Path(cache)/data.NAME/'photon.npz'))
    slab=CPU(work/'build',m,layers=reference.SLAB_LAYERS,densities=b['densities'],edges=e,cum=c)
    h,v,_,ag=slab.histories(2000,core.SEED2,details=True)
    core.check_visits(v)
    from .p8f_colab import run_health
    if not run_health(ag) or int(ag['maxdepth'])>18:raise AssertionError('seed slab CPU health')
    frozen.save(work/'cpu_slab.npz',h,dict(n=2000,seed=core.SEED2,start=0,sources=core.fingerprint()))
    report=dict(status='pass',kernel=kernel,components=checks,bundles=b,
                regression=[[20000,0],[2000,7000]],gpu_slab='pending T4 export gate')
    write_json(Path(out)/'check.json',report)
    return report


def export(work, cache=data.CACHE, out=reference.OUT, layout_name='stack',
           n=CALL_N, seed=PILOT_SEED, phase='pilot'):
    data.verify_protected(out)
    data.verified('bundle',out)
    gate=data.read(Path(out)/'check.json')
    kernel_check()
    if gate['status']!='pass':raise ValueError('CPU component gate required')
    m,b=data.load_bundles(cache,out)
    if gate['bundles']!=b:raise ValueError('CPU gate bundle identity')
    from ..run_p3a import load_spectrum
    from ..photon_transport_p8e import Material
    e,_,c=load_spectrum()
    layers={'old':reference.p8e.LAYERS,'slab':reference.SLAB_LAYERS,'stack':reference.LAYERS}[layout_name]
    record,arrays=input_record(m,b,e,c,layers,n,seed,call_n=2000 if phase=='component' else CALL_N)
    record['phase']=phase
    if phase=='component':
        if n!=2000 or seed!=core.SEED2:
            raise ValueError('component gate requires 2000 histories and P8g-2 SEED2')
    elif phase=='production':
        forecast=data.read(Path(out)/'gpu_projection.json')
        if forecast['status']!='ready' or n!=forecast['histories'] or seed!=GPU_SEED:
            raise ValueError('production must match reviewed allocation')
    elif seed!=PILOT_SEED or n!=CALL_N:
        raise ValueError('pilot must use independent fixed stream and 200000 histories')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    write_json(work/'input.json',record,immutable=True)
    np.savez(work/'inputs.npz',**arrays)
    for name,material in zip(record['materials'],m):
        np.savez(work/(name+'_bundle.npz'),**dict(zip(Material._fields,material)))
    cpu=CPU(work/'build',m,layers=layers,densities=b['densities'],edges=e,cum=c)
    frozen.save(work/'cpu_reference.npz',cpu.histories(2000,core.SEED2),dict(input=record,n=2000,seed=core.SEED2,start=0))
    with zipfile.ZipFile(work/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        names=set(core.fingerprint())|{p.name for p in HERE.glob('*.py')}
        # Imported CPU/data modules must exist even though Colab never executes acquisition.
        for name in sorted(names):z.write(HERE/name,'prototypes/mv_electron/gpu/'+name)
        for p in HERE.parent.glob('*.py'):z.write(p,'prototypes/mv_electron/'+p.name)
        z.writestr('prototypes/__init__.py','')
    files=[work/'input.json',work/'inputs.npz',work/'source.zip',work/'cpu_reference.npz',work/'cpu_reference.json',
           *(work/(name+'_bundle.npz') for name in record['materials'])]
    write_json(work/'manifest.json',dict(input=record,files={p.name:core.sha(p) for p in files}),immutable=True)
    data.verify_protected(out)
    return dict(status='exported',folder=str(work),input=record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('local','export'))
    p.add_argument('--work',type=Path,default=data.CACHE/'gpu')
    p.add_argument('--cache',type=Path,default=data.CACHE)
    p.add_argument('--out',type=Path,default=reference.OUT)
    p.add_argument('--layout',choices=('old','slab','stack'),default='stack')
    p.add_argument('--n',type=int,default=CALL_N)
    p.add_argument('--seed',type=int,default=PILOT_SEED)
    p.add_argument('--phase',choices=('component','pilot','production'),default='pilot')
    a=p.parse_args()
    result=local(a.work,a.cache,a.out) if a.command=='local' else export(a.work,a.cache,a.out,a.layout,a.n,a.seed,a.phase)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
