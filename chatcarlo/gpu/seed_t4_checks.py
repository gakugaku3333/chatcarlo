"""B0–B2 on fresh T4. No old T0 or historical V3 exception is inherited."""
from pathlib import Path
import numpy as np
from . import seed_geometry as core,seed_geometry_checks as checks,seed_checks,seed_gates,seed_validation,seed_boundary_checks,seed_run,contracts,therapy
from .seed_core import CPU
from .seed_cuda import GPU


class HostProbe:
    def __init__(self,e):self.e=e;self.grids=e.grids
    def clear(self):self.e.clear()
    def boundary(self,a):return self.e.boundary(a)
    def source(self,s,d):return self.e.source(s,d)
    def score(self,*a,**kw):return self.e.score(*a,**kw)
    @property
    def values(self):return [self.e.cp.asnumpy(v) for v in self.e.values]


def retry(name,fn):
    attempts=[]
    for seed in (9502101,9502102):
        try:
            r=fn(seed);attempts.append(dict(seed=seed,result=r))
            if r['status']=='pass':return dict(r,attempts=attempts)
            if name=='V2b' and r['status']=='insufficient-statistics':continue
            raise AssertionError(name+' failed')
        except AssertionError as exc:
            attempts.append(dict(seed=seed,error=str(exc)))
            if seed==9502102:raise
    return dict(status='insufficient-statistics',fallback='V2a',attempts=attempts)


def faults(factory):
    reports={};g=core.Grid((-2.,)*3,(.1,)*3,(40,)*3);coarse=core.Grid((-20.,)*3,(40.,)*3,(1,)*3)
    state=np.zeros((1,23));state[0,:3]=(.125,.125,.125);state[0,3]=1.;state[0,6]=.004
    for fault in (0,1,2,3,4):
        e=factory(fault,grids=(g,coarse),energy=.004,cylinders=())
        try:
            r=e.run(1,9502101,injected=state)
            values=r['grids'][0]
            if hasattr(e,'cp'):values=e.cp.asnumpy(values)
            want=np.zeros(g.shape);want[g.index(state[0,:3])]=.004
            np.testing.assert_allclose(values,want,rtol=1e-12,atol=0)
        except (AssertionError,ValueError):
            if fault==0:raise
            reports[str(fault)]='detected'
        else:
            if fault:raise AssertionError('undetected seed fault '+str(fault))
    return dict(status='pass',faults=reports)


def run(cp,materials,inputs,work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    common=dict(densities=inputs['densities'],constants=inputs['constants'],strength=inputs['strength'])
    tiny=(checks.SMALL,core.Grid((-20.,)*3,(10.,)*3,(4,)*3))
    e=GPU(cp,materials,**common,grids=(checks.SMALL,checks.SMALL),cylinders=((2,2,2,.05,.25,4),))
    hp=HostProbe(e);report=dict(V0=checks.v0(hp),V1a=checks.v1a(hp),V1b=seed_gates.v1b(e),V2a=checks.v2a(hp))
    actual=GPU(cp,materials,**common,grids=(checks.SMALL,checks.SMALL));report['V6']=checks.v6(HostProbe(actual));report['G3_grazing']=checks.grazing_seed(HostProbe(actual));report['V6b']=seed_validation.v6b()
    def acceptance(seed):
        report=seed_gates.v3_acceptance(seed)
        sampler=GPU(cp,materials,**common,grids=tiny,beam='prostate')
        for f,angle in enumerate(core.ANGLES):
            sampler.g.angle=float(np.deg2rad(angle));sampler.upload();r=sampler.run(100000,seed,field=f)
            accepted=100000;trials=accepted+int(r['source_counts'][:,0].sum())
            p=report['angles'][str(angle)]['expected'];observed=accepted/trials;sigma=float(np.sqrt(p*(1-p)/trials))
            if abs(observed-p)>3*sigma:raise AssertionError('V3 native CUDA acceptance at '+str(angle))
            report['angles'][str(angle)]['native']=dict(accepted=accepted,trials=trials,observed=observed,sigma=sigma)
        return report
    report['V3']=retry('V3',acceptance)
    for beam,angle in (('rectangle',0),('prostate',45)):
        kw=dict(common,edges=inputs['mohan10_edges'],cum=inputs['mohan10_cum'],grids=tiny,beam=beam,angle=angle)
        cpu=CPU(work/'build',materials,**kw);gpu=GPU(cp,materials,**kw)
        a=cpu.run(2000,9502101,field=core.ANGLES.index(angle));b=gpu.run(2000,9502101,field=core.ANGLES.index(angle))
        report['V4_'+beam]=seed_checks.compare_histories(a['histories'],b['histories'])
        # B0 observes actual device table contents and zero-based seed ID 4.
        contracts.check_observation(b['observed'],a['observed'],'B0 CPU/GPU native input')
    report['B0']=dict(status='pass',material_ids=[0,1,2,3,4])
    report['V5']=seed_validation.v5(work,materials,inputs['densities'],inputs['constants'],inputs['strength'])
    # Transport needs a coarse grid that covers the body; SMALL/SMALL is for probes only.
    actual=GPU(cp,materials,**common,grids=tiny)
    actual.adds=[cp.zeros(g.shape,np.uint64) for g in actual.grids];actual.upload()
    actual.run(100,9502101);first=[cp.asnumpy(v) for v in actual.values];adds=[cp.asnumpy(v) for v in actual.adds]
    actual.run(100,9502101)
    for j,v in enumerate(actual.values):
        other=cp.asnumpy(v);bound=(adds[j]+cp.asnumpy(actual.adds[j]))*2**-52*np.maximum(first[j],other)
        if np.any(abs(first[j]-other)>bound):raise AssertionError('V5 GPU identical pair')
    report['V5_GPU']=dict(status='pass')
    report['B1']=dict(status='pass',cpu=faults(lambda fault,**kw:CPU(work/'faults',materials,**common,fault=fault,**kw)),gpu=faults(lambda fault,**kw:GPU(cp,materials,**common,fault=fault,**kw)))
    cache={}
    def factory(kw):
        key=repr(kw)
        if key not in cache:cache[key]=GPU(cp,materials,**{**common,**kw},grids=tiny)
        return cache[key]
    report['G3']=seed_boundary_checks.check(factory,inputs['densities'])
    fg=therapy.FieldGeometry()
    for name in ('lo','hi','origins','widths','axes','sizes','offsets'):getattr(fg,name)[:]=inputs['field_'+name]
    record=dict(inputs,densities=inputs['densities'],edges=inputs['mohan10_edges'],cum=inputs['mohan10_cum'],scoring=fg)
    report['V1c']=retry('V1c',lambda seed:seed_gates.v1c(cp,materials,dict(record,seed=seed-1),work))
    report['V2b']=retry('V2b',lambda seed:seed_gates.v2b(cp,materials,dict(record,seed=seed-2),work))
    report['G5']=dict(status='pass',source='V1c region/grid and six fixed transport sites')
    seed_run.require_gates(report);contracts.write(work/'gates.json',report);return report
