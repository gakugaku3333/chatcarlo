"""Single-invocation P9b CUDA job: G3 → G4 → G5 → G1 → G2; P9B_DONE on completion."""
import argparse
import ctypes as ct
import json
import time
from pathlib import Path
import numpy as np
from . import p9b_core as core
from .p9b_cases import cases,check_case,rejections
from .p9b import DP,IP

class GPU:
    def __init__(self,cp,materials,**kw):
        self.cp=cp;self.kw=kw;self.g,self.arrays,self.grid=core.geometry(**kw)
        if len(materials)<=int(self.arrays['ids'].max()):raise ValueError('material IDs')
        self.device_arrays={k:cp.asarray(v) for k,v in self.arrays.items()}
        g=core.Geometry.from_buffer_copy(bytes(self.g))
        for k,t in core.SlabGeometry._fields_[:8]:setattr(g,k,ct.cast(int(self.device_arrays[k].data.ptr),t))
        g.seeds=ct.cast(int(self.device_arrays['seeds'].data.ptr),DP);g.parents=ct.cast(int(self.device_arrays['parents'].data.ptr),IP)
        self.geometry=cp.asarray(np.frombuffer(bytes(g),np.uint8).copy())
        data,meta=core.core.pack(materials)
        from .. import physics
        constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA])
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(constants))
        options=core.core.CUDA_OPTIONS+('-DP8G_STACK_CAP=18','-DP8G2_FAULT=0','-DP8G2_CHECK=0')
        self.module=cp.RawModule(code=core.cuda_source(),options=options,backend='nvrtc')
        self.kernel=self.module.get_function('p9b_kernel');init=self.module.get_function('p9b_init')
        init((1,),(1,),self.args+(np.int32(len(materials)),));cp.cuda.Stream.null.synchronize()

    def probe(self,states,directions=None,mode=0,seed=core.core.SEED2,start=0):
        cp=self.cp;x=np.ascontiguousarray(states,float).reshape(-1,23);n=len(x)
        v=np.ascontiguousarray(x[:,3:6] if directions is None else directions,float).reshape(n,3)
        if mode not in (0,1) or not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('probe inputs')
        out=cp.zeros((n,74));self.module.get_function('p9b_probe_kernel')(((n+31)//32,),(32,),self.args+(self.geometry,cp.asarray(x),cp.asarray(v),np.int32(mode),np.uint64(seed),np.uint64(start),np.int32(n),out));return cp.asnumpy(out)

    def injected_probe(self,state,seed=42):
        cp=self.cp;out=cp.zeros(15);adds=cp.zeros(3420,np.uint64)
        self.module.get_function('p9b_injected_kernel')((1,),(1,),self.args+(self.geometry,cp.asarray(state,dtype=float),np.uint64(seed),out,adds))
        return cp.asnumpy(out)

    def score_probe(self,p0,p1,energy,region,site):
        cp=self.cp
        if site not in range(1,7) or not 0<=region<self.g.nl+self.g.ns:raise ValueError('deposit probe')
        result=dict(field=cp.zeros(3420),depth=cp.zeros(self.g.nb),additions=cp.zeros(3420,np.uint64),sites=cp.zeros(6,np.int64),ledger=cp.zeros(6))
        self.module.get_function('p9b_score_probe_kernel')((1,),(1,),(self.geometry,np.int32(region),np.int32(site),cp.asarray(p0,dtype=float),cp.asarray(p1,dtype=float),np.float64(energy),*result.values()))
        return {k:cp.asnumpy(v) for k,v in result.items()}

    def run(self,n,seed,start=0,history=False,maxsteps=10000000,injected=None):
        if type(n) is not int or not 0<n<=200000 or n%100 or not 0<=seed<2**64 or not 0<=start<start+n<2**64 or maxsteps<0:raise ValueError('history range')
        cp=self.cp;g=self.g;dw=g.nb+g.nc+g.nf+37;iw=79+g.nb
        x=None if injected is None else np.ascontiguousarray(injected,float)
        if x is not None and (x.shape!=(n,23) or not np.isfinite(x).all()):raise ValueError('injected states')
        hd=cp.zeros((n if history else 1,dw));hi=cp.zeros((n if history else 1,iw),np.int64)
        groups=cp.zeros((100,dw));gi=cp.zeros((100,iw),np.uint64);visits=cp.zeros(n,np.int64)
        fields=cp.zeros((100,3420));hf=cp.zeros((n if history else 1,3420));adds=cp.zeros((100,3420),np.uint64)
        sites=cp.zeros(6+(6*n if history else 0),np.uint64);ledger=cp.zeros(sites.shape);lateral=cp.zeros(1,np.uint64);next_id=cp.zeros(1,np.int32)
        before,after=cp.cuda.Event(),cp.cuda.Event();before.record()
        self.kernel((40,),(256,),self.args+(self.geometry,np.uint64(seed),np.uint64(start),np.int32(n),next_id,np.int32(history),np.int32(maxsteps),np.uint64(0) if x is None else cp.asarray(x),hd,hi,groups,gi,visits,fields,hf,adds,sites,ledger,lateral))
        after.record();after.synchronize();core.core.check_visits(cp.asnumpy(visits))
        ag=core.core.reduce_records(cp.asnumpy(groups),cp.asnumpy(gi),g.nb,g.nc,g.nf,n,self.grid['mass_g_cm2'])
        ag.update(field=cp.asnumpy(fields),additions_field=cp.asnumpy(adds),sites=cp.asnumpy(sites[:6]),ledger=cp.asnumpy(ledger[:6]),lateral=cp.asnumpy(lateral),seconds=np.array(cp.cuda.get_elapsed_time(before,after)/1000))
        h=None
        if history:
            h=core.core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),g.nb,g.nc,g.nf)
            h.update(field=cp.asnumpy(hf),sites=cp.asnumpy(sites[6:]).reshape(n,6),ledger=cp.asnumpy(ledger[6:]).reshape(n,6))
        return ag,h


def g3_gpu(bundle,cp,materials,densities):
    from .p9b_cases import g3
    engines={}
    def factory(kw):
        key=repr(kw)
        if key not in engines:engines[key]=GPU(cp,materials,**kw)
        return engines[key]
    result=g3(None,materials,densities,factory=factory)
    # Compare the actual random scattering and fixed-angle versions to CPU probes
    # exported locally, without requiring a compiler or original spectra on Colab.
    with np.load(Path(bundle)/'g3_reference.npz',allow_pickle=False) as f:
        for j,case in enumerate(cases(densities)):
            engine=factory(case['kw'])
            for mode in (0,1):
                actual=engine.probe([case['state']],[case['direction']],mode=mode)[0];expected=f[f'{j}_{mode}']
                if not np.array_equal(actual[[39,47,48,49,50]],expected[[39,47,48,49,50]]) or not np.allclose(actual,expected,rtol=0,atol=1e-12):raise AssertionError('G3 CPU/GPU '+case['name'])
    return result


def g5_check(ag,h):
    if not all(np.isfinite(v).all() for r in (ag,h) for v in r.values()):raise AssertionError('G5 nonfinite output')
    if ag['additions_field'].shape!=(100,3420) or ag['additions_field'].dtype!=np.uint64:raise AssertionError('G5 addition count shape/type')
    n=len(h['field']);rows=np.arange(n)*100//n;expected=np.zeros((100,3420));absolute=expected.copy()
    np.add.at(expected,rows,h['field']);np.add.at(absolute,rows,abs(h['field']))
    K=np.bincount(rows,minlength=100)[:,None]
    bound=(ag['additions_field']+K)*2.**-52*absolute
    failures=int(np.count_nonzero(abs(expected-ag['field'])>bound))
    if failures:raise AssertionError(f'G5 {failures} bins exceed actual-addition bound')
    if not np.array_equal(h['sites'].sum(0),ag['sites']):raise AssertionError('G5 site counts')
    limit=(int(h['sites'].sum())+3*n)*2.**-52
    if not np.allclose(h['ledger'].sum(0),ag['ledger'],rtol=limit,atol=1e-12):raise AssertionError('G5 site ledger aggregation')
    if np.any(abs(h['ledger'].sum(1)-h['dep'])>1e-12*h['sources']):raise AssertionError('G5 deposit/source/escape ledger')
    if np.any(abs(h['ledger'].sum(1)-(h['sources']-h['totals'][:,2]-h['totals'][:,3]-h['totals'][:,4]))>1e-12*h['sources']):raise AssertionError('G5 conservation')
    whole=h['field'][:,420:540].sum(1)
    if np.any(abs(whole-h['dep'])>1e-9*h['sources']):raise AssertionError('G5 whole-slab scoring ledger')
    return dict(status='pass',histories=n,exceeding=failures,sites=ag['sites'].tolist(),max_difference=float(np.max(abs(expected-ag['field']))))


def speed_baseline(cp,materials,record,p9b_seconds):
    from . import p9a as old, p9a_colab
    from .. import physics,p9a as ref
    g,arrays,grid=old.geometry(ref.LAYERS,record['densities'],record['edges'],record['cum'])
    arrays.update(sizes=np.array([g.nl,g.nb,g.nc,g.nf,g.nspec],np.int64),mass_g_cm2=grid['mass_g_cm2'],constants=np.array([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA]))
    engine=p9a_colab.GPU(cp,materials,arrays,capacity=18,blocks=40,threads=256)
    ag,_,visits,_=engine.run(2000,core.core.SEED2)
    core.core.check_visits(visits);seconds=float(ag['seconds'])
    return dict(histories=2000,launch=[40,256],P8g2_us=seconds*1e6/2000,P9b_z_us=float(p9b_seconds)*1e6/2000,speed_ratio=seconds/float(p9b_seconds),note='report only; same Colab session')


def run(bundle,output):
    from .p9b_io import load,transport_cases,known_scores
    from .p8f_colab import run_health
    import cupy as cp
    bundle,output=Path(bundle),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('refuse job output overwrite')
    output.mkdir(parents=True,exist_ok=True)
    record,materials=load(bundle);densities=record['densities'];report={}
    report['G3']=g3_gpu(bundle,cp,materials,densities)
    (output/'G3.json').write_text(json.dumps(report['G3'],indent=2)+'\n');print('G3 PASS',flush=True)
    if report['G3']['status']!='pass':raise AssertionError('G3 failed')
    specs=transport_cases(record);engines={};g4={};runs={}
    with np.load(bundle/'cpu_reference.npz',allow_pickle=False) as f:
        for name,kw in specs.items():
            engine=GPU(cp,materials,**kw);engines[name]=engine;ag,h=engine.run(2000,core.core.SEED2,history=True);runs[name]=(ag,h)
            old={k:f[name+'_'+k] for k in core.core.unpack(np.zeros((1,engine.g.nb+engine.g.nc+engine.g.nf+37)),np.zeros((1,79+engine.g.nb),np.int64),engine.g.nb,engine.g.nc,engine.g.nf)}
            old['field']=f[name+'_field']
            g4[name]=core.core.compare_histories(old,h)
            delta=abs(old['field']-h['field'])
            g4[name]['report_only']['field']=dict(fraction=float(np.mean(np.all(delta==0.,axis=1))),max_abs=float(np.max(delta)))
    report['G4']=g4;(output/'G4.json').write_text(json.dumps(g4,indent=2)+'\n')
    if any(r['status']!='pass' for r in g4.values()):raise AssertionError('G4 failed; no production')
    print('G4 PASS',flush=True)
    checks={name:g5_check(*runs[name]) for name in ('seed',)}
    checks['known']=known_scores(engines['seed']);report['G5']=checks
    (output/'G5.json').write_text(json.dumps(checks,indent=2)+'\n');print('G5 PASS',flush=True)
    report['speed_report_only']=speed_baseline(cp,materials,record,runs['z'][0]['seconds'])
    (output/'speed.json').write_text(json.dumps(report['speed_report_only'],indent=2)+'\n')
    # Production starts only after all three preceding gates returned successfully.
    jobs=[('G1_x',dict(specs['x'],score_field=False),13600000,930101),('G1_y',dict(specs['z'],axis=1,score_field=False),13600000,930102),('G2',dict(specs['seed'],seeds=()),20000000,930103)]
    for name,kw,n,seed in jobs:
        dest=output/name;dest.mkdir(exist_ok=True);engine=GPU(cp,materials,**kw)
        config=dict(n=n,seed=seed,call_n=200000,kw=kw,source=record['sources'])
        (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        for j,start in enumerate(range(0,n,200000)):
            ag,_=engine.run(200000,seed,start=start)
            if not run_health(ag):raise AssertionError(name+' production health')
            np.savez_compressed(dest/f'call_{j:03}.npz',**ag)
            print(name,j+1,n//200000,'seconds',float(ag['seconds']),flush=True)
    from .p9b_io import sha
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    (output/'job.json').write_text(json.dumps(dict(status='complete',gates=report,sources=record['sources'],files=files),indent=2)+'\n')
    print('P9B_DONE',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.bundle,a.output)

if __name__=='__main__':main()
