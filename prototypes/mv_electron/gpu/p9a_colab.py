"""P9a T4 execution; shape and invocation come from the sealed input record."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import p8g2_core as core, p9a_spec as driver, p9a_store as store
from .p8f_colab import run_health
from .p5b_colab import environment

class GPU:
    def __init__(self,cp,materials,arrays,capacity=18,fault=0,check=False,blocks=40,threads=128):
        if blocks<1 or threads not in (32,128,256) or not 1<=capacity<=4096:raise ValueError('launch/capacity')
        self.cp,self.arrays,self.blocks,self.threads=cp,arrays,blocks,threads
        self.capacity,self.fault,self.check=capacity,fault,check;self.sizes=tuple(map(int,arrays['sizes']))
        nl,nb,nc,nf,nspec=self.sizes
        if not (1 <= nl <= 16 and 0 < nb <= 220 and 0 < nc <= 108 and 0 < nf <= 430):raise ValueError('P9a scoring capacity')
        data,meta=core.pack(materials)
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(arrays['constants']))
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.options=core.CUDA_OPTIONS+(f'-DP8G_STACK_CAP={capacity}',f'-DP8G2_FAULT={fault}',f'-DP8G2_CHECK={int(check)}')
        self.kernel=cp.RawKernel(core.cuda_source(),'p8g2_kernel',options=self.options,backend='nvrtc');self.kernel.compile()
        init=cp.RawKernel(core.cuda_source(),'p8g2_init',options=self.options,backend='nvrtc');init.compile()
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record();init((1,),(1,),self.args+(np.int32(len(materials)),));last.record();last.synchronize()
        self.initialization_ms=float(cp.cuda.get_elapsed_time(first,last))

    def run(self,n,seed,start=0,history=False,maxsteps=10000000,reference=False,injected=None):
        cp=self.cp;nl,nb,nc,nf,nspec=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        if not 0<n<=200000 or not 0<=start<start+n<2**64 or not 0<=seed<2**64 or maxsteps<0:raise ValueError('history range')
        if injected is not None:
            injected=np.ascontiguousarray(injected,float)
            if injected.shape!=(n,23) or not np.isfinite(injected).all():raise ValueError('injected states')
        hd=cp.zeros((n if history else 1,dw));hi=cp.zeros((n if history else 1,iw),np.int64)
        groups=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64);absolute=cp.zeros((100,dw))
        next=cp.zeros(1,np.int32);visits=cp.zeros(n,np.int64);hits=cp.zeros(80,np.uint64)
        args=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),next,np.int32(history),np.int32(maxsteps),np.int32(reference),np.uint64(0) if injected is None else cp.asarray(injected),hd,hi,groups,ints,absolute,visits,hits)
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record();self.kernel((self.blocks,),(self.threads,),args);last.record();last.synchronize()
        raw=cp.asnumpy(groups);rawi=cp.asnumpy(ints)
        ag=core.reduce_records(raw,rawi,nb,nc,nf,n,self.arrays['mass_g_cm2'])
        ag.update(seconds=np.array(float(cp.cuda.get_elapsed_time(first,last))/1000),_groups_d=raw,_groups_i=rawi,_absolute=cp.asnumpy(absolute))
        h=core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),nb,nc,nf) if history else None
        return ag,h,cp.asnumpy(visits),cp.asnumpy(hits)


def load(bundle):
    bundle=Path(bundle)
    manifest=store.read_json(bundle/'manifest.json')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or core.sha(bundle/name)!=digest:
            raise ValueError('export hash '+name)
    record=store.read_json(bundle/'input.json')
    if manifest['input']!=record or record['sources']!=core.fingerprint():
        raise ValueError('export input/kernel changed')
    materials=[]
    for name in record['materials']:
        with np.load(bundle/(name+'_bundle.npz'),allow_pickle=False) as f:
            materials.append(SimpleNamespace(**{k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in core.FIELDS}))
    with np.load(bundle/'inputs.npz',allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    # Every transferred array is derived locally from, and bound to, one input.
    expected=record['arrays']
    if set(arrays)!=set(expected) or any(not np.array_equal(arrays[k],expected[k]) for k in expected):
        raise ValueError('export geometry differs from input')
    if record['schedule']['shapes']!=driver.shapes_from_sizes(arrays['sizes']):
        raise ValueError('export shapes differ from geometry')
    return record,materials,arrays


def faults(result,record):
    import copy
    import tempfile
    checks=[]
    def rejected(action,label):
        try:action()
        except (ValueError,AssertionError):checks.append(label);return
        raise AssertionError('fault not rejected '+label)
    for key,index,value in [('violations',0,1),('counts',5,1),('totals',3,1),('maxima',0,1.)]:
        bad={k:v.copy() for k,v in result.items()};bad[key][index]=value
        rejected(lambda:store.check_result(bad,record['schedule']),key)
    bad={k:v.copy() for k,v in result.items()};bad['dose'][0]=np.nan
    rejected(lambda:store.check_result(bad,record['schedule']),'nonfinite')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);schedule=record['schedule']
        sid=store.new_session(root,record,schedule,'CPU test',dict(status='pass'),dict(status='new')) if schedule['mode']=='cpu' else store.new_session(root,record,schedule,'T4',dict(status='pass',n=2000,seed=core.SEED2,comparison='P9a CPU/GPU history comparison'),dict(status='new'))
        actual=store.invocation(0,schedule)
        store.save_call(root,0,result,np.ones(schedule['n'],np.int64),actual,record,schedule,sid)
        receipt=store.call_path(root,0).with_suffix('.json')
        original=store.read_json(receipt)
        bad=copy.deepcopy(original);bad['actual']['start']+=1;store.write_json(receipt,bad)
        rejected(lambda:store.load_call(root,0,record,schedule),'actual interval')
        store.write_json(receipt,original)
        changed=copy.deepcopy(schedule);changed['seed']+=1
        rejected(lambda:store.available(root,record,changed),'schedule')
        path=store.call_path(root,0);path.write_bytes(path.read_bytes()+b'corrupt')
        rejected(lambda:store.load_call(root,0,record,schedule),'artifact hash')
    return checks


def require_trial(root,record):
    trial=store.read_json(Path(root)/'trial.json')
    pilot=trial['input']
    stable=lambda r:{k:v for k,v in r.items() if k not in ('phase','schedule')}
    if (trial['status']!='pass' or stable(pilot)!=stable(record)
            or pilot['schedule']['seed']!=driver.PILOT_SEED
            or pilot['schedule']['count']*pilot['schedule']['n']!=driver.CALL_N
            or trial['resume']['status']!='pass' or not trial['faults']
            or trial['previous_session']==trial['new_session']):
        raise ValueError('matching T4 pilot/restart trial required')
    store.verify(root,pilot,pilot['schedule'])
    return trial


def run(bundle,output,trial=None,limit=None):
    import cupy as cp
    record,m,a=load(bundle);output=Path(output);schedule=record['schedule']
    env=environment(cp)
    if 'T4' not in env['gpu']:raise ValueError('T4 required')
    if record['phase']=='production':require_trial(trial,record)
    gpu=GPU(cp,m,a,blocks=40,threads=256)
    from .p8f_colab import read
    expected=read(Path(bundle)/'cpu_reference.npz',
                dict(input=record,n=2000,seed=core.SEED2,start=0))
    ag,h,v,_=gpu.run(2000,core.SEED2,history=True);core.check_visits(v)
    comparison=core.compare_histories(expected,h)
    if comparison['status']!='pass' or not run_health(ag) or int(ag['maxdepth'])>record['capacity']:
        raise AssertionError('P9a CPU/GPU gate')
    # Validate aggregate merging including the computed signed brems bound.
    core.check_group_records(ag,h,2000)
    gate_result,gate_visits=ag,v
    def runner(**kw):
        if record['phase']=='component':
            if kw!=dict(n=2000,seed=core.SEED2,start=0):raise ValueError('component invocation')
            return gate_result,gate_visits
        ag,_,v,_=gpu.run(**kw);return ag,v
    existing=store.available(output,record,schedule)
    resume=store.resume_check(output,runner,record,schedule)
    sid=store.new_session(output,record,schedule,env['gpu'],
        dict(status='pass',n=2000,seed=core.SEED2,comparison='P9a CPU/GPU history comparison',details=comparison),resume)
    store.drive(output,runner,record,schedule,sid,limit)
    if len(store.available(output,record,schedule))==schedule['count']:
        store.verify(output,record,schedule)
        result,_=store.load_call(output,0,record,schedule)
        if record['phase']=='pilot':
            report=dict(status='pass' if resume['status']=='pass' else 'restart-required',input=record,
                resume=resume,faults=faults(result,record),
                previous_session=next(iter(existing.values()))['session_id'] if existing else None,
                new_session=sid,us_per_history=float(result['seconds'])*1e6/schedule['n'])
            store.write_json(output/'trial.json',report)
    print('P9A_DONE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--trial',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.bundle,a.output,a.trial,a.limit)

if __name__=='__main__':main()
