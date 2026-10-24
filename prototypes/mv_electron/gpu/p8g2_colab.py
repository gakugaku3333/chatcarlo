"""Claude-only T4 gates, alternating timing, and one full ncu capture."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from . import p8g2_core as core,p8g_colab as old

write_json,save,read=old.write_json,old.save,old.read

class GPU:
    def __init__(self,cp,materials,arrays,capacity=18,fault=0,check=False,blocks=40,threads=128):
        if blocks<1 or threads not in (32,128,256) or not 1<=capacity<=4096:raise ValueError('launch/capacity')
        self.cp,self.arrays,self.blocks,self.threads=cp,arrays,blocks,threads
        self.capacity,self.fault,self.check=capacity,fault,check;self.sizes=tuple(map(int,arrays['sizes']))
        nl,nb,nc,nf,nspec=self.sizes
        if (nl,nb,nc,nf)!=(6,220,108,430):raise ValueError('registered stack only')
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
    bundle=Path(bundle);manifest=json.loads((bundle/'p8g2_manifest.json').read_text())
    if manifest['status']!='pass' or manifest['capacity']!=18 or manifest['identity']['sources']!=core.fingerprint():raise ValueError('sealed current export required')
    for name,digest in manifest['files'].items():
        if core.sha(bundle/name)!=digest:raise ValueError('export hash '+name)
    with np.load(bundle/'p8f_inputs.npz') as f:arrays={k:f[k].copy() for k in f.files}
    materials=[]
    for name in ('water','lung','bone','lead'):
        with np.load(bundle/(name+'_bundle.npz')) as f:materials.append(SimpleNamespace(**{k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in core.FIELDS}))
    return manifest,materials,arrays


def no_spectrum(arrays):
    arrays={**arrays,'sizes':arrays['sizes'].copy(),'edges':np.empty(0),'cum':np.empty(0)};arrays['sizes'][-1]=0
    return arrays


def reject(action,label):
    try:action()
    except AssertionError as error:return str(error)
    raise AssertionError(label+' not rejected')


def run(bundle,output):
    import cupy as cp
    from .p5b_colab import environment
    bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest,materials,arrays=load(bundle);identity=manifest['identity'];env=environment(cp)
    if 'T4' not in env['gpu']:raise ValueError('T4 required')
    ref=read(bundle/'p8g2_ref_gate2.npz',dict(identity=identity,n=20000,seed=core.SEED2,start=0))
    original=old.GPU(cp,materials,arrays,18,threads=256)
    candidates=[];receipts=[]
    for blocks,threads in core.LAUNCHES:
        gpu=GPU(cp,materials,arrays,blocks=blocks,threads=threads);candidates.append(gpu)
        ag,h,v,_=gpu.run(20000,core.SEED2,history=True);core.check_visits(v)
        comparison=core.compare_histories(ref,h)
        if comparison['status']!='pass':raise AssertionError('2a CPU/GPU')
        _,expected=original.run(20000,core.SEED2,history=True);core.exact_histories(h,expected,'2b same VM P8g-1');core.check_group_records(ag,h,20000)
        # Timed path uses the same transitions/finalization/atomic accumulation.
        ag,_,v,_=gpu.run(20000,core.SEED2,history=False);core.check_visits(v);core.check_group_records(ag,h,20000)
        slots=blocks*threads
        for n,start in ((slots-1,0),(slots,0),(slots+1,0),(slots+1,7000),(slots+1,7000)):
            ag,h,v,_=gpu.run(n,core.SEED2,start,history=True);core.check_visits(v)
            _,expected=original.run(n,core.SEED2,start,history=True);core.exact_histories(h,expected,f'2d {blocks}/{threads}/{n}/{start}');core.check_group_records(ag,h,n)
        receipts.append(dict(blocks=blocks,threads=threads,capacity=18,options=gpu.options,comparison=comparison))
        write_json(output/f'gate2_{blocks}x{threads}.json',dict(status='pass',identity=identity,configuration=receipts[-1]))
    # Check builds keep every atomic counter in caller-allocated global memory.
    checked=GPU(cp,materials,arrays,check=True)
    _,ch,v,_=checked.run(2000,core.SEED2,7000,history=True);core.check_visits(v)
    _,expected=original.run(2000,core.SEED2,7000,history=True);core.exact_histories(ch,expected,'checked normal')
    paths=[];total=np.zeros(80,np.uint64);fixed_arrays=no_spectrum(arrays)
    for case in core.fixed_cases():
        gpu=GPU(cp,materials,fixed_arrays,capacity=case['cap'],check=True)
        kw=dict(start=case['start'],maxsteps=case['maxsteps'],injected=np.array([case['state']]),history=True)
        _,h,v,hits=gpu.run(1,core.SEED2,**kw);core.check_visits(v)
        _,expected,rv,_=gpu.run(1,core.SEED2,reference=True,**kw);core.check_visits(rv);core.exact_histories(h,expected,'2e '+case['name'])
        for idx in case['targets']:
            if not hits[idx]:raise AssertionError('unreached '+case['name']+' '+core.PATHS[idx])
        total+=hits;paths.append(dict(case,counts=hits.tolist()))
    for name,idx in core.TRANSITIONS.items():
        if not total[idx]:raise AssertionError('missing transition '+name)
    # 2000-history rare-path reference wrappers, unrestricted capacity and maxsteps.
    for cap,maxsteps in ((18,1),(3,10000000)):
        gpu=GPU(cp,materials,arrays,capacity=cap,check=True)
        _,h,v,_=gpu.run(2000,core.SEED2,7000,history=True,maxsteps=maxsteps);core.check_visits(v)
        _,expected,rv,_=gpu.run(2000,core.SEED2,7000,history=True,maxsteps=maxsteps,reference=True);core.check_visits(rv);core.exact_histories(h,expected,'2e small limits')
    write_json(output/'gate2_paths.json',dict(status='pass',identity=identity,cases=paths,transitions={k:int(total[v]) for k,v in core.TRANSITIONS.items()}))
    faults=[];low=core.fixed_cases()[0];states=np.tile(low['state'],(64,1))
    good=GPU(cp,materials,fixed_arrays,blocks=1,threads=32,check=True)
    ag,expected,v,_=good.run(64,core.SEED2,history=True,injected=states);core.check_visits(v);core.check_group_records(ag,expected,64)
    for label,fault,target in (('g2',2,31),('g3a',3,32),('g3b',6,33),('g4',4,34),('g5',5,35)):
        bad=GPU(cp,materials,fixed_arrays,fault=fault,blocks=1,threads=32,check=True)
        ag,h,v,hits=bad.run(64,core.SEED2,history=True,injected=states)
        if not hits[target]:raise AssertionError('fault target unreached '+label)
        action=(lambda:core.exact_histories(h,expected,'2f g2')) if fault==2 else (lambda:core.check_visits(v)) if fault in (3,6) else (lambda:core.check_group_records(ag,h,64))
        faults.append(dict(fault=label,build='success',target=int(hits[target]),rejected_by=reject(action,label)))
    write_json(output/'gate2_faults.json',dict(status='pass',identity=identity,faults=faults))
    write_json(output/'gate2.json',dict(status='pass',identity=identity,environment=env,configurations=receipts))
    # Every correctness gate must finish before warmup/alternating timing.
    variants=[('p8f',40,128,old.GPU(cp,materials,arrays,18,original=True)),('p8g1',40,256,original)]+[('p8g2',g.blocks,g.threads,g) for g in candidates]
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id);launches=[];timings=[]
    for name,b,t,gpu in variants:
        attrs=gpu.kernel.attributes if name=='p8g2' else gpu.kernels[gpu.prefix+'_kernel'].attributes
        begin=time.perf_counter();gpu.run(20000,core.SEED2);warmup=time.perf_counter()-begin
        launches.append(dict(name=name,blocks=b,threads=t,capacity=18,options=gpu.options,attributes=attrs,initialization_ms=gpu.initialization_ms,warmup_wall_seconds=warmup,resident_blocks_per_SM=old.resident_blocks(attrs,t,props)))
    for n in (20000,200000):
        for repeat in range(3):
            for name,b,t,gpu in variants:
                result=gpu.run(n,core.SEED2)
                if name=='p8g2':core.check_visits(result[2])
                from .p8f_colab import run_health
                if not run_health(result[0]):raise AssertionError('timing health '+name)
                timings.append(dict(name=name,blocks=b,threads=t,n=n,repeat=repeat,us_per_history=float(result[0]['seconds'])*1e6/n))
                write_json(output/'timing_partial.json',dict(status='report',identity=identity,environment=env,launches=launches,alternating=timings))
    all_medians=[dict(name=name,blocks=b,threads=t,n=n,median_us=float(np.median([r['us_per_history'] for r in timings if r['name']==name and r['blocks']==b and r['threads']==t and r['n']==n]))) for n in (20000,200000) for name,b,t,_ in variants]
    baseline_attrs=original.kernels['p8g_kernel'].attributes
    resource_notes=[]
    for launch in launches:
        if launch['name']!='p8g2':continue
        base_resident=old.resident_blocks(baseline_attrs,launch['threads'],props)
        if launch['resident_blocks_per_SM']<base_resident:resource_notes.append(dict(configuration=[launch['blocks'],launch['threads']],resident_blocks_decreased=True,baseline=base_resident,actual=launch['resident_blocks_per_SM']))
        if launch['attributes']['local_size_bytes']>baseline_attrs['local_size_bytes']:resource_notes.append(dict(configuration=[launch['blocks'],launch['threads']],local_memory_increased=True,baseline_bytes=baseline_attrs['local_size_bytes'],actual_bytes=launch['attributes']['local_size_bytes'],inspect='ncu local loads/stores and related stalls; no register cap'))
    medians=[]
    for n in (20000,200000):
        base=float(np.median([r['us_per_history'] for r in timings if r['n']==n and r['name']=='p8g1']))
        for b,t in core.LAUNCHES:
            med=float(np.median([r['us_per_history'] for r in timings if r['n']==n and r['name']=='p8g2' and r['blocks']==b and r['threads']==t]))
            medians.append(dict(n=n,blocks=b,threads=t,median_us=med,speedup_from_P8g1_to_B=base/med))
    write_json(output/'timing.json',dict(status='report',label='P8g-1 から B 全体への変更',identity=identity,environment=env,launches=launches,alternating=timings,medians=medians,all_medians=all_medians,resource_notes=resource_notes))
    ncu=shutil.which('ncu')
    if not ncu:raise RuntimeError('ncu unavailable; full capture remains incomplete')
    subprocess.run([ncu,'--set','full','--target-processes','all','--kernel-name','regex:p8g2_kernel','--launch-skip','1','--launch-count','1','--force-overwrite','--export',str(output/'p8g2_full'),'--csv','--log-file',str(output/'p8g2_full.csv'),sys.executable,'-m','prototypes.mv_electron.gpu.p8g2_colab',str(bundle),str(output),'--profile-only'],check=True,timeout=1800)
    write_json(output/'profile.json',dict(status='report',identity=identity,configuration=[40,256],raw={name:core.sha(output/name) for name in ('p8g2_full.ncu-rep','p8g2_full.csv')},inspect=['active threads','occupancy','local memory loads/stores','local-memory-related stalls'],comparison=['p8f_prof','P5k F']))


def profile(bundle,output):
    import cupy as cp
    manifest,m,a=load(bundle);gate=json.loads((Path(output)/'gate2.json').read_text())
    if gate['status']!='pass' or gate['identity']!=manifest['identity']:raise ValueError('matching gate2 required')
    gpu=GPU(cp,m,a,blocks=40,threads=256)
    gpu.run(20000,core.SEED2);gpu.run(20000,core.SEED2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--profile-only',action='store_true');args=parser.parse_args()
    (profile if args.profile_only else run)(args.bundle,args.output)

if __name__=='__main__':main()
