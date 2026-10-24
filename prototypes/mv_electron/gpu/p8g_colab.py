"""Single Run-all T4 workflow; material and reference arrays are hash checked."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from . import p8g_core as core
from . import p8f_core as frozen_core


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda o:o.item() if hasattr(o,'item') else o.tolist())+'\n');temp.replace(path)

def save(path,values,identity):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending.npz');np.savez(temp,**values);temp.replace(path)
    write_json(path.with_suffix('.json'),dict(identity,sha256=core.sha(path)))

def read(path,identity):
    receipt=json.loads(path.with_suffix('.json').read_text())
    if receipt['sha256']!=core.sha(path) or any(receipt.get(k)!=v for k,v in identity.items()):raise ValueError('sealed result mismatch '+str(path))
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}


class GPU:
    def __init__(self,cp,materials,arrays,capacity,fault=0,check=False,blocks=40,threads=128,original=False):
        self.cp=cp;self.arrays=arrays;self.fault=fault
        if blocks<1 or threads not in (128,256):raise ValueError("launch")
        self.blocks,self.threads=blocks,threads;self.original=original
        self.prefix="p8f" if original else "p8g"
        if capacity<16:raise ValueError('measured stack capacity required')
        self.sizes=tuple(map(int,arrays['sizes']));nl,nb,nc,nf,nspec=self.sizes
        if (nl,nb,nc,nf)!=(6,220,108,430):raise ValueError('only the registered P8e stack is supported')
        data,meta=(frozen_core if original else core).pack(materials)
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(arrays['constants']))
        self.options=core.CUDA_OPTIONS+(f'-DP8F_STACK_CAP={capacity}',f'-DP8G_STACK_CAP={capacity}',f'-DP8G_FAULT={fault}',f'-DP8G_CHECK={int(check)}')
        self.kernels={};self.launches={}
        for suffix in ('kernel','fixed_kernel','probe_kernel'):
            name=self.prefix+'_'+suffix
            kernel=cp.RawKernel((frozen_core if original else core).cuda_source(),name,options=self.options,backend='nvrtc');kernel.compile();self.kernels[name]=kernel
            self.launches[name]=dict(options=list(self.options),attributes={k:int(v) for k,v in kernel.attributes.items()})

        self.initialization_ms=0.
        if not original:
            init=cp.RawKernel(core.cuda_source(),'p8g_init',options=self.options,backend='nvrtc');init.compile()
            first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
            init((1,),(1,),self.args+(np.int32(len(materials)),));last.record();last.synchronize()
            self.initialization_ms=float(cp.cuda.get_elapsed_time(first,last))
        self.meta=meta

    def counters(self):
        data=self.cp.asnumpy(self.args[0])
        return np.array([data[q[133]:q[133]+37].view(np.uint64) for q in self.meta])

    def values(self,mid):
        cp=self.cp;q=self.meta[mid];out=cp.full((int(q[133]-q[132]),5),np.nan)
        kernel=cp.RawKernel(core.cuda_source(),'p8g_values',options=self.options,backend='nvrtc')
        kernel((1,),(1,),(*self.args,np.int32(mid),out))
        return cp.asnumpy(self.args[0][q[132]:q[133]]),cp.asnumpy(out)

    def run(self,n,seed,start=0,history=False):
        cp=self.cp;nl,nb,nc,nf,nspec=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        if n<=0 or n>200000 or start<0 or start+n>=2**64:raise ValueError('registered run range')
        hd=cp.zeros((n if history else 1,dw));hi=cp.zeros((n if history else 1,iw),np.int64)
        groups=cp.zeros((100,dw));absolute=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64)
        next=cp.array([self.blocks*self.threads],np.int32)
        args=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),next,np.int32(history),hd,hi,groups,ints,absolute)
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.kernels[self.prefix+'_kernel']((self.blocks,),(self.threads,),args);last.record();last.synchronize()
        seconds=float(cp.cuda.get_elapsed_time(first,last))/1000
        raw=cp.asnumpy(groups);rawi=cp.asnumpy(ints)
        aggregate=core.reduce_records(raw,rawi,nb,nc,nf,n,self.arrays['mass_g_cm2']);aggregate['seconds']=np.array(seconds);aggregate['_groups_d']=raw;aggregate['_groups_i']=rawi;aggregate['_absolute']=cp.asnumpy(absolute)
        histories=core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),nb,nc,nf) if history else None
        return aggregate,histories

    def probe(self,probe,seed):
        cp=self.cp;x=cp.asarray([probe['x']],dtype=np.float64);out=cp.zeros((1,128));state=cp.zeros(626,np.int64);next=cp.zeros(16)
        self.kernels[self.prefix+'_probe_kernel']((1,),(1,),(np.int32(probe['which']),*self.args,np.int32(probe['mid']),x,np.int32(1),np.int32(x.shape[1]),np.uint64(seed),out,state,next))
        return cp.asnumpy(out),cp.asnumpy(state),cp.asnumpy(next)

    def fixed(self,state,seed):
        cp=self.cp;nl,nb,nc,nf,_=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        hd=cp.zeros((1,dw));hi=cp.zeros((1,iw),np.int64);events=cp.zeros((16,10));transitions=cp.zeros((16,5),np.int64)
        self.kernels[self.prefix+'_fixed_kernel']((1,),(1,),self.args+(self.geometry,self.mapping,*map(np.int32,(nl,nb,nc,nf)),np.uint64(seed),cp.asarray(state),hd,hi,events,transitions))
        return core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),nb,nc,nf),cp.asnumpy(events),cp.asnumpy(transitions)


def probe_many(gpu,which,x,mid):
    cp=gpu.cp;x=cp.asarray(x,dtype=np.float64);out=cp.zeros((len(x),128));state=cp.zeros(626,np.int64);nxt=cp.zeros(16)
    gpu.kernels[gpu.prefix+'_probe_kernel']((1,),(1,),(np.int32(which),*gpu.args,np.int32(mid),x,np.int32(len(x)),np.int32(x.shape[1]),np.uint64(core.SEED0),out,state,nxt))
    return cp.asnumpy(out),cp.asnumpy(state),cp.asnumpy(nxt)


def components(gpu,old,materials):
    for which,mid,x in core.component_cases(materials):
        for j,(a,b) in enumerate(zip(probe_many(gpu,which,x,mid),probe_many(old,which,x,mid))):
            core.bitwise(a,b,f'2f function {which} material {mid} output {j}')

    cp=gpu.cp;kernel=cp.RawKernel(core.cuda_source(),'p8g_parts',options=gpu.options,backend='nvrtc')
    for mid,m in enumerate(materials):
        x=core.nodes(m.brems_E);out=cp.zeros((len(x),40))
        kernel((1,),(1,),(*gpu.args,np.int32(mid),cp.asarray(x),np.int32(len(x)),out))
        v=cp.asnumpy(out);core.bitwise(v[:,:20].copy(),v[:,20:].copy(),'2f raw quantities')


def synthetic(gpu):
    cp=gpu.cp;a=10.;b=np.nextafter(a,np.inf);x=np.array([1.,a,b,b,20.]);y=np.array([2.,3.,4.,5.,6.]);e=core.nodes(x)
    out=cp.zeros((len(e),4));kernel=cp.RawKernel(core.cuda_source(),'p8g_interp_kernel',options=gpu.options,backend='nvrtc')
    kernel((1,),(1,),(cp.asarray(x),cp.asarray(y),np.int32(len(x)),cp.asarray(e),np.int32(len(e)),out))
    values=cp.asnumpy(out);core.bitwise(values[:,0].copy(),values[:,1].copy(),'2f synthetic right')
    core.bitwise(values[:,2].copy(),values[:,3].copy(),'2f synthetic left')


def resident_blocks(attributes,threads,properties):
    """Turing register allocation is rounded to 256 registers per warp."""
    warp_regs=((int(attributes['num_regs'])*32+255)//256)*256
    registers=warp_regs*((threads+31)//32)
    shared=int(attributes['shared_size_bytes'])
    limits=[16,int(properties['maxThreadsPerMultiProcessor'])//threads]
    if registers:limits.append(int(properties['regsPerMultiprocessor'])//registers)
    if shared:limits.append(int(properties['sharedMemPerMultiprocessor'])//shared)
    return min(limits)


def run(bundle,output):
    import cupy as cp
    from .p5b_colab import environment
    bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((bundle/'p8g_manifest.json').read_text());identity=manifest['identity']
    if manifest['status']!='pass' or core.fingerprint()!=identity['sources']:raise ValueError('current complete export required')
    for name,digest in manifest['files'].items():
        if core.sha(bundle/name)!=digest:raise ValueError('export hash '+name)
    env=environment(cp)
    if 'T4' not in env['gpu']:raise ValueError('T4 required')
    with np.load(bundle/'p8f_inputs.npz') as f:arrays={k:f[k].copy() for k in f.files}
    materials=[]
    for name in ('water','lung','bone','lead'):
        with np.load(bundle/(name+'_bundle.npz')) as f:
            materials.append(SimpleNamespace(**{k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in core.FIELDS}))
    ref=read(bundle/'p8g_ref_gate2.npz',dict(identity=identity,n=20000,seed=core.SEED2,start=0))
    cap=manifest['capacity'];gpu=GPU(cp,materials,arrays,cap);old=GPU(cp,materials,arrays,cap,original=True)
    groups,h=gpu.run(20000,core.SEED2,history=True)
    comparison=core.compare_histories(ref,h);write_json(output/'gate2a.json',comparison)
    if comparison['status']!='pass':raise AssertionError('2a CPU/GPU comparison')
    _,oh=old.run(20000,core.SEED2,history=True);core.exact_histories(h,oh,'2b GPU P8f')
    write_json(output/'gate2b.json',dict(status='pass',identity=identity,n=20000,seed=core.SEED2))
    for threads in (128,256):
        candidate=gpu if threads==128 else GPU(cp,materials,arrays,cap,threads=threads)
        ag,hh=candidate.run(20000,core.SEED2,history=True);core.exact_histories(hh,h,'launch')
        core.check_group_records(ag,hh,20000)
        # Before/equal/after total slots, nonzero start, and consecutive calls.
        slots=40*threads
        for n,start in ((slots-1,0),(slots,0),(slots+1,0),(slots+1,7000),(slots+1,7000)):
            _,actual=candidate.run(n,core.SEED2,start,history=True)
            _,expected=old.run(n,core.SEED2,start,history=True)
            core.exact_histories(actual,expected,f'2d {threads}/{n}/{start}')
    write_json(output/'gate2cd.json',dict(status='pass',identity=identity,launches=[[40,128],[40,256]]))
    checked=GPU(cp,materials,arrays,cap,check=True)
    _,ch=checked.run(2000,core.SEED2,7000,history=True)
    _,nh=gpu.run(2000,core.SEED2,7000,history=True);_,fh=old.run(2000,core.SEED2,7000,history=True)
    core.exact_histories(ch,nh,'2e checked');core.exact_histories(ch,fh,'2e P8f')
    counts=checked.counters()
    if counts[:,1:36:2].any():raise AssertionError('2e invalid reads')
    write_json(output/'gate2e.json',dict(status='pass',identity=identity,sites=core.SITES,counts=counts.tolist()))
    # The synthetic probe keeps its read counters in a thread-local array; check-build atomics on local memory
    # fail on the GPU (2026-10-24 T4 launch failure), so it runs on the normal build only.
    components(gpu,old,materials);components(checked,old,materials);synthetic(gpu)
    faults=[]
    for fault in (1,3):
        bad=GPU(cp,materials,arrays,cap,check=True,fault=fault)
        if fault==1:
            try:components(bad,old,materials)
            except AssertionError as error:rejected=str(error)
            else:raise AssertionError('2f f1 not rejected')
        else:
            bad.run(2000,core.SEED2,7000,history=True)
            if not bad.counters()[:,1:36:2].any():raise AssertionError('2f f3 not rejected')
            rejected='invalid reads'
        c=bad.counters()
        if not c[:,36].sum():raise AssertionError('fault target not reached')
        faults.append(dict(fault=fault,rejected_by=rejected,target_reads=int(c[:,36].sum()),invalid_reads=int(c[:,1:36:2].sum())))
    write_json(output/'gate2f.json',dict(status='pass',identity=identity,faults=faults))
    write_json(output/'gate2.json',dict(status='pass',identity=identity,environment=env))
    # Timing starts only after every correctness gate. No production run.
    variants=[(name,t,GPU(cp,materials,arrays,cap,threads=t,original=name=='p8f')) for name,t in (('p8f',128),('p8f',256),('p8g',128),('p8g',256))]
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id);launches=[];timings=[]
    for name,t,obj in variants:
        attrs=obj.kernels[obj.prefix+'_kernel'].attributes
        launches.append(dict(name=name,blocks=40,threads=t,attributes=attrs,initialization_ms=obj.initialization_ms,resident_blocks_per_SM=resident_blocks(attrs,t,props),options=obj.options))
    for n in (20000,200000):
        for repeat in range(3):
            for name,t,obj in variants:
                result,_=obj.run(n,core.SEED2)
                from .p8f_colab import run_health
                if not run_health(result):raise AssertionError('timing health')
                timings.append(dict(name=name,threads=t,n=n,repeat=repeat,us_per_history=float(result['seconds'])*1e6/n))
                write_json(output/'timing_partial.json',dict(status='report',environment=env,launches=launches,alternating=timings))
    ratios=[]
    for n in (20000,200000):
        for t in (128,256):
            med={name:float(np.median([r['us_per_history'] for r in timings if r['n']==n and r['threads']==t and r['name']==name])) for name in ('p8f','p8g')}
            ratios.append(dict(n=n,threads=t,median_us=med,speedup=med['p8f']/med['p8g']))
    write_json(output/'timing.json',dict(status='report',identity=identity,environment=env,launches=launches,alternating=timings,ratios=ratios))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();run(args.bundle,args.output)

if __name__=='__main__':main()
