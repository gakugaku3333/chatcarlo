"""Single Run-all T4 workflow; material and reference arrays are hash checked."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from . import p8f_core as core


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
    def __init__(self,cp,materials,arrays,capacity,fault=0):
        self.cp=cp;self.arrays=arrays;self.fault=fault
        if capacity<16:raise ValueError('measured stack capacity required')
        self.sizes=tuple(map(int,arrays['sizes']));nl,nb,nc,nf,nspec=self.sizes
        if (nl,nb,nc,nf)!=(6,220,108,430):raise ValueError('only the registered P8e stack is supported')
        data,meta=core.pack(materials)
        self.geometry=cp.asarray(np.concatenate([arrays[k] for k in ('faces','rho','bins','strength','edges','cum')]))
        self.mapping=cp.asarray(np.r_[arrays['ids'],arrays['owner']])
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray(arrays['constants']))
        self.options=core.CUDA_OPTIONS+(f'-DP8F_STACK_CAP={capacity}',f'-DP8F_FAULT={fault}')
        self.kernels={};self.launches={}
        for name in ('p8f_kernel','p8f_fixed_kernel','p8f_probe_kernel'):
            kernel=cp.RawKernel(core.cuda_source(),name,options=self.options,backend='nvrtc');kernel.compile();self.kernels[name]=kernel
            self.launches[name]=dict(options=list(self.options),attributes={k:int(v) for k,v in kernel.attributes.items()})

    def run(self,n,seed,start=0,history=False):
        cp=self.cp;nl,nb,nc,nf,nspec=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        if n<=0 or n>200000 or start<0 or start+n>=2**64:raise ValueError('registered run range')
        hd=cp.zeros((n if history else 1,dw));hi=cp.zeros((n if history else 1,iw),np.int64)
        groups=cp.zeros((100,dw));absolute=cp.zeros((100,dw));ints=cp.zeros((100,iw),np.uint64)
        next=cp.array([5120],np.int32)
        args=self.args+(self.geometry,self.mapping,*map(np.int32,self.sizes),np.uint64(seed),np.uint64(start),np.int32(n),next,np.int32(history),hd,hi,groups,ints,absolute)
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.kernels['p8f_kernel']((40,),(128,),args);last.record();last.synchronize()
        seconds=float(cp.cuda.get_elapsed_time(first,last))/1000
        raw=cp.asnumpy(groups);rawi=cp.asnumpy(ints)
        aggregate=core.reduce_records(raw,rawi,nb,nc,nf,n,self.arrays['mass_g_cm2']);aggregate['seconds']=np.array(seconds);aggregate['_groups_d']=raw;aggregate['_groups_i']=rawi;aggregate['_absolute']=cp.asnumpy(absolute)
        histories=core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),nb,nc,nf) if history else None
        return aggregate,histories

    def probe(self,probe,seed):
        cp=self.cp;x=cp.asarray([probe['x']],dtype=np.float64);out=cp.zeros((1,128));state=cp.zeros(626,np.int64);next=cp.zeros(16)
        self.kernels['p8f_probe_kernel']((1,),(1,),(np.int32(probe['which']),*self.args,np.int32(probe['mid']),x,np.int32(1),np.int32(x.shape[1]),np.uint64(seed),out,state,next))
        return cp.asnumpy(out),cp.asnumpy(state),cp.asnumpy(next)

    def fixed(self,state,seed):
        cp=self.cp;nl,nb,nc,nf,_=self.sizes;dw=nb+nc+nf+37;iw=79+nb
        hd=cp.zeros((1,dw));hi=cp.zeros((1,iw),np.int64);events=cp.zeros((16,10));transitions=cp.zeros((16,5),np.int64)
        self.kernels['p8f_fixed_kernel']((1,),(1,),self.args+(self.geometry,self.mapping,*map(np.int32,(nl,nb,nc,nf)),np.uint64(seed),cp.asarray(state),hd,hi,events,transitions))
        return core.unpack(cp.asnumpy(hd),cp.asnumpy(hi),nb,nc,nf),cp.asnumpy(events),cp.asnumpy(transitions)


def require_pass(result,name):
    if result['status']!='pass':raise AssertionError(name+' failed; inspect saved diagnostics before proceeding')


def fixed_gate(gpu,reference):
    probes,states=core.fixed_inputs();rows=[]
    for j,probe in enumerate(probes):
        actual,state,next=gpu.probe(probe,core.SEED2)
        expected=reference[f'probe_{j}_out'];ccalls=int(reference[f'probe_{j}_state'][625])
        row=core.compare_probe_gpu(probe['which'],expected,actual,ccalls,int(state[625]));row['name']=probe['name'];rows.append(row)
    for j,state in enumerate(states):
        actual,events,transitions=gpu.fixed(state,core.SEED2)
        expected={key[len(f'case_{j}_'):]:value for key,value in reference.items() if key.startswith(f'case_{j}_')}
        integer_ok=all(np.array_equal(actual[key][0],expected[key]) for key in ('counts','violations','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts'))
        ee=expected['events'];ge=events[:len(ee)]
        branch_ok=np.array_equal(ee[:,:2],ge[:,:2]) and actual['steps'][0]==expected['steps'][0]
        integer_ok=integer_ok and np.array_equal(transitions,expected['transitions']) and actual['maxdepth'][0]==expected['maxdepth']
        ok=integer_ok and branch_ok and actual['calls'][0]==expected['mt'][625] and core.history_health(actual)
        # For the sub-cutoff photon there are no transcendental operations.
        if state[19]==0:
            for key in ('totals','maxima','brems','p8_max'):ok &= actual[key][0].tobytes()==expected[key].tobytes()
        floats={}
        for key in ('coarse','fine','totals','maxima','brems','p8_max'):
            a=actual[key][0];e=expected[key][0] if key in ('coarse','fine') else expected[key]
            bad=np.flatnonzero(a!=e)
            floats[key]=dict(first=int(bad[0]) if len(bad) else None,max_relative=float(np.max(np.abs(a-e)/np.maximum(np.maximum(np.abs(a),np.abs(e)),1e-300),initial=0)))
        rows.append(dict(name='history-'+str(j),status='pass' if ok else 'fail',integer_transition=integer_ok,forced_branch=branch_ok,
            calls=[int(expected['mt'][625]),int(actual['calls'][0])],floating_states=floats,first_float_difference=(np.argwhere(ee!=ge)[0].tolist() if np.any(ee!=ge) else None)))
    return dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',cases=rows)


def run(bundle,output):
    import cupy as cp
    from .p5b_colab import environment
    bundle=Path(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((bundle/'p8f_manifest.json').read_text());identity=manifest['identity']
    if manifest['status']!='pass':raise ValueError('export incomplete')
    for name,digest in manifest['files'].items():
        if core.sha(bundle/name)!=digest:raise ValueError('missing/changed export '+name)
    if core.fingerprint()!=identity['sources']:raise ValueError('kernel source identity changed')
    env=environment(cp)
    if 'T4' not in env['gpu']:raise ValueError('registered device is T4')
    with np.load(bundle/'p8f_inputs.npz') as f:arrays={k:f[k].copy() for k in f.files}
    materials=[]
    for name in ('water','lung','bone','lead'):
        with np.load(bundle/(name+'_bundle.npz')) as f:
            materials.append(SimpleNamespace(**{k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in core.FIELDS}))
    gpu=GPU(cp,materials,arrays,manifest['capacity']);write_json(output/'environment.json',dict(environment=env,launches=gpu.launches))
    ref=read(bundle/'p8f_ref_gate2.npz',dict(identity=identity,n=20000,seed=core.SEED2,start=0))
    groups,hist=gpu.run(20000,core.SEED2,history=True)
    c2=core.compare_histories(ref,hist);write_json(output/'gate2a.json',c2);require_pass(c2,'2a')
    core.check_local_bins(hist['histories'],hist['dep']);core.check_aggregation(groups['p8_batches'],hist['histories'],hist['additions'])
    groups2,hist2=gpu.run(15377,core.SEED2,history=True)
    c2b=core.compare_histories({k:v[:15377] for k,v in ref.items()},hist2);write_json(output/'gate2b.json',c2b);require_pass(c2b,'2b')
    core.check_local_bins(hist2['histories'],hist2['dep']);core.check_aggregation(groups2['p8_batches'],hist2['histories'],hist2['additions'])
    badgpu=GPU(cp,materials,arrays,manifest['capacity'],fault=7);_,bad=badgpu.run(15377,core.SEED2,history=True)
    try:core.check_local_bins(bad['histories'],bad['dep'])
    except AssertionError:pass
    else:raise AssertionError('2b local-reset fault not detected')
    fixed=read(bundle/'p8f_ref_fixed.npz',dict(identity=identity,seed=core.SEED2))
    result=fixed_gate(gpu,fixed);write_json(output/'gate2e.json',result);require_pass(result,'2e')
    write_json(output/'gate2.json',dict(status='pass',aggregation='deposit-count bound, all finite',local_reset_fault='rejected after normal pass',identity=identity,us_per_history=float(groups['seconds'])*1e6/20000,blocks=40,threads=128,tail_n=15377))
    paired_cpu=[];paired_gpu=[]
    for b in range(100):
        cpu=read(bundle/f'p8f_ref_batch_{b:03}.npz',dict(identity=identity,n=100000,seed=core.SEED3,start=b*100000,batch=b))
        path=output/f'gate3/batch_{b:03}.npz';receipt=dict(identity=identity,n=100000,seed=core.SEED3,start=b*100000,batch=b)
        if path.exists():actual=read(path,receipt)
        else:
            actual,_=gpu.run(100000,core.SEED3,b*100000)
            if not run_health(actual):raise AssertionError('gate3 health')
            save(path,actual,receipt)
        if not run_health(actual):raise AssertionError('saved gate3 health')
        paired_cpu.append(cpu['dose']);paired_gpu.append(actual['dose'])
    result=core.paired_blocks(np.asarray(paired_cpu),np.asarray(paired_gpu),arrays)
    write_json(output/'gate3.json',dict(result,identity=identity));require_pass(result,'3')
    # Same-session alternation with the unchanged P5k F water slab.
    from .p5k_colab import GPU as WaterGPU
    from .p5b_core import health as water_health
    with np.load(bundle/'p8f_water_baseline.npz') as f:water_data={k:f[k].copy() for k in f.files}
    water=WaterGPU(cp,water_data,variant='F',compile_fault=False);timings=[]
    for _ in range(3):
        result,t=water.measure(20000)
        require_pass(water_health(result),'water timing health')
        material,_=gpu.run(20000,core.SEED2)
        if not run_health(material):raise AssertionError('material timing health')
        timings.append(dict(water_us=t['kernel_ms']*1000/20000,material_us=float(material['seconds'])*1e6/20000))
    write_json(output/'timing.json',dict(status='report',alternating=timings,projected_production_seconds=float(groups['seconds'])*1000))
    # Every outer batch is sealed independently, including the first replay.
    resumed=(output/'production/batch_000.npz').exists()
    if resumed:
        old=read(output/'production/batch_000.npz',dict(identity=identity,n=200000,seed=core.SEED_PRODUCTION,start=0,batch=0))
        repeated,_=gpu.run(200000,core.SEED_PRODUCTION)
        core.check_resume(old,repeated,200000)
        write_json(output/'resume.json',dict(status='pass',identity=identity))
    for b in range(100):
        path=output/f'production/batch_{b:03}.npz';receipt=dict(identity=identity,n=200000,seed=core.SEED_PRODUCTION,start=b*200000,batch=b)
        if path.exists():
            if not run_health(read(path,receipt)):raise AssertionError('saved batch health')
            continue
        result,_=gpu.run(200000,core.SEED_PRODUCTION,b*200000)
        if not run_health(result):raise AssertionError('production unhealthy; no judgement')
        save(path,result,receipt);print('production',b,'complete',flush=True)
    write_json(output/'complete.json',dict(status='pass',identity=identity,seed=core.SEED_PRODUCTION,n=20000000,nbatch=100))


def run_health(r):
    return bool(all(np.isfinite(v).all() for v in r.values()) and np.all(r['dose']>=0) and r['totals'][3]==0 and r['maxima'][0]<=1e-12 and not np.any(r['violations']) and not np.any(r['counts'][5:7]) and not np.any(r['brems'][2:4]) and r['p8_max'][0]<=1e-12)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();run(args.bundle,args.output)

if __name__=='__main__':main()
