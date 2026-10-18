"""Single Run-all P6 T4 workflow: gates 2/3 then fixed production; no analysis."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from . import p6_core as h, p5b_core as core, p5j as maps
from .p5b_colab import GPU as BaseGPU, environment


class GPU(BaseGPU):
    def __init__(self,cp,data,fault=0,blocks=40):
        self.options=core.OPTIONS+(f'-DP6_FAULT={fault}',)
        self.blocks=blocks;self.fault=fault;self.launches={};kernels={}
        text=h.source()
        for name in ('p6_kernel','p6_score_kernel','p5j_initialize_kernel'):
            kernel=cp.RawKernel(text,name,options=self.options,backend='nvrtc');kernel.compile();kernels[name]=kernel
            self.launches[name]=dict(options=list(self.options),block=256,grid=blocks,
                attributes={k:int(kernel.attributes[k]) for k in ('num_regs','local_size_bytes','shared_size_bytes','max_threads_per_block')})
        BaseGPU.__init__(self,cp,data,kernels)
        layout=maps.layout(data);self.logs=cp.empty(int(layout['lengths'].sum())+3,np.float64)
        self.counts=cp.zeros((20,2),np.uint64);self.args+=(self.logs,self.counts)
        self.kernels['p5j_initialize_kernel']((1,),(1,),self.args);cp.cuda.Device().synchronize()
        self.generated=core.describe(dict(logs=cp.asnumpy(self.logs),**layout))
        self.geometry=cp.asarray(np.frombuffer(bytes(h.geometry()[0]),np.uint8));self.next=cp.zeros(1,np.int32)

    def output(self,shapes):
        return {k:self.cp.zeros(shape,dtype=np.int32 if k=='maxdepth' else (np.uint64 if k in h.INTEGER else np.float64))
                for k,shape in shapes.items()}

    def measure(self,n,seed,start=0,history=False,field=True,regions=True,maxsteps=h.MAXSTEPS,capacity=0,count_additions=False):
        core.check_range(n,start,maxsteps=maxsteps)
        if capacity and not history:raise ValueError('trace requires histories')
        if not 0<=seed<2**64:raise ValueError('invalid seed')
        cp=self.cp;self.next.fill(0);self.counts.fill(0)
        histories=self.output(h.history_shapes(n) if history else h.history_shapes(1))
        production=self.output(h.run_shapes(len(self.data['cum'])))
        trace=cp.zeros((n,capacity,10)) if capacity else np.uint64(0)
        meta=cp.zeros((n,2),np.int64) if capacity else np.uint64(0)
        additions=cp.zeros((100,h.BINS),np.uint64) if count_additions else np.uint64(0)
        args=self.args+(self.geometry,np.uint64(seed),np.uint64(start),np.int32(n),np.int32(maxsteps),
                        np.int32(field),np.int32(regions),self.next,np.int32(history))
        args+=tuple(histories[k] for k in core.HISTORY_ORDER)+tuple(production[k] for k in core.RUN_ORDER)
        args+=tuple(histories[k] for k in ('regions','region_ledger','sources'))+tuple(production[k] for k in ('regions','region_ledger'))
        args+=(trace,np.int32(capacity),meta,additions)
        cp.cuda.Device().synchronize();begin=time.perf_counter();first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.kernels['p6_kernel']((self.blocks,),(256,),args);last.record();last.synchronize()
        r=self.download(histories if history else production)
        h.validate(r,history,n,len(self.data['cum']))
        timing=dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin,n=n,seed=seed,start=start)
        self.region_additions=cp.asnumpy(additions) if count_additions else None
        if capacity:return r,timing,cp.asnumpy(trace),cp.asnumpy(meta)
        return r,timing

    def histories(self,n,seed,start=0,**kwargs):return self.measure(n,seed,start,history=True,**kwargs)[0]
    def run(self,n,seed,start=0,**kwargs):return self.measure(n,seed,start,**kwargs)[0]

    def scorer(self,records):
        cp=self.cp;n=len(records);out=cp.zeros((n,h.BINS));sums=cp.zeros((n,15))
        self.kernels['p6_score_kernel'](((n+127)//128,),(128,),
            (self.geometry,cp.asarray(records),np.int32(n),out,sums))
        return dict(regions=cp.asnumpy(out),sums=cp.asnumpy(sums))


def load_bundle(bundle):
    bundle=Path(bundle);manifest=json.loads((bundle/'p6_manifest.json').read_text())
    if manifest.get('status')!='pass':raise h.Stop('incomplete export')
    for name,info in manifest['files'].items():
        path=bundle/name
        if path.stat().st_size!=info['bytes'] or core.file_sha(path)!=info['sha256']:raise h.Stop('bundle size/hash mismatch: '+name)
    data=core.read_npz(bundle/'p6_inputs.npz')
    if h.identity(data)!=manifest['identity']:raise h.Stop('bundle source/input/settings mismatch')
    h.require_report(bundle/'local.json',data,'local');h.require_report(bundle/'gate1.json',data,'gate1')
    # Input zip is read only; extracted references are checked against their original manifest.
    target=bundle/'reference'
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle/'p6_reference.zip') as z:
        for info in z.infolist():
            rel=Path(info.filename)
            if rel.is_absolute() or '..' in rel.parts:raise h.Stop('invalid reference zip member')
            path=target/rel
            content=z.read(info)
            if path.exists():
                if path.read_bytes()!=content:raise h.Stop('extracted reference differs; retain and stop')
            else:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    report=json.loads((target/'reference.json').read_text())
    if report!=manifest['reference']:raise h.Stop('reference manifest mismatch')
    for name,digest in report['files'].items():
        if core.file_sha(target/name)!=digest:raise h.Stop('reference hash mismatch')
    return data,target,manifest


def require_gate2(report,ident):
    if report.get('status')!='pass' or report.get('identity')!=ident or 'T4' not in report.get('environment',{}).get('gpu',''):
        raise h.Stop('current T4 gate2 required')
    if set(report.get('checks',{}))!={'2a','2b','2c','2d','2e'}:raise h.Stop('incomplete gate2')
    for check in report['checks'].values():core.require(check,'gate2')


def require_gate3(report,ident):
    if report.get('status')!='pass' or report.get('identity')!=ident or report.get('batches')!=100:raise h.Stop('current complete gate3 required')
    core.require(report.get('comparison',core.verdict(False)),'gate3 comparison')


def gate2(cp,data,reference,manifest,root):
    from . import p5k_colab as baseline
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ident=h.identity(data);report=dict(status='incomplete',identity=ident,environment=environment(cp),checks={})
    if 'T4' not in report['environment']['gpu']:raise h.Stop('T4 required')
    gpu=GPU(cp,data);cpu=core.read_npz(Path(reference)/'histories.npz');meta=json.loads(str(cpu.pop('metadata')))
    if meta!=dict(identity=ident,seed=h.SEED,start=0,n=20000,outputs=core.describe(cpu)):raise h.Stop('history reference metadata mismatch')
    conversion=1.602176634e-10
    try:
        normal=gpu.histories(20000,h.SEED)
        check=h.c2(cpu,normal,manifest['Dref'],conversion);report['checks']['2a']=check;core.require(check,'2a CPU/GPU histories')
        # 1 block has 256 slots. Three full populations necessarily exercise refill.
        small=GPU(cp,data,blocks=1);n=3*256
        expected={k:v[:n].copy() for k,v in cpu.items()}
        actual=small.histories(n,h.SEED)
        ordinary=h.c2(expected,actual,manifest['Dref'],conversion);core.require(ordinary,'2b ordinary refill')
        faulty=GPU(cp,data,fault=4,blocks=1).histories(n,h.SEED)
        failed=h.c2(expected,faulty,manifest['Dref'],conversion)
        # A stale score array violates the independent per-history slab/deposit ledger.
        fault=core.verdict(failed['status']=='fail' and not core.bits(actual['regions'],faulty['regions']),comparison=failed)
        core.require(fault,'2b numerical fault detection');report['checks']['2b']=core.verdict(True,normal=ordinary,fault=fault,slots=256,n=n)
        total=gpu.run(20000,h.SEED,count_additions=True);check=h.c3(normal,total,region_additions=gpu.region_additions);report['checks']['2c']=check;core.require(check,'2c atomic merge')
        vectors=core.read_npz(Path(reference)/'scorer.npz');records=vectors.pop('records')
        normal_scores=gpu.scorer(records);good=h.exact(normal_scores,vectors);core.require(good,'2d CPU/GPU scorer')
        faults={}
        for fault in (5,6,7):
            bad=GPU(cp,data,fault=fault).scorer(records);comparison=h.exact(bad,vectors)
            faults[str(fault)]=core.verdict(comparison['status']=='fail',comparison=comparison)
            core.require(faults[str(fault)],'2d planted scorer fault')
        report['checks']['2d']=core.verdict(True,normal=good,faults=faults,inputs=len(records))
        old=baseline.GPU(cp,data,variant='F',compile_fault=False)
        expected=old.per_history(20000)
        actual=gpu.histories(20000,h.SEED,field=False,regions=False)
        check=h.exact(actual,expected,core.HISTORY_ORDER);report['checks']['2e']=check;core.require(check,'2e no-field GPU regression')
        report.update(status='pass',launches=gpu.launches,generated=gpu.generated)
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/'gate2.json',report)
    return report


def gate3(gpu,reference,manifest,root,gate2_report):
    root=Path(root);ident=h.identity(gpu.data);require_gate2(gate2_report,ident)
    report=dict(status='incomplete',identity=ident,batches=0,environment=environment(gpu.cp))
    try:
        found=h.resume(root/'gate3',ident,'gate3')
        for b in range(100):
            if b not in found:
                r=gpu.run(100000,h.SEED,start=100000*b);path=root/'gate3'/f'batch_{b:03}.npz'
                h.save_checkpoint(path,ident,'gate3',b,r);found[b]=path
            print('gate3',b+1,'/100',flush=True)
        actual=[h.read_checkpoint(found[b],ident,'gate3',b) for b in range(100)]
        expected=[h.read_checkpoint(Path(reference)/'batches'/f'batch_{b:03}.npz',ident,'gate3',b) for b in range(100)]
        report['comparison']=h.c4(expected,actual,manifest['Dref']);core.require(report['comparison'],'gate3 scale')
        report.update(status='pass',batches=100)
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/'gate3.json',report)
    return report


def timing(gpu,root,gate2_report):
    from . import p5k_colab as baseline
    require_gate2(gate2_report,h.identity(gpu.data));old=baseline.GPU(gpu.cp,gpu.data,variant='F',compile_fault=False)
    for obj in (old,gpu):
        r=obj.production(1000) if obj is old else obj.run(1000,h.SEED)
        core.require(core.health(r) if obj is old else h.health(r),'timing warmup')
    records=[]
    for repeat in range(3):
        for label in ('P5k F','P6'):
            if label=='P5k F':r,t=old.measure(100000)
            else:r,t=gpu.measure(100000,h.SEED)
            core.require(core.health(r) if label=='P5k F' else h.health(r),'timing health')
            records.append(dict(label=label,repeat=repeat,microseconds_per_history=t['kernel_ms']*1000/100000,**t))
    core.write_json(Path(root)/'timing.json',dict(status='report',measurements=records,environment=environment(gpu.cp)))


def production(gpu,root,gate2_report,gate3_report):
    root=Path(root);ident=h.identity(gpu.data);require_gate2(gate2_report,ident);require_gate3(gate3_report,ident)
    found=h.resume(root/'production',ident,'production');timings=[]
    for b in range(100):
        if b not in found:
            r,t=gpu.measure(460000,h.MAIN_SEED,start=460000*b);path=root/'production'/f'batch_{b:03}.npz'
            h.save_checkpoint(path,ident,'production',b,r);found[b]=path;timings.append(dict(batch=b,**t))
        print('production',b+1,'/100',flush=True)
    rows=[h.read_checkpoint(found[b],ident,'production',b) for b in range(100)]
    result=h.stacked(rows,'production');path=root/'production/result.npz'
    metadata=dict(identity=ident,contract=h.contract('production'),outputs=core.describe(result))
    if path.exists():
        saved=core.read_npz(path);old=json.loads(str(saved.pop('metadata')))
        if old!=metadata or h.exact(saved,result)['status']!='pass':raise h.Stop('existing production result mismatch')
    else:core.save_npz(path,result,metadata)
    core.write_json(root/'production_complete.json',dict(status='complete; analysis deferred',identity=ident,
        contract=h.contract('production'),sha256=core.file_sha(path),current_session_timings=timings))
    return path


def run_all(bundle,results):
    data,reference,manifest=load_bundle(bundle)
    import cupy as cp
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    # Re-run small gate2 on resume, then verify every stored batch before reuse.
    r2=gate2(cp,data,reference,manifest,root);gpu=GPU(cp,data)
    timing(gpu,root,r2)
    r3=gate3(gpu,reference,manifest,root,r2)
    return production(gpu,root,r2,r3)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--results',required=True)
    a=p.parse_args();run_all(a.bundle,a.results)
