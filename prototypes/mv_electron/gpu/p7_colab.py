"""Single Run-all P7 T4 workflow: per condition gates 2/3, then production for A/B/C; no analysis."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from . import p7_core as h, p5b_core as core, p5j as maps
from .p5b_colab import GPU as BaseGPU, environment

CONVERSION=1.602176634e-10


class GPU(BaseGPU):
    def __init__(self,cp,data,name,fault=0,blocks=40):
        self.options=core.OPTIONS+(f'-DP6_FAULT={fault}',);self.condition=h.condition(name)
        self.blocks=blocks;self.fault=fault;self.launches={};kernels={}
        text=h.source()
        for k in ('p7_kernel','p7_score_kernel','p5j_initialize_kernel'):
            kernel=cp.RawKernel(text,k,options=self.options,backend='nvrtc');kernel.compile();kernels[k]=kernel
            self.launches[k]=dict(options=list(self.options),block=256,grid=blocks,
                attributes={a:int(kernel.attributes[a]) for a in ('num_regs','local_size_bytes','shared_size_bytes','max_threads_per_block')})
        BaseGPU.__init__(self,cp,data,kernels)
        layout=maps.layout(data);self.logs=cp.empty(int(layout['lengths'].sum())+3,np.float64)
        self.counts=cp.zeros((20,2),np.uint64);self.args+=(self.logs,self.counts)
        self.kernels['p5j_initialize_kernel']((1,),(1,),self.args);cp.cuda.Device().synchronize()
        self.generated=core.describe(dict(logs=cp.asnumpy(self.logs),**layout))
        self.geometry=cp.asarray(np.frombuffer(bytes(h.geometry('p7')[0]),np.uint8));self.next=cp.zeros(1,np.int32)
        self.nspec=len(data['cum'])

    def output(self,shapes):
        return {k:self.cp.zeros(s,dtype=np.int32 if k=='maxdepth' else (np.uint64 if k in h.INTEGER else np.float64)) for k,s in shapes.items()}

    def measure(self,n,seed,start=0,history=False,field=True,regions=True,maxsteps=h.MAXSTEPS,count_additions=False):
        core.check_range(n,start,maxsteps=maxsteps)
        if not 0<=seed<2**64:raise ValueError('invalid seed')
        cp=self.cp;self.next.fill(0);self.counts.fill(0);c=self.condition
        histories=self.output(h.history_shapes(n) if history else h.history_shapes(1))
        production=self.output(h.run_shapes(self.nspec))
        additions=cp.zeros((100,h.BINS),np.uint64) if count_additions else np.uint64(0)
        args=self.args+(self.geometry,np.uint64(seed),np.uint64(start),np.int32(n),np.int32(maxsteps),
                        np.int32(field),np.float64(c['field_half']),np.float64(c['ssd']),np.int32(regions),self.next,np.int32(history))
        args+=tuple(histories[k] for k in core.HISTORY_ORDER)+tuple(production[k] for k in core.RUN_ORDER)
        args+=tuple(histories[k] for k in ('regions','region_ledger','sources'))+tuple(production[k] for k in ('regions','region_ledger'))
        args+=(np.uint64(0),np.int32(0),np.uint64(0),additions)
        cp.cuda.Device().synchronize();begin=time.perf_counter();first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.kernels['p7_kernel']((self.blocks,),(256,),args);last.record();last.synchronize()
        r=self.download(histories if history else production)
        h.validate(r,history,n,self.nspec)
        timing=dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin,n=n,seed=seed,start=start)
        self.region_additions=cp.asnumpy(additions) if count_additions else None
        return r,timing

    def histories(self,n,seed,start=0,**kw):return self.measure(n,seed,start,history=True,**kw)[0]
    def run(self,n,seed,start=0,**kw):return self.measure(n,seed,start,**kw)[0]

    def scorer(self,records):
        cp=self.cp;n=len(records);out=cp.zeros((n,h.BINS));sums=cp.zeros((n,15))
        self.kernels['p7_score_kernel'](((n+127)//128,),(128,),(self.geometry,cp.asarray(records),np.int32(n),out,sums))
        return dict(regions=cp.asnumpy(out),sums=cp.asnumpy(sums))


def load_bundle(bundle):
    bundle=Path(bundle);manifest=json.loads((bundle/'p7_manifest.json').read_text())
    if manifest.get('status')!='pass':raise h.Stop('incomplete export')
    for name,info in manifest['files'].items():
        path=bundle/name
        if path.stat().st_size!=info['bytes'] or core.file_sha(path)!=info['sha256']:raise h.Stop('bundle size/hash mismatch: '+name)
    target=bundle/'reference';target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle/'p7_reference.zip') as z:
        for info in z.infolist():
            rel=Path(info.filename)
            if rel.is_absolute() or '..' in rel.parts:raise h.Stop('invalid reference zip member')
            path=target/rel;content=z.read(info)
            if path.exists():
                if path.read_bytes()!=content:raise h.Stop('extracted reference differs; retain and stop')
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    data={}
    for name in 'ABC':
        d=core.read_npz(bundle/f'p7_inputs_{name}.npz')
        if h.identity(d,name)!=manifest['conditions'][name]['identity']:raise h.Stop('bundle source/input/settings mismatch: '+name)
        report=json.loads((target/name/'pilot.json').read_text())
        if report!=manifest['conditions'][name]['pilot']:raise h.Stop('reference manifest mismatch: '+name)
        for rel,digest in report['files'].items():
            if core.file_sha(target/name/rel)!=digest:raise h.Stop('reference hash mismatch: '+name)
        data[name]=d
    return data,target,manifest


def require_gate(report,ident,kind):
    if report.get('status')!='pass' or report.get('identity')!=ident:raise h.Stop('current '+kind+' required')
    if kind=='gate2' and ('T4' not in report.get('environment',{}).get('gpu','') or set(report.get('checks',{}))!={'2a','2b','2c','2d','2e'}):
        raise h.Stop('incomplete gate2')
    if kind=='gate3' and report.get('batches')!=100:raise h.Stop('incomplete gate3')
    for check in report.get('checks',{}).values():core.require(check,kind)
    if kind=='gate3':core.require(report.get('comparison',core.verdict(False)),'gate3 comparison')


def gate2(cp,data,name,reference,manifest,root):
    from . import p5k_colab as baseline
    root=Path(root);root.mkdir(parents=True,exist_ok=True);ident=h.identity(data,name);dref=manifest['conditions'][name]['Dref']
    report=dict(status='incomplete',identity=ident,environment=environment(cp),checks={})
    if 'T4' not in report['environment']['gpu']:raise h.Stop('T4 required')
    gpu=GPU(cp,data,name);ref=Path(reference)/name;nspec=len(data['cum'])
    cpu=core.read_npz(ref/'histories.npz');meta=json.loads(str(cpu.pop('metadata')))
    if meta!=dict(identity=ident,seed=h.SEED,start=0,n=20000,outputs=core.describe(cpu)):raise h.Stop('history reference metadata mismatch')
    try:
        normal=gpu.histories(20000,h.SEED)
        check=h.c2(cpu,normal,dref,CONVERSION,nspec);report['checks']['2a']=check;core.require(check,'2a CPU/GPU histories')
        small=GPU(cp,data,name,blocks=1);n=3*256;expected={k:v[:n].copy() for k,v in cpu.items()}
        actual=small.histories(n,h.SEED);ordinary=h.c2(expected,actual,dref,CONVERSION,nspec);core.require(ordinary,'2b ordinary refill')
        faulty=GPU(cp,data,name,fault=4,blocks=1).histories(n,h.SEED);failed=h.c2(expected,faulty,dref,CONVERSION,nspec)
        fault=core.verdict(failed['status']=='fail' and not core.bits(actual['regions'],faulty['regions']),comparison=failed['status'])
        core.require(fault,'2b numerical fault detection');report['checks']['2b']=core.verdict(True,normal=ordinary['status'],fault=fault,slots=256,n=n)
        total=gpu.run(20000,h.SEED,count_additions=True);check=h.c3(normal,total,nspec,region_additions=gpu.region_additions)
        report['checks']['2c']=check;core.require(check,'2c atomic merge')
        vectors=core.read_npz(ref/'scorer.npz');records=vectors.pop('records')
        good=h.exact(gpu.scorer(records),vectors);core.require(good,'2d CPU/GPU scorer');faults={}
        for f in (5,6,7):
            comparison=h.exact(GPU(cp,data,name,fault=f).scorer(records),vectors)
            faults[str(f)]=core.verdict(comparison['status']=='fail',comparison=comparison['status']);core.require(faults[str(f)],'2d planted scorer fault')
        report['checks']['2d']=core.verdict(True,normal=good['status'],faults=faults,inputs=len(records))
        if name=='A':
            # 2e uses the 10 MV inputs of P5k F; condition A carries the 10 MV spectrum.
            old=baseline.GPU(cp,data,variant='F',compile_fault=False);exp=old.per_history(20000)
            act=gpu.histories(20000,h.SEED,field=False,regions=False)
            check=h.exact(act,exp,core.HISTORY_ORDER)
        else:check=core.verdict(True,note='2e is run once, on condition A (10 MV inputs)')
        report['checks']['2e']=check;core.require(check,'2e no-field GPU regression')
        report.update(status='pass',launches=gpu.launches,generated=gpu.generated)
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/f'gate2_{name}.json',report)
    return report


def gate3(gpu,name,reference,manifest,root,r2):
    root=Path(root);ident=h.identity(gpu.data,name);require_gate(r2,ident,'gate2')
    report=dict(status='incomplete',identity=ident,batches=0,environment=environment(gpu.cp));c=h.contract('gate3',name)
    try:
        found=h.resume(root/f'gate3_{name}',ident,'gate3',name)
        for b in range(100):
            if b not in found:
                r=gpu.run(c['n'],c['seed'],start=c['n']*b);path=root/f'gate3_{name}'/f'batch_{b:03}.npz'
                h.save_checkpoint(path,ident,'gate3',name,b,r);found[b]=path
        actual=[h.read_checkpoint(found[b],ident,'gate3',name,b) for b in range(100)]
        expected=[h.read_checkpoint(Path(reference)/name/'batches'/f'batch_{b:03}.npz',ident,'gate3',name,b) for b in range(100)]
        report['comparison']=h.c4(expected,actual,manifest['conditions'][name]['Dref'],name);core.require(report['comparison'],'gate3 scale')
        report.update(status='pass',batches=100);print('gate3',name,'pass',flush=True)
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/f'gate3_{name}.json',report)
    return report


def production(gpu,name,manifest,root,r2,r3):
    root=Path(root);ident=h.identity(gpu.data,name);require_gate(r2,ident,'gate2');require_gate(r3,ident,'gate3')
    N=manifest['conditions'][name]['N_main'];c=h.contract('production',name,N)
    found=h.resume(root/f'production_{name}',ident,'production',name,N);timings=[]
    for b in range(100):
        if b not in found:
            r,t=gpu.measure(c['n'],c['seed'],start=c['n']*b);path=root/f'production_{name}'/f'batch_{b:03}.npz'
            h.save_checkpoint(path,ident,'production',name,b,r,N);found[b]=path;timings.append(dict(batch=b,**t))
        if (b+1)%10==0:print('production',name,b+1,'/100',flush=True)
    rows=[h.read_checkpoint(found[b],ident,'production',name,b,N) for b in range(100)]
    result=h.stacked(rows,'production',name,N);path=root/f'production_{name}'/'result.npz'
    metadata=dict(identity=ident,contract=c,outputs=core.describe(result))
    if path.exists():
        saved=core.read_npz(path);old=json.loads(str(saved.pop('metadata')))
        if old!=metadata or h.exact(saved,result)['status']!='pass':raise h.Stop('existing production result mismatch')
    else:core.save_npz(path,result,metadata)
    core.write_json(root/f'production_{name}_complete.json',dict(status='complete; analysis deferred',identity=ident,
        contract=c,sha256=core.file_sha(path),current_session_timings=timings))
    return path


def timing(cp,data,root):
    """Report only: P6-equivalent condition (10 MV) alternating with P7 condition A, 100k histories, 3 repeats."""
    gpu=GPU(cp,data['A'],'A');records=[]
    for repeat in range(3):
        for name in 'ABC':
            g=gpu if name=='A' else GPU(cp,data[name],name)
            r,t=g.measure(100000,h.SEED);core.require(h.health(r),'timing health')
            records.append(dict(condition=name,repeat=repeat,microseconds_per_history=t['kernel_ms']*1000/100000,**t))
    core.write_json(Path(root)/'timing.json',dict(status='report',measurements=records,environment=environment(cp)))


def run_all(bundle,results):
    data,reference,manifest=load_bundle(bundle)
    import cupy as cp
    root=Path(results);root.mkdir(parents=True,exist_ok=True);gates={}
    for name in 'ABC':
        r2=gate2(cp,data[name],name,reference,manifest,root);gpu=GPU(cp,data[name],name)
        gates[name]=(gpu,r2,gate3(gpu,name,reference,manifest,root,r2))
    timing(cp,data,root)
    for name in sorted('ABC',key=lambda n:manifest['conditions'][n]['N_main']):
        gpu,r2,r3=gates[name];production(gpu,name,manifest,root,r2,r3)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--results',required=True)
    a=p.parse_args();run_all(a.bundle,a.results)
