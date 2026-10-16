"""T4 gates 0–3, alternating timing and source maps in one run; NumPy-only import."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from . import p5b_core as core, p5d_core as d, p5i_core as h, p5i as maps
from . import p5g_colab as previous, p5g_core as g
from .p5b_colab import GPU as BaseGPU, environment as base_environment


def environment(cp):
    from cupy_backends.cuda.libs import nvrtc
    return dict(base_environment(cp),nvrtc=list(nvrtc.getVersion()))


class GPU(previous.GPU):
    def __init__(self, cp, data, baseline=False, check=False, lineinfo=False, compile_fault=True):
        self.baseline=baseline;self.checked=check
        if baseline:
            previous.GPU.__init__(self,cp,data,lineinfo=lineinfo,compile_fault=compile_fault)
            self.baseline=True  # P5g reference; it has no P5i counters.
            return
        self.options=core.OPTIONS+(f'-DP5I_CHECK={int(check)}',)+(('--generate-line-info',) if lineinfo else ())
        names=h.KERNELS;self.entries=names[:2];self.launches={};kernels={}
        for name in (*names,*((names[0]+'_fault',) if compile_fault else ())):
            opts=self.options+(('-DP5B_MERGE_FAULT=1',) if name.endswith('_fault') else ())
            k=cp.RawKernel(h.source(),name.removesuffix('_fault'),options=opts,backend='nvrtc')
            k.compile();kernels[name]=k
            self.launches[name]=dict(block=128,grid='ceil(n/128)',options=list(opts),
                attributes={key:int(k.attributes[key]) for key in
                    ('num_regs','local_size_bytes','shared_size_bytes','max_threads_per_block')})
        BaseGPU.__init__(self,cp,data,kernels)
        self.maps=maps.layout(data)
        self.logs=cp.empty(int(self.maps['lengths'].sum()),np.float64)
        self.counts=cp.zeros((9,2),np.uint64)
        self.args+=(self.logs,self.counts)
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.launch('p5i_initialize_kernel',1,self.args);last.record();last.synchronize()
        self.initialization_ms=float(cp.cuda.get_elapsed_time(first,last))
        self.generated=core.describe(dict(logs=cp.asnumpy(self.logs),**self.maps))

    def measure(self,n,start=0,history=False,fault=False,maxsteps=core.MAXSTEPS,audit_capacity=0):
        if not self.baseline:self.counts.fill(0)
        core.check_range(n,start,maxsteps=maxsteps)
        if audit_capacity and (self.baseline or not history):raise ValueError('audit requires new history kernel')
        cp=self.cp;cp.cuda.Device().synchronize();begin=time.perf_counter()
        r=self.output(core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        name=self.entries[int(history)]+('_fault' if fault else '')
        args=self.args+(np.uint64(core.SEED),np.uint64(start),np.int32(n),np.int32(maxsteps))
        args+=tuple(r[k] for k in (core.HISTORY_ORDER if history else core.RUN_ORDER))
        if audit_capacity:
            trace=cp.zeros((n,audit_capacity,10));meta=cp.zeros((n,2),np.int64)
            args+=(trace,np.int32(audit_capacity),meta);name='p5i_audit_kernel'
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.launch(name,n,args);last.record();out=self.download(r);last.synchronize()
        times=dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin)
        core.validate_output(out,core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        if audit_capacity:return out,times,cp.asnumpy(trace),cp.asnumpy(meta)
        return out,times


    def counter_report(self):
        return h.counters(self.cp.asnumpy(self.counts))


def gate0(gpu,root,randoms=1000,variants=True,functions=range(6)):
    cp=gpu.cp;root=Path(root);gpu.counts.fill(0);checks={}
    grids=(gpu.data['brems_e'],gpu.data['rates'][:,0] if gpu.data['rates'].shape[1]==3 else gpu.data['ts'],
           gpu.data['ts'],gpu.data['pes'],gpu.data['pes'],gpu.data['pes'])
    for which in functions:
        grid=grids[which]
        x=maps.vectors(grid,gpu.data['thresholds'],randoms);out=cp.zeros((len(x),40))
        gpu.launch('p5i_gate0_kernel',len(x),gpu.args+(np.int32(which),cp.asarray(x),np.int32(len(x)),out))
        out=cp.asnumpy(out)
        check=core.verdict(core.bits(out[:,:20].copy(),out[:,20:].copy()),inputs=len(x))
        checks[str(which)]=check;core.require(check,'T4 gate0 function '+str(which))
    count=h.counters(cp.asnumpy(gpu.counts),require_coverage=len(functions)==6);core.require(count,'T4 gate0 unit coverage')
    diag=cp.empty((len(gpu.logs),5))
    gpu.launch('p5i_diagnostic_kernel',1,gpu.args+(diag,))
    init_diag=cp.empty_like(diag)
    gpu.launch('p5i_initialization_diagnostic_kernel',1,gpu.args+(init_diag,))
    report=dict(status='pass',checks=checks,counters=count,randoms_per_interval=randoms,
        auxiliary=dict(intermediates_equal=core.bits(cp.asnumpy(diag),cp.asnumpy(init_diag)),equal=core.bits(cp.asnumpy(diag[:,0]).copy(),cp.asnumpy(gpu.logs)),formal=False,
            nonpositive_arguments=int(cp.sum(diag[:,4]<=0).item()),
            recorded_columns=['log','quotient','q','q_cube','argument']),
        generated=gpu.generated,initialization_ms=gpu.initialization_ms,extra_bytes=gpu.logs.nbytes,
        identity=h.identity(gpu.data,environment(cp),gpu.launches,gpu.options))
    if variants:
        variant=GPU(cp,maps.synthetic(gpu.data),check=True,compile_fault=False)
        report['synthetic']=gate0(variant,root/'synthetic',randoms,False,(1,3,4,5))
    core.save_npz(root/'p5i_tables.npz',dict(logs=cp.asnumpy(gpu.logs),auxiliary=cp.asnumpy(diag),initialization_records=cp.asnumpy(init_diag),**gpu.maps))
    core.write_json(root/'gate0_gpu.json',report)
    return report


def gate2(bundle,results,gate1):
    data,_,_=d.load_bundle(bundle);h.require_gate1(gate1,data)
    import cupy as cp
    root=Path(results);root.mkdir(parents=True,exist_ok=True);env=environment(cp)
    if 'T4' not in env['gpu']:raise core.Stop('T4 required')
    new,checked,old=GPU(cp,data),GPU(cp,data,check=True),GPU(cp,data,baseline=True)
    report=dict(status='incomplete',checks={},identity=h.identity(data,env,new.launches))
    try:
        report['gate0']=gate0(checked,root/'gate0')
        expected=old.per_history(20000);core.save_npz(root/'p5g_history.npz',expected)
        for label,gpu in (('ordinary',new),('checked',checked)):
            out=gpu.per_history(20000);core.save_npz(root/(label+'.npz'),out)
            report['checks'][label]=h.exact(out,expected,core.HISTORY_ORDER)
            core.require(report['checks'][label],label+' T4 bits')
        report['checks']['counters']=checked.counter_report()
        core.require(report['checks']['counters'],'T4 log coverage')
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/'gate2.json',report)
    return report


def require_gate2(path, data, env, gpu):
    r=json.loads(Path(path).read_text())
    if r.get('status')!='pass' or r.get('identity')!=h.identity(data,env,gpu.launches):
        raise core.Stop('current gate2 required')
    for v in r['checks'].values():core.require(v,'gate2 check')


def gate3_identity(data, env, gpu):
    """Gate3 checkpoint identity; keep the gate1/2 identity unchanged."""
    return dict(h.identity(data,env,gpu.launches),nspec=20)


def require_gate3(path, data, env, gpu):
    r=json.loads(Path(path).read_text())
    if (r.get('overall')!='pass; adoption pending user decision'
        or r.get('identity')!=gate3_identity(data,env,gpu)):
        raise core.Stop('current gate3 required')
    return r


def gate3(bundle,results,gate1,gate2_report):

    import cupy as cp
    data,ref,manifest=d.load_bundle(bundle);h.require_gate1(gate1,data);gpu=GPU(cp,data);env=environment(cp)
    require_gate2(gate2_report,data,env,gpu)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    identity=gate3_identity(data,env,gpu)
    path=root/'identity.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(identity)):
        raise core.Stop('C0 identity mismatch')
    core.write_json(path,identity)
    report=dict(overall='incomplete',identity=identity,checks={})
    def record(k,v):core.record(report,k,v)
    def save(k,v):core.save_npz(root/(k+'.npz'),v,dict(identity=identity,outputs=core.describe(v)))
    try:
        record('C0',core.verdict(True))
        values=gpu.checks(ref);save('c1',values);record('C1',core.c1(ref,values))
        histories=gpu.per_history(20000);save('c2',histories)
        record('C2',core.c2({k:ref['history_'+k] for k in core.HISTORY_ORDER},histories,float(ref['Dref']),float(ref['conversion'])))
        prod=gpu.production(20000);save('c3',prod);normal=core.c3(histories,prod,20)
        faulty=gpu.production(20000,fault=True);save('c3_fault',faulty);fault=core.c3(histories,faulty,20)
        pieces=[gpu.per_history(5000,start) for start in range(0,20000,5000)]
        split={k:np.concatenate([r[k] for r in pieces]) for k in core.HISTORY_ORDER};save('c3_split',split)
        record('C3',core.verdict(normal['status']=='pass' and fault['status']=='fail' and core.split_check(histories,split)['status']=='pass',ordinary=normal,merge_fault=fault))
        found=core.resume(root/'batches',identity)
        for b in range(100):
            if b not in found:
                out=gpu.production(100000,100000*b);core.require(core.health(out),'C4 health')
                if not core.bits(out['nspec'],ref['c4_nspec'][b]):raise core.Stop('C4 spectrum')
                path=root/'batches'/f'batch_{b:03}.npz'
                core.save_npz(path,out,core.checkpoint_meta(identity,b,out));found[b]=path
        record('C4',core.c4(ref,[core.read_checkpoint(found[b],identity,b) for b in range(100)],20))
        report['overall']='pass; adoption pending user decision'
    except Exception as exc:
        report.update(overall='not adopted',stop=str(exc));raise
    finally:core.write_json(root/'gate3.json',report)
    return report


def timing(bundle,results,gate1,gate2_report,gate3_report):

    import cupy as cp
    data,_,manifest=d.load_bundle(bundle);h.require_gate1(gate1,data);new,old=GPU(cp,data),GPU(cp,data,baseline=True)
    require_gate2(gate2_report,data,environment(cp),new)
    require_gate3(gate3_report,data,environment(cp),new)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',measurements=[],identity=gate3_identity(data,environment(cp),new),
                decision='report only; adoption is a user decision',
                initialization_ms=dict(P5g=0.,P5i=new.initialization_ms),
                additional_bytes=dict(P5g=0,P5i=new.logs.nbytes))
    try:
        for gpu in (old,new):core.require(core.health(gpu.production(1000)),'warmup health')
        for repeat in range(3):
            for label,gpu in ([('P5g',old),('P5i',new)] if repeat%2==0 else [('P5i',new),('P5g',old)]):
                out,times=gpu.measure(100000);core.require(core.health(out),'timing health')
                core.save_npz(root/f'{label}_{repeat}.npz',out)
                report['measurements'].append(dict(variant=label,repeat=repeat,n=100000,**times))
        report['status']='pass'
    finally:core.write_json(root/'timing.json',report)
    return report



def target(bundle,root,variant,lineinfo):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str(root/'nvrtc_cache')
    import cupy as cp
    data,_,_=d.load_bundle(bundle)
    gpu=GPU(cp,data,baseline=variant=='P5g',lineinfo=lineinfo,compile_fault=False)
    text=g.source() if variant=='P5g' else h.source()
    (root/'compiled.cu').write_text(text)
    core.save_npz(root/'history.npz',gpu.per_history(20000))
    gpu.production(1000);gpu.production(100000);cp.cuda.Device().synchronize()
    core.write_json(root/'target.json',dict(environment=environment(cp),options=list(gpu.options),launches=gpu.launches,
        identity=h.identity(data,environment(cp),gpu.launches,gpu.options),
        generated=None if gpu.baseline else gpu.generated,seed=core.SEED,start=0,
        history_n=20000,warmup_n=1000,measured_n=100000,initialization_ms=0. if gpu.baseline else gpu.initialization_ms))


def divergence_map(bundle,results,gate1,gate2_report,gate3_report,ncu='ncu'):
    data,_,_=d.load_bundle(bundle);h.require_gate1(gate1,data)
    import cupy as cp
    gpu=GPU(cp,data);env=environment(cp)
    require_gate2(gate2_report,data,env,gpu);require_gate3(gate3_report,data,env,gpu)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',variants={},profiles={},note='instruction counts are not a time bound')
    def command(variant,folder,lineinfo):
        cmd=[sys.executable,'-m',__name__,'--target',variant,'--bundle',str(bundle),'--results',str(folder)]
        return cmd+(['--line-info'] if lineinfo else [])
    for variant,kernel in (('P5g','p5g_run_kernel'),('P5i','p5i_run_kernel')):
        state=dict(capture='pending',export='pending',parse='pending');report['profiles'][variant]=state
        try:
            plain,line=root/(variant+'_plain'),root/(variant+'_line')
            for folder,info in ((plain,False),(line,True)):
                subprocess.run(command(variant,folder,info),check=True)
            bits=h.exact(core.read_npz(plain/'history.npz'),core.read_npz(line/'history.npz'),core.HISTORY_ORDER)
            attrs=[json.loads((folder/'target.json').read_text())['launches'][kernel]['attributes'] for folder in (plain,line)]
            try:sass_equal=previous.sass(plain,kernel)==previous.sass(line,kernel)
            except core.Stop:sass_equal=None
            report['variants'][variant]=dict(bits=bits,sass_equal=sass_equal,resources=attrs,
                resources_equal=attrs[0]==attrs[1],attribution=variant if bits['status']=='pass' and sass_equal and attrs[0]==attrs[1] else 'line-info configuration only')
        except Exception as exc:
            report['variants'][variant]=dict(status='unavailable',stop=str(exc))
        folder=root/(variant+'_profile');folder.mkdir(parents=True,exist_ok=True)
        stage='capture'
        try:
            cmd=[ncu,'--section','SourceCounters','--replay-mode','kernel','--cache-control','all',
                '--kernel-name',kernel,'--launch-skip','1','--launch-count','1',
                '--export',str(folder/'instructions'),*command(variant,folder,True)]
            r=subprocess.run(cmd,capture_output=True,text=True)
            (folder/'capture.log').write_text(r.stdout+r.stderr)
            state['capture']='pass' if r.returncode==0 else 'fail'
            if r.returncode:raise core.Stop('ncu capture unavailable')
            stage='export'
            r=subprocess.run([ncu,'--import',str(folder/'instructions.ncu-rep'),'--page','source',
                '--print-source','cuda,sass','--csv','--resolve-source-file',str(folder/'compiled.cu')],capture_output=True,text=True)
            (folder/'source.csv').write_text(r.stdout);(folder/'export.log').write_text(r.stderr)
            state['export']='pass' if r.returncode==0 else 'fail'
            if r.returncode:raise core.Stop('ncu export unavailable')
            stage='parse'
            records=previous.parse_source_csv(r.stdout,function=kernel)
            core.write_json(folder/'sass_records.json',records)
            core.write_json(folder/'line_summary.json',g.weighted_map(records));state['parse']='pass'
            state['all_warp_instructions']=sum(r['warp'] for r in records)
            state['fp64_warp_instructions']=sum(r['warp'] for r in records if __import__('re').search(r'\bD(?:ADD|MUL|FMA|DIV|SQRT|SET|SETP|MNMX|RCP)\b',r['instruction']))
        except Exception as exc:
            state[stage]='fail'
            for key in ('capture','export','parse'):
                if state[key]=='pending':state[key]='skipped'
            state['stop']=str(exc)
        core.write_json(root/'map.json',report)
    report['status']='pass' if all(v['parse']=='pass' for v in report['profiles'].values()) else 'unavailable'
    core.write_json(root/'map.json',report)
    return report


def run_all(bundle,results,gate1):
    root=Path(results)
    gate2(bundle,root/'gate2',gate1)
    gate3(bundle,root/'gate3',gate1,root/'gate2/gate2.json')
    timing(bundle,root/'timing',gate1,root/'gate2/gate2.json',root/'gate3/gate3.json')
    return divergence_map(bundle,root/'map',gate1,root/'gate2/gate2.json',root/'gate3/gate3.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('P5g','P5i'))
    p.add_argument('--line-info',action='store_true')
    p.add_argument('--bundle',required=True);p.add_argument('--results',required=True)
    p.add_argument('--gate1')
    a=p.parse_args()
    if a.target:target(a.bundle,a.results,a.target,a.line_info)
    elif a.gate1:run_all(a.bundle,a.results,a.gate1)
    else:p.error('--gate1 required for all gates')
