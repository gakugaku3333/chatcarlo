"""Run-all T4 gates, timing and raw ncu capture. Never parses CSV on Colab."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from . import p5b_core as core,p5d_core as d,p5j_core as j,p5l_core as h
from . import p5j_colab as baseline
from . import p5k_colab as fbaseline
from .p5b_colab import GPU as BaseGPU


class GPU(baseline.GPU):
    def __init__(self,cp,data,variant='G1',check=False,lineinfo=False,compile_fault=True,blocks=40):
        baseline.GPU.__init__(self,cp,data,check=check,lineinfo=lineinfo,compile_fault=False)
        self.variant=variant;self.blocks=blocks
        text=h.source(mapped=lineinfo)
        for name in ('p5l_kernel','p5l_schedule_kernel','p5l_size_kernel','p5l_contract_kernel'):
            kernel=cp.RawKernel(text,name,options=self.options,backend='nvrtc');kernel.compile();self.kernels[name]=kernel
            self.launches[name]=dict(block=256,grid=blocks,options=list(self.options),
                attributes={k:int(kernel.attributes[k]) for k in ('num_regs','local_size_bytes','shared_size_bytes','max_threads_per_block')})
        if compile_fault:
            kernel=cp.RawKernel(text,'p5l_kernel',options=self.options+('-DP5B_MERGE_FAULT=1',),backend='nvrtc')
            kernel.compile();self.kernels['p5l_kernel_fault']=kernel
        size=cp.zeros(1,np.int32);self.kernels['p5l_size_kernel']((1,),(1,),(size,))
        self.slot_bytes=int(cp.asnumpy(size)[0]);self.arena=None;self.arena_bytes=0
        self.next=cp.zeros(1,np.int32)

    def storage(self,R,B=None):
        needed=self.slot_bytes*R*(self.blocks if B is None else B)
        if self.arena is None or self.arena_bytes<needed:
            self.arena=self.cp.empty(max(1,needed),np.uint8);self.arena_bytes=needed
        return self.arena

    def measure(self,n,start=0,history=False,fault=False,maxsteps=core.MAXSTEPS,audit_capacity=0):
        core.check_range(n,start,maxsteps=maxsteps)
        if audit_capacity and not history:raise ValueError('audit requires history')
        cp=self.cp;self.counts.fill(0);self.next.fill(0);cp.cuda.Device().synchronize();begin=time.perf_counter()
        histories=self.output(core.history_shapes(n) if history else core.history_shapes(1))
        production=self.output(core.run_shapes(len(self.data['cum'])))
        R=0 if self.variant=='FK4' else 256*h.CONFIG[h.VARIANTS.index(self.variant)][2]
        arena=self.storage(R) if R else cp.empty(1,np.uint8)
        trace=cp.zeros((n,audit_capacity,10)) if audit_capacity else cp.empty(0)
        meta=cp.zeros((n,2),np.int64) if audit_capacity else cp.empty(0,np.int64)
        args=self.args+(np.uint64(core.SEED),np.uint64(start),np.int32(n),np.int32(maxsteps),
            np.int32(h.VARIANTS.index(self.variant)),arena,self.next,np.int32(history))
        args+=tuple(histories[k] for k in core.HISTORY_ORDER)+tuple(production[k] for k in core.RUN_ORDER)
        # A zero-sized device array has a non-null allocation in CuPy: pass a true null pointer for no audit.
        args+=(trace if audit_capacity else np.uint64(0),np.int32(audit_capacity),meta if audit_capacity else np.uint64(0))
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.kernels['p5l_kernel'+('_fault' if fault else '')]((self.blocks,),(256,),args)
        last.record();last.synchronize()
        out=self.download(histories if history else production)
        times=dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin,
                   slot_bytes=self.slot_bytes,required_slot_bytes=R*self.blocks*self.slot_bytes,
                   retained_arena_bytes=self.arena_bytes,blocks=self.blocks,R=R)
        core.validate_output(out,core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        if audit_capacity:return out,times,cp.asnumpy(trace),cp.asnumpy(meta)
        return out,times

    def per_history(self,n,start=0,maxsteps=core.MAXSTEPS):return self.measure(n,start,history=True,maxsteps=maxsteps)[0]
    def production(self,n,start=0,fault=False):return self.measure(n,start,fault=fault)[0]

    def scheduler(self,n,variant,B=2,length=-1,phase=0):
        m=h.CONFIG[variant][2]
        cp=self.cp;self.next.fill(0);arena=self.storage(256*m,B)
        visits=cp.zeros(n,np.int32);rows=cp.full(n,-1,np.int32);histories=cp.zeros(n,np.uint64)
        ordered=cp.full(B*8*(32*m+1),-1,np.int32)
        self.kernels['p5l_schedule_kernel']((B,),(256,),
            (arena,np.int32(variant),np.int32(n),np.uint64(7000),self.next,visits,rows,histories,
             np.int32(length),np.int32(phase),ordered))
        if length>=0:return cp.asnumpy(ordered).reshape(-1,32*m+1)
        return core.verdict(bool(np.all(cp.asnumpy(visits)==1) and
            np.array_equal(cp.asnumpy(rows),np.arange(n)*100//n) and
            np.array_equal(cp.asnumpy(histories),np.arange(n,dtype=np.uint64)+7000)))


def gpu_management(gpu):
    for v,cfg in zip(h.VARIANTS,h.CONFIG):
        m=cfg[2]
        for B in (1,2,40):
            R=B*256*m
            for n in (1,R-1,R,R+1,2*R+17):core.require(gpu.scheduler(n,h.VARIANTS.index(v),B),'T4 ownership '+v)
        for phase in range(4):
            for length in (0,1,31,32,33,63,64,65,127,128):
                if length>32*m:continue
                out=gpu.scheduler(1,h.VARIANTS.index(v),2,length,phase)
                core.require(core.verdict(bool(np.all(out[:,-1]==length) and
                    np.all(out[:,:length]==np.arange(length)))),'T4 list endpoint')
    synthetic(gpu)
    return dict(status='pass',variants=list(h.VARIANTS),synthetic=core.verdict(True))


def synthetic(gpu):
    cp=gpu.cp
    out=cp.zeros(8*gpu.blocks*4,np.int32)
    gpu.kernels['p5l_contract_kernel']((gpu.blocks,),(256,),(gpu.storage(1024),out))
    core.require(core.verdict(bool(np.all(cp.asnumpy(out)==0))),'T4 synthetic contracts')


def refill_audit(new,old):
    # 2 blocks, 8 warps each; n exceeds even W4's 2048 slots.
    saved=new.blocks;new.blocks=2
    try:
        n=2081
        out,_,tr,meta=new.measure(n,7000,history=True,audit_capacity=8192)
        ref,_,rt,rm=old.measure(n,7000,history=True,audit_capacity=8192)
        core.require(h.exact(out,ref),'refill audit fields')
        core.require(h.traces(tr,meta,rt,rm),'refill audit complete traces')
    finally:new.blocks=saved
    return core.verdict(True,n=n,blocks=2,overflow=0)


def transport_edges(new,old):
    for steps in (1,2):
        for n in (529,1,529):
            out,_,tr,meta=new.measure(n,7000,history=True,maxsteps=steps,audit_capacity=32)
            ref,_,rt,rm=old.measure(n,7000,history=True,maxsteps=steps,audit_capacity=32)
            core.require(h.exact(out,ref),'T4 failure/reuse fields');core.require(h.traces(tr,meta,rt,rm),'T4 failure/reuse trace')
    _,_,tr,meta=new.measure(1,7000,history=True,audit_capacity=1)
    core.require(core.verdict(h.traces(tr,meta,tr,meta)['status']=='fail'),'T4 overflow rejection')
    return core.verdict(True)


def gate2(cp,data,root):
    root=Path(root);report=dict(status='incomplete',checks={},gate0={},identity=h.identity(data),environment=baseline.environment(cp))
    if 'T4' not in report['environment']['gpu']:raise core.Stop('T4 required')
    try:
        for label,checked in (('ordinary',False),('checked',True)):
            old=baseline.GPU(cp,data,check=checked)
            new=GPU(cp,data,check=checked)
            report['gate0'][label]=dict(functions=baseline.gate0(new,root/'gate0'/label),management=gpu_management(new))
            report.setdefault('generated',{})[label]=new.generated
            report.setdefault('launches',{})[label]=new.launches
            expected=old.per_history(20000);reference_counts=cp.asnumpy(old.counts)
            report['checks'][label]={}
            core.require(core.c3(expected,old.production(20000),20),'P5j aggregate against histories')
            for v in h.VARIANTS:
                new.variant=v;out=new.per_history(20000)
                count=cp.asnumpy(new.counts)
                fields=h.exact(out,expected);core.require(fields,'T4 '+v+' histories')
                core.require(core.verdict(core.bits(count,reference_counts)),'T4 log counts')
                aggregate=core.c3(expected,new.production(20000),20);core.require(aggregate,'T4 '+v+' aggregate')
                # All 20,000 audit traces, chunked to bound device/host memory.
                for first in range(0,20000,32):
                    n=min(32,20000-first)
                    out,_,tr,meta=new.measure(n,first,history=True,audit_capacity=8192)
                    ref,_,rt,rm=old.measure(n,first,history=True,audit_capacity=8192)
                    core.require(h.exact(out,ref),'T4 audit fields');core.require(h.traces(tr,meta,rt,rm),'T4 audit trace')
                edge=transport_edges(new,old)
                refill=refill_audit(new,old)
                report['checks'][label][v]=dict(fields=fields,aggregate=aggregate,traces=core.verdict(True),edges=edge,refill=refill)
                core.write_json(root/'gate2.json',report)
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/'gate2.json',report)
    return report


def require_gate2(report,data):
    if report.get('status')!='pass' or report.get('identity')!=h.identity(data):raise core.Stop('current complete gate2 required')
    for label in ('ordinary','checked'):
        gate=report.get('gate0',{}).get(label,{})
        j.require_gate0(gate.get('functions',{}),label=='checked')
        core.require(gate.get('management',core.verdict(False)),'gate2 management')
        if not report.get('generated',{}).get(label) or not report.get('launches',{}).get(label):
            raise core.Stop('gate2 generated constants and launch records required')
        checks=report.get('checks',{}).get(label,{})
        if set(checks)!=set(h.VARIANTS):raise core.Stop('gate2 variants incomplete')
        for v in checks.values():
            for k in ('fields','aggregate','traces','edges','refill'):core.require(v[k],'gate2 '+k)


def timing(cp,data,root,gate2_report):
    require_gate2(gate2_report,data);root=Path(root)
    old=fbaseline.GPU(cp,data,'F');new=GPU(cp,data);records=[]
    labels=('F',*h.VARIANTS)
    for gpu in (old,new):core.require(core.health(gpu.production(1000)),'warmup')
    for repeat in range(3):
        order=labels[repeat:]+labels[:repeat]
        if repeat==1:order=order[::-1]
        for label in order:
            gpu=old if label=='F' else new
            if label!='F':new.variant=label
            out,t=gpu.measure(100000);core.require(core.health(out),'timing health')
            records.append(dict(variant=label,repeat=repeat,n=100000,**t))
    medians={v:float(np.median([r['kernel_ms'] for r in records if r['variant']==v])) for v in labels}
    fastest=min(h.VARIANTS,key=medians.get)
    selected=fastest if medians[fastest]<medians['F'] else 'W4K4'
    if medians[fastest]<medians['F']:
        for label in ('F',fastest):
            gpu=old if label=='F' else new;new.variant=fastest
            out,t=gpu.measure(1000000);core.require(core.health(out),'million health')
            records.append(dict(variant=label,n=1000000,**t))
    report=dict(status='pass',identity=h.identity(data),measurements=records,medians_ms=medians,
        selected=selected,fastest=fastest,candidate_status='gate2 passed; gate3 unconfirmed',
        decision='report only; adoption is the user decision',
        interpretation='Differences include compiler register allocation and the whole driver method.')
    core.write_json(root/'timing.json',report);return report


def gate3(gpu,reference,root,gate2_report):
    require_gate2(gate2_report,gpu.data);root=Path(root)
    identity=dict(h.identity(gpu.data,gpu.generated),variant=gpu.variant,environment=baseline.environment(gpu.cp),launches=gpu.launches)
    path=root/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise core.Stop('C0 identity mismatch')
    core.write_json(path,identity);report=dict(overall='incomplete',identity=identity,checks={})
    def record(k,v):core.record(report,k,v)
    def save(k,v):core.save_npz(root/(k+'.npz'),v)
    try:
        record('C0',core.verdict(True));values=gpu.checks(reference);save('c1',values);record('C1',core.c1(reference,values))
        histories=gpu.per_history(20000);save('c2',histories)
        record('C2',core.c2({k:reference['history_'+k] for k in core.HISTORY_ORDER},histories,float(reference['Dref']),float(reference['conversion'])))
        prod=gpu.production(20000);save('c3',prod);normal=core.c3(histories,prod,20)
        faulty=gpu.production(20000,fault=True);save('c3_fault',faulty);fault=core.c3(histories,faulty,20)
        pieces=[gpu.per_history(5000,start) for start in range(0,20000,5000)]
        split={k:np.concatenate([r[k] for r in pieces]) for k in core.HISTORY_ORDER}
        record('C3',core.verdict(normal['status']=='pass' and fault['status']=='fail' and core.split_check(histories,split)['status']=='pass',ordinary=normal,merge_fault=fault))
        found=core.resume(root/'batches',identity)
        for b in range(100):
            if b not in found:
                out=gpu.production(100000,100000*b);core.require(core.health(out),'C4 health')
                if not core.bits(out['nspec'],reference['c4_nspec'][b]):raise core.Stop('C4 spectrum')
                path=root/'batches'/f'batch_{b:03}.npz';core.save_npz(path,out,core.checkpoint_meta(identity,b,out));found[b]=path
        record('C4',core.c4(reference,[core.read_checkpoint(found[b],identity,b) for b in range(100)],20))
        report['overall']='pass; adoption pending user decision'
    except Exception as exc:
        report.update(overall='not adopted',stop=str(exc));raise
    finally:core.write_json(root/'gate3.json',report)
    return report


def profile(bundle,root,variant):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str(root/'nvrtc_cache')
    import cupy as cp
    data,_,_=d.load_bundle(bundle)
    gpu=baseline.GPU(cp,data,lineinfo=True,compile_fault=False) if variant=='P5j' else GPU(cp,data,variant,lineinfo=True,compile_fault=False)
    (root/'compiled.cu').write_text(j.source(True) if variant=='P5j' else h.source(True))
    gpu.production(1000);gpu.production(100000)
    core.write_json(root/'target.json',dict(variant=variant,launches=gpu.launches,environment=baseline.environment(cp)))


def raw_map(bundle,root,selected):
    root=Path(root);report=dict(status='incomplete',profiles={},parse='deferred to local; raw CSV only')
    # Sections retain requested/actual memory traffic, hits, barriers, occupancy and resources alongside source counters.
    sections=('SpeedOfLight','SourceCounters','MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables','WarpStateStats','Occupancy','LaunchStats','SchedulerStats','InstructionStats','ComputeWorkloadAnalysis')
    other=('W4K4' if selected=='W2K4' else 'W2K4') if selected.startswith('W') else 'W4K4'
    for variant in (selected,other):
        folder=root/variant;folder.mkdir(parents=True,exist_ok=True);kernel='p5j_run_kernel' if variant=='P5j' else 'p5l_kernel'
        cmd=['ncu','--replay-mode','kernel','--cache-control','all','--kernel-name',kernel,'--launch-skip','1','--launch-count','1']
        for section in sections:cmd+=['--section',section]
        cmd+=['--export',str(folder/'profile'),sys.executable,'-m',__name__,'--target',variant,'--bundle',str(bundle),'--results',str(folder)]
        capture=subprocess.run(cmd,capture_output=True,text=True);(folder/'capture.log').write_text(capture.stdout+capture.stderr)
        state=dict(capture_returncode=capture.returncode,exports={});report['profiles'][variant]=state
        if capture.returncode==0:
            for page in ('source','details','raw'):
                cmd=['ncu','--import',str(folder/'profile.ncu-rep'),'--page',page,'--csv']
                if page=='source':cmd+=['--print-source','cuda,sass','--resolve-source-file',str(folder/'compiled.cu')]
                out=subprocess.run(cmd,capture_output=True,text=True);(folder/(page+'.csv')).write_text(out.stdout);(folder/(page+'.log')).write_text(out.stderr)
                state['exports'][page]=out.returncode
        core.write_json(root/'map.json',report)
    report['status']='captured' if all(v['capture_returncode']==0 and len(v['exports'])==3 and not any(v['exports'].values()) for v in report['profiles'].values()) else 'unavailable'
    core.write_json(root/'map.json',report);return report


def run_all(bundle,results,gate1):
    data,ref,_=d.load_bundle(bundle);h.require_gate1(gate1,data)
    import cupy as cp
    root=Path(results);r=gate2(cp,data,root/'gate2')
    t=timing(cp,data,root/'timing',r)
    gpu=GPU(cp,data,t['selected']);gate3(gpu,ref,root/'gate3',r)
    return raw_map(bundle,root/'map',t['fastest'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target',choices=('P5j',*h.VARIANTS))
    p.add_argument('--bundle',required=True);p.add_argument('--results',required=True);p.add_argument('--gate1');a=p.parse_args()
    if a.target:profile(a.bundle,a.results,a.target)
    elif a.gate1:run_all(a.bundle,a.results,a.gate1)
    else:p.error('--gate1 required')
