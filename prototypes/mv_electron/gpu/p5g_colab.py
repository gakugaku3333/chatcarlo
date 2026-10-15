"""P5f-1 map, gate 2, timing and independent gate 3. No CUDA imports at import time."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from . import p5b_core as core, p5d_core as d, p5g_core as g
from .p5b_colab import GPU as BaseGPU, environment


class GPU(BaseGPU):
    def __init__(self, cp, data, baseline=False, cap=64, lineinfo=False, compile_fault=True):
        self.baseline=baseline
        self.options=core.OPTIONS[:-1]+(f'-DP5_STACK_CAP={cap}',)+(('--generate-line-info',) if lineinfo else ())
        text=core.source() if baseline else g.source()
        names=core.KERNELS if baseline else g.KERNELS
        self.entries=names[:2];kernels={};self.launches={}
        for name in (*names,*((names[0]+'_fault',) if compile_fault else ())):
            opts=self.options+(('-DP5B_MERGE_FAULT=1',) if name.endswith('_fault') else ())
            k=cp.RawKernel(text,name.removesuffix('_fault'),options=opts,backend='nvrtc')
            k.compile();kernels[name]=k
            self.launches[name]=dict(block=128,grid='ceil(n/128)',options=list(opts),
                attributes={k0:int(k.attributes[k0]) for k0 in
                    ('num_regs','local_size_bytes','shared_size_bytes','max_threads_per_block')})
        super().__init__(cp,data,kernels)

    def measure(self,n,start=0,history=False,fault=False,maxsteps=core.MAXSTEPS,audit_capacity=0):
        core.check_range(n,start,maxsteps=maxsteps)
        if audit_capacity and (self.baseline or not history):raise ValueError('audit requires new history kernel')
        cp=self.cp;cp.cuda.Device().synchronize();begin=time.perf_counter()
        r=self.output(core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        name=self.entries[int(history)]+('_fault' if fault else '')
        args=self.args+(np.uint64(core.SEED),np.uint64(start),np.int32(n),np.int32(maxsteps))
        args+=tuple(r[k] for k in (core.HISTORY_ORDER if history else core.RUN_ORDER))
        if audit_capacity:
            trace=cp.zeros((n,audit_capacity,10));meta=cp.zeros((n,2),np.int64)
            args+=(trace,np.int32(audit_capacity),meta);name='p5g_audit_kernel'
        first,last=cp.cuda.Event(),cp.cuda.Event();first.record()
        self.launch(name,n,args);last.record();out=self.download(r);last.synchronize()
        times=dict(kernel_ms=float(cp.cuda.get_elapsed_time(first,last)),call_s=time.perf_counter()-begin)
        core.validate_output(out,core.history_shapes(n) if history else core.run_shapes(len(self.data['cum'])))
        if audit_capacity:return out,times,cp.asnumpy(trace),cp.asnumpy(meta)
        return out,times

    def per_history(self,n,start=0,maxsteps=core.MAXSTEPS):
        return self.measure(n,start,history=True,maxsteps=maxsteps)[0]

    def production(self,n,start=0,fault=False):
        return self.measure(n,start,fault=fault)[0]


def gate2(bundle,results,gate1):
    """Only after current sources pass the CPU 100-batch gate."""
    g.require_gate1(Path(gate1))
    import cupy as cp
    data,ref,manifest=d.load_bundle(bundle);root=Path(results);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',checks={},environment=environment(cp))
    if 'T4' not in report['environment']['gpu']:raise core.Stop('gate 2 requires T4')
    new,old=GPU(cp,data),GPU(cp,data,baseline=True)
    report['identity']=g.identity(manifest,report['environment'],new.launches)
    def check(name,a,b,start=0):
        core.save_npz(root/(name+'_new.npz'),a);core.save_npz(root/(name+'_v0.npz'),b)
        report['checks'][name]=g.exact(a,b,core.HISTORY_ORDER)
        if report['checks'][name]['status']!='pass':
            indexes=[]
            for k in core.HISTORY_ORDER:
                matched=core.bit_elements(a[k],b[k]).reshape(len(a[k]),-1).all(axis=1)
                indexes.extend(np.flatnonzero(~matched).tolist())
            first=min(indexes) if indexes else 0
            _,_,tr,meta=new.measure(1,start+first,history=True,audit_capacity=100000)
            core.save_npz(root/(name+'_mismatch_trace.npz'),dict(trace=tr,meta=meta))
        core.require(report['checks'][name],name)
    try:
        check('specified_20000',new.per_history(20000),old.per_history(20000))
        for i,(n,start) in enumerate(((1,0),(127,7000),(128,7000),(129,7000),(129,7000))):
            a,_,trace,meta=new.measure(n,start,history=True,audit_capacity=20000)
            check('range_'+str(i),a,old.per_history(n,start),start=start)
            core.save_npz(root/f'range_{i}_trace.npz',dict(trace=trace,meta=meta))
            for j in range(n):core.require(g.trace_check(trace[j],meta[j],stored=True),'GPU lifecycle')
        report['unusual']=unusual_gpu(cp,data,root/'unusual')
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(root/'gate2.json',report)
    return report


def unusual_gpu(cp,data,root):
    root.mkdir(parents=True,exist_ok=True);evidence=set();checks={}
    for cap,maxsteps,label,n in ((64,2,'photon_limit',100),(64,1,'charged_limit',100),
                                 (3,core.MAXSTEPS,'stack',100),(64,core.MAXSTEPS,'ordinary',2000)):
        fixture=dict(data,edges=np.array([.015,.016] if label=='photon_limit' else [9.99,10.]),cum=np.array([1.]))
        new,old=GPU(cp,fixture,cap=cap),GPU(cp,fixture,baseline=True,cap=cap)
        a,b=new.per_history(n,maxsteps=maxsteps),old.per_history(n,maxsteps=maxsteps)
        core.save_npz(root/(label+'_new.npz'),a);core.save_npz(root/(label+'_v0.npz'),b)
        checks[label]=g.exact(a,b,core.HISTORY_ORDER);core.require(checks[label],label)
        for h in range(n):
            # Diagnostic path only: full transitions, bounded and overflow checked.
            out,_,tr,meta=new.measure(1,h,history=True,maxsteps=maxsteps,audit_capacity=20000)
            core.require(g.exact(out,{k:v[h:h+1] for k,v in a.items()},core.HISTORY_ORDER),'audit vs ordinary')
            core.require(g.trace_check(tr[0],meta[0],stored=True),'GPU unusual trace')
            v=tr[0,:int(meta[0,0])];pairs=v[:,:2].astype(int);reason=v[:,7].astype(int)
            seen=set()
            if np.any((pairs[:,0]==0)&(pairs[:,1]==3)&((reason&1)>0)) and label=='photon_limit':seen.add('photon_limit')
            if np.any((pairs[:,0]==1)&(pairs[:,1]==3)&((reason&1)>0)) and label=='charged_limit':seen.add('charged_limit')
            if cap==3 and out['counts'][0,6]>0:seen.add('stack')
            seen.update(g.trace_events(v))
            if seen-evidence:
                core.save_npz(root/f'{label}_trace_{h:04}.npz',dict(trace=v,meta=meta[0]))
                evidence.update(seen)
    required={'photon_limit','charged_limit','stack','hard','brems','reentry','rest','flight','escape','hard_resume','brems_resume'}
    result=core.verdict(required<=evidence,checks=checks,evidence=sorted(evidence),missing=sorted(required-evidence))
    core.write_json(root/'unusual.json',result);core.require(result,'GPU unusual');return result


def require_gate2(path,manifest,env,gpu):
    report=json.loads(Path(path).read_text())
    # JSON roundtrip normalizes tuple/list representation in state specification.
    identity=json.loads(json.dumps(g.identity(manifest,env,gpu.launches)))
    if report.get('status')!='pass' or report.get('identity')!=identity:
        raise core.Stop('gate 2 missing, failed or current configuration differs')


def require_gate3(path,manifest,env,gpu):
    report=json.loads(Path(path).read_text())
    expected=json.loads(json.dumps(g.identity(manifest,env,gpu.launches)))
    if report.get('overall')!='pass; adoption pending user decision' or report.get('identity')!=expected:
        raise core.Stop('gate 3 missing, failed or configuration differs')


def timing(bundle,results,gate1,gate2_report,gate3_report):
    g.require_gate1(Path(gate1))
    import cupy as cp
    data,_,manifest=d.load_bundle(bundle);new,old=GPU(cp,data),GPU(cp,data,baseline=True)
    require_gate2(gate2_report,manifest,environment(cp),new)
    require_gate3(gate3_report,manifest,environment(cp),new)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',measurements=[],identity=g.identity(manifest,environment(cp),new.launches),
                decision='report only; adoption is a user decision')
    try:
        for gpu in (old,new):core.require(core.health(gpu.production(1000)),'warmup health')
        for repeat in range(3):
            for label,gpu in ([('V0',old),('P5g',new)] if repeat%2==0 else [('P5g',new),('V0',old)]):
                out,times=gpu.measure(100000);core.require(core.health(out),'timing health')
                core.save_npz(root/f'{label}_{repeat}.npz',out)
                report['measurements'].append(dict(variant=label,repeat=repeat,n=100000,**times))
        report['status']='pass'
    finally:core.write_json(root/'timing.json',report)
    return report


def gate3(bundle,results,gate1,gate2_report):
    g.require_gate1(Path(gate1))
    import cupy as cp
    data,ref,manifest=d.load_bundle(bundle);gpu=GPU(cp,data);env=environment(cp)
    require_gate2(gate2_report,manifest,env,gpu)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    identity=g.identity(manifest,env,gpu.launches)
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


def parse_source_csv(text,function=None):
    """Fail closed if this ncu version does not expose all three instruction counts."""
    aliases=dict(address=('Address','SASS Address','Program Counter','PC'),
        instruction=('SASS','Instruction','SASS Instruction'),
        warp=('Instructions Executed','Warp Instructions Executed'),
        thread=('Thread Instructions Executed',),
        predicated=('Predicated-On Thread Instructions Executed','Predicated On Thread Instructions Executed'),
        line=('Line','Source Line','Line Number'),source=('File','Source File'),function=('Function','Kernel Name'))
    records=[];columns=None
    for row in csv.reader(io.StringIO(text)):
        lookup={v.strip():i for i,v in enumerate(row)}
        found={k:next((lookup[a] for a in names if a in lookup),None) for k,names in aliases.items()}
        if all(found[k] is not None for k in ('address','instruction','warp','thread','predicated')):
            columns=found;continue
        if columns is None:continue
        try:
            value={k:row[i].strip() if i is not None else '' for k,i in columns.items()}
            if not value['address'] or not value['instruction']:continue
            value['function']=value['function'] or function or ''
            value['source']=value['source'] or '<unmapped>'
            for k in ('warp','thread','predicated'):value[k]=float(value[k].replace(',',''))
            value['line']=int(value['line']) if value['line'] else 0
            if not np.isfinite([value[k] for k in ('warp','thread','predicated')]).all():continue
            records.append(value)
        except (IndexError,ValueError):continue
    if not records or any(not r['function'] for r in records) or not any(r['line'] for r in records):
        raise core.Stop('ncu source CSV schema/counters/mapping unavailable; retain raw export, do not infer lane counts')
    return records


def source_mapping(text):
    import difflib
    lines=text.splitlines();mapping=[]
    names=('p5a0_common.h','p5_rng.h','p5_kernel.h','p5b_wrap.h')
    for index,name in enumerate(names):
        begin=lines.index('// '+name)+1
        end=lines.index('// '+names[index+1]) if index+1<len(names) else len(lines)
        original=(core.HERE/name).read_text().splitlines()
        for block in difflib.SequenceMatcher(None,original,lines[begin:end],autojunk=False).get_matching_blocks():
            for offset in range(block.size):
                mapping.append(dict(compiled_line=begin+block.b+offset+1,header=name,header_line=block.a+offset+1))
    return mapping


def target(bundle,root,variant,lineinfo):
    """Separate process: retained NVRTC cache holds cubins for SASS inspection."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str(root/'nvrtc_cache')
    import cupy as cp
    data,_,_=d.load_bundle(bundle);gpu=GPU(cp,data,baseline=variant=='V0',lineinfo=lineinfo,compile_fault=False)
    text=core.source() if variant=='V0' else g.source()
    (root/'compiled.cu').write_text(text)
    if variant=='V0':core.write_json(root/'source_mapping.json',source_mapping(text))
    history=gpu.per_history(20000);core.save_npz(root/'history.npz',history)
    gpu.production(1000);cp.cuda.Device().synchronize()
    gpu.production(100000);cp.cuda.Device().synchronize()
    core.write_json(root/'target.json',dict(environment=environment(cp),options=list(gpu.options),launches=gpu.launches))


def sass(root,kernel):
    binary=shutil.which('cuobjdump')
    if not binary:raise core.Stop('cuobjdump unavailable')
    for i,path in enumerate(sorted((root/'nvrtc_cache').glob('*.cubin'))):
        r=subprocess.run([binary,'--dump-sass',str(path)],capture_output=True,text=True)
        (root/f'sass_{i}.txt').write_text(r.stdout+r.stderr)
        if r.returncode:continue
        match=re.search(r'Function : '+re.escape(kernel)+r'\s*\n(.*?)(?=Function : |\Z)',r.stdout,re.S)
        if match:
            # Source line annotations are deliberately excluded; compare instruction text and encodings.
            instructions=[line.strip() for line in match[1].splitlines() if re.search(r'/\*\s*(?:0x)?[0-9a-fA-F]+\s*\*/',line)]
            if instructions:return instructions
    raise core.Stop('target SASS unavailable in retained NVRTC cubins')


def divergence_map(bundle,results,gate1,ncu='ncu',compare=False,gate2_report=None,gate3_report=None):
    g.require_gate1(Path(gate1))
    if compare:
        import cupy as cp
        data,_,manifest=d.load_bundle(bundle);gpu=GPU(cp,data)
        require_gate2(gate2_report,manifest,environment(cp),gpu)
        require_gate3(gate3_report,manifest,environment(cp),gpu)
    root=Path(results);root.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',cause='not classified; instruction locations only',profiles={})
    try:
        for label,lineinfo in (('v0_plain',False),('v0_line',True)):
            folder=root/label
            cmd=[sys.executable,'-m',__name__,'--target','V0','--bundle',str(bundle),'--results',str(folder)]
            if lineinfo:cmd+=['--line-info']
            subprocess.run(cmd,check=True)
        a,b=(core.read_npz(root/k/'history.npz') for k in ('v0_plain','v0_line'))
        report['lineinfo_bits']=g.exact(a,b,core.HISTORY_ORDER)
        core.require(report['lineinfo_bits'],'line-info 20000 history bits')
        same_sass=sass(root/'v0_plain','p5b_run_kernel')==sass(root/'v0_line','p5b_run_kernel')
        attrs=[json.loads((root/k/'target.json').read_text())['launches']['p5b_run_kernel']['attributes'] for k in ('v0_plain','v0_line')]
        report.update(sass_equal=same_sass,resources_equal=attrs[0]==attrs[1],resources=attrs,
            attribution='V0' if same_sass and attrs[0]==attrs[1] else '行の情報付きの構成の地図。V0への直接の帰属は保留')
        for variant,kernel in ((('V0','p5b_run_kernel'),('P5g','p5g_run_kernel')) if compare else (('V0','p5b_run_kernel'),)):
            folder=root/(variant+'_profile');folder.mkdir(parents=True,exist_ok=True)
            cmd=[ncu,'--section','SourceCounters','--replay-mode','kernel','--cache-control','all',
                 '--kernel-name',kernel,'--launch-skip','1','--launch-count','1',
                 '--export',str(folder/'instructions'),sys.executable,'-m',__name__,
                 '--target',variant,'--line-info','--bundle',str(bundle),'--results',str(folder)]
            r=subprocess.run(cmd,capture_output=True,text=True)
            (folder/'capture.log').write_text(r.stdout+r.stderr)
            record=report['profiles'][variant]=dict(command=cmd,returncode=r.returncode,status='unavailable')
            if r.returncode:raise core.Stop('ncu counters unavailable; retained capture.log')
            export=[ncu,'--import',str(folder/'instructions.ncu-rep'),'--page','source','--print-source','cuda,sass','--csv','--resolve-source-file',str(folder/'compiled.cu')]
            r=subprocess.run(export,capture_output=True,text=True)
            (folder/'source.csv').write_text(r.stdout);(folder/'export.log').write_text(r.stderr)
            if r.returncode:raise core.Stop('ncu source export unavailable')
            records=parse_source_csv(r.stdout,function=kernel)
            core.write_json(folder/'sass_records.json',records)
            core.write_json(folder/'line_summary.json',g.weighted_map(records))
            record['status']='pass'
        report['status']='pass'
    except Exception as exc:
        report.update(status='unavailable',stop=str(exc));raise
    finally:core.write_json(root/'map.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['V0','P5g'],required=True)
    p.add_argument('--line-info',action='store_true')
    p.add_argument('--bundle',required=True);p.add_argument('--results',required=True)
    args=p.parse_args();target(args.bundle,args.results,args.target,args.line_info)
