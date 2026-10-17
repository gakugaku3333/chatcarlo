"""P5k CPU phase queues. gate1 is explicit and never invoked by local/export."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from . import p5b_core as core, p5d_core as d, p5j, p5k_core as h


class CPU(p5j.CPU):
    def __init__(self,work,data,check=False):
        # Build only the new translation unit, which includes the frozen P5j CPU entry points.
        core.fingerprints();core.check_inputs(data)
        self.checked=check;self.data=data;self.tables=self.make_tables(data);self.maps=p5j.layout(data)
        self.logs=np.empty(int(self.maps['lengths'].sum())+3);self.counts=np.zeros((20,2),np.uint64)
        self.flags=d.FLAGS+[f'-DP5J_CHECK={int(check)}'];exe=shutil.which('clang++')
        if exe is None:raise core.Stop('clang++ unavailable')
        self.compiler=subprocess.check_output([exe,'--version'],text=True).splitlines()[0]
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        digest=core.sha(json.dumps(dict(identity=h.identity(data),flags=self.flags,compiler=self.compiler),sort_keys=True).encode())
        path=work/('p5k_'+digest[:16]+('.dylib' if sys.platform=='darwin' else '.so'))
        build=subprocess.run([exe,*self.flags,str(core.HERE/'p5k_cpu.cpp'),'-o',str(path)],capture_output=True,text=True)
        (work/'build.log').write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(path));dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.lib.p5j_configure.argtypes=[ct.POINTER(d.Tables),dp,p5j.UP,ct.c_int]
        self.lib.p5j_gate0.argtypes=[ct.c_int,dp,ct.c_int,dp];self.lib.p5j_contract.argtypes=[dp]
        for name in ('p5j_configure','p5j_gate0','p5j_contract'):getattr(self.lib,name).restype=None
        self.lib.p5k_run.argtypes=[ct.c_int,ct.c_int,ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int]+[
            dp if k not in core.INTEGER else (ct.POINTER(ct.c_int) if k=='maxdepth' else ip) for k in core.HISTORY_ORDER]+[dp,ct.c_int,ip]
        self.lib.p5k_run.restype=None
        self.lib.p5k_schedule.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.c_uint64,ct.c_int,
            ct.POINTER(ct.c_int),ct.POINTER(ct.c_int),ct.POINTER(ct.c_uint64)]
        self.lib.p5k_schedule.restype=None
        self.lib.p5k_lists.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_int)]
        self.lib.p5k_slot_bytes.restype=ct.c_int
        import time
        begin=time.perf_counter();self.configure(True);self.initialization_s=time.perf_counter()-begin
        self.generated=core.describe(dict(logs=self.logs[:-3],constants=self.logs[-3:],**self.maps))

    def histories(self,n,variant='Q1',start=7000,blocks=2,maxsteps=core.MAXSTEPS,capacity=0):
        core.check_range(n,start,maxsteps=maxsteps)
        if blocks<1:raise ValueError('blocks must be positive')
        self.counts.fill(0);self.configure()
        r=core.empty_output(core.history_shapes(n));depth=np.zeros(n,np.int32)
        trace=np.zeros((n,capacity,10));meta=np.zeros((n,2),np.int64)
        args=[depth.ctypes.data_as(ct.POINTER(ct.c_int)) if k=='maxdepth' else r[k] for k in core.HISTORY_ORDER]
        v=-1 if variant=='P5j' else h.VARIANTS.index(variant)
        self.lib.p5k_run(v,blocks,core.SEED,start,n,maxsteps,*args,trace,capacity,meta)
        r['maxdepth']=depth.astype(np.int64)
        core.validate_output(r,core.history_shapes(n))
        return (r,trace,meta) if capacity else r

    def schedule(self,variant,n,blocks=2,start=7000,fault=0):
        visits=np.zeros(n,np.int32);rows=np.full(n,-1,np.int32);histories=np.zeros(n,np.uint64)
        self.lib.p5k_schedule(h.VARIANTS.index(variant),blocks,n,start,fault,
            visits.ctypes.data_as(ct.POINTER(ct.c_int)),rows.ctypes.data_as(ct.POINTER(ct.c_int)),
            histories.ctypes.data_as(ct.POINTER(ct.c_uint64)))
        return core.verdict(bool(np.all(visits==1) and np.array_equal(rows,np.arange(n)*100//n)
            and np.array_equal(histories,np.arange(n,dtype=np.uint64)+start)))


def management(cpu):
    checks={}
    for v in h.VARIANTS:
        R=512 if v=='Q1-512' else 256
        for B in (1,2,3):
            for n in (1,31,R-1,R,R+1,B*R-1,B*R,B*R+1,2*B*R+17,1,B*R+1):
                key=f'{v}/{B}/{n}/{len(checks)}';checks[key]=cpu.schedule(v,n,B)
                core.require(checks[key],key)
        for fault in (1,2,3):
            bad=cpu.schedule(v,2*R+17,fault=fault)
            core.require(core.verdict(bad['status']=='fail'),'planted scheduler fault')
    for R in (256,512):
        for p in range(4):
            for length in (0,1,31,32,33,255,256,257,511,512):
                if length>R:continue
                out=np.full(R,-1,np.int32)
                count=cpu.lib.p5k_lists(R,p,length,out.ctypes.data_as(ct.POINTER(ct.c_int)))
                core.require(core.verdict(count==length and np.array_equal(out[:length],np.arange(length))), 'list endpoint')
    return dict(status='pass',cases=len(checks),faults=15,list_lengths=[0,1,31,32,33,255,256,257,511,512])


def compare_small(cpu,n=2000,start=7000):
    expected=cpu.histories(n,'P5j',start);checks={}
    for v in h.VARIANTS:
        out=cpu.histories(n,v,start);checks[v]=h.exact(out,expected);core.require(checks[v],v+' histories')
    # Bounded chunks keep complete traces without an n*capacity allocation.
    trace_checks={v:True for v in h.VARIANTS};maxtrace=0
    for first in range(0,n,16):
        size=min(16,n-first);ref,rt,rm=cpu.histories(size,'P5j',start+first,capacity=8192)
        maxtrace=max(maxtrace,int(rm[:,0].max()))
        for v in h.VARIANTS:
            out,tr,meta=cpu.histories(size,v,start+first,capacity=8192)
            core.require(h.exact(out,ref),v+' audit fields')
            result=h.traces(tr,meta,rt,rm);core.require(result,v+' audit trace')
    # Same arena large -> small -> large, including forced failures and immediate reuse.
    unusual={};evidence=set()
    for steps in (1,2):
        for v in h.VARIANTS:
            for size in (529,1,529):
                out,tr,meta=cpu.histories(size,v,start,maxsteps=steps,capacity=32)
                ref,rt,rm=cpu.histories(size,'P5j',start,maxsteps=steps,capacity=32)
                core.require(h.exact(out,ref),'failure/reuse fields');core.require(h.traces(tr,meta,rt,rm),'failure/reuse traces')
                for tr0,m in zip(tr,meta):
                    pairs=tr0[:int(m[0]),:2]
                    for pair in ((0,3),(1,3),(3,4)):
                        if np.any(np.all(pairs==pair,axis=1)):evidence.add(pair)
            unusual[f'{v}/{steps}']=core.verdict(True)
    core.require(core.verdict({(0,3),(1,3),(3,4)}<=evidence),'P-H and C-H-D failure evidence')
    for v in h.VARIANTS:
        _,tr,meta=cpu.histories(1,v,start,capacity=1)
        core.require(core.verdict(h.traces(tr,meta,tr,meta)['status']=='fail'),'trace overflow rejection')
    return dict(status='pass',n=n,start=start,fields=checks,traces=trace_checks,max_trace_records=maxtrace,unusual=unusual)


def local(work,data):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);report=dict(status='incomplete',modes={})
    try:
        for label,check in (('ordinary',False),('checked',True)):
            cpu=CPU(work/label,data,check)
            report['modes'][label]=dict(management=management(cpu),transport=compare_small(cpu))
            report['slot_bytes']=cpu.lib.p5k_slot_bytes();report['identity']=h.identity(data,cpu.generated)
        from .p5k_emulate import reproduce
        report['emulation']=reproduce(cpu,4096)
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'local.json',report)
    return report


def gate1(work,saved):
    from . import p5b, p5j_core
    work=Path(work);data,config=p5b.load_local_inputs(Path(saved))
    cpus={k:CPU(work/k,data,c) for k,c in (('ordinary',False),('checked',True))}
    cpu=cpus['ordinary']
    if cpu.compiler!=config['compiler']:raise core.Stop('saved compiler mismatch')
    report=dict(status='incomplete',batches=100,identity=h.identity(data,cpu.generated),checks={},gate0={},
                compiler=cpu.compiler,flags={k:v.flags for k,v in cpus.items()})
    try:
        for mode,cpu in cpus.items():
            report['gate0'][mode]=dict(management=management(cpu),transport=compare_small(cpu))
        for b in range(100):
            saved_out=p5b.saved_batch(Path(saved),b,config);modes={}
            for mode,cpu in cpus.items():
                sums={v:core.empty_output(core.run_shapes(20)) for v in h.VARIANTS}
                reference_counts=np.zeros((20,2),np.uint64)
                count_sums={v:np.zeros((20,2),np.uint64) for v in h.VARIANTS}
                for first in range(0,100000,200):
                    ref=cpu.histories(200,'P5j',b*100000+first)
                    reference_counts+=cpu.counts
                    for v in h.VARIANTS:
                        out=cpu.histories(200,v,b*100000+first)
                        eq=h.exact(out,ref)
                        if eq['status']!='pass':
                            bad=next(i for i in range(200) if any(not core.bits(out[k][i:i+1],ref[k][i:i+1]) for k in core.HISTORY_ORDER))
                            core.save_npz(work/f'mismatch_{mode}_{v}_{b}_{first+bad}.npz',
                                {f'{prefix}_{k}':r[k][bad:bad+1] for prefix,r in (('actual',out),('reference',ref)) for k in core.HISTORY_ORDER})
                            raise core.Stop('gate1 history mismatch')
                        count_sums[v]+=cpu.counts
                        accumulate(sums[v],out,first,100000)
                modes[mode]=dict(variants={})
                for v in h.VARIANTS:
                    result=dict(histories=core.verdict(True,fields={k:True for k in core.HISTORY_ORDER}),
                        saved=saved_match(sums[v],saved_out),
                        counters=core.verdict(core.bits(count_sums[v],reference_counts)),log_counts=count_sums[v].tolist())
                    # The historical saved-batch rule is bitwise except brems[1].
                    for key in ('saved','counters'):core.require(result[key],'gate1 '+key)
                    modes[mode]['variants'][v]=result
            report['checks'][str(b)]=modes;core.write_json(work/'gate1.json',report)
        report['status']='pass'
    except Exception as exc:
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'gate1.json',report)
    return report


def saved_match(actual,expected):
    fields={k:core.bits(actual[k],expected[k]) for k in d.OUTPUTS if k!='brems'}
    mask=np.arange(10)!=1
    fields['brems_except_1']=core.bits(actual['brems'][mask].copy(),expected['brems'][mask].copy())
    m=max(0.,float(expected['brems'][0])-1)
    bound=m*2**-52*float(expected['brems'][1])/(1-m*2**-53)
    delta=abs(float(actual['brems'][1]-expected['brems'][1]))
    return core.verdict(all(fields.values()) and delta<=bound,fields=fields,brems_1_difference=delta,brems_1_bound=bound)


def accumulate(total,r,offset,n,absolute=False):
    for i in range(len(r['steps'])):
        row=(offset+i)*100//n
        for k in ('batches','fine_batches'):total[k][row]+=np.abs(r[k][i]) if absolute else r[k][i]
        for k in ('totals','brems'):total[k]+=np.abs(r[k][i]) if absolute else r[k][i]
        for k in ('counts','violations'):total[k]+=r[k][i]
        total['maxima']=np.maximum(total['maxima'],r['maxima'][i]);total['maxdepth'][...]=max(int(total['maxdepth']),int(r['maxdepth'][i]))
        total['step_batches'][row]+=r['steps'][i];total['nspec'].flat[int(r['nspec_index'][i])]+=1;total['nan_count'][...]+=r['nan_count'][i]


def export_sources(work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(work/'p5k_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(core.HERE.iterdir()):
            if path.suffix in ('.py','.h','.cpp','.cu','.ipynb'):z.write(path,Path('prototypes/mv_electron/gpu')/path.name)
    return work/'p5k_sources.zip'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('local','gate1','export'))
    p.add_argument('--work',type=Path,required=True);p.add_argument('--saved',type=Path);a=p.parse_args()
    if a.command=='export':export_sources(a.work)
    elif a.command=='gate1':
        if a.saved is None:p.error('--saved required')
        gate1(a.work,a.saved)
    else:
        from .p5a import load_inputs
        local(a.work,load_inputs(full=True))


if __name__=='__main__':main()
