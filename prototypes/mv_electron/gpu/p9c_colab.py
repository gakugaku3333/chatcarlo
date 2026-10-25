"""P9c single invocation: --mode t0 OR --mode production (gates then run)."""
import argparse
import ctypes as ct
import json
import time
import resource
import threading
from pathlib import Path
import numpy as np
from . import p9c_core as core
from .p9c import DP,IP,bind_grids,field_seed,check_result
from .p9c_stats import Checkpoint
from .p9c_validation import DOORS

BEAMS=('rectangle','prostate')
INITIAL_GATES={'V0','V1a','V1b','V3','G3_grazing','V4_rectangle','V4_prostate','V5_GPU'}

def selected_beams(beams):
    if isinstance(beams,str):beams=beams.split(',')
    beams=tuple(beams)
    if not beams or len(set(beams))!=len(beams) or any(b not in BEAMS for b in beams):
        raise ValueError('beams must be a nonempty subset of rectangle,prostate')
    return beams

def validate_gates(report,mode,required=False):
    if not isinstance(report,dict) or not report:raise ValueError('empty or invalid gates')
    needed=INITIAL_GATES|({'V1c','V2b'} if mode=='production' else set())
    if required and not needed<=report.keys():raise ValueError('missing required gates')
    for gate,value in report.items():
        if not isinstance(value,dict) or (value.get('status')!='pass' and not (
            gate=='V2b' and value.get('status')=='insufficient-statistics' and value.get('fallback')=='V2a')):
            raise AssertionError(gate+' failed; no production')

def saved_gates(folder,record,mode):
    """Read a source-bound gate record, including legacy records with job.json."""
    folder=Path(folder);saved=json.loads((folder/'gates.json').read_text())
    if 'gates' in saved:
        identity=saved;report=saved['gates']
    else:
        identity=json.loads((folder/'job.json').read_text());report=saved
    if identity.get('sources')!=record['sources'] or identity.get('mode')!=mode:
        raise ValueError('gates source/mode mismatch')
    validate_gates(report,mode,required=True)
    return report

def require_t0(folder,record):
    folder=Path(folder);prior=json.loads((folder/'job.json').read_text())
    if prior.get('mode')!='t0' or prior.get('sources')!=record['sources']:
        raise ValueError('matching completed T0 required')
    if prior.get('status')=='complete':return
    reason=prior.get('reason')
    if prior.get('status')!='partial-accepted' or not isinstance(reason,str) or not reason.strip():
        raise ValueError('completed or partial-accepted T0 with nonempty reason required')
    saved_gates(folder,record,'t0')
    if not (folder/'rectangle_summary.json').is_file():raise ValueError('T0 rectangle_summary.json required')

class GPU:
    def __init__(self,cp,materials,**kw):
        from .. import physics
        self.cp=cp;self.g,self.arrays,self.grids=core.geometry(**kw)
        self.values=[cp.zeros(g.shape) for g in self.grids];self.adds=None
        self.device_arrays={k:cp.asarray(v) for k,v in self.arrays.items()}
        data,meta=core.core.pack(materials)
        self.args=(cp.asarray(data),cp.asarray(meta),cp.asarray([physics.M,physics.C,physics.I,physics.RE,physics.NE,physics.ALPHA]))
        self.module=cp.RawModule(code=core.cuda_source(),options=core.core.CUDA_OPTIONS+('-DP8G_STACK_CAP=18','-DP8G2_CHECK=0'),backend='nvrtc')
        self.module.get_function('p9c_init')((1,),(1,),self.args+(np.int32(len(materials)),));self.upload()
    def upload(self):
        cp=self.cp;g=core.Geometry.from_buffer_copy(bytes(self.g))
        for k,t in core.previous.SlabGeometry._fields_[:8]:setattr(g,k,ct.cast(int(self.device_arrays[k].data.ptr),t))
        g.seeds=ct.cast(int(self.device_arrays['seeds'].data.ptr),DP);g.parents=ct.cast(int(self.device_arrays['parents'].data.ptr),IP)
        bind_grids(g,self.grids,self.values,self.adds,device=True)
        self.geometry=cp.asarray(np.frombuffer(bytes(g),np.uint8).copy())
    def clear(self):
        for x in self.values:x.fill(0)
        if self.adds is not None:
            for x in self.adds:x.fill(0)
    def boundary(self,states):
        cp=self.cp;a=cp.asarray(np.ascontiguousarray(states,float).reshape(-1,23));out=cp.zeros((len(a),6))
        self.module.get_function('p9c_boundary_probe')(((len(a)+31)//32,),(32,),(self.geometry,a,np.int32(len(a)),out));return cp.asnumpy(out)
    def score(self,a,b,loss=1.,site=3,repeats=1):
        cp=self.cp;self.module.get_function('p9c_score_probe')((40,),(256,),(self.geometry,cp.asarray(a,dtype=float),cp.asarray(b,dtype=float),np.float64(loss),np.int32(site),np.int32(repeats)))
        return self.values
    def run(self,n,seed,start=0,field=0,maxsteps=10000000,injected=None,regions=False):
        cp=self.cp
        if type(n) is not int or not 0<n<=200000 or not 0<=start<start+n<2**64:raise ValueError('history range')
        self.clear();d=cp.zeros((n,37));i=cp.zeros((n,79),np.int64);ledger=cp.zeros((n,6));sites=cp.zeros((n,6),np.int64);counts=cp.zeros((n,2),np.int64)
        r=cp.zeros((n,3420)) if regions else None;x=None if injected is None else cp.asarray(injected,dtype=float)
        begin,end=cp.cuda.Event(),cp.cuda.Event();begin.record()
        self.module.get_function('p9c_kernel')((40,),(256,),self.args+(self.geometry,np.uint64(field_seed(seed,field)),np.uint64(start),np.int32(n),np.int32(maxsteps),np.uint64(0) if x is None else x,d,i,ledger,sites,counts,np.uint64(0) if r is None else r))
        end.record();end.synchronize();h=core.core.unpack(cp.asnumpy(d),cp.asnumpy(i),0,0,0);ld=cp.asnumpy(ledger);sc=cp.asnumpy(counts);check_result(h,ld,sc)
        if not self.g.beam and np.any(sc[:,1]):raise AssertionError('rectangle body miss: investigate source geometry')
        if abs(float(self.values[1].sum())-h['dep'].sum())>1e-9*h['sources'].sum():raise AssertionError('coarse grid sum / ledger')
        return dict(histories=h,ledger=ld,sites=cp.asnumpy(sites),source_counts=sc,regions=None if r is None else cp.asnumpy(r),grids=self.values,seconds=cp.cuda.get_elapsed_time(begin,end)/1000)

class HostProbe:
    """Small validation adapters copy only probe grids, never production grids."""
    def __init__(self,engine):self.engine=engine;self.grids=engine.grids
    def clear(self):self.engine.clear()
    def boundary(self,x):return self.engine.boundary(x)
    def score(self,*a,**kw):return self.engine.score(*a,**kw)
    @property
    def values(self):return [self.engine.cp.asnumpy(x) for x in self.engine.values]

def json_write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(path).with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def gpu_identical_pair(engine,seed):
    cp=engine.cp;engine.adds=[cp.zeros(g.shape,np.uint64) for g in engine.grids];engine.upload()
    engine.run(100,seed)
    first=[cp.asnumpy(v) for v in engine.values];adds=[cp.asnumpy(v) for v in engine.adds]
    engine.run(100,seed)
    for j,v in enumerate(engine.values):
        second=cp.asnumpy(v);K=adds[j]+cp.asnumpy(engine.adds[j]);bound=K*2.**-52*np.maximum(first[j],second)
        if np.any(abs(first[j]-second)>bound):raise AssertionError('V5 GPU identical pair / atomic bound')
    return dict(status='pass',histories=100,comparison='actual-addition roundoff bound')


def initial_gates(cp,materials,record,bundle):
    from . import p9c_checks as checks
    from .p9c_gates import v1b,v3_acceptance
    tiny=(checks.SMALL,checks.SMALL);kw=dict(densities=record['densities'],grids=tiny)
    e=GPU(cp,materials,**kw,cylinders=((2,2,2,.05,.25,4),))
    report={'V0':checks.v0(HostProbe(e)),'V1a':checks.v1a(HostProbe(e)),'V1b':v1b(e),'V3':v3_acceptance()}
    with np.load(Path(bundle)/'cpu_reference.npz',allow_pickle=False) as f:
        for name,beam,angle in (('rectangle','rectangle',0),('prostate','prostate',45)):
            e=GPU(cp,materials,densities=record['densities'],grids=(checks.SMALL,core.Grid((-20.,)*3,(10.,)*3,(4,)*3)),edges=record['edges'],cum=record['cum'],beam=beam,angle=angle)
            if name=='prostate':report['G3_grazing']=checks.grazing_seed(HostProbe(e))
            actual=e.run(2000,record['seed'],field=core.ANGLES.index(angle))['histories']
            old={k:f[name+'_'+k] for k in actual}
            report['V4_'+name]=core.core.compare_histories(old,actual)
            if name=='rectangle':
                report['V5_GPU']=gpu_identical_pair(e,record['seed'])
    if any(r['status']!='pass' for r in report.values()):raise AssertionError('initial gate failed')
    return report

def selections(grid):
    x,y,z=grid.centers();xx=x[:,None,None];yy=y[None,:,None];zz=z[None,None,:]
    mask=core.water_mask(grid)
    prostate=((xx/2.25)**2+(yy/1.75)**2+(zz/1.5)**2<=1)&mask
    rectum=(xx**2+(zz-3.)**2<=1.5**2)&(abs(yy)<=1.75)&mask
    lines=[grid.index((a,b,0))[:2] for a,b in ((0,-.2),(.5,.5),(-1.,-.9))]
    ss=core.seeds();ids=[]
    for a,b,c in ((0,-.2,0),(.5,.5,-.5),(-1.,-.9,0)):
        matches=[j for j,s in enumerate(ss) if s[:3]==(a,b,c)]
        if len(matches)!=1:raise ValueError('registered line seed absent')
        ids.append(matches[0])
    return mask,prostate,rectum,lines,ids

def extract(array,selection):
    mask,prostate,rectum,lines,_=selection
    arr=np.asarray(array);profiles=np.array([arr[i,j,:] for i,j in lines]);valid=np.array([mask[i,j,:] for i,j in lines])
    return np.array([arr[prostate].mean(),arr[rectum].mean()]),np.where(valid,profiles,0.)

class MemoryMonitor:
    """NVML samples include CUDA context/local memory outside the CuPy pool."""
    class Info(ct.Structure):
        _fields_=[('total',ct.c_ulonglong),('free',ct.c_ulonglong),('used',ct.c_ulonglong)]
    def __init__(self,index):
        self.lib=ct.CDLL('libnvidia-ml.so.1');self.handle=ct.c_void_p();self.peak=0;self.total=0;self.samples=0;self.errors=0;self.done=threading.Event()
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes=[ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo.argtypes=[ct.c_void_p,ct.POINTER(self.Info)]
        if self.lib.nvmlInit_v2()!=0 or self.lib.nvmlDeviceGetHandleByIndex_v2(index,ct.byref(self.handle))!=0:raise RuntimeError('NVML memory monitor unavailable')
    def sample(self):
        info=self.Info()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ct.byref(info))!=0:self.errors+=1;return
        self.samples+=1;self.peak=max(self.peak,int(info.used));self.total=int(info.total)
    def start(self):
        self.sample()
        if not self.samples:
            self.lib.nvmlShutdown();raise RuntimeError('NVML memory sampling failed')
        def watch():
            while not self.done.wait(.05):self.sample()
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
    def stop(self):
        self.done.set();self.thread.join();self.sample();self.lib.nvmlShutdown()
    def report(self):return dict(device_peak_used_bytes=self.peak,device_total_bytes=self.total,samples=self.samples,sample_errors=self.errors,interval_seconds=.05,host_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def run_pairs(cp,materials,record,output,n,grids,seed,beams=BEAMS):
    beams=selected_beams(beams)
    monitor=MemoryMonitor(cp.cuda.runtime.getDevice());monitor.start()
    try:return _run_pairs(cp,materials,record,output,n,grids,seed,beams)
    finally:
        monitor.stop();json_write(Path(output)/'memory.json',monitor.report())


def _run_pairs(cp,materials,record,output,n,grids,seed,beams=BEAMS):
    from .p9c_gates import summarize,source_acceptance
    if type(n) is not int or n<=0 or n%100:raise ValueError('histories per field must be a positive multiple of 100')
    fine,coarse=grids;selection=selections(fine);mask,_,_,lines,ids=selection
    identity=dict(sources=record['sources'],bundles=record['bundles'],n=n,seed=seed,grids=[g.record() for g in grids],angles=list(core.ANGLES),lines_seed_ids=ids)
    shapes={'fine':fine.shape,'coarse':coarse.shape,'aggregate':(2,),'lines':(3,fine.shape[2]),'representative':(3,fine.shape[2]),'shadow':(3,)}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);reports={}
    z=fine.centers()[2];shadow=[]
    for sid,(i,j) in zip(ids,lines):
        end=core.seeds()[sid][2]+.05;shadow.append((z>=end+.1)&(z<end+.5)&mask[i,j,:])
    # One persistent device pair, reused across every door; only finalized batch
    # moments and small measurement records survive a restart.
    engines=[]
    for side in (True,False):engines.append(GPU(cp,materials,densities=record['densities'],edges=record['edges'],cum=record['cum'],grids=grids,with_seeds=side))
    for beam in selected_beams(beams):
        checkpoint=Checkpoint(out/(beam+'.npz'),dict(identity,beam=beam),shapes)
        times_path=out/(beam+'_measurements.json');measurements=json.loads(times_path.read_text()) if times_path.exists() else {}
        for batch in range(len(checkpoint.completed),100):
            door_set=set();representatives=[];nb=n//100
            # These are the two batch grids; run into the same device arrays for
            # each door, then add to the host batch sums, avoiding 7 grid copies.
            sums=[[np.zeros(g.shape) for g in grids] for _ in (0,1)];timings=[];rejected=0;missed=0
            for f,angle in enumerate(core.ANGLES):
                for side,engine in zip((True,False),engines):
                    k=0 if side else 1;engine.g.angle=np.deg2rad(angle);engine.g.beam=int(beam=='prostate');engine.upload()
                    # Chunk a door if §8 chooses >200k histories per batch.
                    rep=np.zeros((3,fine.shape[2]));seconds=0.;door_rejected=0;door_missed=0
                    for start in range(batch*nb,(batch+1)*nb,200000):
                        count=min(200000,(batch+1)*nb-start);result=engine.run(count,seed,start=start,field=f);seconds+=result['seconds']
                        for j in range(2):sums[k][j]+=cp.asnumpy(result['grids'][j])
                        if f==0:
                            arr=cp.asnumpy(result['grids'][0]);rep+=np.array([arr[i,j,:] for i,j in lines])
                        door_rejected+=int(result['source_counts'][:,0].sum());door_missed+=int(result['source_counts'][:,1].sum())
                        if side:
                            rejected+=int(result['source_counts'][:,0].sum());missed+=int(result['source_counts'][:,1].sum())
                    if f==0:representatives.append(np.where(np.array([mask[i,j,:] for i,j in lines]),rep/nb,0.))
                    door_set.add((f,side));timings.append(dict(field=f,angle=angle,with_seeds=side,seconds=seconds,rejected=door_rejected,body_missed=door_missed))
                    print('P9C_HEARTBEAT',beam,batch,f,int(side),flush=True)
            pairs={};agg=[];profiles=[];shadows=[]
            for k in (0,1):
                sums[k]=[a/(7*nb) for a in sums[k]];a,p=extract(sums[k][0],selection);agg.append(a);profiles.append(p)
                shadows.append(np.array([p[j,sm].mean() if np.any(sm) else 0. for j,sm in enumerate(shadow)]))
            for j,key in enumerate(('fine','coarse')):pairs[key]=(sums[0][j],sums[1][j])
            pairs.update(aggregate=tuple(agg),lines=tuple(profiles),representative=tuple(representatives),shadow=tuple(shadows))
            checkpoint.commit(batch,pairs,door_set)
            measurements[str(batch)]=dict(timings=timings,rejected=rejected,body_missed=missed,device_pool_bytes=int(cp.get_default_memory_pool().total_bytes()),cuda_free_total=[int(x) for x in cp.cuda.runtime.memGetInfo()],host_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            json_write(times_path,measurements)
            print('P9C_BATCH',batch,'beam',beam,flush=True)
        reports[beam]=summarize(checkpoint,fine,selection,measurements)
        if beam=='prostate':reports[beam]['V3_CUDA_acceptance']=source_acceptance(measurements,n//100)
        json_write(out/(beam+'_summary.json'),reports[beam])
    return reports

def run(bundle,output,mode,n=None,grids=(core.FINE,core.COARSE),t0=None,seed=None,beams=BEAMS):
    from .p9c_io import load
    from .p9c_gates import production_gates
    if mode not in ('t0','production'):raise ValueError('mode')
    beams=selected_beams(beams)
    if mode=='production' and (n is None or t0 is None):raise ValueError('production requires --histories-per-field and --t0')
    if mode=='production' and (type(n) is not int or n<=0 or n%100):raise ValueError('histories per field must be a positive multiple of 100')
    if seed is not None:field_seed(seed,0)
    if mode=='t0' and (n is not None or grids!=(core.FINE,core.COARSE)):raise ValueError('T0 fixes 100000 histories/field and candidate grids')
    import cupy as cp
    record,materials=load(bundle);seed=record['seed'] if seed is None else seed
    if mode=='production':require_t0(t0,record)
    if (Path(output)/'gates.json').exists():report=saved_gates(output,record,mode)
    else:
        report=initial_gates(cp,materials,record,bundle)
        if mode=='production':report.update(production_gates(cp,materials,record,output))
        validate_gates(report,mode)
        json_write(Path(output)/'gates.json',dict(sources=record['sources'],mode=mode,gates=report))
    results=run_pairs(cp,materials,record,output,100000 if mode=='t0' else n,grids,seed,beams)
    complete=all(results.get(beam,{}).get('batches')==100 for beam in beams)
    job=dict(status='complete' if complete else 'partial',mode=mode,sources=record['sources'],gates=report,beams=list(beams),results=results)
    json_write(Path(output)/'job.json',job)
    if complete:print('P9C_DONE',flush=True)
    return job

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('output',type=Path);p.add_argument('--mode',required=True,choices=('t0','production'));p.add_argument('--histories-per-field',type=int);p.add_argument('--t0',type=Path);p.add_argument('--seed',type=int)
    p.add_argument('--fine-step-cm',type=float,default=.025)
    p.add_argument('--beams',type=selected_beams,default=BEAMS)
    a=p.parse_args();grids=(core.FINE,core.COARSE)
    if a.fine_step_cm!=.025:
        step=a.fine_step_cm;extent=core.FINE.hi-np.array(core.FINE.lo);shape=np.rint(extent/step).astype(int)
        if step<=0 or not np.allclose(shape*step,extent,rtol=0,atol=1e-12):p.error('fine spacing must divide each extent')
        grids=(core.Grid(core.FINE.lo,(step,)*3,tuple(int(x) for x in shape)),core.COARSE)
    run(a.bundle,a.output,a.mode,a.histories_per_field,grids,a.t0,a.seed,a.beams)
if __name__=='__main__':main()
