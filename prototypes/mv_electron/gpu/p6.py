"""Local P6 gates, CPU references, Drive bundle, and deferred production analysis.

Default outputs: ~/.cache/chatcarlo/p6. --work selects a writable local directory.
No command launches EGS5 or Colab. Analysis is an explicit separate command.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes as ct
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from . import p6_core as h, p5b_core as core, p5d_core as d, p5j

DEFAULT_WORK=Path('~/.cache/chatcarlo/p6').expanduser()
SAVED=Path('~/.cache/chatcarlo/p3b/main_proto').expanduser()
OUT=h.ROOT/'docs/validation/mv/p6'
DP=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
IP=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
UP=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
GP=ct.POINTER(h.Geometry)


class CPU:
    def __init__(self,work,data=None,rng='philox',fault=0):
        if rng not in ('mt','philox') or fault not in range(8):raise ValueError('invalid build')
        self.data=data;self.rng=rng;self.fault=fault;self.geometry=h.geometry()[0]
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        exe=shutil.which('clang++')
        if exe is None:raise h.Stop('clang++ unavailable')
        self.compiler=subprocess.check_output([exe,'--version'],text=True).splitlines()[0]
        self.flags=['-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_'+rng.upper(),
                    '-DP5_TRANSPORT_FAULT=0','-DP5_STACK_CAP=64',f'-DP6_FAULT={fault}']
        digest=core.sha(json.dumps(dict(flags=self.flags,compiler=self.compiler,
            sources={n:core.file_sha(h.HERE/n) for n in (*h.j.KERNEL_SOURCES,*h.SOURCES)}),sort_keys=True).encode())
        library=work/('p6_'+digest[:20]+('.dylib' if sys.platform=='darwin' else '.so'))
        if not library.exists():
            build=subprocess.run([exe,*self.flags,str(h.HERE/'p6_cpu.cpp'),'-o',str(library)],capture_output=True,text=True)
            (work/('build_'+digest[:20]+'.log')).write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(library))
        signatures={
            'p6_score':[DP,ct.c_int,ct.c_int,DP,DP,DP],
            'p6_sinks':[GP,DP,IP,ct.c_int,ct.c_int,DP,DP,DP,DP,UP],
            'p6_mt_run':[ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,ct.c_int,
                DP,DP,DP,DP,IP,IP,IP,IP,DP,IP,DP,IP,DP,IP,ct.c_int,DP,ct.c_int,IP,IP,ct.c_void_p],
            'p6_histories':[ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,ct.c_int,ct.c_int]+[
                DP if k not in core.INTEGER else (ct.POINTER(ct.c_int) if k=='maxdepth' else IP) for k in core.HISTORY_ORDER]+[
                DP,IP,DP,DP,ct.c_int,IP],
            'p6_run':[ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,ct.c_int]+[
                DP if k not in core.INTEGER else (ct.POINTER(ct.c_int) if k=='maxdepth' else IP) for k in core.RUN_ORDER]+[DP,IP,ct.c_void_p],
            'p5j_configure':[ct.POINTER(d.Tables),DP,p5j.UP,ct.c_int]}
        for name,args in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=None
        if data is not None:
            core.check_inputs(data);self.tables=d.CPU.make_tables(data);maps=p5j.layout(data)
            self.logs=np.empty(int(maps['lengths'].sum())+3);self.log_counts=np.zeros((20,2),np.uint64)
            self.configure(True)
            self.generated=core.describe(dict(logs=self.logs,**maps))

    def configure(self,initialize=False):
        if self.data is None:raise ValueError('transport tables required')
        self.lib.p5j_configure(ct.byref(self.tables),self.logs,self.log_counts.ctypes.data_as(p5j.UP),int(initialize))

    def score(self,x,nb=72):
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=16 or nb<=0:raise ValueError('invalid scorer rows')
        out=np.zeros((len(x),nb));points=out.copy();values=np.zeros((len(x),4))
        self.lib.p6_score(x,len(x),nb,out,points,values)
        return out,points,values

    def mt(self,n,seed,field=True,regions=True,maxsteps=h.MAXSTEPS,record_capacity=0,source_capacity=0,source_state=None):
        if self.rng!='mt':raise ValueError('MT library required')
        core.check_range(n,0,maxsteps=maxsteps);check_seed(seed,mt=True);self.configure()
        r=h.empty(h.run_shapes(len(self.data['cum'])));depth=np.zeros(1,np.int64)
        if source_state is not None:
            source_state=np.ascontiguousarray(source_state,dtype=np.float64)
            if (field or source_state.shape!=(23,) or not np.all(np.isfinite(source_state))
                    or source_state[19]!=-1 or source_state[6]<=0 or not 0<=source_state[2]<30):
                raise ValueError('fixed source requires a finite 23-field electron state without field')
        records=np.zeros((record_capacity,9));sites=np.zeros(record_capacity,np.int64);site_counts=np.zeros(5,np.int64)
        sources=np.zeros((source_capacity,7));meta=np.zeros(4,np.int64)
        self.lib.p6_mt_run(seed,n,maxsteps,ct.byref(self.geometry),int(field),int(regions),
            r['batches'],r['fine_batches'],r['totals'],r['maxima'],r['violations'],r['counts'],r['nspec'],
            r['step_batches'],r['brems'],depth,r['regions'],r['region_ledger'],records,sites,record_capacity,sources,source_capacity,meta,
            site_counts,None if source_state is None else source_state.ctypes.data_as(ct.c_void_p))
        r['maxdepth']=np.array(depth[0]);h.validate(r,nspec=len(self.data['cum']))
        return r,dict(deposits=records[:min(record_capacity,int(meta[0]))],sites=sites[:min(record_capacity,int(meta[0]))],
                      sources=sources[:min(source_capacity,int(meta[2]))],meta=meta,site_counts=site_counts)

    def histories(self,n,seed,start=0,field=True,regions=True,maxsteps=h.MAXSTEPS,fresh=False,capacity=0):
        if self.rng!='philox':raise ValueError('Philox library required')
        core.check_range(n,start,maxsteps=maxsteps);check_seed(seed);self.configure()
        r=h.empty(h.history_shapes(n));depth=np.zeros(n,np.int32);trace=np.zeros((n,capacity,10));meta=np.zeros((n,2),np.int64)
        args=[depth.ctypes.data_as(ct.POINTER(ct.c_int)) if k=='maxdepth' else r[k] for k in core.HISTORY_ORDER]
        self.lib.p6_histories(seed,start,n,maxsteps,ct.byref(self.geometry),int(field),int(regions),int(fresh),
            *args,r['regions'],r['region_ledger'],r['sources'],trace,capacity,meta)
        r['maxdepth']=depth.astype(np.int64);h.validate(r,True,n,len(self.data['cum']))
        return (r,trace,meta) if capacity else r

    def run(self,n,seed,start=0,field=True,regions=True,maxsteps=h.MAXSTEPS,count_additions=False):
        if self.rng!='philox':raise ValueError('Philox library required')
        core.check_range(n,start,maxsteps=maxsteps);check_seed(seed);self.configure()
        r=h.empty(h.run_shapes(len(self.data['cum'])));depth=np.zeros(1,np.int32)
        args=[depth.ctypes.data_as(ct.POINTER(ct.c_int)) if k=='maxdepth' else r[k] for k in core.RUN_ORDER]
        self.region_additions=np.zeros((100,h.BINS),np.uint64) if count_additions else None
        additions=self.region_additions.ctypes.data_as(ct.c_void_p) if count_additions else None
        self.lib.p6_run(seed,start,n,maxsteps,ct.byref(self.geometry),int(field),int(regions),*args,r['regions'],r['region_ledger'],additions)
        r['maxdepth']=np.array(depth[0],np.int64);h.validate(r,nspec=len(self.data['cum']))
        return r


def check_seed(seed,mt=False):
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**(32 if mt else 64):raise ValueError('invalid seed')


def scorer_vectors(randoms=100000):
    rng=np.random.default_rng(20261018)
    x=np.zeros((randoms,16));x[:,:6]=rng.uniform(-12,12,(randoms,6));x[:,6]=rng.uniform(0,10,randoms)
    x[:,7]=rng.integers(0,3,randoms);x[:,8]=-9.;x[:,9]=.25;x[:,10:13]=-9.;x[:,13:16]=9.
    # Boundary hits, reverse direction, crossings, parallel, coincident, same exterior face.
    endpoints=[([-9,0,0],[9,0,0]),([9,0,0],[-9,0,0]),([0,-9,0],[0,9,0]),
               ([0,0,-12],[0,0,12]),([0,0,0],[0,0,0]),([9,0,0],[9,0,0]),
               ([-9,0,0],[-9,0,0]),([10,0,0],[11,1,1]),([-10,0,0],[-11,1,1]),
               ([0,0,9],[0,0,-9]),([0,0,0],[0,0,9])]
    edges=[]
    for axis in range(3):
        for p0,p1 in endpoints:
            edges.append([*p0,*p1,1.,axis,-9.,.25,-9.,-9.,-9.,9.,9.,9.])
    return np.ascontiguousarray(np.r_[np.array(edges,float),x])


def check_scorer(cpu,randoms=100000):
    from numba import njit
    from .. import scoring_p3b as s
    @njit
    def reference(x,nb):
        bins=np.zeros((len(x),nb));points=bins.copy();values=np.zeros((len(x),4))
        for i in range(len(x)):
            v=x[i];values[i,0],values[i,1]=s.clip(v[:3],v[3:6],v[10:13],v[13:16])
            values[i,2]=s.score_box_bins(bins[i],v[:3],v[3:6],v[6],int(v[7]),v[8],v[9],v[10:13],v[13:16])
            values[i,3]=s.score_box_point(points[i],v[:3],v[6],int(v[7]),v[8],v[9],v[10:13],v[13:16])
        return bins,points,values
    x=scorer_vectors(randoms);actual=cpu.score(x);expected=reference(x,72)
    result=core.verdict(all(core.bits(a,b) for a,b in zip(actual,expected)),randoms=randoms,edges=len(x)-randoms,
                        outputs=['clip','score_box_bins','score_box_point','arrays'])
    core.require(result,'0a scorer');return result


def check_sinks(cpu,work=None):
    nh=321;n=20000;rng=np.random.default_rng(2026101801)
    records=np.zeros((n,9));histories=np.sort(rng.integers(0,nh,n)).astype(np.int64)
    records[:,0]=histories;records[:,1]=rng.integers(0,2,n);records[:,2:8]=rng.uniform([-10,-10,0]*2,[10,10,30]*2,(n,6))
    records[:,8]=rng.uniform(0,2,n);points=records[:,1]==1;records[points,5:8]=records[points,2:5]
    groups=np.zeros((100,h.BINS));local=np.zeros((nh,h.BINS));sg=np.zeros((nh,15));sl=sg.copy()
    additions=np.zeros(groups.shape,np.uint64)
    cpu.lib.p6_sinks(ct.byref(cpu.geometry),records,histories,n,nh,groups,local,sg,sl,additions)
    total=np.zeros_like(groups);absolute=total.copy();rows=np.arange(nh)*100//nh
    np.add.at(total,rows,local);np.add.at(absolute,rows,np.abs(local))
    bound=additions*2**-52*absolute
    result=core.verdict(core.bits(sg,sl) and bool(np.all(np.abs(total-groups)<=bound)),
        region_sums_bitwise=core.bits(sg,sl),histories=nh,deposits=n,bin_exceeding=int(np.count_nonzero(np.abs(total-groups)>bound)))
    result.update(max_abs_difference=float(np.abs(total-groups).max()),max_histories_per_group=int(np.bincount(rows).max()))
    result.update(seed=2026101801,inputs=core.describe(dict(records=records,histories=histories)),
        max_additions_per_bin=int(additions.max()),bound_rule='per-bin deposits in group * 2^-52 * sum_abs')
    if work is not None:core.write_json(Path(work)/'sinks.json',result)
    if work is not None:
        core.save_npz(Path(work)/('sinks_failure.npz' if result['status']=='fail' else 'sinks.npz'),dict(records=records,histories=histories,groups=groups,
            local=local,sums_group=sg,sums_local=sl,total=total,bound=bound,additions=additions))
    core.require(result,'0b sinks');return result


def numba_reference(n,seed,records=False,source_state=None):
    from .. import run_p3s as s, photon_transport_p3b as t, batch_kernels_p3b as b
    from ..batch_kernels_p3u import spectrum
    if source_state is None:
        edges,cum=spectrum(h.PARAMS)
        source=dict(spectrum_edges=edges,spectrum_cum=cum,field_half=5.,ssd=100.)
    else:source=dict(source_state=source_state)
    values=t.run_coupled(10. if source_state is None else float(source_state[6]),n,seed,.05,.025,30.,*s.inputs('elsepa'),annihilate_in_flight=True,
        **source,brems_mode=2,score_kerma=False,score_pair=False,dz=.25,dz_fine=.1,regions=h.PARAMS['regions'],
        record_deposits=2000000 if records else 0,record_sources=n if records else 0,**s.rule_kwargs('own'))
    return b.reduce(values),values


def site_coverage(records):
    return {str(site):int(count) for site,count in enumerate(records['site_counts'],1)}


def check_endpoint(cpu,work=None):
    # Terminal electron: its actual energy runs out before its energy hinge.
    # The terminal flag disables scattering/hard/brems events, as in production.
    source=np.zeros(23);source[:8]=[.25,-.5,10.,0.,0.,1.,.02,.02]
    source[8]=.04;source[18]=1.;source[19]=-1.
    _,values=numba_reference(1,h.SEED,True,source_state=source)
    actual,rec=cpu.mt(1,h.SEED,field=False,source_state=source,record_capacity=16,source_capacity=1)
    deposits=core.bits(rec['deposits'],values[21]);sources=core.bits(rec['sources'],values[23])
    coverage=site_coverage(rec)
    good=(deposits and sources and coverage['5']>0 and not values[22] and not values[24]
          and rec['meta'].tolist()[1:]==[0,1,0] and h.health(actual)['status']=='pass')
    result=core.verdict(bool(good),source_state=source.tolist(),seed=h.SEED,n=1,
        deposits_bitwise=deposits,sources_bitwise=sources,coverage=coverage,cpp_meta=rec['meta'].tolist(),
        numba_deposit_overflow=int(values[22]),numba_source_overflow=int(values[24]),health=h.health(actual))
    if work is not None:
        core.save_npz(Path(work)/'endpoint_addendum2.npz',dict(source_state=source,numba_deposits=values[21],
            cpp_deposits=rec['deposits'],cpp_sites=rec['sites'],cpp_site_counts=rec['site_counts'],
            numba_sources=values[23],cpp_sources=rec['sources'],cpp_meta=rec['meta']))
    return result


def local(work,data):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    report=dict(status='incomplete',identity=h.identity(data),checks={})
    try:
        mt=CPU(work/'build',data,'mt');cpu=CPU(work/'build',data)
        report['checks']['0a']=check_scorer(mt);report['checks']['0b']=check_sinks(mt,work)
        expected,values=numba_reference(2000,h.SEED,True)
        actual,rec=mt.mt(2000,h.SEED,record_capacity=2000000,source_capacity=2000)
        fields=h.exact(h.reduced(actual),expected);core.require(fields,'0c outputs')
        deposits=core.bits(rec['deposits'],values[21]);sources=core.bits(rec['sources'],values[23])
        coverage=site_coverage(rec);endpoint=check_endpoint(mt,work)
        counts_match=all(coverage[str(s)]==int(np.count_nonzero(rec['sites']==s)) for s in range(1,6))
        good=(deposits and sources and not values[22] and not values[24] and rec['meta'].tolist()[1:]==[0,2000,0]
              and all(coverage[str(s)]>0 for s in range(1,5)) and counts_match and endpoint['status']=='pass')
        check=core.verdict(bool(good),deposits_bitwise=deposits,sources_bitwise=sources,coverage=coverage,
                           site_counts_match_records=counts_match,endpoint=endpoint,
                           cpp_meta=rec['meta'].tolist(),numba_deposit_overflow=int(values[22]),numba_source_overflow=int(values[24]),outputs=fields)
        report['checks']['0c']=check;core.require(check,'0c records')
        bad=CPU(work/'build',data,'mt',fault=2);faulty,_=bad.mt(2000,h.SEED)
        check=core.verdict(h.health(actual)['status']=='pass' and np.any(faulty['region_ledger']) and h.health(faulty)['status']=='fail',
                           fault_ledger=faulty['region_ledger'].tolist(),numeric_mismatch=h.exact(h.reduced(actual),expected)['status']=='pass' and not core.bits(actual['regions'],faulty['regions']))
        report['checks']['0d']=check;core.require(check,'0d ledger rejection')
        checks={};evidence=set()
        for steps in (1,2):
            for n in (529,1,529):
                a,tr,meta=cpu.histories(n,h.SEED,start=7000,maxsteps=steps,capacity=32)
                b,rt,rm=cpu.histories(n,h.SEED,start=7000,maxsteps=steps,fresh=True,capacity=32)
                fields=h.exact(a,b);trace=h.k.traces(tr,meta,rt,rm)
                core.require(fields,'0e fields');core.require(trace,'0e trace')
                for v,m in zip(tr,meta):
                    for pair in ((0,3),(1,3),(3,4)):
                        if np.any(np.all(v[:int(m[0]),:2]==pair,axis=1)):evidence.add(pair)
                checks[f'{steps}/{n}/{len(checks)}']=dict(fields=fields,trace=trace)
        core.require(core.verdict({(0,3),(1,3),(3,4)}<=evidence),'0e failure paths')
        report['checks']['0e']=core.verdict(True,checks=checks,evidence=sorted(evidence))
        report.update(status='pass',compiler=mt.compiler,flags=mt.flags,generated=mt.generated)
    except Exception as exc:
        if '0b' not in report['checks'] and (work/'sinks.json').exists():
            report['checks']['0b']=json.loads((work/'sinks.json').read_text())
        report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'local.json',report)
    return report


def verify_saved(saved):
    from .. import batch_kernels_p3b as b, batch_runner_p3u as runner, run_p3a as io
    config=json.loads((Path(saved)/'config.json').read_text())
    expected=io.plain(dict(kernel='p3b_field',params=h.PARAMS,N=46000000,seed=20291762,nbatch=100,
                           source_hashes=runner.source_hashes(),fingerprint=b.fingerprint(h.PARAMS)))
    if config!=expected:raise h.Stop('saved P3b input/code/table/spectrum/environment fingerprint differs')
    return config,runner.batch_seeds(20291762,100)


def gate1(work,data,saved=SAVED):
    from ..batch_runner_p3u import read_batch
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    h.require_report(work/'local.json',data,'local');config,seeds=verify_saved(saved)
    report=dict(status='incomplete',identity=h.identity(data),checks={},saved_config_hash=core.file_sha(Path(saved)/'config.json'))
    try:
        mt=CPU(work/'build',data,'mt');batches={}
        for b in (0,2,99):
            expected=read_batch(Path(saved)/f'batch_{b:03}.npz','p3b_field',460000,seeds[b])
            start=time.perf_counter();actual,rec=mt.mt(460000,seeds[b]);values=h.reduced(actual)
            check=h.exact(values,expected,tuple(values));core.require(check,'1a batch '+str(b))
            batches[str(b)]=dict(check,seed=seeds[b],n=460000,seconds=time.perf_counter()-start,
                                 coverage=site_coverage(rec),saved_sha256=core.file_sha(Path(saved)/f'batch_{b:03}.npz'))
            print('gate1 batch',b,'pass','deposit sites',batches[str(b)]['coverage'],flush=True)
        coverage={str(s):sum(b['coverage'][str(s)] for b in batches.values()) for s in range(1,6)}
        report['checks']['1a']=core.verdict(True,batches=batches,coverage=coverage)
        from .p5k import CPU as P5CPU
        cpu=CPU(work/'build',data);old=P5CPU(work/'p5j_build',data)
        a=cpu.histories(20000,h.SEED,field=False,regions=False);b=old.histories(20000,'P5j',start=0)
        result=h.exact(a,b,core.HISTORY_ORDER);core.require(result,'1b no-field CPU');report['checks']['1b']=result
        expected,_=numba_reference(20000,h.SEED)
        normal,_=mt.mt(20000,h.SEED);check=h.exact(h.reduced(normal),expected);core.require(check,'1c normal')
        faults={}
        for fault in (1,2,3):
            bad=CPU(work/'build',data,'mt',fault=fault);actual,_=bad.mt(20000,h.SEED)
            # Compare numerics directly; ledger rejection is additional evidence, not a comparison substitute.
            values=raw_reduced(actual);result=h.exact(values,expected,tuple(expected))
            faults[str(fault)]=dict(status='pass' if result['status']=='fail' else 'fail',comparison=result)
            core.require(faults[str(fault)],'1c planted fault '+str(fault))
        report['checks']['1c']=core.verdict(True,normal=check,faults=faults,n=20000)
        verify_saved(saved);report['status']='pass'
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'gate1.json',report)
    return report


def raw_reduced(r):
    # Fault-injection comparison only; production saving always uses h.reduced/health.
    return h._reduced(r)


_REF_CPU=None

def scorer_records():
    vectors=scorer_vectors(0)
    records=np.zeros((len(vectors)*2+8,9))
    for i,v in enumerate(vectors):
        for kind in (0,1):
            row=records[2*i+kind];row[0]=2*i+kind;row[1]=kind
            row[2:5]=v[:3];row[5:8]=v[3:6] if kind==0 else v[:3];row[8]=v[6]
    # Signed, asymmetric deposits inside profile layers make axis/sign faults visible.
    for i,(p0,p1) in enumerate([([.3,.6,5.],[.7,.8,5.]),([-1.,.2,5.],[1.,.4,5.]),
                              ([.2,-.8,10.],[.4,.8,10.]),([.8,.4,20.],[.9,.6,20.])]):
        for kind in (0,1):
            j=2*len(vectors)+2*i+kind
            records[j]=[j,kind,*p0,*(p1 if kind==0 else p0),1.]
    return records


def scorer_fixture(cpu):
    records=scorer_records();n=len(records)
    groups=np.zeros((100,h.BINS));local=np.zeros((n,h.BINS));sg=np.zeros((n,15));sl=sg.copy()
    additions=np.zeros(groups.shape,np.uint64)
    cpu.lib.p6_sinks(ct.byref(cpu.geometry),records,np.arange(n,dtype=np.int64),len(records),n,groups,local,sg,sl,additions)
    return dict(records=records,regions=local,sums=sl)


def _reference_init(work,data):
    global _REF_CPU
    _REF_CPU=CPU(work,data)


def _reference_batch(b):
    t=time.perf_counter();r=_REF_CPU.run(100000,h.SEED,start=b*100000)
    core.require(h.health(r),'CPU reference health');return b,r,time.perf_counter()-t


def reference(work,data):
    work=Path(work);h.require_report(work/'local.json',data,'local');h.require_report(work/'gate1.json',data,'gate1')
    ident=h.identity(data);target=work/'reference';target.mkdir(parents=True,exist_ok=True)
    found=h.resume(target/'batches',ident,'gate3')
    cpu=CPU(work/'build',data);path=target/'histories.npz'
    if path.exists():
        histories=core.read_npz(path);meta=json.loads(str(histories.pop('metadata')))
        h.validate(histories,True,20000)
        if meta!=dict(identity=ident,seed=h.SEED,start=0,n=20000,outputs=core.describe(histories)):raise h.Stop('history reference mismatch')
        core.require(h.health(histories),'saved history reference health')
    else:
        histories=cpu.histories(20000,h.SEED);core.require(h.health(histories),'history reference health')
        core.save_npz(path,histories,dict(identity=ident,seed=h.SEED,start=0,n=20000,outputs=core.describe(histories)))
    core.save_npz(target/'scorer.npz',scorer_fixture(cpu))
    # At most two spawn workers; only the parent writes final checkpoints.
    missing=[b for b in range(100) if b not in found]
    if missing:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=_reference_init,
                                 initargs=(work/'build',data)) as pool:
            for b,r,seconds in pool.map(_reference_batch,missing,chunksize=1):
                path=target/'batches'/f'batch_{b:03}.npz';h.save_checkpoint(path,ident,'gate3',b,r);found[b]=path
                print(f'reference {len(found)}/100 ({seconds:.2f} s)',flush=True)
    rows=[h.read_checkpoint(found[b],ident,'gate3',b) for b in range(100)]
    core.save_npz(target/'result.npz',h.stacked(rows,'gate3'),dict(identity=ident,contract=h.contract('gate3')))
    if ident!=h.identity(data):raise h.Stop('reference inputs changed')
    report=dict(status='pass',identity=ident,batches=100,n=10000000,histories=20000,seed=h.SEED,
                files={str(p.relative_to(target)):core.file_sha(p) for p in [target/'histories.npz',target/'scorer.npz',target/'result.npz',*(found[b] for b in range(100))]})
    core.write_json(target/'reference.json',report);return report


def export(work,data,destination=None):
    work=Path(work);h.require_report(work/'local.json',data,'local');h.require_report(work/'gate1.json',data,'gate1')
    ref=work/'reference';report=json.loads((ref/'reference.json').read_text())
    if report.get('status')!='pass' or report.get('identity')!=h.identity(data) or report.get('batches')!=100 or report.get('histories')!=20000:
        raise h.Stop('complete current CPU references required')
    if any(core.file_sha(ref/p)!=digest for p,digest in report['files'].items()):raise h.Stop('reference hashes differ')
    for b in range(100):h.read_checkpoint(ref/'batches'/f'batch_{b:03}.npz',h.identity(data),'gate3',b)
    dest=Path(destination) if destination else work/'export';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    core.save_npz(dest/'p6_inputs.npz',data);files.append(dest/'p6_inputs.npz')
    with zipfile.ZipFile(dest/'p6_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        # Include only source/dependencies, never licensed spectrum values or transport tables.
        for p in sorted(h.HERE.iterdir()):
            if p.suffix in ('.py','.h','.cpp','.cu','.ipynb'):z.write(p,Path('prototypes/mv_electron/gpu')/p.name)
        p=h.ROOT/'prototypes/mv_electron/analysis_p3b3.py';z.write(p,p.relative_to(h.ROOT))
    files.append(dest/'p6_sources.zip')
    with zipfile.ZipFile(dest/'p6_reference.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ref.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(ref))
    files.append(dest/'p6_reference.zip')
    for name in ('local.json','gate1.json'):
        shutil.copyfile(work/name,dest/name);files.append(dest/name)
    shutil.copyfile(h.HERE/'p6_colab.ipynb',dest/'p6_colab.ipynb');files.append(dest/'p6_colab.ipynb')
    # Lock every saved comparison input now; analysis verifies these before computing comparisons.
    comparison_paths=[h.ROOT/'docs/validation/mv/p3b'/p for p in (
        'pilot_lock_p3b3.json','runs/main_proto/result.npz','runs/main_proto/config.json',
        'egs5/main_egs/result.npz','egs5/main_egs/config.json','egs5/main_egs/summary.json')]
    pilot=json.loads(comparison_paths[0].read_text())
    manifest=dict(status='pass',identity=h.identity(data),files={p.name:dict(sha256=core.file_sha(p),bytes=p.stat().st_size) for p in files},
                  reference=report,Dref=pilot['Dref'],comparison_inputs={str(p.relative_to(h.ROOT)):core.file_sha(p) for p in comparison_paths})
    core.write_json(dest/'p6_manifest.json',manifest)
    print(str(dest),flush=True);return manifest


def analyse(work,data,results=None,bundle=None):
    # Deliberately explicit; neither export nor run_all calls this entry point.
    from .. import analysis_p3b3 as stats, run_p3b3 as p3
    from .p6_colab import load_bundle,require_gate2,require_gate3
    work=Path(work);bundle=Path(bundle) if bundle else work/'export';results=Path(results) if results else work/'results'
    _,_,manifest=load_bundle(bundle);ident=h.identity(data)
    require_gate2(json.loads((results/'gate2.json').read_text()),ident)
    require_gate3(json.loads((results/'gate3.json').read_text()),ident)
    hashes=manifest['comparison_inputs']
    if any(core.file_sha(h.ROOT/p)!=s for p,s in hashes.items()):raise h.Stop('saved comparison input hashes changed')
    rows=[h.read_checkpoint(results/'production'/f'batch_{b:03}.npz',ident,'production',b) for b in range(100)]
    path=results/'production/result.npz';gpu=core.read_npz(path);meta=json.loads(str(gpu.pop('metadata')))
    expected=h.stacked(rows,'production')
    if meta!=dict(identity=ident,contract=h.contract('production'),outputs=core.describe(expected)) or h.exact(gpu,expected)['status']!='pass':
        raise h.Stop('production stacked data or metadata mismatch')
    lock=json.loads((h.ROOT/'docs/validation/mv/p3b/pilot_lock_p3b3.json').read_text())
    if lock['Dref']!=manifest['Dref'] or lock['N_main']!=46000000:raise h.Stop('pilot lock changed')
    proto=p3.read_proto('main_proto',46000000,20291762);egs=p3.read_egs('main_egs',46000000,20291763)
    # Saved settings are independently compared against the P3b production contract.
    pc=json.loads((p3.OUT/'runs/main_proto/config.json').read_text())
    ec=json.loads((p3.OUT/'egs5/main_egs/config.json').read_text())
    from ..batch_kernels_p3u import spectrum
    edges,cum=spectrum(h.PARAMS)
    if pc!=p3.proto_config(46000000,20291762) or ec!=p3.egs_config(46000000,20291763,edges,cum):
        raise h.Stop('saved comparison settings mismatch')
    gd=stats.dose(gpu);rd=stats.dose(egs);pd=stats.dose(proto)
    comparisons={'A':stats.compare(gd,rd,lock['Dref']),'B':stats.compare(gd,pd,lock['Dref'])}
    statuses=[x['status'] for c in comparisons.values() for group in c.values() for x in group]
    verdict='統計不足' if '統計不足' in statuses else ('合格' if all(s in ('1%以内','0.1 cm以内') for s in statuses) else '不合格または判定不能')
    result=dict(status=verdict,Dref=lock['Dref'],comparisons=comparisons,
                B_scope='CPU→GPU の移植全体の比較。乱数の変更だけを分離して確かめたとは結論しない。',
                report_only=dict(A=stats.reports(gd,rd,lock['Dref']),B=stats.reports(gd,pd,lock['Dref']),
                                 symmetry={k:stats.symmetry(v['profile']) for k,v in (('GPU',gd),('EGS5',rd),('MT',pd))}),
                comparison_inputs=hashes,production_sha256=core.file_sha(path))
    if any(core.file_sha(h.ROOT/p)!=s for p,s in hashes.items()):raise h.Stop('comparison inputs changed during analysis')
    core.write_json(OUT/'analysis.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('local','gate1','reference','export','analyse'))
    p.add_argument('--work',type=Path,default=DEFAULT_WORK);p.add_argument('--saved',type=Path,default=SAVED)
    p.add_argument('--destination',type=Path);p.add_argument('--results',type=Path);p.add_argument('--bundle',type=Path)
    a=p.parse_args()
    from .p5a import load_inputs
    data=load_inputs(full=True)
    if a.command=='gate1':r=gate1(a.work,data,a.saved)
    elif a.command=='export':r=export(a.work,data,a.destination)
    elif a.command=='analyse':r=analyse(a.work,data,a.results,a.bundle)
    else:r=globals()[a.command](a.work,data)
    print(json.dumps(dict(status=r.get('status'),work=str(a.work)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
