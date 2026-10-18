"""P7 CPU library (C++ MT / Philox) and the local deterministic gates 0 and 1.

Entry point: prototypes/mv_electron/run_p7.py. No command launches EGS5 or Colab here.
"""
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import math
import numpy as np
from . import p7_core as h, p6_core as h6, p5b_core as core, p5d_core as d, p5j
from . import p6

DP=p6.DP;IP=p6.IP;UP=p6.UP;GP=p6.GP


class CPU:
    """C++ build of p7_cpu.cpp. Geometry layout 'p7' (common) or 'p6' (gate 1a replay)."""
    def __init__(self,work,data,rng='philox',fault=0,p7_fault=0,layout='p7'):
        if rng not in ('mt','philox') or fault not in range(8) or p7_fault not in (0,8):raise ValueError('invalid build')
        self.data=data;self.rng=rng;self.fault=fault;self.p7_fault=p7_fault;self.layout=layout
        self.geometry=h.geometry(layout)[0];self.bins=h.BINS
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        exe=shutil.which('clang++')
        if exe is None:raise h.Stop('clang++ unavailable')
        self.compiler=subprocess.check_output([exe,'--version'],text=True).splitlines()[0]
        self.flags=['-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_'+rng.upper(),
                    '-DP5_TRANSPORT_FAULT=0','-DP5_STACK_CAP=64',f'-DP6_FAULT={fault}',f'-DP7_FAULT={p7_fault}']
        digest=core.sha(json.dumps(dict(flags=self.flags,compiler=self.compiler,
            sources={n:core.file_sha(h.HERE/n) for n in (*h.j.KERNEL_SOURCES,*h.SOURCES)}),sort_keys=True).encode())
        library=work/('p7_'+digest[:20]+('.dylib' if sys.platform=='darwin' else '.so'))
        if not library.exists():
            build=subprocess.run([exe,*self.flags,str(h.HERE/'p7_cpu.cpp'),'-o',str(library)],capture_output=True,text=True)
            (work/('build_'+digest[:20]+'.log')).write_text(build.stdout+build.stderr);build.check_returncode()
        self.lib=ct.CDLL(str(library))
        D=ct.c_double
        signatures={
            'p7_score':[DP,ct.c_int,ct.c_int,DP,DP,DP],
            'p7_sinks':[GP,DP,IP,ct.c_int,ct.c_int,DP,DP,DP,DP,UP],
            'p7_mt_run':[ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,D,D,ct.c_int,
                DP,DP,DP,DP,IP,IP,IP,IP,DP,IP,DP,IP,DP,IP,ct.c_int,DP,ct.c_int,IP,IP,ct.c_void_p],
            'p7_histories':[ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,D,D,ct.c_int,ct.c_int]+[
                DP if x not in core.INTEGER else (ct.POINTER(ct.c_int) if x=='maxdepth' else IP) for x in core.HISTORY_ORDER]+[
                DP,IP,DP,DP,ct.c_int,IP],
            'p7_run':[ct.c_uint64,ct.c_uint64,ct.c_int,ct.c_int,GP,ct.c_int,D,D,ct.c_int]+[
                DP if x not in core.INTEGER else (ct.POINTER(ct.c_int) if x=='maxdepth' else IP) for x in core.RUN_ORDER]+[DP,IP,ct.c_void_p],
            'p5j_configure':[ct.POINTER(d.Tables),DP,p5j.UP,ct.c_int]}
        for name,args in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=None
        core.check_inputs(data);self.tables=d.CPU.make_tables(data);maps=p5j.layout(data)
        self.logs=np.empty(int(maps['lengths'].sum())+3);self.log_counts=np.zeros((20,2),np.uint64)
        self.configure(True);self.generated=core.describe(dict(logs=self.logs,**maps))
        self.nspec=len(data['cum'])

    def configure(self,initialize=False):
        self.lib.p5j_configure(ct.byref(self.tables),self.logs,self.log_counts.ctypes.data_as(p5j.UP),int(initialize))

    def score(self,x,nb):
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=16 or nb<=0:raise ValueError('invalid scorer rows')
        out=np.zeros((len(x),nb));points=out.copy();values=np.zeros((len(x),4))
        self.lib.p7_score(x,len(x),nb,out,points,values)
        return out,points,values

    def mt(self,n,seed,field_half,ssd=100.,field=True,regions=True,maxsteps=h.MAXSTEPS,record_capacity=0,source_capacity=0,source_state=None):
        if self.rng!='mt':raise ValueError('MT library required')
        core.check_range(n,0,maxsteps=maxsteps);p6.check_seed(seed,mt=True);self.configure()
        r=h.empty(h.run_shapes(self.nspec));depth=np.zeros(1,np.int64)
        if source_state is not None:
            source_state=np.ascontiguousarray(source_state,dtype=np.float64)
            if field or source_state.shape!=(23,):raise ValueError('fixed source requires 23-field state without field')
        records=np.zeros((record_capacity,9));sites=np.zeros(record_capacity,np.int64);site_counts=np.zeros(5,np.int64)
        sources=np.zeros((source_capacity,7));meta=np.zeros(4,np.int64)
        self.lib.p7_mt_run(seed,n,maxsteps,ct.byref(self.geometry),int(field),float(field_half),float(ssd),int(regions),
            r['batches'],r['fine_batches'],r['totals'],r['maxima'],r['violations'],r['counts'],r['nspec'],
            r['step_batches'],r['brems'],depth,r['regions'],r['region_ledger'],records,sites,record_capacity,sources,source_capacity,meta,
            site_counts,None if source_state is None else source_state.ctypes.data_as(ct.c_void_p))
        r['maxdepth']=np.array(depth[0]);h.validate(r,nspec=self.nspec)
        m=int(meta[0]);s=int(meta[2])
        return r,dict(deposits=records[:min(record_capacity,m)],sites=sites[:min(record_capacity,m)],
                      sources=sources[:min(source_capacity,s)],meta=meta,site_counts=site_counts)

    def histories(self,n,seed,field_half,ssd=100.,start=0,field=True,regions=True,maxsteps=h.MAXSTEPS,fresh=False,capacity=0):
        if self.rng!='philox':raise ValueError('Philox library required')
        core.check_range(n,start,maxsteps=maxsteps);p6.check_seed(seed);self.configure()
        r=h.empty(h.history_shapes(n));depth=np.zeros(n,np.int32);trace=np.zeros((n,capacity,10));meta=np.zeros((n,2),np.int64)
        args=[depth.ctypes.data_as(ct.POINTER(ct.c_int)) if x=='maxdepth' else r[x] for x in core.HISTORY_ORDER]
        self.lib.p7_histories(seed,start,n,maxsteps,ct.byref(self.geometry),int(field),float(field_half),float(ssd),int(regions),int(fresh),
            *args,r['regions'],r['region_ledger'],r['sources'],trace,capacity,meta)
        r['maxdepth']=depth.astype(np.int64);h.validate(r,True,n,self.nspec)
        return (r,trace,meta) if capacity else r

    def run(self,n,seed,field_half,ssd=100.,start=0,field=True,regions=True,maxsteps=h.MAXSTEPS,count_additions=False):
        if self.rng!='philox':raise ValueError('Philox library required')
        core.check_range(n,start,maxsteps=maxsteps);p6.check_seed(seed);self.configure()
        r=h.empty(h.run_shapes(self.nspec));depth=np.zeros(1,np.int32)
        args=[depth.ctypes.data_as(ct.POINTER(ct.c_int)) if x=='maxdepth' else r[x] for x in core.RUN_ORDER]
        self.region_additions=np.zeros((100,h.BINS),np.uint64) if count_additions else None
        additions=self.region_additions.ctypes.data_as(ct.c_void_p) if count_additions else None
        self.lib.p7_run(seed,start,n,maxsteps,ct.byref(self.geometry),int(field),float(field_half),float(ssd),int(regions),
            *args,r['regions'],r['region_ledger'],additions)
        r['maxdepth']=np.array(depth[0],np.int64);h.validate(r,nspec=self.nspec)
        return r


# ---------------- references ----------------

def numba_reference(n,seed,name,layout='p7',records=False,source_state=None):
    """P3b Numba kernel with this condition's field, spectrum and the given layout."""
    from .. import run_p3s as s, photon_transport_p3b as t, batch_kernels_p3b as b
    c=h.condition(name);L=h.layout(layout)
    if source_state is None:
        edges,_,cum=h.spectrum(c['spectrum'])
        source=dict(spectrum_edges=edges,spectrum_cum=cum,field_half=c['field_half'],ssd=c['ssd'])
    else:source=dict(source_state=source_state)
    values=t.run_coupled(10. if source_state is None else float(source_state[6]),n,seed,.05,.025,30.,*s.inputs('elsepa'),
        annihilate_in_flight=True,**source,brems_mode=2,score_kerma=False,score_pair=False,dz=.25,dz_fine=.1,
        regions=L['regions'],record_deposits=2000000 if records else 0,record_sources=n if records else 0,**s.rule_kwargs('own'))
    return b.reduce(values),values


def scorer_vectors(randoms=100000,extent=20.):
    """P6's 0a inputs on the +-20 cm box, plus outside-+-9 cm and outermost-bin edges."""
    rng=np.random.default_rng(20261018)
    x=np.zeros((randoms,16));x[:,:6]=rng.uniform(-24,24,(randoms,6));x[:,6]=rng.uniform(0,10,randoms)
    x[:,7]=rng.integers(0,3,randoms);x[:,8]=-extent;x[:,9]=.25;x[:,10:13]=-extent;x[:,13:16]=extent
    e=extent
    endpoints=[([-e,0,0],[e,0,0]),([e,0,0],[-e,0,0]),([0,-e,0],[0,e,0]),
               ([0,0,-24],[0,0,24]),([0,0,0],[0,0,0]),([e,0,0],[e,0,0]),
               ([-e,0,0],[-e,0,0]),([21,0,0],[22,1,1]),([-21,0,0],[-22,1,1]),
               ([0,0,e],[0,0,-e]),([0,0,0],[0,0,e]),
               # outside +-9 cm and the outermost bins (19.75..20, exactly on 20)
               ([9.5,0,0],[15.,0,0]),([-15.,0,0],[-9.5,0,0]),([19.8,0,0],[19.9,0,0]),
               ([19.75,0,0],[20.,0,0]),([-20.,0,0],[-19.75,0,0]),([19.9,.1,.1],[20.5,.2,.2])]
    edges=[]
    for axis in range(3):
        for p0,p1 in endpoints:edges.append([*p0,*p1,1.,axis,-e,.25,-e,-e,-e,e,e,e])
    return np.ascontiguousarray(np.r_[np.array(edges,float),x]),len(endpoints)*3


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
    x,edges=scorer_vectors(randoms);actual=cpu.score(x,160);expected=reference(x,160)
    outer=bool(np.any(actual[0][:edges,140:]!=0) and np.any(actual[0][:edges,-1]!=0))
    result=core.verdict(all(core.bits(a,b) for a,b in zip(actual,expected)) and outer,randoms=randoms,edges=edges,
                        outer_bins_hit=outer,outputs=['clip','score_box_bins','score_box_point','arrays'])
    core.require(result,'0a scorer');return result


def check_sinks(cpu,work=None):
    nh=321;n=20000;rng=np.random.default_rng(2026101801)
    records=np.zeros((n,9));histories=np.sort(rng.integers(0,nh,n)).astype(np.int64)
    records[:,0]=histories;records[:,1]=rng.integers(0,2,n);records[:,2:8]=rng.uniform([-21,-21,0]*2,[21,21,30]*2,(n,6))
    records[:,8]=rng.uniform(0,2,n);points=records[:,1]==1;records[points,5:8]=records[points,2:5]
    groups=np.zeros((100,h.BINS));local=np.zeros((nh,h.BINS));sg=np.zeros((nh,15));sl=sg.copy()
    additions=np.zeros(groups.shape,np.uint64)
    cpu.lib.p7_sinks(ct.byref(cpu.geometry),records,histories,n,nh,groups,local,sg,sl,additions)
    total=np.zeros_like(groups);absolute=total.copy();rows=np.arange(nh)*100//nh
    np.add.at(total,rows,local);np.add.at(absolute,rows,np.abs(local))
    bound=additions*2**-52*absolute;over=int(np.count_nonzero(np.abs(total-groups)>bound))
    outer=bool(np.any(local[:,h.layout('p7')['offsets'][3]+140:h.layout('p7')['offsets'][3]+160]!=0))
    result=core.verdict(core.bits(sg,sl) and over==0 and outer,region_sums_bitwise=core.bits(sg,sl),histories=nh,deposits=n,
                        bin_exceeding=over,outer_bins_hit=outer,bound_rule='per-bin deposits in group * 2^-52 * sum_abs')
    core.require(result,'0b sinks');return result


def site_coverage(rec):return {str(s):int(c) for s,c in enumerate(rec['site_counts'],1)}


def check_endpoint(mt):
    """Addendum 2 of P6: site 5 checked with a fixed terminal-electron state."""
    source=np.zeros(23);source[:8]=[.25,-.5,10.,0.,0.,1.,.02,.02];source[8]=.04;source[18]=1.;source[19]=-1.
    _,values=numba_reference(1,h.SEED,'B',records=True,source_state=source)
    actual,rec=mt.mt(1,h.SEED,10.,field=False,source_state=source,record_capacity=16,source_capacity=1)
    good=(core.bits(rec['deposits'],values[21]) and core.bits(rec['sources'],values[23]) and site_coverage(rec)['5']>0
          and rec['meta'].tolist()[1:]==[0,1,0] and h.health(actual)['status']=='pass')
    return core.verdict(bool(good),coverage=site_coverage(rec))


def check_records(mt):
    """0c: condition B, C++ MT vs Numba deposit and source records, 2,000 histories."""
    c=h.condition('B');expected,values=numba_reference(2000,h.SEED,'B',records=True)
    actual,rec=mt.mt(2000,h.SEED,c['field_half'],record_capacity=2000000,source_capacity=2000)
    fields=h.exact(h.reduced(actual),expected);core.require(fields,'0c outputs')
    deposits=core.bits(rec['deposits'],values[21]);sources=core.bits(rec['sources'],values[23])
    coverage=site_coverage(rec);endpoint=check_endpoint(mt)
    good=(deposits and sources and not values[22] and not values[24] and rec['meta'].tolist()[1:]==[0,2000,0]
          and all(coverage[str(s)]>0 for s in range(1,5)) and endpoint['status']=='pass')
    result=core.verdict(bool(good),deposits_bitwise=deposits,sources_bitwise=sources,coverage=coverage,endpoint=endpoint,
                        cpp_meta=rec['meta'].tolist(),outputs=fields)
    core.require(result,'0c records');return result,actual,expected


def check_spectrum_reader(tmp,mt):
    """0f: synthetic 3-bin file read by the P7 reader vs hand-computed values; bad files rejected;
    real files' SHA/bins; C++ source energies lie inside their bin (lower edge + u x width, 0<=u<1)."""
    tmp=Path(tmp);tmp.mkdir(parents=True,exist_ok=True);path=tmp/'synthetic.spectrum'
    path.write_text('synthetic\n 3, 0.00, 1\n 0.5, 2\n 1.5, 1\n 3.5, 1\n')
    # widths 0.5/1/2, values 2/1/1 -> value x width 1/1/2 -> probabilities 1/4,1/4,1/2; cumulative .25/.5/1
    edges,prob,cum=h.read_spectrum(path,core.file_sha(path),3,None)
    synthetic=bool(np.array_equal(edges,[0.,.5,1.5,3.5]) and np.array_equal(prob,[.25,.25,.5]) and np.array_equal(cum,[.25,.5,1.]))
    rejected={}
    for label,text in (('mode','x\n 3, 0.00, 2\n 0.5, 2\n 1.5, 1\n 3.5, 1\n'),('rows','x\n 3, 0.00, 1\n 0.5, 2\n 1.5, 1\n'),
                       ('lower','x\n 3, 0.10, 1\n 0.5, 2\n 1.5, 1\n 3.5, 1\n'),('zero','x\n 3, 0.00, 1\n 0.5, 0\n 1.5, 1\n 3.5, 1\n')):
        bad=tmp/(label+'.spectrum');bad.write_text(text)
        try:h.read_spectrum(bad,core.file_sha(bad),3,None);rejected[label]=False
        except h.Stop:rejected[label]=True
    try:h.read_spectrum(h.SPECTRA['mohan4']['path'],'0'*64,16,.25);rejected['sha']=False
    except h.Stop:rejected['sha']=True
    real={name:len(h.spectrum(name)[2]) for name in h.SPECTRA}
    # Source energies from the C++ MT build with the 4 MV inputs (mt carries condition C inputs).
    c=h.condition('C');actual,rec=mt.mt(2000,h.SEED,c['field_half'],source_capacity=2000)
    e,_,cm=h.spectrum('mohan4');E=rec['sources'][:,6];j=np.searchsorted(e,E,side='right')-1
    u=(E-e[j])/(e[j+1]-e[j]);inside=bool(np.all((j>=0)&(j<16)&(u>=0)&(u<1)))
    counts=np.bincount(j,minlength=16);histogram=bool(np.array_equal(counts,actual['nspec'].reshape(16,10).sum(1)))
    good=synthetic and all(rejected.values()) and real==dict(mohan10=20,mohan4=16) and inside and histogram
    result=core.verdict(bool(good),synthetic=dict(prob=prob.tolist(),cum=cum.tolist()),rejected=rejected,real=real,
                        source_inside_bin=inside,source_histogram_matches_nspec=histogram,top=float(E.max()))
    core.require(result,'0f spectrum');return result


# ---------------- 0h / 0i / 0j : pure-Python contracts (also covered by tests) ----------------

def check_analysis_ranges():
    """0h: synthetic profiles with hand-known central/outside membership."""
    from .. import analysis_p7 as A
    out={}
    for name in 'ABC':
        c=h.condition(name);s=A.spec(c['field_half']);nb=s['radius_bins']
        # Plateau 1 inside the central range, 0.6 between it and the edge, 0 beyond; +x only marker in outside bins.
        prof=np.zeros((100,2,3,2*nb));fine=np.zeros((100,2,3,4*nb))
        for j in range(3):
            m=s['central'][j];edge=int(math.ceil(A.half_width(c['field_half'],h.DEPTHS[j])/.25))
            radial=np.zeros(nb);radial[:m]=1.;radial[m:edge]=.6
            for axis in (0,1):
                prof[:,axis,j,nb:]=radial;prof[:,axis,j,:nb]=radial[::-1]
                rf=np.repeat(radial,2);fine[:,axis,j,2*nb:]=rf;fine[:,axis,j,:2*nb]=rf[::-1]
        d=A.directions(prof)
        means=[float(A.fold(prof)[:,j,:s['central'][j]].mean()) for j in range(3)]
        cross=[A.crossing(A.fold(prof).mean(0)[j],.25,s['central'][j]) for j in range(3)]
        crossf=[A.crossing(A.fold(fine).mean(0)[j],.125,s['central'][j]) for j in range(3)]
        expected=[(int(math.ceil(A.half_width(c['field_half'],h.DEPTHS[j])/.25))-.5)*.25+.25*(.6-.5)/.6 for j in range(3)]
        ok=means==[1.,1.,1.] and np.allclose(cross,expected,atol=1e-12) and np.allclose(crossf,
            [ (2*int(math.ceil(A.half_width(c['field_half'],h.DEPTHS[j])/.25))-.5)*.125+.125*(.6-.5)/.6 for j in range(3)],atol=1e-12)
        # +x vs -x: a marker in +x only must appear in direction 0 and nowhere else.
        mark=prof.copy();mark[:,0,0,nb+s['central'][0]-1]+=1.
        dm=A.directions(mark);sign=bool(dm[:,0,0,s['central'][0]-1].mean()==2. and dm[:,1,0,s['central'][0]-1].mean()==1.
                                        and dm[:,2,0,s['central'][0]-1].mean()==1.)
        # outside: last bin of B at 20 cm included.
        lo,hi=s['outside'][2];last=hi==nb if name=='B' else True
        # bootstrap: a crossing in the mean but not in some samples must stop.
        stop=False
        if name=='A':
            batches=np.ones((100,nb));batches[:50,s['central'][0]:]=.3;batches[50:,s['central'][0]:]=.7
            try:A.stop_on_missing(dict(J3=[dict(proto=A.bootstrap_position(batches,.25,s['central'][0]),egs=dict(position=1.,missing=0))]))
            except ValueError:stop=True
        else:stop=True
        out[name]=dict(central=s['central'],outside=s['outside'],ok=bool(ok),sign=sign,last_outside_bin=last,stop_on_missing=stop)
    result=core.verdict(all(v['ok'] and v['sign'] and v['last_outside_bin'] and v['stop_on_missing'] for v in out.values()),conditions=out)
    core.require(result,'0h analysis ranges');return result


def check_sizes():
    """0i: 1.6e8 < N <= 2.4e8 allowed, > 2.4e8 stops; 2e6 rounding boundaries."""
    cases={}
    cases['below_old_cap']=h.main_size(1.6e8/1.5)==160000000
    cases['between']=h.main_size(2.0e8/1.5)==200000000
    cases['at_cap']=h.main_size(2.4e8/1.5)==240000000
    try:h.main_size(2.4e8/1.5+1);cases['above_cap_stops']=False
    except h.Stop:cases['above_cap_stops']=True
    cases['exact_quantum']=h.round_up(4000000)==4000000
    cases['one_over']=h.round_up(4000001)==6000000
    result=core.verdict(all(cases.values()),cases=cases);core.require(result,'0i sizes');return result


def check_contracts(work,data):
    """0j: swaps and edits are rejected; C's 16-bin data re-read; resume skips finished batches."""
    work=Path(work)/'contracts';work.mkdir(parents=True,exist_ok=True);cases={}
    ids={n:h.identity(h.condition_inputs(data,n),n) for n in 'ABC'}
    def fake(name,n):
        r=h.empty(h.run_shapes(h.condition(name)['nspec']));r['nspec'][0,0]=n;r['totals'][0]=1.;return r
    for name in 'ABC':
        d=work/name;d.mkdir(exist_ok=True)
        r=fake(name,40000);path=d/'batch_000.npz'
        if not path.exists():h.save_checkpoint(path,ids[name],'gate3',name,0,r)
        cases[name+'_reread']=bool(h.read_checkpoint(path,ids[name],'gate3',name,0)['nspec'].shape==(h.condition(name)['nspec'],10))
    for label,args in (('A_as_B',(ids['B'],'gate3','B',0)),('gate3_as_production',(ids['A'],'production','A',0,4000000)),
                       ('wrong_batch',(ids['A'],'gate3','A',1))):
        try:h.read_checkpoint(work/'A'/'batch_000.npz',*args);cases[label]=False
        except (h.Stop,ValueError):cases[label]=True
    edited=dict(ids['A']);edited['params']=dict(edited['params'],field_half=5.)
    try:h.read_checkpoint(work/'A'/'batch_000.npz',edited,'gate3','A',0);cases['edited_identity']=False
    except h.Stop:cases['edited_identity']=True
    try:h.contract('production','A',4000001);cases['bad_N']=False
    except h.Stop:cases['bad_N']=True
    try:h.save_checkpoint(work/'A'/'batch_000.npz',ids['A'],'gate3','A',0,fake('A',40000));cases['no_overwrite']=False
    except h.Stop:cases['no_overwrite']=True
    found=h.resume(work/'A',ids['A'],'gate3','A');cases['resume_finds_done']=sorted(found)==[0]
    try:h.resume(work/'A',ids['B'],'gate3','B');cases['resume_other_condition']=False
    except h.Stop:cases['resume_other_condition']=True
    result=core.verdict(all(cases.values()),cases=cases);core.require(result,'0j contracts');return result


# ---------------- gate 0 / gate 1 ----------------

def local(work,data):
    """Gate 0 (deterministic, local). `data` is the frozen P5 input set (10 MV spectrum)."""
    import math  # noqa: F401  (used by check_analysis_ranges)
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    dB=h.condition_inputs(data,'B');dC=h.condition_inputs(data,'C')
    report=dict(status='incomplete',identity={n:h.identity(h.condition_inputs(data,n),n) for n in 'ABC'},checks={})
    try:
        mt=CPU(work/'build',dB,'mt');cpu=CPU(work/'build',dB)
        report['checks']['0a']=check_scorer(mt);report['checks']['0b']=check_sinks(mt)
        c0,actual,expected=check_records(mt);report['checks']['0c']=c0
        bad=CPU(work/'build',dB,'mt',fault=2);faulty,_=bad.mt(2000,h.SEED,10.)
        check=core.verdict(h.health(actual)['status']=='pass' and bool(np.any(faulty['region_ledger'])) and h.health(faulty)['status']=='fail'
                           and not core.bits(actual['regions'],faulty['regions']),fault_ledger=faulty['region_ledger'].tolist())
        report['checks']['0d']=check;core.require(check,'0d ledger rejection')
        checks={};evidence=set()
        for steps in (1,2):
            for n in (529,1,529):
                a,tr,meta=cpu.histories(n,h.SEED,10.,start=7000,maxsteps=steps,capacity=32)
                b,rt,rm=cpu.histories(n,h.SEED,10.,start=7000,maxsteps=steps,fresh=True,capacity=32)
                fields=h.exact(a,b);trace=h6.k.traces(tr,meta,rt,rm)
                core.require(fields,'0e fields');core.require(trace,'0e trace')
                for v,m in zip(tr,meta):
                    for pair in ((0,3),(1,3),(3,4)):
                        if np.any(np.all(v[:int(m[0]),:2]==pair,axis=1)):evidence.add(pair)
                checks[f'{steps}/{n}/{len(checks)}']=dict(fields=fields['status'],trace=trace['status'])
        core.require(core.verdict({(0,3),(1,3),(3,4)}<=evidence),'0e failure paths')
        report['checks']['0e']=core.verdict(True,checks=checks,evidence=sorted(evidence))
        report['checks']['0f']=check_spectrum_reader(work/'spectra',CPU(work/'build',dC,'mt'))
        report['checks']['0g']=check_analysis_identity()
        report['checks']['0h']=check_analysis_ranges()
        report['checks']['0i']=check_sizes()
        report['checks']['0j']=check_contracts(work,data)
        report['checks']['0k']=check_kernel_diff()
        report.update(status='pass',compiler=mt.compiler,flags=mt.flags)
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'local.json',report)
    return report


def check_analysis_identity():
    """0g: P6 layout + 16 central bins + outside 28:36 on the saved P3b-3 data equals analysis_p3b3."""
    from .. import analysis_p3b3 as A, analysis_p7 as B, run_p3b3 as p3
    proto=p3.read_proto('main_proto',46000000,20291762);egs=p3.read_egs('main_egs',46000000,20291763)
    D=json.loads((p3.OUT/'pilot_lock_p3b3.json').read_text())['Dref'];s=B.spec(legacy=True)
    l,r=A.dose(proto),A.dose(egs);l2,r2=B.dose(proto,s),B.dose(egs,s)
    old=json.dumps(dict(c=A.compare(l,r,D),s=A.symmetry(l['profile']),r=A.reports(l,r,D)),sort_keys=True)
    c=B.compare(l2,r2,D,s);rep=B.reports(l2,r2,D,s)
    for x in c['J2']:x.pop('central_cm')
    for x in rep['profiles']:x.pop('outside_cm')
    new=json.dumps(dict(c=c,s=B.symmetry(l2['profile'],s),r=rep),sort_keys=True)
    result=core.verdict(old==new,characters=len(old));core.require(result,'0g analysis identity');return result


# Plan section 7: the only permitted edits after the P6->P7 identifier renames.
PERMITTED={
 'p7_kernel.h':[
  ("                  bool field,P6Records *r,P5jTrace *tr=0) {","                  bool field,double field_half,double ssd,P6Records *r,P5jTrace *tr=0) {"),
  ("        double x=(2*rng_uniform(s)-1)*5.,y=(2*rng_uniform(s)-1)*5.;","#if P7_FAULT == 8\n        field_half=5.;\n#endif\n        double x=(2*rng_uniform(s)-1)*field_half,y=(2*rng_uniform(s)-1)*field_half;"),
  ("        double norm=sqrt(x*x+y*y+100.*100.);\n        a[0]=x;a[1]=y;a[3]=x/norm;a[4]=y/norm;a[5]=100./norm;","        double norm=sqrt(x*x+y*y+ssd*ssd);\n        a[0]=x;a[1]=y;a[3]=x/norm;a[4]=y/norm;a[5]=ssd/norm;"),
  ("uint64_t seed,uint64_t start,bool field,double *trace=0,int capacity=0) {","uint64_t seed,uint64_t start,bool field,double field_half,double ssd,double *trace=0,int capacity=0) {"),
  ("p7_init(&s->q,t,&s->rng,0,&o,field,&records,trace?&tr:0);","p7_init(&s->q,t,&s->rng,0,&o,field,field_half,ssd,&records,trace?&tr:0);")],
 'p7_cpu.cpp':[
  ("const P6Geometry *g,int field,int enabled,","const P6Geometry *g,int field,double field_half,double ssd,int enabled,"),
  ("p7_init(&q,&configured,&rng,b,&o,field,&rec);","p7_init(&q,&configured,&rng,b,&o,field,field_half,ssd,&rec);"),
  ("seed,start,field,trace,capacity);","seed,start,field,field_half,ssd,trace,capacity);"),
  ("seed,start,field);","seed,start,field,field_half,ssd);")],
 'p7_gpu.cu':[
  (" int field,int enabled,int *next,"," int field,double field_half,double ssd,int enabled,int *next,"),
  ("seed,start,field,trace,capacity);","seed,start,field,field_half,ssd,trace,capacity);")]}


def _rename(s):
    import re
    s=(s.replace('P6_KERNEL_H','P7_KERNEL_H').replace('"p6_field.h"','"p7_field.h"').replace('"p6_kernel.h"','"p7_kernel.h"')
       .replace('P6State','P7State').replace('P6Slot','P7Slot').replace('P6_BINS','P7_BINS').replace('P6_RUN_','P7_RUN_'))
    return re.sub(r'\bp6_(init|particle|charged|transition|output|trace|initialize|work|finalize|mt_run|histories|run|emit|kernel|score_kernel|sinks|score)\b',r'p7_\1',s)


def check_kernel_diff():
    """0k: P6 text + identifier renames + the PERMITTED edits must equal the P7 text exactly."""
    out={}
    for new in PERMITTED:
        old=_rename((h.HERE/new.replace('p7_','p6_')).read_text())
        for a,b in PERMITTED[new]:old=old.replace(a,b)
        out[new]=old==(h.HERE/new).read_text()
    result=core.verdict(all(out.values()),files=out);core.require(result,'0k kernel diff');return result


def gate1(work,data,saved=p6.SAVED):
    """Gate 1 (local): 1a saved P3b-3 batches 0/2/99 with the P6 layout; 1b each condition vs Numba;
    1c planted faults; 1d no-field regression vs P5j CPU."""
    from ..batch_runner_p3u import read_batch
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    loc=json.loads((work/'local.json').read_text())
    idents={n:h.identity(h.condition_inputs(data,n),n) for n in 'ABC'}
    if loc.get('status')!='pass' or loc.get('identity')!=idents or set(loc.get('checks',{}))!={'0'+x for x in 'abcdefghijk'}:
        raise h.Stop('current complete gate 0 required')
    config,seeds=p6.verify_saved(saved)
    report=dict(status='incomplete',identity=idents,checks={},saved_config_hash=core.file_sha(Path(saved)/'config.json'))
    try:
        old=CPU(work/'build',data,'mt',layout='p6');batches={}
        for b in (0,2,99):
            expected=read_batch(Path(saved)/f'batch_{b:03}.npz','p3b_field',460000,seeds[b])
            t=time.perf_counter();actual,rec=old.mt(460000,seeds[b],5.);values=h.reduced(actual,'p6')
            # Only the first 1,836 bins exist in the P6 layout; the rest of the 3,420 capacity must stay zero.
            unused=bool(not np.any(actual['regions'][:,h.layout('p6')['bins']:]))
            check=h.exact(values,expected,tuple(values));check['unused_capacity_zero']=unused
            if not unused:check['status']='fail'
            core.require(check,'1a batch '+str(b))
            batches[str(b)]=dict(check,seed=seeds[b],n=460000,seconds=time.perf_counter()-t,
                                 saved_sha256=core.file_sha(Path(saved)/f'batch_{b:03}.npz'))
            print('gate1 1a batch',b,'pass',f"{batches[str(b)]['seconds']:.1f} s",flush=True)
        report['checks']['1a']=core.verdict(True,batches=batches)
        conditions={};refs={}
        for name in 'ABC':
            d=h.condition_inputs(data,name);c=h.condition(name);mt=CPU(work/'build',d,'mt')
            expected,_=numba_reference(20000,h.SEED,name);refs[name]=expected
            actual,_=mt.mt(20000,h.SEED,c['field_half']);check=h.exact(h.reduced(actual),expected)
            core.require(check,'1b condition '+name);conditions[name]=dict(check,n=20000)
            print('gate1 1b',name,'pass',flush=True)
        report['checks']['1b']=core.verdict(True,conditions=conditions)
        faults={};dB=h.condition_inputs(data,'B')
        for fault in (1,2,3):
            bad=CPU(work/'build',dB,'mt',fault=fault);actual,_=bad.mt(20000,h.SEED,10.)
            r=h.exact(h._reduced(actual),refs['B'],tuple(refs['B']))
            faults[str(fault)]=dict(status='pass' if r['status']=='fail' else 'fail',comparison=r['status'])
            core.require(faults[str(fault)],'1c fault '+str(fault))
        dA=h.condition_inputs(data,'A');bad=CPU(work/'build',dA,'mt',p7_fault=8);actual,_=bad.mt(20000,h.SEED,2.5)
        r=h.exact(h._reduced(actual),refs['A'],tuple(refs['A']))
        faults['8']=dict(status='pass' if r['status']=='fail' else 'fail',comparison=r['status'],condition='A')
        core.require(faults['8'],'1c fault 8 (field half left at 5)')
        report['checks']['1c']=core.verdict(True,faults=faults,n=20000)
        from .p5k import CPU as P5CPU
        cpu=CPU(work/'build',data);p5=P5CPU(work/'p5j_build',data)
        a=cpu.histories(20000,h.SEED,5.,field=False,regions=False);b=p5.histories(20000,'P5j',start=0)
        r=h.exact(a,b,core.HISTORY_ORDER);core.require(r,'1d no-field CPU');report['checks']['1d']=r
        p6.verify_saved(saved);report['status']='pass'
    except Exception as exc:report.update(status='fail',stop=str(exc));raise
    finally:core.write_json(work/'gate1.json',report)
    return report


# ---------------- pilot (prototype side): C++ CPU Philox 4M per condition ----------------
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

_REF=None


def _ref_init(work,data,name):
    global _REF
    _REF=(CPU(work,h.condition_inputs(data,name)),h.condition(name))


def _ref_batch(b):
    cpu,c=_REF;t=time.perf_counter();n=40000
    r=cpu.run(n,c['seeds']['pilot_proto'],c['field_half'],c['ssd'],start=b*n)
    core.require(h.health(r),'CPU pilot health');return b,r,time.perf_counter()-t


def stacked(rows,stage,name,N=None):
    if len(rows)!=100:raise h.Stop('100 outer batches required')
    vals=[h.reduced(r) for r in rows];c=h.contract(stage,name,N)
    return dict(**{x:np.stack([v[x] for v in vals]) for x in vals[0]},N=np.array(c['n']*100),
                nbatch=np.array(100),seed=np.array(c['seed']),starts=np.arange(100,dtype=np.int64)*c['n'],
                ends=np.arange(1,101,dtype=np.int64)*c['n'],params=np.array(json.dumps(h.params(name),sort_keys=True)))


def scorer_fixture(cpu):
    """Deposits from the scorer vectors (incl. outside +-9 cm and outermost bins) through the CPU sinks."""
    vectors,_=scorer_vectors(0);records=np.zeros((len(vectors)*2+8,9))
    for i,v in enumerate(vectors):
        for kind in (0,1):
            row=records[2*i+kind];row[0]=2*i+kind;row[1]=kind
            row[2:5]=v[:3];row[5:8]=v[3:6] if kind==0 else v[:3];row[8]=v[6]
    for i,(p0,p1) in enumerate([([.3,.6,5.],[.7,.8,5.]),([-1.,.2,5.],[1.,.4,5.]),([.2,-.8,10.],[.4,.8,10.]),([15.,.6,20.],[19.9,.8,20.])]):
        for kind in (0,1):
            j=2*len(vectors)+2*i+kind;records[j]=[j,kind,*p0,*(p1 if kind==0 else p0),1.]
    n=len(records);groups=np.zeros((100,h.BINS));local=np.zeros((n,h.BINS));sg=np.zeros((n,15));sl=sg.copy()
    cpu.lib.p7_sinks(ct.byref(cpu.geometry),records,np.arange(n,dtype=np.int64),n,n,groups,local,sg,sl,np.zeros(groups.shape,np.uint64))
    return dict(records=records,regions=local,sums=sl)


def require_gates(work,data):
    work=Path(work);idents={n:h.identity(h.condition_inputs(data,n),n) for n in 'ABC'}
    for name,keys in (('local.json',{'0'+x for x in 'abcdefghijk'}),('gate1.json',{'1a','1b','1c','1d'})):
        r=json.loads((work/name).read_text())
        if r.get('status')!='pass' or r.get('identity')!=idents or set(r.get('checks',{}))!=keys:raise h.Stop('current complete '+name+' required')
    return idents


def pilot_proto(work,data,name):
    """4M CPU Philox histories (gate-3 contract) + 20k per-history reference + scorer fixture."""
    work=Path(work);idents=require_gates(work,data);ident=idents[name]
    d=h.condition_inputs(data,name);c=h.condition(name);target=work/'pilot'/name;target.mkdir(parents=True,exist_ok=True)
    found=h.resume(target/'batches',ident,'gate3',name);cpu=CPU(work/'build',d)
    path=target/'histories.npz';meta=dict(identity=ident,seed=h.SEED,start=0,n=20000)
    if path.exists():
        hist=core.read_npz(path);m=json.loads(str(hist.pop('metadata')))
        if m!=dict(meta,outputs=core.describe(hist)):raise h.Stop('history reference mismatch')
    else:
        hist=cpu.histories(20000,h.SEED,c['field_half'],c['ssd']);core.require(h.health(hist),'history reference health')
        core.save_npz(path,hist,dict(meta,outputs=core.describe(hist)))
    if not (target/'scorer.npz').exists():core.save_npz(target/'scorer.npz',scorer_fixture(cpu))
    missing=[b for b in range(100) if b not in found]
    if missing:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=_ref_init,
                                 initargs=(work/'build',data,name)) as pool:
            for b,r,seconds in pool.map(_ref_batch,missing,chunksize=1):
                p=target/'batches'/f'batch_{b:03}.npz';h.save_checkpoint(p,ident,'gate3',name,b,r);found[b]=p
                if len(found)%10==0:print(f'pilot {name} {len(found)}/100 ({seconds:.1f} s/batch)',flush=True)
    rows=[h.read_checkpoint(found[b],ident,'gate3',name,b) for b in range(100)]
    result=stacked(rows,'gate3',name);core.save_npz(target/'result.npz',result,dict(identity=ident,contract=h.contract('gate3',name)))
    report=dict(status='pass',identity=ident,batches=100,n=4000000,histories=20000,
                files={str(p.relative_to(target)):core.file_sha(p) for p in [target/'histories.npz',target/'scorer.npz',target/'result.npz',*(found[b] for b in range(100))]})
    core.write_json(target/'pilot.json',report);return report
