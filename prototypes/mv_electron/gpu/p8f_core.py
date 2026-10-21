"""P8f numeric packing and pre-registered comparisons (NumPy only)."""
import hashlib
import json
from pathlib import Path
import numpy as np

SEED0, SEED2, SEED3, SEED_PRODUCTION = 2026102101, 2026102102, 2026102103, 820101
DREF = 0.060387587985606006
CPU_FLAGS = ('-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC')
CUDA_OPTIONS = ('-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false')
FIELDS = ('ne excitation c z nj stop_T stop_delta stop_collision stop_rad gs_T gs_K gs_u gs_y rates photo_E photo pair_E pair_n pair_e thresholds dl bpar delcm delpos bindings pm shell_nj yields lines line_counts brems_E brems_V log_stop_T').split()
KEYS = ('dose','totals','maxima','violations','counts','brems','p8_max','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','steps')
HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint():
    # No plans, reports, notebooks, launch or measurement drivers.
    names = ('p8f_kernel.h','p8f_materials.h','p8f_cpu.cpp','p8f_gpu.cu','p8f_core.py',
             'p5_kernel.h','p5_rng.h','p5a0_common.h','p5d_gxx_compat.h')
    return {name: sha(HERE/name) for name in names}

def pack(materials):
    chunks, descriptors, offset = [], [], 0
    for m in materials:
        row = []
        for field in FIELDS:
            a = np.asarray(getattr(m, field), dtype=np.float64)
            if not np.isfinite(a).all():
                raise ValueError('nonfinite material '+field)
            shape = a.shape
            if field in ('ne','excitation','c','bpar','delcm','delpos'):
                if shape != (): raise ValueError('scalar '+field)
            elif a.ndim == 0: raise ValueError('array '+field)
            row.append((offset, a.size, shape[0] if shape else 1,
                        shape[2] if field == 'lines' else shape[1] if len(shape)>1 else 1))
            chunks.append(a.ravel());offset += a.size
        nz = len(m.z)
        shapes = dict(bindings=(nz,5),pm=(nz,5,4),shell_nj=(nz,),yields=(nz,4),
                      lines=(nz,4,m.lines.shape[2],4),line_counts=(nz,4),
                      gs_y=(len(m.gs_T),len(m.gs_K),len(m.gs_u)),
                      brems_V=(len(m.brems_E),5),dl=(6,6),thresholds=(2,))
        if not 1 <= nz <= 30 or any(np.shape(getattr(m,k))!=s for k,s in shapes.items()):
            raise ValueError('material shape')
        for k in ('stop_delta','stop_rad','log_stop_T'):
            if np.shape(getattr(m,k))!=np.shape(m.stop_T): raise ValueError('stopping shape')
        if len(m.stop_collision) not in (0,len(m.stop_T)):raise ValueError('collision shape')
        if m.rates.ndim!=2 or m.rates.shape[1] not in (2,3):raise ValueError('rate shape')
        for k in ('stop_T','gs_T','gs_K','gs_u','brems_E'):
            a=getattr(m,k)
            if a.ndim!=1 or len(a)<2 or np.any(np.diff(a)<=0):raise ValueError('grid '+k)
        for x,y in (('photo_E','photo'),('pair_E','pair_n'),('pair_E','pair_e')):
            if np.shape(getattr(m,x))!=np.shape(getattr(m,y)):raise ValueError('photon shape')
        descriptors.append(row)
    if not descriptors:raise ValueError('empty materials')
    return np.ascontiguousarray(np.concatenate(chunks)),np.asarray(descriptors,np.int64)

def bitwise(actual, expected, label='output'):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        mismatch = np.flatnonzero(a.ravel()!=b.ravel()) if a.shape==b.shape else []
        raise AssertionError(f'{label}: byte mismatch; first={list(mismatch[:1])}')

def aggregation_bound(histories, deposits, group_size):
    return (np.asarray(deposits)+group_size)*2.**-52*np.abs(histories).sum(axis=0)

def check_aggregation(groups, histories, deposit_counts):
    h=np.asarray(histories);n=len(h)
    if not np.isfinite(h).all() or not np.isfinite(groups).all():raise AssertionError('nonfinite aggregate')
    for b in range(100):
        selected=np.minimum(99,np.arange(n)*100//n)==b
        bound=aggregation_bound(h[selected],np.asarray(deposit_counts)[selected].sum(axis=0),int(selected.sum()))
        if np.any(np.abs(groups[b]-h[selected].sum(axis=0))>bound):raise AssertionError('aggregate differs from history outputs')

def check_local_bins(histories,dep):
    if not np.isfinite(histories).all() or np.any(np.abs(np.asarray(histories).sum(axis=1)-dep)>1e-12):
        raise AssertionError('actual output local bins do not balance dep')

def judge(prototype, reference, grid, *, dref):
    from prototypes.mv_electron.run_p8e import block_statistics, classify_blocks
    if dref!=DREF:raise ValueError('fixed EGS Dref required')
    _,pm,ps=block_statistics(prototype,grid);_,rm,rs=block_statistics(reference,grid)
    difference=(pm-rm)/dref;sem=np.hypot(ps,rs)/dref
    return dict(Dref=dref,difference=difference.tolist(),SEM=sem.tolist(),status=classify_blocks(difference,sem))

def paired_blocks(cpu,gpu,grid):
    cpu,gpu=np.asarray(cpu),np.asarray(gpu)
    shape=(100,len(grid['mass_g_cm2']))
    if cpu.shape!=shape or gpu.shape!=shape or not np.isfinite(cpu).all() or not np.isfinite(gpu).all() or np.any(cpu<0) or np.any(gpu<0):raise ValueError('100 paired outer batches required')
    a=cpu@grid['block_weights'].T;b=gpu@grid['block_weights'].T
    delta=b-a;diff=delta.mean(axis=0)/DREF;sem=delta.std(axis=0,ddof=1)/10/DREF
    return dict(difference=diff.tolist(),SEM=sem.tolist(),status='pass' if np.all(np.abs(diff)<=.005) and np.all(sem<=.002) else 'insufficient-statistics' if np.any(sem>.002) else 'fail')

def unpack(d,i,nb,nc,nf):
    sizes=[nb,nc,nf,6,3,10,2,1,1,14];names=['histories','coarse','fine','totals','maxima','brems','p8_max','sources','dep','diagnostic']
    r={};off=0
    for k,s in zip(names,sizes):r[k]=np.ascontiguousarray(d[:,off:off+s]);off+=s
    r['diagnostic']=r['diagnostic'].reshape(-1,2,7)
    off=0
    for k,s in zip(('violations','counts','steps','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches','calls','diagnostic_count','maxdepth','nspec_index','additions'),(3,16,1,5,18,18,8,6,1,1,1,1,nb)):
        r[k]=np.ascontiguousarray(i[:,off:off+s]);off+=s
        if s==1:r[k]=r[k][:,0]
    r['dep']=r['dep'][:,0];r['sources']=r['sources'][:,0]
    return r

def history_health(r):
    finite=all(np.isfinite(v).all() for v in r.values())
    return bool(finite and np.all(r['histories']>=0) and not np.any(r['counts'][:,5:7]) and not np.any(r['violations']) and not np.any(r['brems'][:,2:4]) and not np.any(r['totals'][:,3]) and np.all(r['maxima'][:,0]<=1e-12) and np.all(r['p8_max'][:,0]<=1e-12))

def compare_histories(cpu,gpu):
    from .p5b_core import c2_calls
    checks=dict(cpu_health=history_health(cpu),gpu_health=history_health(gpu),
                source=bool(np.array_equal(cpu['sources'],gpu['sources']) and np.array_equal(cpu['nspec_index'],gpu['nspec_index'])),
                calls=c2_calls(dict(diag_used=cpu['diagnostic_count'],diagnostic=cpu['diagnostic']),dict(diag_used=gpu['diagnostic_count'],diagnostic=gpu['diagnostic'])))
    reports={}
    for k in ('histories','coarse','fine','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts'):
        same=np.all(cpu[k]==gpu[k],axis=1);bad=np.flatnonzero(~same)
        reports[k]=dict(fraction=float(same.mean()),first=int(bad[0]) if len(bad) else None,max_abs=float(np.max(np.abs(cpu[k]-gpu[k]),initial=0)))
    return dict(status='pass' if checks['cpu_health'] and checks['gpu_health'] and checks['source'] and checks['calls']['status']=='pass' else 'fail',checks=checks,report_only=reports)

def cuda_source():
    def expand(name,seen):
        path=HERE/name;text=path.read_text();lines=[]
        for line in text.splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child in seen:continue
                seen.add(child);lines.append(expand(child,seen))
            else:lines.append(line)
        return '\n'.join(lines)
    probe=(HERE/'p8f_cpu.cpp').read_text()
    probe=probe[probe.index('extern "C" void p8f_probe('):probe.index('extern "C" void p8f_score_probe(')]
    probe=probe.replace('extern "C" void p8f_probe(', 'extern "C" __global__ void p8f_probe_kernel(')
    probe=probe.replace('P8Material m=p8f_material', 'if(blockIdx.x || threadIdx.x)return;P8Material m=p8f_material',1)
    return (HERE/'p8f_gpu.cu').read_text().replace('//@@COMMON@@',expand('p8f_kernel.h',set())).replace('//@@PROBE@@',probe)

def reduce_records(d,i,nb,nc,nf,n,masses):
    r=unpack(d,np.asarray(i,dtype=np.int64),nb,nc,nf)
    out=dict(p8_batches=r['histories'],coarse=r['coarse'],fine=r['fine'],
        dose=r['histories'].sum(axis=0)/n/masses,steps=r['steps'],maxdepth=np.array(r['maxdepth'].max(),np.int64),additions=r['additions'])
    for k in ('totals','brems','violations','counts','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches'):out[k]=r[k].sum(axis=0)
    for k in ('maxima','p8_max'):out[k]=r[k].max(axis=0)
    return out

def fixed_inputs():
    """Representative for every 0d category, also replayed unchanged on T4."""
    probes=[dict(name='shell-cumulative-tie',which=24,x=[.5,.5,.5,0,0,0],mid=0),dict(name='shell-last-positive',which=24,x=[.99,.1,.2,0,0,0],mid=0),dict(name='event-priority',which=25,x=[.25,.25,.25,.25,.25,.25,.25],mid=0)]
    for shell in (0,1,4):probes.append(dict(name='relax'+str(shell),which=9,x=[0.,float(shell),1.],mid=3))
    probes += [dict(name='select',which=7,x=[.1],mid=3),dict(name='brems',which=8,x=[1.],mid=3)]
    for strength,terminal,mu in ((0.,0.,1.),(.01,0.,-1.),(.01,1.,1.),(.0001,0.,1.)):
        a=np.zeros(29);a[5]=1.;a[6]=a[7]=1.;a[14]=strength;a[18]=terminal;a[23:]=[0,1,10.,1.,.025,mu]
        probes.append(dict(name=f'boundary-{strength}-{terminal}-{mu}',which=17,x=a.tolist(),mid=0))
    states=[]
    for flag in (2.,3.):
        a=np.zeros(23);a[2]=21.2;a[5]=1.;a[6]=.007;a[16]=5;a[17]=flag;states.append(a)
    for offset in (0.,np.nextafter(1e-13,0.),np.nextafter(1e-13,np.inf),2e-13):
        a=np.zeros(23);a[2]=5.;a[5]=1.;a[6]=a[7]=1.;a[9]=.025;a[10]=.025;a[11]=.05;a[8]=offset;a[12]=0.;a[13]=.025;a[14]=.05;a[15]=0.;a[19]=-1;states.append(a)
    # A horizontal 10 MeV electron in lead: residual energy/scattering/hard
    # clocks are beyond the brems flight, so the first event is emission and
    # the suspended parent must subsequently resume with its own clocks.
    a=np.zeros(23);a[2]=21.25;a[3]=1.;a[6]=a[7]=10.;a[8]=10.;a[9]=a[10]=.25;a[11]=.5;a[12]=a[15]=1e100;a[14]=.025;a[16]=5;a[19]=-1.;states.append(a)
    return probes,np.asarray(states)

def compare_probe_gpu(which,cpu,gpu,ccalls,gcalls):
    # Outputs free of device math must retain their exact input bits.
    integer={24:[0,1],25:[0],7:[0,1],8:[1],9:[0,1,*range(35,59)],17:[16,26]}.get(which,[])
    exact={24:[0,1],25:[0],7:[0,1],8:[1],9:[0,1,2,*range(3,35,4),*range(35,59)],17:[*range(3),*range(6,23),23,25,26,27]}.get(which,[])
    c=np.asarray(cpu).ravel();g=np.asarray(gpu).ravel()
    ok=ccalls==gcalls and np.array_equal(c[integer],g[integer]) and c[exact].tobytes()==g[exact].tobytes() and np.isfinite(g).all()
    mismatch=np.flatnonzero(c!=g)
    return dict(status='pass' if ok else 'fail',calls=[int(ccalls),int(gcalls)],first_float_difference=int(mismatch[0]) if len(mismatch) else None,max_relative=float(np.max(np.abs(c-g)/np.maximum(np.maximum(np.abs(c),np.abs(g)),1e-300),initial=0)))

def check_resume(old,new,n):
    # Same histories, different atomic order: every additive quantity is checked
    # with its actual absolute sum and addition count, including signed brems.
    if not np.array_equal(old['_groups_i'],new['_groups_i']):raise AssertionError('resume integer transition mismatch')
    a,b=old['_groups_d'],new['_groups_d'];nb=old['p8_batches'].shape[1]
    f=nb+old['coarse'].shape[1]+old['fine'].shape[1]
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError('resume record mismatch')
    count=np.full(a.shape,n//100,dtype=np.int64);count[:,:nb]=np.maximum(old['additions'],new['additions'])
    bound=(count+n//100)*2.**-52*np.maximum(old['_absolute'],new['_absolute'])
    for j in (*range(f+6,f+9),f+19,f+20):bound[:,j]=0.
    if np.any(np.abs(a-b)>bound):raise AssertionError('resume atomic sum mismatch')
