"""P6 fixed geometry, provenance, numerical checks and checkpoint contracts.

NumPy-only transport-independent module; also used on Colab.
"""
import ctypes as ct
import json
import re
from pathlib import Path
import numpy as np
from . import p5b_core as base, p5j_core as j, p5k_core as k

HERE=base.HERE
ROOT=HERE.parents[2]
SEED=base.SEED
MAIN_SEED=20291764
MAXSTEPS=base.MAXSTEPS
BINS=1836
SIZES=(120,300,120,*([72]*6),*([144]*6))
OFFSETS=np.r_[0,np.cumsum(SIZES)[:-1]].astype(np.int32)
SOURCES=('p6_field.h','p6_kernel.h','p6_cpu.cpp','p6_gpu.cu')
TOOLS=('p6.py','p6_core.py','p6_colab.py','p6_colab.ipynb')
PARAMS=dict(E=10.,backend='elsepa',brems_mode=2,spectrum='mohan',field_half=5.,ssd=100.,
            dz=.25,dz_fine=.1,regions=[0.,0.,1.,1.,9.,.25,.125,1.,5.,10.,20.])
INTEGER=base.INTEGER|{'region_ledger'}
HISTORY_ORDER=base.HISTORY_ORDER+('regions','region_ledger','sources')
RUN_ORDER=base.RUN_ORDER+('regions','region_ledger')
Stop=base.Stop

class Geometry(ct.Structure):
    _fields_=[(n,ct.c_double*m) for n,m in (('lo',45),('hi',45),('origins',15),('widths',15))]
    _fields_ += [(n,ct.c_int*15) for n in ('axes','sizes','offsets')]


def geometry():
    lo=[[-1.,-1.,0.],[-1.,-1.,0.],[-np.inf,-np.inf,0.]]
    hi=[[1.,1.,30.],[1.,1.,30.],[np.inf,np.inf,30.]]
    axes=[2]*3;origins=[0.]*3;widths=[.25,.1,.25]
    for width in (.25,.125):
        for axis in (0,1):
            for depth in (5.,10.,20.):
                low=[-1.,-1.,depth-.5];high=[1.,1.,depth+.5]
                low[axis]=-9.;high[axis]=9.
                lo.append(low);hi.append(high);axes.append(axis);origins.append(-9.);widths.append(width)
    arrays=dict(lo=np.array(lo).ravel(),hi=np.array(hi).ravel(),origins=np.array(origins),widths=np.array(widths),
                axes=np.array(axes,np.int32),sizes=np.array(SIZES,np.int32),offsets=OFFSETS.copy())
    g=Geometry()
    for name,values in arrays.items():getattr(g,name)[:]=values
    return g,arrays


def identity(data):
    k.frozen();base.check_inputs(data)
    return dict(sources={n:base.file_sha(HERE/n) for n in (*j.KERNEL_SOURCES,*k.SOURCES,*SOURCES)},
                tools={n:base.file_sha(HERE/n) for n in TOOLS},inputs=base.describe(data),
                geometry=base.describe(geometry()[1]),options=list(base.OPTIONS),params=PARAMS,
                gate3=dict(seed=SEED,n=100000,batches=100),production=dict(seed=MAIN_SEED,n=460000,batches=100))


def source():
    text=j.source()
    for name in ('p6_field.h','p6_kernel.h','p6_gpu.cu'):
        s=(HERE/name).read_text();s=re.sub(r'^#include "p[56][^\n]+\n','',s,flags=re.M)
        text+='\n'+s
    return text


def history_shapes(n):return dict(base.history_shapes(n),regions=(n,BINS),region_ledger=(n,15),sources=(n,7))
def run_shapes(nspec=20):return dict(base.run_shapes(nspec),regions=(100,BINS),region_ledger=(15,))
def empty(shapes):return {n:np.zeros(s,np.int64 if n in INTEGER else np.float64) for n,s in shapes.items()}


def validate(r,history=False,n=None,nspec=20):
    shapes=history_shapes(n) if history else run_shapes(nspec)
    if set(r)!=set(shapes):raise Stop('P6 fields mismatch')
    base.validate_output({k:r[k] for k in (base.HISTORY_ORDER if history else base.RUN_ORDER)},
                         base.history_shapes(n) if history else base.run_shapes(nspec))
    for name in set(shapes)-set(base.HISTORY_ORDER if history else base.RUN_ORDER):
        if r[name].shape!=shapes[name] or r[name].dtype!=(np.int64 if name in INTEGER else np.float64):
            raise Stop('P6 shape/dtype mismatch: '+name)
    return r


def health(r):
    old={k:r[k] for k in (base.HISTORY_ORDER if 'calls' in r else base.RUN_ORDER)}
    return base.verdict(base.health(old)['status']=='pass' and not np.any(r['region_ledger'])
                        and all(np.isfinite(v).all() for v in r.values())
                        and region_output_ledger(r))


def region_output_ledger(r):
    # Independently verify the emitted slab array. This detects stale local arrays
    # even when the per-history region_sums were reset correctly during refill.
    slab=r['regions'][...,420:540].sum(axis=-1)
    if 'calls' in r:
        return bool(np.all(np.abs(slab-r['totals'][:,1])<=1e-9*r['totals'][:,0]))
    return bool(abs(float(slab.sum()-r['totals'][1]))<=1e-9*float(r['totals'][0]))


def exact(a,b,keys=None):
    keys=tuple(a) if keys is None else keys
    fields={k:base.bits(a[k],b[k]) for k in keys}
    return base.verdict(all(fields.values()),fields=fields)


def unpack_regions(packed):
    parts=[packed[...,int(o):int(o)+s] for o,s in zip(OFFSETS,SIZES)]
    return dict(axis_coarse=parts[0].copy(),axis_fine=parts[1].copy(),slab_coarse=parts[2].copy(),
                profile=np.stack(parts[3:9],axis=-2).reshape((*packed.shape[:-1],2,3,72)),
                profile_fine=np.stack(parts[9:15],axis=-2).reshape((*packed.shape[:-1],2,3,144)))


def reduced(r):
    """Collapse the 100 inner groups to one outer batch, as P3b.reduce does."""
    validate(r,nspec=r['nspec'].shape[0]);base.require(health(r),'P6 save health')
    return _reduced(r)


def _reduced(r):
    # Pure reshaping/reduction, also used to inspect intentionally unhealthy faults.
    out=dict(coarse=r['batches'].sum(0),fine=r['fine_batches'].sum(0),pair=np.zeros(600),
             kerma=np.zeros(600),collision_kerma=np.zeros(600),totals=r['totals'],maxima=r['maxima'],
             violations=r['violations'],counts=r['counts'],maxdepth=r['maxdepth'],nspec=r['nspec'],
             lateral=np.zeros(100),steps=r['step_batches'],brems=r['brems'],first_record=np.zeros(8),
             region_ledger=r['region_ledger'])
    out.update({k:v.sum(0) for k,v in unpack_regions(r['regions']).items()})
    return out


def c2(cpu,gpu,dref,conversion):
    n=len(cpu['steps'])
    validate(cpu,True,n);validate(gpu,True,n)
    a={k:cpu[k] for k in base.HISTORY_ORDER};b={k:gpu[k] for k in base.HISTORY_ORDER}
    checks=dict(cpu_health=health(cpu),gpu_health=health(gpu),source=base.c2_source(a,b),calls=base.c2_calls(a,b))
    # Sources include signed x/y and all three direction cosines (not just energy).
    source_delta=np.abs(cpu['sources']-gpu['sources'])
    checks['field_source']=base.verdict(bool(np.all(source_delta<=1e-13)),max_abs=float(source_delta.max(initial=0)))
    bins=base.c2_bins(a,b,dref=dref,conversion=conversion)
    regions={}
    for index,(off,size) in enumerate(zip(OFFSETS,SIZES)):
        delta=gpu['regions'][:,off:off+size]-cpu['regions'][:,off:off+size]
        volume=(1.,.4,.25,*([.5]*6),*([.25]*6))[index]
        delta=delta*conversion/volume
        regions[str(index)]=dict(status='report',paired=base.paired(delta,dref),
            bit_fraction=float(np.mean(base.bit_elements(cpu['regions'][:,off:off+size],gpu['regions'][:,off:off+size]))))
    return base.verdict(all(v['status']=='pass' for v in checks.values()),checks=checks,legacy_bins=bins,regions=regions)


def c3(histories,production,nspec=20,*,region_additions):
    n=len(histories['steps']);validate(histories,True,n,nspec);validate(production,False,nspec=nspec)
    old=base.c3({k:histories[k] for k in base.HISTORY_ORDER},{k:production[k] for k in base.RUN_ORDER},nspec)
    total=np.zeros((100,BINS));absolute=total.copy();rows=np.arange(n)*100//n
    np.add.at(total,rows,histories['regions']);np.add.at(absolute,rows,np.abs(histories['regions']))
    K=np.bincount(rows,minlength=100)[:,None]
    if region_additions.shape!=total.shape or region_additions.dtype!=np.uint64:
        raise Stop('2c requires per-group per-bin deposit addition counts (uint64)')
    # Addendum 1: local deposit additions plus the history-to-group atomic merge.
    delta=np.abs(total-production['regions']);bound=(region_additions+K)*2**-52*absolute
    ledger=base.bits(histories['region_ledger'].sum(0),production['region_ledger'])
    finite=all(np.isfinite(v).all() for r in (histories,production) for v in r.values())
    failures=int(np.count_nonzero(~(delta<=bound)))
    return base.verdict(old['status']=='pass' and failures==0 and ledger and finite,legacy=old,
                        region_exceeding=failures,ledger_exact=ledger,finite=finite)


def contract(stage):
    if stage=='gate3':return dict(seed=SEED,n=100000,batches=100,stage=stage)
    if stage=='production':return dict(seed=MAIN_SEED,n=460000,batches=100,stage=stage)
    raise ValueError('unknown stage')


def checkpoint_meta(ident,stage,b,r):
    c=contract(stage)
    if not isinstance(b,int) or not 0<=b<100:raise Stop('invalid batch index')
    return dict(identity=ident,contract=c,batch=b,start=c['n']*b,end=c['n']*(b+1),outputs=base.describe(r))


def read_checkpoint(path,ident,stage,b):
    r=base.read_npz(path)
    try:meta=json.loads(str(r.pop('metadata')))
    except (KeyError,ValueError) as exc:raise Stop('missing checkpoint metadata') from exc
    validate(r);base.require(health(r),'checkpoint health')
    if int(r['nspec'].sum())!=contract(stage)['n']:raise Stop('checkpoint history count')
    if meta!=checkpoint_meta(ident,stage,b,r):raise Stop('checkpoint identity/seed/range/settings/hash mismatch')
    return r


def save_checkpoint(path,ident,stage,b,r):
    validate(r);base.require(health(r),'save health')
    if int(r['nspec'].sum())!=contract(stage)['n']:raise Stop('save history count')
    if Path(path).exists():raise Stop('refuse checkpoint overwrite')
    base.save_npz(path,r,checkpoint_meta(ident,stage,b,r))


def resume(directory,ident,stage):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if any('pending' in p.name or p.suffix in ('.tmp','.part') for p in directory.iterdir()):raise Stop('unfinished checkpoint; retain and stop')
    config=dict(identity=ident,contract=contract(stage))
    path=directory/'config.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise Stop('resume config mismatch')
    else:base.write_json(path,config)
    found={}
    for path in directory.glob('batch_*.npz'):
        if not re.fullmatch(r'batch_\d{3}\.npz',path.name):raise Stop('invalid checkpoint name')
        b=int(path.stem[-3:]);read_checkpoint(path,ident,stage,b);found[b]=path
    return found


def stacked(rows,stage):
    if len(rows)!=100:raise Stop('100 outer batches required')
    vals=[reduced(r) for r in rows];c=contract(stage)
    return dict(**{k:np.stack([v[k] for v in vals]) for k in vals[0]},N=np.array(c['n']*100),
                nbatch=np.array(100),seed=np.array(c['seed']),starts=np.arange(100,dtype=np.int64)*c['n'],
                ends=np.arange(1,101,dtype=np.int64)*c['n'],params=np.array(json.dumps(PARAMS,sort_keys=True)))


def c4(reference,actual,dref):
    from .. import analysis_p3b3 as stats
    left=stacked(reference,'gate3');right=stacked(actual,'gate3')
    ld,rd=stats.dose(left),stats.dose(right)
    # Exact analogue of core.c4: paired outer-batch differences, not independent SEMs.
    legacy_left=np.stack([base.deep_blocks(r['batches'].sum(0),100000,stats.MEV_TO_GY) for r in reference])
    legacy_right=np.stack([base.deep_blocks(r['batches'].sum(0),100000,stats.MEV_TO_GY) for r in actual])
    box_left=np.stack([ld['axis_coarse'][:,lo:hi].mean(1) for lo,hi in stats.BLOCKS],axis=1)
    box_right=np.stack([rd['axis_coarse'][:,lo:hi].mean(1) for lo,hi in stats.BLOCKS],axis=1)
    profile_left=stats.fold(ld['profile'])[:,:,:16].mean(2);profile_right=stats.fold(rd['profile'])[:,:,:16].mean(2)
    quantities={};passed=True
    for label,a,b in (('legacy_deep',legacy_left,legacy_right),('axis_blocks',box_left,box_right),('profile_central',profile_left,profile_right)):
        p=base.paired(b-a,dref);ok=(np.abs(p['mean_pct'])<=.5)&(np.array(p['SEM_pct'])<=.2)
        quantities[label]=dict(p,passed=ok.tolist());passed &= bool(np.all(ok))
    spectra=all(base.bits(a['nspec'],b['nspec']) for a,b in zip(reference,actual))
    healthy=all(health(r)['status']=='pass' for r in reference+actual)
    return base.verdict(passed and spectra and healthy,quantities=quantities,spectra_exact=spectra,healthy=healthy)


def require_report(path,data,kind):
    r=json.loads(Path(path).read_text())
    if r.get('status')!='pass' or r.get('identity')!=identity(data):raise Stop('current '+kind+' required')
    required={'0a','0b','0c','0d','0e'} if kind=='local' else {'1a','1b','1c'}
    if set(r.get('checks',{}))!=required:raise Stop('incomplete '+kind)
    for check in r['checks'].values():base.require(check,kind)
    checks=r['checks']
    if kind=='local':
        if checks['0a'].get('randoms')!=100000 or checks['0a'].get('edges',0)<6:raise Stop('0a incomplete')
        c=checks['0c']
        if (not c.get('deposits_bitwise') or not c.get('sources_bitwise') or c.get('cpp_meta',[None]*4)[1:]!=[0,2000,0]
            or set(c.get('coverage',{}))!={str(i) for i in range(1,6)} or not all(c['coverage'][str(i)]>0 for i in range(1,5))
            or c.get('endpoint',{}).get('status')!='pass'):raise Stop('0c incomplete')  # addendum 2
        if not any(checks['0d'].get('fault_ledger',[])) or not checks['0d'].get('numeric_mismatch'):raise Stop('0d incomplete')
    else:
        batches=checks['1a'].get('batches',{})
        if set(batches)!={'0','2','99'}:raise Stop('1a incomplete')
        for batch in batches.values():
            if batch.get('n')!=460000 or len(batch.get('fields',{}))!=21 or not all(batch['fields'].values()):raise Stop('1a incomplete fields')
        c=checks['1c'];base.require(c.get('normal',base.verdict(False)),'1c normal')
        if c.get('n')!=20000 or set(c.get('faults',{}))!={'1','2','3'}:raise Stop('1c incomplete')
        for fault in c['faults'].values():
            base.require(fault,'1c fault')
            if fault.get('comparison',{}).get('status')!='fail':raise Stop('1c numerical fault evidence required')
    return r
