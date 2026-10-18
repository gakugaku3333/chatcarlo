"""P7 conditions, common +-20 cm layout, spectrum input, shapes and checkpoint contracts.

NumPy-only transport-independent module; also used on Colab. P6 files are not edited.
Plan: docs/ai/plans/2026-10-18-mv-p7-field-energy-variants.md
"""
import ctypes as ct
import json
import math
import re
from pathlib import Path
import numpy as np
from . import p5b_core as base, p5j_core as j, p5k_core as k, p6_core as p6

HERE=base.HERE
ROOT=HERE.parents[2]
SEED=base.SEED
MAXSTEPS=base.MAXSTEPS
Stop=base.Stop
BINS=3420
SOURCES=('p6_field.h','p7_field.h','p7_kernel.h','p7_cpu.cpp','p7_gpu.cu')
INTEGER=p6.INTEGER
HISTORY_ORDER=p6.HISTORY_ORDER
RUN_ORDER=p6.RUN_ORDER
CACHE=Path('~/.cache/chatcarlo/p7').expanduser()
SPECTRA=dict(
    mohan10=dict(path=ROOT/'docs/validation/mv/p0/data/mohan10.spectrum',
                 sha256='06fe2ba471dbfd64ca4305b5f1560479439ea4b565f993ecb21dcda86a6d8065',bins=20,width=.5),
    mohan4=dict(path=CACHE/'spectra/mohan4.spectrum',
                sha256='9e7f8bee03be8ce64156439e8f8be97f0d4bdb23dbdd778d58dcd73260cfa484',bins=16,width=.25))
# Plan section 1/7: field half-widths at the surface, SSD 100 cm, seeds.
CONDITIONS=dict(
    A=dict(spectrum='mohan10',field_half=2.5,ssd=100.,seeds=dict(pilot_egs=20291801,pilot_proto=20291811,main_egs=20291821,main_gpu=20291831)),
    B=dict(spectrum='mohan10',field_half=10.,ssd=100.,seeds=dict(pilot_egs=20291802,pilot_proto=20291812,main_egs=20291822,main_gpu=20291832)),
    C=dict(spectrum='mohan4',field_half=5.,ssd=100.,seeds=dict(pilot_egs=20291803,pilot_proto=20291813,main_egs=20291823,main_gpu=20291833)))
DEPTHS=(5.,10.,20.)
# Layouts: 'p6' reproduces the saved P3b-3 geometry (gate 1a only); 'p7' is the common one.
LAYOUTS=dict(p6=dict(extent=9.,coarse=.25,fine=.125),p7=dict(extent=20.,coarse=.25,fine=.125))


def layout(name):
    if name not in LAYOUTS:raise ValueError('unknown layout')
    L=LAYOUTS[name];nc=int(round(2*L['extent']/L['coarse']));nf=int(round(2*L['extent']/L['fine']))
    sizes=(120,300,120,*([nc]*6),*([nf]*6))
    offsets=np.r_[0,np.cumsum(sizes)[:-1]].astype(np.int32)
    return dict(name=name,sizes=sizes,offsets=offsets,bins=int(sum(sizes)),coarse_bins=nc,fine_bins=nf,
                volumes=(1.,.4,.25,*([2*L['coarse']]*6),*([2*L['fine']]*6)),
                regions=[0.,0.,1.,1.,L['extent'],L['coarse'],L['fine'],1.,*DEPTHS],**L)


def geometry(name='p7'):
    L=layout(name);e=L['extent']
    lo=[[-1.,-1.,0.],[-1.,-1.,0.],[-np.inf,-np.inf,0.]]
    hi=[[1.,1.,30.],[1.,1.,30.],[np.inf,np.inf,30.]]
    axes=[2]*3;origins=[0.]*3;widths=[.25,.1,.25]
    for width in (L['coarse'],L['fine']):
        for axis in (0,1):
            for depth in DEPTHS:
                low=[-1.,-1.,depth-.5];high=[1.,1.,depth+.5]
                low[axis]=-e;high[axis]=e
                lo.append(low);hi.append(high);axes.append(axis);origins.append(-e);widths.append(width)
    arrays=dict(lo=np.array(lo).ravel(),hi=np.array(hi).ravel(),origins=np.array(origins),widths=np.array(widths),
                axes=np.array(axes,np.int32),sizes=np.array(L['sizes'],np.int32),offsets=L['offsets'].copy())
    if L['bins']>BINS:raise Stop('layout exceeds P7 capacity')
    g=p6.Geometry()
    for key,values in arrays.items():getattr(g,key)[:]=values
    return g,arrays


def read_spectrum(path,sha256,bins,width):
    """Mode-1 spectrum: probability = value x bin width, normalised; uniform inside each bin."""
    path=Path(path)
    if base.file_sha(path)!=sha256:raise Stop('spectrum SHA256 mismatch: '+path.name)
    lines=path.read_text().splitlines()
    header=[float(x) for x in lines[1].replace(',',' ').split()]
    if header!=[float(bins),0.,1.]:raise Stop('spectrum header must be bins, lower edge 0, mode 1')
    rows=np.array([[float(x) for x in s.replace(',',' ').split()] for s in lines[2:] if s.strip()])
    if rows.shape!=(bins,2) or not np.all(np.isfinite(rows)) or np.any(rows[:,1]<=0):raise Stop('invalid spectrum rows')
    edges=np.r_[0.,rows[:,0]]
    weights=rows[:,1]*np.diff(edges);prob=weights/weights.sum()
    cum=np.cumsum(prob);cum[-1]=1.
    if width is not None and not np.array_equal(edges,np.arange(bins+1)*width):raise Stop('spectrum edges differ from preregistration')
    if np.any(np.diff(cum)<=0) or cum[0]<=0:raise Stop('invalid cumulative spectrum')
    return edges,prob,cum


def spectrum(name):
    s=SPECTRA[name];return read_spectrum(s['path'],s['sha256'],s['bins'],s['width'])


def condition(name):
    if name not in CONDITIONS:raise ValueError('unknown condition')
    c=dict(CONDITIONS[name]);s=SPECTRA[c['spectrum']]
    return dict(c,name=name,spectrum_sha256=s['sha256'],nspec=s['bins'],spectrum_top=s['bins']*s['width'],layout='p7')


def condition_inputs(data,name):
    """Replace only the spectrum arrays of the frozen P5 inputs; every other table is unchanged."""
    edges,_,cum=spectrum(condition(name)['spectrum'])
    out=dict(data);out['edges']=np.ascontiguousarray(edges);out['cum']=np.ascontiguousarray(cum)
    base.check_inputs(out);return out


def params(name):
    c=condition(name)
    return dict(E=10.,backend='elsepa',brems_mode=2,spectrum=c['spectrum'],spectrum_sha256=c['spectrum_sha256'],
                field_half=c['field_half'],ssd=c['ssd'],dz=.25,dz_fine=.1,regions=layout('p7')['regions'])


def identity(data,name):
    """Fingerprint of sources, inputs, geometry and condition. Launch tools are excluded (P5g/P6 lesson)."""
    k.frozen();base.check_inputs(data);c=condition(name)
    # Locally the spectrum file is re-read; on Colab (no file) the input hashes in `inputs` are
    # compared against the manifest written locally after this same check.
    if SPECTRA[c['spectrum']]['path'].exists() and not np.array_equal(data['cum'],spectrum(c['spectrum'])[2]):
        raise Stop('inputs do not carry this condition spectrum')
    if len(data['cum'])!=c['nspec']:raise Stop('inputs do not carry this condition spectrum')
    return dict(condition=name,sources={n:base.file_sha(HERE/n) for n in (*j.KERNEL_SOURCES,*k.SOURCES,*SOURCES)},
                inputs=base.describe(data),geometry=base.describe(geometry('p7')[1]),options=list(base.OPTIONS),params=params(name))


def source():
    text=j.source()
    for n in ('p6_field.h','p7_field.h','p7_kernel.h','p7_gpu.cu'):
        s=(HERE/n).read_text();s=re.sub(r'^#include "p[567][^\n]+\n','',s,flags=re.M)
        text+='\n'+s
    return text


def history_shapes(n,bins=BINS):return dict(base.history_shapes(n),regions=(n,bins),region_ledger=(n,15),sources=(n,7))
def run_shapes(nspec,bins=BINS):return dict(base.run_shapes(nspec),regions=(100,bins),region_ledger=(15,))
def empty(shapes):return {n:np.zeros(s,np.int64 if n in INTEGER else np.float64) for n,s in shapes.items()}


def validate(r,history=False,n=None,nspec=None,bins=BINS):
    if nspec is None:raise ValueError('nspec must come from the condition (no default)')
    shapes=history_shapes(n,bins) if history else run_shapes(nspec,bins)
    if set(r)!=set(shapes):raise Stop('P7 fields mismatch')
    base.validate_output({x:r[x] for x in (base.HISTORY_ORDER if history else base.RUN_ORDER)},
                         base.history_shapes(n) if history else base.run_shapes(nspec))
    for key in set(shapes)-set(base.HISTORY_ORDER if history else base.RUN_ORDER):
        if r[key].shape!=shapes[key] or r[key].dtype!=(np.int64 if key in INTEGER else np.float64):
            raise Stop('P7 shape/dtype mismatch: '+key)
    return r


def health(r):
    # The first three regions (axis coarse/fine, slab) have the same offsets as P6.
    return p6.health(r)


def exact(a,b,keys=None):return p6.exact(a,b,keys)


def unpack_regions(packed,name='p7'):
    L=layout(name);parts=[packed[...,int(o):int(o)+s] for o,s in zip(L['offsets'],L['sizes'])]
    if packed.shape[-1]<L['bins']:raise Stop('packed regions shorter than layout')
    lead=packed.shape[:-1]
    return dict(axis_coarse=parts[0].copy(),axis_fine=parts[1].copy(),slab_coarse=parts[2].copy(),
                profile=np.stack(parts[3:9],axis=-2).reshape((*lead,2,3,L['coarse_bins'])),
                profile_fine=np.stack(parts[9:15],axis=-2).reshape((*lead,2,3,L['fine_bins'])))


def reduced(r,name='p7'):
    base.require(health(r),'P7 save health');return _reduced(r,name)


def _reduced(r,name='p7'):
    out=dict(coarse=r['batches'].sum(0),fine=r['fine_batches'].sum(0),pair=np.zeros(600),
             kerma=np.zeros(600),collision_kerma=np.zeros(600),totals=r['totals'],maxima=r['maxima'],
             violations=r['violations'],counts=r['counts'],maxdepth=r['maxdepth'],nspec=r['nspec'],
             lateral=np.zeros(100),steps=r['step_batches'],brems=r['brems'],first_record=np.zeros(8),
             region_ledger=r['region_ledger'])
    out.update({x:v.sum(0) for x,v in unpack_regions(r['regions'],name).items()})
    return out


# ---- history counts (plan section 3 / check 0i) ----
QUANTUM=2000000
MAXIMUM_N=240000000
MARGIN=1.5


def round_up(n):
    if not math.isfinite(n) or n<=0:raise Stop('invalid required history count')
    return int(math.ceil(n/QUANTUM-1e-12))*QUANTUM


def main_size(required):
    """required: histories needed by the worst of the 11 quantities (extrapolated from the pilot).
    Returns N_main = round_up(1.5 x required); stop above 2.4e8."""
    n=round_up(MARGIN*float(required))
    if n>MAXIMUM_N:raise Stop(f'N_main {n} exceeds {MAXIMUM_N}; 相談')
    return n


def contract(stage,name,N=None):
    c=condition(name)
    if stage=='gate3':return dict(condition=name,seed=c['seeds']['pilot_proto'],n=40000,batches=100,stage=stage)
    if stage=='production':
        if not isinstance(N,int) or N<=0 or N%QUANTUM or N>MAXIMUM_N:raise Stop('invalid production N_main')
        return dict(condition=name,seed=c['seeds']['main_gpu'],n=N//100,batches=100,stage=stage)
    raise ValueError('unknown stage')


def checkpoint_meta(ident,stage,name,b,r,N=None):
    c=contract(stage,name,N)
    if ident.get('condition')!=name:raise Stop('identity condition mismatch')
    if not isinstance(b,int) or not 0<=b<100:raise Stop('invalid batch index')
    return dict(identity=ident,contract=c,batch=b,start=c['n']*b,end=c['n']*(b+1),outputs=base.describe(r))


def read_checkpoint(path,ident,stage,name,b,N=None):
    r=base.read_npz(path)
    try:meta=json.loads(str(r.pop('metadata')))
    except (KeyError,ValueError) as exc:raise Stop('missing checkpoint metadata') from exc
    validate(r,nspec=condition(name)['nspec']);base.require(health(r),'checkpoint health')
    if int(r['nspec'].sum())!=contract(stage,name,N)['n']:raise Stop('checkpoint history count')
    if meta!=checkpoint_meta(ident,stage,name,b,r,N):raise Stop('checkpoint identity/condition/seed/range/settings/hash mismatch')
    return r


def save_checkpoint(path,ident,stage,name,b,r,N=None):
    validate(r,nspec=condition(name)['nspec']);base.require(health(r),'save health')
    if int(r['nspec'].sum())!=contract(stage,name,N)['n']:raise Stop('save history count')
    if Path(path).exists():raise Stop('refuse checkpoint overwrite')
    base.save_npz(path,r,checkpoint_meta(ident,stage,name,b,r,N))


def resume(directory,ident,stage,name,N=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if any('pending' in p.name or p.suffix in ('.tmp','.part') for p in directory.iterdir()):raise Stop('unfinished checkpoint; retain and stop')
    config=dict(identity=ident,contract=contract(stage,name,N))
    path=directory/'config.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise Stop('resume config mismatch')
    else:base.write_json(path,config)
    found={}
    for path in directory.glob('batch_*.npz'):
        if not re.fullmatch(r'batch_\d{3}\.npz',path.name):raise Stop('invalid checkpoint name')
        b=int(path.stem[-3:]);read_checkpoint(path,ident,stage,name,b,N);found[b]=path
    return found


# ---- CPU/GPU comparisons (gates 2 and 3), generalised from P6 to the P7 layout ----

def c2(cpu,gpu,dref,conversion,nspec):
    """P6's c2: health, sources, RNG calls are pass/fail; region bin differences are report only."""
    n=len(cpu['steps']);validate(cpu,True,n,nspec);validate(gpu,True,n,nspec)
    a={x:cpu[x] for x in base.HISTORY_ORDER};b={x:gpu[x] for x in base.HISTORY_ORDER}
    checks=dict(cpu_health=health(cpu),gpu_health=health(gpu),source=base.c2_source(a,b),calls=base.c2_calls(a,b))
    delta=np.abs(cpu['sources']-gpu['sources'])
    checks['field_source']=base.verdict(bool(np.all(delta<=1e-13)),max_abs=float(delta.max(initial=0)))
    L=layout('p7');regions={}
    for index,(off,size) in enumerate(zip(L['offsets'],L['sizes'])):
        d=(gpu['regions'][:,off:off+size]-cpu['regions'][:,off:off+size])*conversion/L['volumes'][index]
        regions[str(index)]=dict(status='report',paired=base.paired(d,dref),
            bit_fraction=float(np.mean(base.bit_elements(cpu['regions'][:,off:off+size],gpu['regions'][:,off:off+size]))))
    return base.verdict(all(v['status']=='pass' for v in checks.values()),checks=checks,
                        legacy_bins=base.c2_bins(a,b,dref=dref,conversion=conversion),regions=regions)


def c3(histories,production,nspec,*,region_additions):
    n=len(histories['steps']);validate(histories,True,n,nspec);validate(production,False,nspec=nspec)
    old=base.c3({x:histories[x] for x in base.HISTORY_ORDER},{x:production[x] for x in base.RUN_ORDER},nspec)
    total=np.zeros((100,BINS));absolute=total.copy();rows=np.arange(n)*100//n
    np.add.at(total,rows,histories['regions']);np.add.at(absolute,rows,np.abs(histories['regions']))
    K=np.bincount(rows,minlength=100)[:,None]
    if region_additions.shape!=total.shape or region_additions.dtype!=np.uint64:raise Stop('2c requires per-bin addition counts')
    delta=np.abs(total-production['regions']);bound=(region_additions+K)*2**-52*absolute
    ledger=base.bits(histories['region_ledger'].sum(0),production['region_ledger'])
    finite=all(np.isfinite(v).all() for r in (histories,production) for v in r.values())
    failures=int(np.count_nonzero(~(delta<=bound)))
    return base.verdict(old['status']=='pass' and failures==0 and ledger and finite,legacy=old,region_exceeding=failures,ledger_exact=ledger,finite=finite)


def stacked(rows,stage,name,N=None):
    if len(rows)!=100:raise Stop('100 outer batches required')
    vals=[reduced(r) for r in rows];c=contract(stage,name,N)
    return dict(**{x:np.stack([v[x] for v in vals]) for x in vals[0]},N=np.array(c['n']*100),
                nbatch=np.array(100),seed=np.array(c['seed']),starts=np.arange(100,dtype=np.int64)*c['n'],
                ends=np.arange(1,101,dtype=np.int64)*c['n'],params=np.array(json.dumps(params(name),sort_keys=True)))


def c4(reference,actual,dref,name):
    """Gate 3: paired outer-batch differences (mean <= 0.5 %, SEM <= 0.2 % of Dref) on the legacy deep
    blocks, the axis-box 5 cm blocks and the condition's central profile means."""
    from .. import analysis_p7 as stats
    s=stats.spec(condition(name)['field_half']);n=contract('gate3',name)['n']
    left=stacked(reference,'gate3',name);right=stacked(actual,'gate3',name)
    ld,rd=stats.dose(left,s),stats.dose(right,s)
    legacy_left=np.stack([base.deep_blocks(r['batches'].sum(0),n,stats.MEV_TO_GY) for r in reference])
    legacy_right=np.stack([base.deep_blocks(r['batches'].sum(0),n,stats.MEV_TO_GY) for r in actual])
    box_left=np.stack([ld['axis_coarse'][:,lo:hi].mean(1) for lo,hi in stats.BLOCKS],axis=1)
    box_right=np.stack([rd['axis_coarse'][:,lo:hi].mean(1) for lo,hi in stats.BLOCKS],axis=1)
    fl,fr=stats.fold(ld['profile']),stats.fold(rd['profile'])
    profile_left=np.stack([fl[:,j,:s['central'][j]].mean(1) for j in range(3)],axis=1)
    profile_right=np.stack([fr[:,j,:s['central'][j]].mean(1) for j in range(3)],axis=1)
    quantities={};passed=True
    for label,a,b in (('legacy_deep',legacy_left,legacy_right),('axis_blocks',box_left,box_right),('profile_central',profile_left,profile_right)):
        p=base.paired(b-a,dref);ok=(np.abs(p['mean_pct'])<=.5)&(np.array(p['SEM_pct'])<=.2)
        quantities[label]=dict(p,passed=ok.tolist());passed &= bool(np.all(ok))
    spectra=all(base.bits(a['nspec'],b['nspec']) for a,b in zip(reference,actual))
    healthy=all(health(r)['status']=='pass' for r in reference+actual)
    return base.verdict(passed and spectra and healthy,quantities=quantities,spectra_exact=spectra,healthy=healthy)
