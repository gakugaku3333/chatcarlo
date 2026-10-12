"""Fixed B7/B8 checks and timing; run separately after the unit-test gate."""
import json
import os
from pathlib import Path
import time
import numpy as np
from numba import njit
from scipy.stats import chisquare
from . import run_p3a as a, run_p3s as s
from .batch_runner_p3u import run_batches
from . import batch_kernels_p3b as k
from .photon_transport_p3b import sample_field, run_coupled

CACHE=Path(os.environ.get('CHATCARLO_P3B_CACHE','~/.cache/chatcarlo/p3b')).expanduser()
OUT=a.ROOT/'docs/validation/mv/p3b/checks_p3b1.json'
N=2000000
SOURCE_SEED=20291740
TRANSPORT_SEED=20291741
TIMING_SEED=20291742
TIMING_N=50000
REGIONS=[0.,0.,1.,1.,9.,.25,.125,1.,5.,10.,20.]


@njit
def source_check(n,seed):
    np.random.seed(seed);counts=np.zeros((10,10),np.int64)
    for h in range(n):
        x,y,ux,uy,uz=sample_field(5.,100.)
        counts[min(9,int(x+5)),min(9,int(y+5))]+=1
    return counts


def params():
    return dict(E=10.,backend='elsepa',brems_mode=2,spectrum='mohan',
                field_half=5.,ssd=100.,dz=.25,dz_fine=.1,regions=REGIONS)


def symmetry(profile):
    centers=-9.+(np.arange(profile.shape[-1])+.5)*.25
    plus=(centers>0)&(centers<=4);minus=(centers<0)&(centers>=-4)
    xp=profile[:,0][:,:,plus].mean(axis=-1);xm=profile[:,0][:,:,minus].mean(axis=-1)
    yp=profile[:,1][:,:,plus].mean(axis=-1);ym=profile[:,1][:,:,minus].mean(axis=-1)
    # Nine contrasts of central means; normal approximation at 100 batches.
    differences=np.stack((xp-xm,yp-ym,(xp+xm-yp-ym)/2),axis=-1)
    sem=differences.std(axis=0,ddof=1)/np.sqrt(len(profile))
    z=np.divide(differences.mean(axis=0),sem,out=np.full_like(sem,np.inf),where=sem>0)
    return dict(z=[[float(v) if np.isfinite(v) else None for v in row] for row in z],sem=sem.tolist(),limit=2.77,
        status='合格' if np.all(np.abs(z)<2.77) else '不合格',
        scope='中央の領域 (|x| <= 4 cm) の平均の対称性。曲線全体や半影は判定しない。100バッチの正規近似。')


def relative_sem(values):
    mean=values.mean(axis=0);sem=values.std(axis=0,ddof=1)/np.sqrt(len(values))
    return dict(mean=mean.tolist(),sem=sem.tolist(),
        relative=np.divide(sem,mean,out=np.zeros_like(mean),where=mean!=0).tolist())


def main():
    if OUT.exists() or CACHE.exists():raise SystemExit('P3b checks/work already exists; no rerun permitted')
    CACHE.mkdir(parents=True)
    config=params();k.check_params(config,N//100)
    fingerprint=k.fingerprint(config)
    counts=source_check(N,SOURCE_SEED);p=float(chisquare(counts.ravel()).pvalue)
    edges,cum=k_spectrum(config)
    common=dict(annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
        brems_mode=2,score_kerma=False,score_pair=False,field_half=5.,ssd=100.,
        dz=.25,dz_fine=.1,**s.rule_kwargs('own'))
    inputs=s.inputs('elsepa')
    sources=run_coupled(10.,1000,SOURCE_SEED,.05,.025,30.,*inputs,record_sources=1000,**common)
    if np.any(sources[7]) or np.any(sources[20]):raise SystemExit('source transport ledger failed')
    state=sources[23];directions=np.column_stack((state[:,0],state[:,1],np.full(1000,100.)))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    direction_error=float(np.max(np.abs(directions-state[:,3:6])))
    b7=dict(N=N,seed=SOURCE_SEED,p=p,counts=counts.tolist(),direction_error=direction_error,
        status='合格' if p>.01 and direction_error<=1e-14 and np.all(np.abs(state[:,:2])<=5.) else '不合格')
    result=dict(B7=b7,fingerprint=fingerprint)
    if b7['status']!='合格':
        a.write(OUT,result);raise SystemExit('B7 failed; no rerun permitted')
    row=a.load(run_batches(CACHE/'symmetry','p3b_field',config,N,TRANSPORT_SEED,2,nbatch=100))
    result['B8']=dict(symmetry(row['profile']),N=N,seed=TRANSPORT_SEED)
    blocks=row['axis_coarse'].reshape(100,6,20).sum(axis=2)
    centers=-9.+(np.arange(row['profile'].shape[-1])+.5)*.25
    central=row['profile'][...,np.abs(centers)<=4].mean(axis=-1)
    result['precision']=dict(axis_5cm=relative_sem(blocks),central_profiles=relative_sem(central))
    if result['B8']['status']!='合格':
        a.write(OUT,result);raise SystemExit('B8 failed; no rerun permitted')
    timings={}
    from .photon_transport_p3t import run_coupled as slab
    slab_common={key:value for key,value in common.items() if key not in ('field_half','ssd','dz','dz_fine')}
    for name,call,kw in (('p3t_slab',slab,slab_common),('field_without_regions',run_coupled,common),
                         ('field_with_regions',run_coupled,dict(common,regions=REGIONS))):
        call(10.,100,TIMING_SEED,.05,.025,30.,*inputs,**kw)  # exclude JIT warmup, as P3u
        start=time.perf_counter()
        values=call(10.,TIMING_N,TIMING_SEED,.05,.025,30.,*inputs,**kw)
        seconds=time.perf_counter()-start
        if np.any(values[7]) or (len(values)>15 and np.any(values[20])):raise SystemExit('timing ledger failed')
        timings[name]=dict(N=TIMING_N,seed=TIMING_SEED,seconds=seconds,us_per_history=seconds/TIMING_N*1e6)
    ratio=timings['field_with_regions']['seconds']/timings['field_without_regions']['seconds']
    if k.fingerprint(config)!=fingerprint:raise SystemExit('fingerprint changed during checks')
    result['timing']=dict(runs=timings,region_ratio=ratio,report_before_p3b2=ratio>1.5)
    a.write(OUT,result);print(json.dumps(result,ensure_ascii=False,indent=2))


def k_spectrum(config):
    from .batch_kernels_p3u import spectrum
    return spectrum(config)


if __name__=='__main__':main()
