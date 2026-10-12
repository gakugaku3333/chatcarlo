"""P3b-1 amendment 1, diagnostic D1: half-space asymmetry of deposits (not a B8 re-judgment).

Usage: python -m prototypes.mv_electron.run_p3b1_diag run PART   (PART = 0 or 1; odd/even chunks)
       python -m prototypes.mv_electron.run_p3b1_diag analyse
Chunks are saved outside the repository; nothing is recomputed if a chunk file exists.
"""
import json,os,sys
from pathlib import Path
import numpy as np
from . import run_p3a as a, run_p3s as s
from .photon_transport_p3b import run_coupled
from .batch_kernels_p3u import spectrum

CACHE=Path(os.environ.get('CHATCARLO_P3B_CACHE','~/.cache/chatcarlo/p3b')).expanduser()/'diag_symmetry'
OUT=a.ROOT/'docs/validation/mv/p3b/diag_symmetry.json'
SEED=20291750;CHUNKS=1000;N=2000;CAPACITY=6000000;HALF=4.;LIMIT=2.87


def chunk(seed):
    edges,cum=spectrum(dict(spectrum='mohan'))
    v=run_coupled(10.,N,seed,.05,.025,30.,*s.inputs('elsepa'),annihilate_in_flight=True,
        spectrum_edges=edges,spectrum_cum=cum,brems_mode=2,score_kerma=False,score_pair=False,
        field_half=5.,ssd=100.,record_deposits=CAPACITY,**s.rule_kwargs('own'))
    if np.any(v[7]) or int(v[22])!=0:raise SystemExit('ledger violation or record overflow')
    r=v[21];mid=(r[:,2:5]+r[:,5:8])/2;loss=r[:,8]
    inside=(np.abs(mid[:,0])<=HALF)&(np.abs(mid[:,1])<=HALF)
    band=np.clip((mid[:,2]/5.).astype(np.int64),0,5)
    sums=np.zeros((6,4))
    for j,mask in enumerate((mid[:,0]>0,mid[:,0]<0,mid[:,1]>0,mid[:,1]<0)):
        sums[:,j]=np.bincount(band[inside&mask],weights=loss[inside&mask],minlength=6)
    return sums,np.array([loss.sum(),(loss*mid[:,0]).sum(),(loss*mid[:,1]).sum(),len(r)])


def run(part):
    CACHE.mkdir(parents=True,exist_ok=True)
    seeds=[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(SEED).spawn(CHUNKS)]
    for i in range(part,CHUNKS,2):
        target=CACHE/f'chunk_{i:04}.npz'
        if target.exists():continue
        sums,moments=chunk(seeds[i]);a.save(target,sums=sums,moments=moments,seed=seeds[i])
        if i%50==part:print(f'chunk {i}',flush=True)


def analyse():
    rows=[a.load(CACHE/f'chunk_{i:04}.npz') for i in range(CHUNKS)]
    sums=np.stack([r['sums'] for r in rows]);moments=np.stack([r['moments'] for r in rows])
    result=dict(N=N*CHUNKS,seed=SEED,chunks=CHUNKS,half_width_cm=HALF,limit=LIMIT,bands=[]);worst=0.
    for b in range(6):
        row=dict(depth_cm=[5*b,5*b+5])
        for name,j in (('x',0),('y',2)):
            plus=sums[:,b,j];minus=sums[:,b,j+1];total=(plus+minus).mean()
            value=(plus-minus).mean()/total
            # Ratio estimator: linearised per-chunk residuals.
            resid=((plus-minus)-value*(plus+minus))/total
            sem=resid.std(ddof=1)/np.sqrt(CHUNKS);z=value/sem;worst=max(worst,abs(z))
            row[name]=dict(asymmetry_pct=value*100,sem_pct=sem*100,z=z)
        result['bands'].append(row)
    total=moments[:,0].mean()
    for name,j in (('mean_x_cm',1),('mean_y_cm',2)):
        value=moments[:,j].mean()/total;resid=(moments[:,j]-value*moments[:,0])/total
        result[name]=dict(value=value,sem=resid.std(ddof=1)/np.sqrt(CHUNKS))
    result['records_per_history']=float(moments[:,3].mean()/N)
    result['max_abs_z']=worst
    result['status']='偏りは検出されない' if worst<LIMIT else '偏りがある'
    a.write(OUT,result);print(json.dumps(result,ensure_ascii=False,indent=1))


if __name__=='__main__':
    run(int(sys.argv[2])) if sys.argv[1]=='run' else analyse()
