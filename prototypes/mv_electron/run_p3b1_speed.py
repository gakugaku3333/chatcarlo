"""P3b-1 amendment 2: output hashes (S1) and interleaved timing (S3).

Usage: python -m prototypes.mv_electron.run_p3b1_speed reference   (once, before the change)
       python -m prototypes.mv_electron.run_p3b1_speed             (after the change)
"""
import hashlib,json,sys,time
import numpy as np
from . import run_p3a as a, run_p3s as s
from .batch_kernels_p3u import spectrum
from .photon_transport_p3b import run_coupled
from .photon_transport_p3t import run_coupled as slab

REFERENCE=a.CODE/'tests/p3b1_reference_hashes.json'
OUT=a.ROOT/'docs/validation/mv/p3b/speed_p3b1.json'
REGIONS=[0.,0.,1.,1.,9.,.25,.125,1.,5.,10.,20.]
SHIFTED=[1.3,-.7,1.,1.,9.,.25,.125,1.,5.,10.,20.]
N=50000;SEED=20291760;REPS=3;LIMIT=1.5


def common():
    edges,cum=spectrum(dict(spectrum='mohan'))
    return dict(annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,brems_mode=2,
                score_kerma=False,score_pair=False,**s.rule_kwargs('own'))


def digest(values):
    out=[]
    for v in values:
        v=np.asarray(v);h=hashlib.sha256(str((v.dtype,v.shape)).encode());h.update(v.tobytes());out.append(h.hexdigest())
    return out


def hashes():
    field=dict(field_half=5.,ssd=100.,dz=.25,dz_fine=.1)
    return dict(
        production=digest(run_coupled(10.,2000,20291761,.05,.025,30.,*s.inputs('elsepa'),**common(),**field,
                                      regions=REGIONS,record_deposits=3000000,record_sources=2000)),
        shifted_mf_mode1=digest(run_coupled(10.,2000,20291762,.05,.025,30.,*s.inputs('mf'),annihilate_in_flight=True,
            brems_mode=1,score_kerma=False,score_pair=False,**s.rule_kwargs('own'),**field,regions=SHIFTED,
            source_offset=(1.3,-.7),record_deposits=3000000)))


def main(mode='measure'):
    if mode=='reference':
        if REFERENCE.exists():raise SystemExit('reference hashes already exist')
        a.write(REFERENCE,hashes());return
    reference=json.loads(REFERENCE.read_text());now=hashes()
    identity={k:now[k]==reference[k] for k in reference}
    # Stop rule: no timing unless every output hash is unchanged.
    if not all(identity.values()):raise SystemExit('S1 failed: '+json.dumps(identity))
    kw=common();inputs=s.inputs('elsepa');field=dict(field_half=5.,ssd=100.,dz=.25,dz_fine=.1)
    arms={'p3t_slab':lambda n:slab(10.,n,SEED,.05,.025,30.,*inputs,**kw),
          'field_without_regions':lambda n:run_coupled(10.,n,SEED,.05,.025,30.,*inputs,**kw,**field),
          'field_with_regions':lambda n:run_coupled(10.,n,SEED,.05,.025,30.,*inputs,**kw,**field,regions=REGIONS)}
    for f in arms.values():f(100)
    times={k:[] for k in arms}
    for i in range(REPS):
        for k in (list(arms) if i%2==0 else list(arms)[::-1]):
            t=time.perf_counter();v=arms[k](N);times[k].append((time.perf_counter()-t)/N*1e6)
            if np.any(v[7]) or (len(v)>15 and np.any(v[20])):raise SystemExit('ledger violation')
    best={k:min(v) for k,v in times.items()};ratio=best['field_with_regions']/best['field_without_regions']
    result=dict(N=N,seed=SEED,reps=REPS,S1=identity,us_per_history=times,best=best,
                S3=dict(regions_over_no_regions=ratio,limit=LIMIT,status='合格' if ratio<=LIMIT else '不合格'),
                regions_over_p3t_slab=best['field_with_regions']/best['p3t_slab'],
                source_sha256={x:a.sha(a.CODE/x) for x in ('photon_transport_p3b.py','scoring_p3b.py','photon_transport_p3t.py')})
    a.write(OUT,result);print(json.dumps(result,ensure_ascii=False,indent=1))


if __name__=='__main__':main(*sys.argv[1:2])
