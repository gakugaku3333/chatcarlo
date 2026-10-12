"""P3t: bit-identity against P3r/P3s and interleaved timing on the Mohan 10 MV slab.

Usage: python -m prototypes.mv_electron.run_p3t [reps] [N]
Writes docs/validation/mv/p3t/timing.json. The spectrum is read at run time (untracked file).
"""
import hashlib,json,sys,time,platform
import numba
import numpy as np
from . import run_p3a as p3a, run_p3s as p3s
from . import photon_transport_p3s as s, photon_transport_p3r as r, photon_transport_p3t as t
OUT=p3a.ROOT/'docs/validation/mv/p3t'
SEED=777
SOURCES=('photon_transport_p3s.py','photon_transport_p3r.py','photon_transport_p3t.py','bremsstrahlung_p3r.py','physics.py','run_p3t.py')

def main(reps=3,N=50000):
    edges,prob,cum=p3a.load_spectrum();args=p3s.inputs('elsepa')
    kw=dict(annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
            score_kerma=False,score_pair=False,**p3s.rule_kwargs('own'))
    def arm(f,**extra):
        return lambda n:f(10.,n,SEED,.05,.025,30.,*args,**kw,**extra)
    arms={'P3s':arm(s.run_coupled),
          'P3r off':arm(r.run_coupled,disable_brems=True),'P3t off':arm(t.run_coupled,disable_brems=True),
          'P3r local':arm(r.run_coupled,brems_mode=1),'P3t local':arm(t.run_coupled,brems_mode=1),
          'P3r transport':arm(r.run_coupled,brems_mode=2),'P3t transport':arm(t.run_coupled,brems_mode=2)}
    for f in arms.values():f(100)  # compile
    times={k:[] for k in arms};last={}
    for i in range(reps):
        for k in (list(arms) if i%2==0 else list(arms)[::-1]):
            t0=time.perf_counter();last[k]=arms[k](N);times[k].append((time.perf_counter()-t0)/N*1e6)
    def equal(a,b,n=None):
        # Bytes, dtype and shape: array_equal alone would accept +0.0 against -0.0.
        if n is None and len(a)!=len(b):return False
        for x,y in zip(a[:n],b[:n]):
            x=np.asarray(x);y=np.asarray(y)
            if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():return False
        return True
    identity={'P3t off == P3s (13 common outputs)':equal(last['P3s'],last['P3t off'],13),
              'P3t off == P3r off':equal(last['P3r off'],last['P3t off']),
              'P3t local == P3r local':equal(last['P3r local'],last['P3t local']),
              'P3t transport == P3r transport':equal(last['P3r transport'],last['P3t transport'])}
    # Stop rule: no timing is reported or saved unless every output is identical.
    if not all(identity.values()):raise SystemExit('identity failed: '+json.dumps(identity))
    rows={k:dict(us_per_history=[round(x,1) for x in v],best=round(min(v),1),
                 charged_steps_per_history=float(last[k][8][0]/N),
                 brems_per_history=(float(last[k][13][0]/N) if len(last[k])>13 else None)) for k,v in times.items()}
    ratio=min(times['P3t off'])/min(times['P3s'])
    code=p3a.ROOT/'prototypes/mv_electron'
    result=dict(N=N,seed=SEED,reps=reps,machine=platform.machine(),python=platform.python_version(),
                numpy=np.__version__,numba=numba.__version__,
                note='single process, arms interleaved, order reversed on odd repetitions; identity is a byte comparison',
                source_sha256={x:hashlib.sha256((code/x).read_bytes()).hexdigest() for x in SOURCES},
                identity=identity,timing=rows,
                T4=dict(off_over_P3s=round(ratio,4),limit=1.03,status='合格' if ratio<=1.03 else '不合格'))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'timing.json').write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=1))
    if ratio>1.03:raise SystemExit('T4 failed')

if __name__=='__main__':
    main(*(int(x) for x in sys.argv[1:3]))
