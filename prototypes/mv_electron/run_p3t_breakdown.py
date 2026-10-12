"""P3t: which of the two removed operations cost how much.

Builds, in memory, the P3r loop with only the unused event_order call removed
("minus a"), and times it between the frozen P3r and P3t. No file is modified.
Usage: python -m prototypes.mv_electron.run_p3t_breakdown [reps] [N]
Writes docs/validation/mv/p3t/breakdown.json.
"""
import hashlib,importlib.util,json,sys,time
import numpy as np
from . import run_p3a as p3a, run_p3s as p3s
from . import photon_transport_p3r as r, photon_transport_p3t as t
from .run_p3t import OUT,SEED
LINE='                priority=event_order(bb,eb,mb,hb,rb,ab)\n'

def minus_a():
    source=(p3a.ROOT/'prototypes/mv_electron/photon_transport_p3r.py').read_text()
    if source.count(LINE)!=1:raise RuntimeError('P3r source changed')
    name='prototypes.mv_electron._p3r_minus_a'
    module=importlib.util.module_from_spec(importlib.util.spec_from_loader(name,loader=None))
    module.__package__='prototypes.mv_electron';sys.modules[name]=module
    exec(compile(source.replace(LINE,''),name,'exec'),module.__dict__)
    return module

def main(reps=3,N=50000):
    edges,prob,cum=p3a.load_spectrum();args=p3s.inputs('elsepa');a=minus_a()
    kw=dict(annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
            score_kerma=False,score_pair=False,**p3s.rule_kwargs('own'))
    settings={'off':dict(disable_brems=True),'local':dict(brems_mode=1),'transport':dict(brems_mode=2)}
    codes={'P3r':r,'P3r minus a':a,'P3t (minus a and b)':t}
    arms={(c,s):(lambda n,f=m.run_coupled,e=extra:f(10.,n,SEED,.05,.025,30.,*args,**kw,**e))
          for s,extra in settings.items() for c,m in codes.items()}
    for f in arms.values():f(100)
    times={k:[] for k in arms};digest={}
    for i in range(reps):
        for k in (list(arms) if i%2==0 else list(arms)[::-1]):
            t0=time.perf_counter();out=arms[k](N);times[k].append((time.perf_counter()-t0)/N*1e6)
            h=hashlib.sha256()
            for x in out:
                x=np.asarray(x);h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
            digest[k]=h.hexdigest()
    # Stop rule: no timing unless the three codes give byte-identical outputs in each setting.
    for s in settings:
        if len({digest[(c,s)] for c in codes})!=1:raise SystemExit('identity failed: '+s)
    rows={}
    for s in settings:
        best={c:min(times[(c,s)]) for c in codes}
        rows[s]=dict(us_per_history={c:[round(x,1) for x in times[(c,s)]] for c in codes},
                     best={c:round(v,1) for c,v in best.items()},
                     cost_a=round(best['P3r']-best['P3r minus a'],1),
                     cost_b=round(best['P3r minus a']-best['P3t (minus a and b)'],1),
                     output_sha256=digest[('P3r',s)])
    result=dict(N=N,seed=SEED,reps=reps,note='single process, nine arms interleaved, order reversed on odd repetitions; '
                'a = unused per-step event_order call, b = duplicated bremsstrahlung table lookup',settings=rows)
    (OUT/'breakdown.json').write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=1))

if __name__=='__main__':
    main(*(int(x) for x in sys.argv[1:3]))
