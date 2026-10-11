"""P3a 追補1: one more EGS5 run with a new seed, then descriptive region comparisons."""
import json
from pathlib import Path
import numpy as np
from prototypes.mv_electron import run_p3a as m
m.require_pilot();edges,prob,cum=m.load_spectrum()
m.run_egs('p3a_egs_seed2',N=10000000,seed=20291422,edges=edges,cum=cum)
R=m.OUT
proto=np.load(R/'runs/p3a_proto/result.npz');e1=np.load(R/'egs5/p3a_egs/result.npz');e2=np.load(R/'egs5/p3a_egs_seed2/result.npz')
def region(a,b,bins):
    x=a[:,bins].mean(1);y=b[:,bins].mean(1);d=y.mean()
    return dict(delta_pct_local=float((x.mean()-y.mean())/d*100),SEM_pct_local=float(np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))/d*100))
mean=(e1['coarse']+e2['coarse'])/2  # batch-wise average of two independent runs
sets=dict(new_minus_old_EGS5=(e2['coarse'],e1['coarse']),proto_minus_old_EGS5=(proto['coarse'],e1['coarse']),
          proto_minus_new_EGS5=(proto['coarse'],e2['coarse']),proto_minus_mean_EGS5=(proto['coarse'],mean))
regions={'0-15cm':slice(0,75),'15-30cm':slice(75,150),'whole':slice(0,150)}
lock=json.loads((R/'pilot_lock.json').read_text())
for k,b in enumerate(lock['blocks']):regions[f'block{k+1}_{b[0]}_{b[-1]}']=b
out={k:{r:region(a,b,bins) for r,bins in regions.items()} for k,(a,b) in sets.items()}
def dep_ratio(r):
    d=r['coarse'].sum(1);i=r['batchin'];q=d.sum()/i.sum();return q,np.std((d-q*i)/i.mean(),ddof=1)/10
q1,s1=dep_ratio(e1);q2,s2=dep_ratio(e2)
out['EGS5_deposit_over_incoming_new_over_old_minus_1_pct']=dict(value=float((q2/q1-1)*100),SEM=float(np.hypot(s1/q1,s2/q2)*100))
out['mean_incident_MeV']=dict(old=float(e1['batchin'].sum()/1e7),new=float(e2['batchin'].sum()/1e7))
out['pairs']=dict(old=float(e1['batchcb'][:,0].sum()),new=float(e2['batchcb'][:,0].sum()))
(R/'supplementary/seed2.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for k in sets:
    print(k,' '.join(f"{r}: {v['delta_pct_local']:+.3f}±{v['SEM_pct_local']:.3f}" for r,v in out[k].items()))
print(out['EGS5_deposit_over_incoming_new_over_old_minus_1_pct'],out['mean_incident_MeV'],out['pairs'])
