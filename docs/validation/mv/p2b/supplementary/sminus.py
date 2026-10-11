"""追補2: run the registered S- job and tabulate the three conditions (descriptive)."""
import json
from pathlib import Path
import numpy as np
from prototypes.mv_electron import reference_p2b as m
mode,noflight,seed=m.RUNS['p2b_Sminus'];m.run('p2b_Sminus',m.N,seed,mode,noflight)
lock=json.loads((m.P2/'pilot_lock.json').read_text())['windows']['6.0'];D=lock['Dref'];deep=lock['deep']
data={n:dict(np.load(m.OUT/'egs5'/n/'result.npz')) for n in m.RUNS};summ={n:json.loads((m.OUT/'egs5'/n/'summary.json').read_text()) for n in m.RUNS}
regions=[('buildup','fine',m.BUILD),('window','coarse',lock['near'])]+[(f'deep_{b[0]}_{b[-1]}','coarse',b) for b in [deep[i:i+25] for i in range(0,100,25)]+[deep[100:]]]
out=dict(status={n:summ[n]['status'] for n in m.RUNS},rows=[])
for l,r in (('p2b_Sminus','p2b_Splus'),('p2b_Cminus','p2b_Sminus'),('p2b_Cminus','p2b_Splus')):
    for name,key,bins in regions:
        d,s=m.region_stats(data[l],data[r],bins,key,D);out['rows'].append(dict(pair=f'{l} - {r}',region=name,delta_pct_Dref=d,SEM_pct_Dref=s))
out['deposit_ratio_to_Splus']={n:summ[n]['ledger'][1]/summ['p2b_Splus']['ledger'][1] for n in m.RUNS}
out['callbacks']={n:summ[n]['callbacks'] for n in m.RUNS}
(m.OUT/'supplementary/sminus.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for r in out['rows']:print(f"{r['pair']:26s} {r['region']:14s} {r['delta_pct_Dref']:+.3f} ± {r['SEM_pct_Dref']:.3f}")
print(out['deposit_ratio_to_Splus'],out['status'])

# Wider regions, chosen after seeing the block table: surface to 10.8 cm, the rest, and the whole slab.
wide=[]
for l,r in (('p2b_Sminus','p2b_Splus'),('p2b_Cminus','p2b_Sminus'),('p2b_Cminus','p2b_Splus')):
    for name,bins in (('0-10.8cm',list(range(0,54))),('10.8-30cm',list(range(54,150))),('whole slab',list(range(150)))):
        d,s=m.region_stats(data[l],data[r],bins,'coarse',D);ref=float(data['p2b_Splus']['coarse'][:,bins].mean()/D*100)
        wide.append(dict(pair=f'{l} - {r}',region=name,delta_pct_Dref=d,SEM_pct_Dref=s,delta_pct_of_local_dose=d/ref*100,SEM_pct_of_local_dose=s/ref*100))
out['wide']=wide
(m.OUT/'supplementary/sminus.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for r in wide:print(f"{r['pair']:26s} {r['region']:11s} {r['delta_pct_Dref']:+.3f} ± {r['SEM_pct_Dref']:.3f} %Dref   {r['delta_pct_of_local_dose']:+.3f} ± {r['SEM_pct_of_local_dose']:.3f} % of local dose")
