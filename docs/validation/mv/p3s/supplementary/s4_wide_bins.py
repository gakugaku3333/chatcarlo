"""S4 data seen in 2 mm groups (4 fine bins). Made after S4 returned 統計不足; descriptive only."""
import glob,json,os,re
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]/'check_runs';runs={}
for d in sorted(glob.glob(str(R/'*s4_*'))):
    m=re.search(r's4_(\d+)_(\w+)$',d)
    if m and os.path.exists(d+'/result.npz'):runs[(int(m[1]),m[2])]=np.load(d+'/result.npz')
out=[]
for T in (2,6,10):
    c=runs[(T,'const')]['fine'];D=c.mean(0).max();bins=np.where(c.mean(0)>=.01*D)[0];g=bins[:(len(bins)//4)*4].reshape(-1,4)
    for rule in ('own','egs'):
        r=runs[(T,rule)]['fine'];x=np.stack([r[:,q].mean(1) for q in g],1);y=np.stack([c[:,q].mean(1) for q in g],1)
        dl=(x.mean(0)-y.mean(0))/D*100;se=np.sqrt(x.var(0,ddof=1)/100+y.var(0,ddof=1)/100)/D*100
        a=r[:,bins].mean(1);b=c[:,bins].mean(1)
        out.append(dict(T=T,rule=rule,groups=len(g),max_abs_delta_pct=float(abs(dl).max()),max_3SEM_pct=float(3*se.max()),max_abs_z=float(abs(dl/se).max()),
            chi2_per_group=float(np.sum((dl/se)**2)/len(g)),whole_delta_pct=float((a.mean()-b.mean())/b.mean()*100),whole_SEM_pct=float(np.hypot(a.std(ddof=1),b.std(ddof=1))/10/b.mean()*100)))
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
for o in out:print(o)
