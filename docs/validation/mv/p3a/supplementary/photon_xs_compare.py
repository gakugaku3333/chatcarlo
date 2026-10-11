"""Direct comparison of the photon data the two codes transport with (no Monte Carlo).

EGS5: piecewise-linear-in-ln(E) fits stored in pgs5job.pegs5dat (GMFP, GBR1, GBR2),
read in the order of egs5_hatch.f lines 435-461. Prototype: photon_data_p2.coefficients
(analytic Klein-Nishina + XCOM photoelectric and pair).
"""
import json,sys
from pathlib import Path
import numpy as np
from prototypes.mv_electron.photon_data_p2 import TABLE,coefficients
work=Path(sys.argv[1]).expanduser()
lines=(work/'pgs5job.pegs5dat').read_text().splitlines()
rlc=float(lines[4].split()[0]);flags=list(map(int,lines[5].split()));nge=flags[1];neke=flags[3]
values=[]
for line in lines[18:]:
    try:values.extend(map(float,line.split()))
    except ValueError:break
    if len(values)>=neke*30+3+nge*6:break
assert len(values)>=neke*30+3+nge*6
ge0,ge1=values[neke*30+1:neke*30+3];g=np.array(values[neke*30+3:neke*30+3+nge*6]).reshape(nge,6)
p=np.load(TABLE);rows=[]
for E in np.geomspace(.02,9.99,400):
    x=np.log(E);j=max(0,min(nge-1,int(ge1*x+ge0)-1))
    mfp=(g[j,0]+g[j,1]*x)*rlc;b1=g[j,2]+g[j,3]*x;b2=g[j,4]+g[j,5]*x
    mu=1/mfp;c=coefficients(float(E),p['E'],p['water'],p['thresholds'])
    rows.append((E,mu,float(np.sum(c)),mu*b1,float(c[2]+c[3]),mu*(b2-b1),float(c[0]),mu*(1-b2),float(c[1])))
r=np.array(rows);np.save(Path(__file__).with_name('photon_xs_compare.npy'),r)
out=[]
for E in (.05,.1,.2,.5,1,1.25,1.5,2,2.5,3,4,5,6,8,9.9):
    i=np.argmin(abs(r[:,0]-E));q=r[i]
    out.append(dict(E=float(q[0]),total_pct=float((q[2]/q[1]-1)*100),pair_pct=float((q[4]/q[3]-1)*100) if q[3]>0 else None,
                    compton_pct=float((q[6]/q[5]-1)*100),pair_share_pct=float(q[3]/q[1]*100),photo_share_pct=float(q[7]/q[1]*100)))
    o=out[-1];print(f"{o['E']:6.3f} MeV  total {o['total_pct']:+.3f}%  Compton {o['compton_pct']:+.3f}%  pair "+(f"{o['pair_pct']:+.2f}%" if o['pair_pct'] is not None else '  -  ')+f"  (pair share {o['pair_share_pct']:.2f}%, photo share {o['photo_share_pct']:.3f}%)")
Path(__file__).with_name('photon_xs_compare.json').write_text(json.dumps(dict(note='prototype / EGS5 - 1, per cent',nge=nge,rows=out,max_abs_total_pct=float(np.max(abs(r[:,2]/r[:,1]-1))*100)),indent=2)+'\n')
print('nge',nge,'max |total diff| %',np.max(abs(r[:,2]/r[:,1]-1))*100,'at',r[np.argmax(abs(r[:,2]/r[:,1]-1)),0])
