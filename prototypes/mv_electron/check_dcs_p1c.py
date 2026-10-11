"""N2/N3, fixed Gauss orders; variant data are never transport acceptance inputs."""
import re
import numpy as np
from numpy.polynomial.legendre import leggauss
from .dcs import moments,get_backend
from .mf import McKinleyFeshbach,born_g1
from .physics import elastic
from .regression_p1c import P,ROOT,write
ENERGIES=[.01,.02,.1,1.,2.,10.]
def back(provider,T,order=32):
    # Always split exactly at y=1; independent backward hemisphere integral.
    edges=provider.edges(T) if hasattr(provider,'edges') else np.r_[0,np.geomspace(min(elastic(T)[0])*1e-5,2,606)]
    cuts=np.unique(np.r_[1,edges[(edges>1)&(edges<2)],2]);x,w=leggauss(order)
    y=(cuts[:-1,None]+cuts[1:,None])/2+(cuts[1:,None]-cuts[:-1,None])*x/2
    return float(np.sum(provider.density(T,y)*w*(cuts[1:,None]-cuts[:-1,None])/2))
def interpolation_midpoints(g):
    grids=[g['single'][:,0],g['T']]
    mid=np.unique(np.concatenate([np.sqrt(ts[:-1]*ts[1:]) for ts in grids]))
    return mid[(mid>.01)&(mid<.02)]

def inputs(extra=()):
    b=McKinleyFeshbach();born=McKinleyFeshbach('born_only');rows=[]
    # Midpoints of the actual converged runtime single/rate energy table.
    for T in sorted(set(ENERGIES+list(map(float,extra)))):
        y,w=b.quadrature(T);G,total,_=moments(b,T);H,t,_=moments(b,T,order=32)
        rel=float(max(abs(np.r_[total,G[1:]]/np.r_[t,H[1:]]-1)))
        numeric=moments(born,T)[0][1];closed=born_g1(T);err=float(abs(numeric/closed-1))
        minval=float(min(b.density(T,np.unique(np.r_[0,y,np.linspace(0,2,10001),2]))))
        rows.append(dict(T=T,minimum_density=minval,doubling_max_relative=rel,born_numeric_G1=float(numeric),born_closed_G1=closed,born_relative=err,pass_=minval>0 and rel<=1e-6 and err<=1e-6))
        rows[-1]['pass']=rows[-1].pop('pass_')
    r={'status':'pass' if all(x['pass'] for x in rows) else 'fail','rows':rows,'positivity_reason':'electron R >= 1-beta² > 0; B>0 and denominator>0','midpoint_energies':list(map(float,extra))};write('n2.json',r);return r

def scattering():
    rows=[]
    for name,b in [('rutherford',get_backend('rutherford')),('mf',McKinleyFeshbach()),('mf_nuclear_only',McKinleyFeshbach('nuclear_only')),('mf_born_only',McKinleyFeshbach('born_only')),('dcslib',get_backend('dcslib'))]:
        for T in ENERGIES:
            G,total,_=moments(b,T,order=32)
            rows.append(dict(backend=name,T=T,total_cm1=total,G1_cm1=float(G[1]),G2_cm1=float(G[2]),back_cm1=back(b,T)))
    # Read preregistered approximate ratios from approved plan, not hand-entered results.
    plan=(ROOT/'docs/ai/plans/2026-10-08-mv-p1c-mott-correction.md').read_text()
    table=plan.split('前提事実4の表',1)[1].split('この表は見込み',1)[0];checks=[]
    for line in table.splitlines():
        cells=[v.strip() for v in line.split('|')[1:-1]]
        if not cells or not re.fullmatch(r'[\d.]+',cells[0]):continue
        T=float(cells[0]);mf=next(r for r in rows if r['backend']=='mf' and r['T']==T);dc=next(r for r in rows if r['backend']=='dcslib' and r['T']==T)
        for field,c in zip(['G1_cm1','G2_cm1','back_cm1'],cells[1:]):
            expected=float(c.split()[0]);actual=mf[field]/dc[field];err=abs(actual/expected-1)
            checks.append(dict(T=T,quantity=field,preregistered=expected,actual=actual,relative=err,within_1e3=err<=1e-3))
    r={'status':'record-only','rows':rows,'precalculation_crosscheck':checks,'note':'back integrated separately on [1,2]; P1b M3 used full-angle Gauss nodes y>1, so its recorded values are not replaced.'};write('n3.json',r);return r
