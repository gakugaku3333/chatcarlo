"""Follow-up on the bremsstrahlung count/energy gap (prototype − EGS5: −0.69%, −1.23%). Changes no verdict.
(a) discrete-emission rate: prototype table vs the EGS5 formula with EGS5's own per-element correction (read at run time
    from the EGS5 data file; no value is stored here).
    The piecewise-linear tables EGS5 uses at run time were also read from the PEGS output, but their total discrete
    cross section does not match the Moller rate below ~2 MeV (10.2 vs 19.2 /cm at 0.1 MeV), so they are not plain
    cross sections and the comparison was dropped as uninterpretable (see DISCUSSION section 4).
(b) share of the bremsstrahlung energy that comes from pair-produced electrons and positrons (first-interaction estimate).
Run from the repository root: PYTHONPATH=. .venv/bin/python docs/validation/mv/p3r/supplementary/brems_gap.py"""
import json,re
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import prototypes.mv_electron.run_p3r as R
from prototypes.mv_electron import bremsstrahlung_p3r as B
from prototypes.mv_electron.physics import M,RE,ALPHA,Z,NJ
from prototypes.mv_electron.photon_data_p2 import coefficients
EGS=R.EGS
num=lambda t:[float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?',t.replace('D','E'))]
# --- EGS5 correction factor (PEGS aprim, IAPRIM=1): log interpolation in Z, linear in total energy / m.
v=num((EGS/'data/aprime.data').read_text());nz,ne=int(v[0]),int(v[1]);en=1+np.array(v[2:2+ne])/M
rows=np.array(v[2+ne:2+ne+nz*(ne+1)]).reshape(nz,ne+1);zp=rows[:,0];tab=rows[:,1:]
def aprim(z,E):
    az=np.array([np.interp(np.log(z),np.log(zp),tab[:,i]) for i in range(ne)]);return float(np.interp(E/M,en,az))
def element(j,T,k):  # k dSigma/dk of one element, PEGS brmsfz without the correction
    z=Z[j];E=T+M;rem=E-k;y=rem/E;a2=(ALPHA*z)**2
    fc=a2*(1/(1+a2)+.20206+a2*(-.0369+a2*(.0083-.002*a2)))
    xi=6.144/(5.31-fc) if z==1 else np.log(1194*z**(-2/3))/(np.log(184.15*z**(-1/3))-fc)
    xl=4/3*np.log(z);d=136*z**(-1/3)*M/E*k/rem
    if d>=np.exp((21.12-xl)/4.184)-.952:return 0.
    if d<=1:s1=20.867+d*(-3.242+.625*d)-xl;s2=20.209+d*(-1.930-.086*d)-xl
    else:s1=s2=21.12-4.184*np.log(d+.952)-xl
    return NJ[j]*RE**2*ALPHA*z*(z+xi)*((1+y*y)*s1-2/3*y*s2)
def egs_formula(T):
    return sum(aprim(Z[j],T+M)*quad(lambda u:element(j,T,np.exp(u)),np.log(.01),np.log(T),limit=200)[0] for j in range(len(Z)))
e,vals=B.tables();rate=[]
for T in (.05,.1,.2,.5,1.,2.,3.,5.,7.,10.):
    p=B.quantities(T,e,vals)[0];f=egs_formula(T)
    rate.append(dict(T=T,prototype=p,EGS5_formula=f,prototype_vs_formula_pct=100*(p/f-1)))
out=dict(rate=rate)
# --- (b) pair share: first interactions of the Mohan spectrum in 30 cm; each of e-/e+ takes half of E-2m;
#     radiated energy = T x (CSDA yield x 1.037, the ratio measured in G7).
edges,prob,cum=R.base.load_spectrum();args=R.inputs('elsepa');es,water,th=args[6:9]
pairs=energy=0.
for i,p in enumerate(prob):
    for E in np.linspace(edges[i],edges[i+1],41)[1::2]:
        if E<=.0101:continue
        c=coefficients(E,es,water,th);mu=c.sum();n=p/20*(1-np.exp(-30*mu))*(c[2]+c[3])/mu
        T=(E-2*M)/2
        if n>0 and T>.01:pairs+=n;energy+=n*2*T*R.csda_yield(T)*1.037
total=json.loads((R.OUT/'analysis.json').read_text())['brems_per_history']['prototype']['energy_MeV']['value']
out['pair_share']=dict(first_interaction_pairs_per_history=pairs,measured_pairs_per_history=.02330,
    brems_energy_from_pair_particles_MeV=energy,total_brems_energy_MeV=total,share=energy/total,
    gap_explained_by_minus_1p4pct_pairs_pct=-1.4*energy/total)
print(json.dumps(R.plain(out),ensure_ascii=False,indent=1))
