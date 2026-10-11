"""Analytic estimate of positron annihilation in flight in water (no Monte Carlo).

Heitler cross section per electron, slowing down with the ESTAR total stopping power
(electron values; the positron's differ by a few per cent). Pair positrons are taken
uniform in kinetic energy between 0 and E-2m (rough Bethe-Heitler shape at these energies).
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
from prototypes.mv_electron.physics import DATA
e=np.load(DATA/'estar.npz')
M=.51099895;R0=2.8179403262e-13;NE=3.3428e23  # MeV, cm, electrons per gram of water
def sigma(T):
    g=1+T/M;s=np.sqrt(g*g-1)
    return np.pi*R0**2/(g+1)*((g*g+4*g+1)/(g*g-1)*np.log(g+s)-(g+3)/s)
def flight(T0,n=4000):
    """Probability of annihilating before slowing to 10 keV, and mean kinetic energy when it happens."""
    T=np.geomspace(.01,T0,n);rate=NE*sigma(T)/np.interp(np.log(T),np.log(e['T']),e['total'])  # per MeV lost
    seg=np.diff(T)*(rate[1:]+rate[:-1])/2;tau=np.concatenate(([0],np.cumsum(seg)))  # optical depth from 10 keV up to T
    surv=np.exp(-(tau[-1]-tau))  # survival from T0 down to T
    p=surv*rate;P=np.sum(np.diff(T)*(p[1:]+p[:-1])/2);TT=np.sum(np.diff(T)*((p*T)[1:]+(p*T)[:-1])/2)
    return P,TT/P
out={}
for E in (6.,10.):
    t0=np.linspace(.02,E-2*M,400);v=np.array([flight(t) for t in t0])
    P=v[:,0].mean();Tm=(v[:,0]*v[:,1]).mean()/P
    out[E]=dict(pair_positron_mean_T_MeV=float(t0.mean()),P_flight=float(P),mean_T_at_annihilation_MeV=float(Tm),
                P_flight_at_mean_T=float(flight(t0.mean())[0]))
B=ROOT/'docs/validation/mv/p2b/egs5';s=json.loads((B/'p2b_Splus/summary.json').read_text());c=json.loads((B/'p2b_Cminus/summary.json').read_text())
cb=s['callbacks'];o=out[6.]
o['EGS5_P_flight']=cb['flight_annihilation']/cb['pairs']
o['EGS5_deposit_difference_MeV']=c['ledger'][1]-s['ledger'][1]
o['EGS5_difference_per_flight_annihilation_MeV']=o['EGS5_deposit_difference_MeV']/cb['flight_annihilation']
o['kinetic_energy_removed_locally_over_deposit_pct']=cb['flight_annihilation']*o['mean_T_at_annihilation_MeV']/s['ledger'][1]*100
o['EGS5_deposit_difference_pct']=o['EGS5_deposit_difference_MeV']/s['ledger'][1]*100
o['implied_reabsorbed_fraction']=1-o['EGS5_difference_per_flight_annihilation_MeV']/o['mean_T_at_annihilation_MeV']
P2=ROOT/'docs/validation/mv/p2/egs5'
for E,j in ((6.,2),(10.,3)):
    a=json.loads((P2/f'main_ref_{j}_0/summary.json').read_text());b=json.loads((P2/f'main_ref_{j}_4/summary.json').read_text())
    out[E]['P2']=dict(P_flight=a['callbacks']['flight_annihilation']/a['callbacks']['pairs'],pairs_per_history=a['callbacks']['pairs']/a['N'],
        deposit_difference_pct=(b['ledger'][1]/b['N']-a['ledger'][1]/a['N'])/(a['ledger'][1]/a['N'])*100,
        removed_locally_pct=a['callbacks']['flight_annihilation']*out[E]['mean_T_at_annihilation_MeV']/a['ledger'][1]*100)
(ROOT/'docs/validation/mv/p2b/supplementary/annihilation_estimate.json').write_text(json.dumps({str(k):v for k,v in out.items()},indent=2)+'\n')
print(json.dumps(out,indent=1))
