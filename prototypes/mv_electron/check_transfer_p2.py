"""Q4 first-interaction transfer only, with an independent KN closed-form integral."""
import json
import numpy as np
from numba import njit
from .physics import M
from .photon_sampling_p2 import compton
from .photon_data_p2 import TABLE,coefficients,kn_transfer,transfer_coefficient,log_interp
from .p2_common import OUT,ROOT,SEED,ENERGIES,write,progress

@njit
def first_transfer(E,n,seed,es,w,th):
    np.random.seed(seed);c=coefficients(E,es,w,th);mu=c.sum();s=0.;s2=0.
    for h in range(n):
        r=np.random.random()*mu
        if r<c[0]:v=compton(E,0.,0.,1.)[4]
        elif r<c[0]+c[1]:v=E
        else:v=E-2*M
        s+=v;s2+=v*v
    mean=s/n;sem=np.sqrt(max(0.,(s2-n*mean*mean)/(n-1))/n)
    return mean*mu/E,sem*mu/E

def q4():
    if (OUT/'q4.json').exists():return json.loads((OUT/'q4.json').read_text())
    progress('T4','running',samples_per_energy=1000000)
    p=np.load(TABLE);es=p['E'];w=p['water'];th=p['thresholds']
    nist=np.loadtxt(ROOT/'chatcarlo/data/nist_xaamdi/water.csv',delimiter=',',comments='#');rows=[];n=1000000
    for j,E in enumerate(ENERGIES):
        measured,sem=first_transfer(E,n,SEED+400+j,es,w,th)
        expected=transfer_coefficient(E,es,w,th);c=coefficients(E,es,w,th)
        contributions=[float(kn_transfer(E)),float(c[1]),float((c[2]+c[3])*(1-2*M/E))]
        muen=log_interp(E,nist[:,0]/1000,nist[:,2]);z=(measured-expected)/sem
        state='統計不足' if sem/expected>.002 else '合格' if abs(z)<=3 else '不合格'
        rows.append(dict(E=E,N=n,seed=SEED+400+j,mu_tr_measured_cm2_g=measured,mu_tr_analytic_cm2_g=expected,
              SEM_cm2_g=sem,SEM_relative=sem/expected,z=z,status=state,
              model_mu_tr_contributions=dict(zip(('compton','photoelectric','pair'),contributions)),
              NIST_mu_en_cm2_g=float(muen),model_mu_tr_over_NIST_mu_en=float(expected/muen)))
    state='不合格' if any(r['status']=='不合格' for r in rows) else '統計不足' if any(r['status']=='統計不足' for r in rows) else '合格'
    result=dict(status=state,rows=rows,NIST_is_diagnostic_only=True,
                radiation_yield_definition='g is the initial charged-particle kinetic-energy-weighted mean radiation yield; NIST mu_en=(1-g)mu_tr',
                omitted_process_contributions='model mu_tr components quantified above; NIST binding/fluorescence/triplet recoil and radiative contributions cannot be separated numerically from the bundled coefficient alone')
    write('q4.json',result);progress('T4','complete',result=state);return result
