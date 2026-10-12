"""G4 (azimuth uniformity, parent -z, seed 20291607) failed with p = 0.00095. Diagnosis only; changes no verdict.
Run from the repository root: PYTHONPATH=. .venv/bin/python docs/validation/mv/p3r/supplementary/g4_diagnosis.py"""
import json
import numpy as np
from numba import njit
from scipy.stats import chisquare
from prototypes.mv_electron.run_p3r import draw_azimuth

@njit
def raw(n,seed):
    np.random.seed(seed);a=np.empty(n)
    for i in range(n):a[i]=np.random.random()
    return a

u=raw(100000,20291607);phi,_=draw_azimuth(1.,0.,0.,-1.,100000,20291607)
others=[float(chisquare(np.histogram(draw_azimuth(1.,0.,0.,-1.,100000,s)[0],np.linspace(-np.pi,np.pi,21))[0]).pvalue)
        for s in range(30000001,30000041)]
out=dict(p_raw_uniform_same_seed=float(chisquare(np.histogram(u,np.linspace(0,1,21))[0]).pvalue),
         spread_of_phi_minus_2pi_u=float(np.ptp(np.unwrap((phi-2*np.pi*u)%(2*np.pi)))),
         other_seeds=dict(first=30000001,count=40,min_p=min(others),median_p=float(np.median(others)),
                          below_threshold=int(sum(p<.01/3 for p in others))))
print(json.dumps(out,indent=1))
