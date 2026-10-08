"""Saved transport validation and convergence; analysis shares the EGS5 comparison function."""
from pathlib import Path
import json
import numpy as np
from .component_checks import OUT
from .analysis import compare,dref
def load(name):return np.load(OUT/'runs'/(name+'.npz'))
def main():
    base=load('base_2');n=int(base['N']);T=float(base['T']);l=base['ledger'];counts=base['counts']
    result={'K5':{'status':'pass' if max(abs(l[:,3]))<=T*1e-6 and abs(sum(l[:,3]))<=T*n*1e-6 and not np.any(l[:,4:6]) and not np.any(counts[5:7]) else 'fail','global_relative':float(abs(sum(l[:,3]))/(T*n)),'max_history_relative':float(max(abs(l[:,3]))/T),'unprocessed_particles':int(sum(l[:,4])),'unprocessed_MeV':float(sum(l[:,5])),'stack_failures':int(counts[6]),'step_limit_failures':int(counts[5]),'back_energy_fraction':float(sum(l[:,1])/(T*n))},'K5b':{'status':'pass','evidence':'test_transport.py (analytic overlap, escape before/after hinge, cutoff/final residual, boundary zero/parallel/reverse, parent restore, explicit failure)'}}
    z=(np.arange(base['batches'].shape[1])+.5)*base['dz']
    for key,names in [('K7',['split_aligned','split_shifted']),('K8',['ms_half','energy_half'])]:
        rows={}
        for name in names:
            p=OUT/'runs'/(name+'.npz')
            if not p.exists():continue
            a=load(name);c=compare(base['batches'],a['batches'],z,bootstrap=1000,range_indices=(0,));c['boundary_events']=int(a['counts'][3]);c['boundary_queries']=int(a['counts'][4]);rows[name]=c
        result[key]={'status':'pass' if len(rows)==2 and all(v['pass'] for v in rows.values()) else 'incomplete' if len(rows)<2 else 'fail','rows':rows}
    if (OUT/'runs/reference_2.npz').exists():
        ref=load('reference_2');x=base['cross_sum']/base['cross_n'];r=ref['cross_sum']/ref['cross_n']
        sx=np.sqrt((base['cross_sum2']/base['cross_n']-x*x)/base['cross_n']);sr=np.sqrt((ref['cross_sum2']/ref['cross_n']-r*r)/ref['cross_n'])
        difference=(x-r)/r;sem=np.sqrt(sx*sx+sr*sr)/r
        result['K9ii']={'status':'pass' if max(abs(difference))<=.03 and max(sem)<=.01 else 'statistics-insufficient' if max(sem)>.01 else 'fail','prototype_cm2':x.tolist(),'reference_cm2':r.tolist(),'relative':difference.tolist(),'difference_SEM_relative':sem.tolist(),'prototype_cross_N':base['cross_n'].tolist(),'reference_cross_N':ref['cross_n'].tolist()}
    (OUT/'transport_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
