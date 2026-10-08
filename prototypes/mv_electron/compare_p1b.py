"""M6/M7 and preregistered shared-EGS5 batch-bootstrap reduction."""
import numpy as np
from .analysis import compare,dref
from .check_egs5 import parse

def reduction(p1,new,ref,mask):
    rng=np.random.default_rng(20261008);rs=[];ds=[]
    for j in range(1000):
        # Same EGS5 realization in both comparisons; re-estimate denominator.
        e=ref[rng.integers(len(ref),size=len(ref))].mean(0)
        p=p1[rng.integers(len(p1),size=len(p1))].mean(0)
        n=new[rng.integers(len(new),size=len(new))].mean(0)
        D=dref(e);dp=max(abs(p[mask]-e[mask]))/D;dn=max(abs(n[mask]-e[mask]))/D
        rs.append(1-dn/dp);ds.append(dn)
    ci=np.percentile(rs,[2.5,97.5]);dci=np.percentile(ds,[2.5,97.5])
    base=parse('base_2');D=dref(base['batches'].mean(0));dn=max(abs(new.mean(0)[mask]-base['batches'].mean(0)[mask]))/D
    import json
    from .component_checks import OUT
    dp=json.loads((OUT/'egs5_checks.json').read_text())['K6']['max_dose_difference_over_Dref']
    status='残差の大部分が消えた' if ci[0]>=.5 else '部分的に低減した' if ci[0]>0 else '低減を確認できない'
    return {'point':float(1-dn/dp),'CI95':ci.tolist(),'d_new_CI95':dci.tolist(),'status':status,'majority_possible':bool(ci[0]<.5<=ci[1]),'bootstrap_N':1000,'seed':20261008,'fixed_mask_indices':np.flatnonzero(mask).tolist(),'samples_r':rs,'samples_d_new':ds}

def main(backend,out):
    from .component_checks import OUT
    ans={}
    for T,key in [(2,'M6'),(10,'M7')]:
        ref=parse(f'base_{T}');p=np.load(out/'runs'/f'base_{T}.npz');mask=ref['q']>=.2*dref(ref['q'])
        c=compare(ref['batches'],p['batches'],ref['z'],tol=.03,range_tol=.03,mask=mask,bootstrap=2000 if T==2 else 1000)
        c['status']=('pass' if c['pass'] else 'fail') if T==2 else 'reference-only'
        c['prototype_back_energy_fraction']=float(p['ledger'][:,1].sum()/(T*int(p['N'])))
        fractions=p['ledger'][:,1]/T
        c['prototype_back_fraction_SEM']=float(fractions.std(ddof=1)/np.sqrt(len(fractions)))
        c['EGS5_back_energy_fraction']=ref['ledger'][2]/ref['ledger'][0]
        N=ref['ledger'][0]/T;f=c['EGS5_back_energy_fraction']
        c['EGS5_back_fraction_SEM_upper_bound']=float(np.sqrt(f*(1-f)/(N-1)))
        ans[key]=c
        if T==2:ans['reduction']=reduction(np.load(OUT/'runs/base_2.npz')['batches'],p['batches'],ref['batches'],mask)
    return ans
