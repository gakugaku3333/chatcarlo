"""Ordered mismatch isolation; no fitting to reference, no algorithm substitution."""
import json
import numpy as np
from .component_checks import OUT
from .analysis import compare
def cross_compare(a,b):
    ma=a['cross_sum']/a['cross_n'];mb=b['cross_sum']/b['cross_n']
    sa=(a['cross_sum2']/a['cross_n']-ma*ma)/a['cross_n'];sb=(b['cross_sum2']/b['cross_n']-mb*mb)/b['cross_n']
    return {'hinge_cm2':ma.tolist(),'single_cm2':mb.tolist(),'relative':((ma-mb)/mb).tolist(),'difference_SEM_relative':(np.sqrt(sa+sb)/mb).tolist(),'hinge_boundary_events':int(a['counts'][3]),'single_boundary_events':int(b['counts'][3]),'hinge_boundary_queries':int(a['counts'][4]),'single_boundary_queries':int(b['counts'][4])}
def main(output_dir=None):
    result={'order':['K3 and K9(i)','class-II disabled, continuous collision+radiation only','no boundary queries','step dependence (K8)'],'facts':[],'hypotheses':[]}
    g=json.loads((OUT/'gs_checks.json').read_text());f=json.loads((OUT/'fixed_checks.json').read_text())
    result['components']={'K3':g['K3']['status'],'K9i':f['K9i']['status']}
    for key,first,second in [('continuous_only','nohard_hinge','nohard_reference'),('boundary_free','boundaryfree_hinge','boundaryfree_reference'),('extended_slab','noboundary_hinge','noboundary_reference')]:
        p=OUT/'runs'/(first+'.npz');q=OUT/'runs'/(second+'.npz')
        if not p.exists() or not q.exists():continue
        a=np.load(p);b=np.load(q);v=cross_compare(a,b)
        v['energy_max_residual_MeV']=float(max(np.max(abs(a['ledger'][:,3])),np.max(abs(b['ledger'][:,3]))))
        if np.any(a['batches']):
            z=(np.arange(a['batches'].shape[1])+.5)*a['dz']
            v['dose']=compare(a['batches'],b['batches'],z,tol=.03,range_tol=.03,bootstrap=1000)
        result[key]=v
    if (OUT/'transport_checks.json').exists():result['step_dependence']=json.loads((OUT/'transport_checks.json').read_text())['K8']
    if (OUT/'model_diagnostic.json').exists():result['coefficient_model_difference']=json.loads((OUT/'model_diagnostic.json').read_text())
    result['facts']=['Independent GS angular-moment/normalization/CDF and fixed-energy displacement checks passed.','The prototype and EGS5 use different elastic DCS; measured coefficient differences are saved separately.','No adjustment to f_E or K1_max to match EGS5; initial values accepted by K8.']
    result['hypotheses']=['The screened-Rutherford versus partial-wave model difference may contribute to the residual depth-dose/backscatter difference. The full response difference has not been causally attributed; changing the prescribed DCS is outside this plan.','GS inverse-CDF interpolation and energy-hinge discretization can contribute small residuals. K8 bounds their observed effect at the tested refinements, but does not establish a causal decomposition of the EGS5 residual.']
    ((OUT if output_dir is None else output_dir)/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
