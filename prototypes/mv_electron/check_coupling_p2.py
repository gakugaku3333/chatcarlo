"""Q3 history balance, rest budget, dual scoring and branch statistics."""
import json
import numpy as np
from .p2_common import OUT,SEED,ENERGIES,write,save,progress
from .run_coupled_p2 import execute
from .photon_transport_p2 import initial,photon_state,accumulate_primary
from .physics import M

def q3():
    if (OUT/'q3.json').exists():return json.loads((OUT/'q3.json').read_text())
    progress('T3','running',backend_cases=12,histories_per_case=10000)
    rows=[]
    for backend in ('dcslib','mf','elsepa'):
        for j,E in enumerate(ENERGIES):
            dest=OUT/'checks'/f'q3_{backend}_{j}.npz'
            if dest.exists():r=dict(np.load(dest))
            else:r=execute(backend,E,10000,SEED+300+j);save(dest,**r)
            rebin=r['fine'].reshape(100,150,4).sum(axis=2)
            err=float(np.max(abs(rebin-r['coarse']))/E)
            good=abs(float(r['totals'][0])-10000*E)<10000*E*1e-12 and not np.any(r['violations']) and not r['counts'][5] and not r['counts'][6] and err<=1e-9
            rows.append(dict(backend=backend,E=E,N=10000,seed=SEED+300+j,history_max_relative=r['maxima'].tolist(),violations=r['violations'].tolist(),
                scoring_rebin_max_relative=err,totals_MeV=r['totals'].tolist(),step_limit=int(r['counts'][5]),stack_limit=int(r['counts'][6]),
                annihilations=int(r['counts'][12]),escaped_positrons=int(r['counts'][13]),status='合格' if good else '不合格'))
            write('q3_checkpoint.json',dict(completed=rows))
    # Artificial primary branching: 100 primaries, 2 perfectly correlated daughters.
    child_a=np.tile(np.arange(10,dtype=float),10);child_b=2*child_a
    primary=child_a+child_b
    branch_batches=np.zeros((100,2))
    for h in range(100):accumulate_primary(branch_batches,h,100,np.array([child_a[h],child_b[h]]))
    batched=branch_batches.sum(axis=1)
    direct=float(np.var(primary,ddof=1)/100);actual=float(np.var(batched,ddof=1)/100)
    a=initial(.005,15.,0.,0.,1.,0,0,.05,.025);a[19]=1.;a[20]=1.
    # override public incident-energy parameter with actual kinetic source in the kernel
    # E argument is the source kinetic energy; use the kernel directly for below-cutoff test.
    from .run_coupled_p2 import inputs,KEYS
    from .photon_transport_p2 import run_coupled
    stopped=dict(zip(KEYS,run_coupled(.005,100,SEED+310,.05,.025,30.,*inputs('mf'),source_state=a)))
    b=initial(1.,0.,0.,0.,1.,0,0,.05,.025);b[19]=1.;b[20]=1.
    # Thin slab cannot be binned at fixed widths; use 30 cm slab and source just inside rear surface.
    b[2]=30.-1e-12;escaped=execute('mf',1.,100,SEED+311,source_state=b)
    rest=dict(stopped_kinetic_before_annihilation_MeV=.005,annihilations=int(stopped['counts'][12]),
              cutoff_bin_dep_MeV=float(stopped['fine'][:,300].sum()),
              source_rest_budget_MeV=float(stopped['totals'][0]-100*.005),
              max_relative_residual=float(stopped['maxima'][0]),
              escaped_positrons=int(escaped['counts'][13]),escaped_rest_budget_MeV=float(escaped['totals'][5]),
              escaped_delta_rest_MeV=float(escaped['totals'][4]))
    tests=dict(primary_variance=actual==direct,
               stopped_cutoff_before_annihilation=stopped['counts'][12]>=100 and rest['cutoff_bin_dep_MeV']>=.5-1e-10 and not np.any(stopped['violations']),
               escaped_rest=escaped['counts'][13]==100 and abs(escaped['totals'][4]-200*M)<1e-9 and not np.any(escaped['violations']))
    result=dict(status='合格' if all(r['status']=='合格' for r in rows) and all(tests.values()) else '不合格',rows=rows,tests=tests,
                branching_variance_direct=direct,branching_variance_history=actual,rest_checks=rest,
                rest_convention='escape and unresolved contain kinetic/photon energy; delta_rest = net pair/medium rest change, +2m per pair, -2m per annihilation')
    write('q3.json',result);progress('T3','complete',result=result['status']);return result
