"""P1c formula and real-table transport isolation/K5b invariants."""
import functools
import numpy as np
import pytest
from prototypes.mv_electron.physics import M,ALPHA
from prototypes.mv_electron.mf import correction,McKinleyFeshbach,born_g1
from prototypes.mv_electron.dcs import moments,get_backend
from prototypes.mv_electron.regression_p1c import P
from prototypes.mv_electron.tests.test_dcs_p1b import isolated

@pytest.mark.parametrize('T',[.01,.02,.1,1.,2.,10.])
@pytest.mark.parametrize('Z',[1,8])
def test_formula_independent(T,Z):
    gamma=1+T/M;beta=np.sqrt(1-gamma**-2)
    assert correction(T,0,Z)==1
    assert correction(T,2,Z)==pytest.approx(gamma**-2,rel=1e-12)
    for theta in [np.pi/6,np.pi/2,2*np.pi/3]:
        expected=1-beta**2*np.sin(theta/2)**2+np.pi*ALPHA*Z*beta*np.sin(theta/2)*(1-np.sin(theta/2))
        assert correction(T,1-np.cos(theta),Z)==pytest.approx(expected,rel=1e-14)
    assert np.all(correction(T,np.linspace(0,2,10001),Z)>0)

def test_interface_and_born_closed_form():
    b=get_backend('mf');assert b.name=='mf' and b.native_energies==()
    for T in [.01,.02,.1,1.,2.,10.]:
        assert np.all(np.diff(b.edges(T))>0) and 1 in b.edges(T)
        assert moments(McKinleyFeshbach('born_only'),T)[0][1]==pytest.approx(born_g1(T),rel=1e-6)

def production_table():
    file=P/'mf/data/gs.npz'
    if not file.exists():pytest.skip('production mf table not yet built; acceptance remains pending')
    return np.load(file)

def test_no_rutherford_with_actual_mf_table():
    from prototypes.mv_electron.transport import run
    from prototypes.mv_electron.reference import fixed
    g=production_table();args=tuple(g[k] for k in ['T','K','u','y']);kw=dict(backend_rates=g['rates'],backend_single=g['single'])
    for mode in [0,1]:
        r=isolated(run)(2.,10,20261007,.05,.05,np.array([0.,1.2]),.01,*args,mode=mode,**kw)
        assert not np.any(r[1][:,4:6])
    for condensed in [False,True]:
        assert np.all(np.isfinite(isolated(fixed)(2.,.05,10,20261007,condensed,.05,*args,**kw)))

def test_k5b_with_actual_mf_table(monkeypatch):
    from prototypes.mv_electron.tests import test_transport as cases
    from prototypes.mv_electron.transport import run
    g=production_table();args=tuple(g[k] for k in ['T','K','u','y'])
    monkeypatch.setattr(cases,'run',functools.partial(run,backend_rates=g['rates'],backend_single=g['single']))
    cases.test_k5_all_descendants(args)
    for thick in [.00001,.0008,.01,.1,1.2]:cases.test_k5b_energy_hinge_escape_final_cutoff(args,thick)
    for first,second in [(.003,.047),(.047,.003),(.02,.02),(0,.12)]:cases.test_k5b_analytic_bin_overlap(first,second)
    cases.test_k5b_faces_parallel_reverse_and_zero()
    for name in ['test_k5b_parent_restore_and_moller_settlement','test_limits_are_failure_not_local_deposition','test_k5b_final_hinge_residual_is_scored_on_track','test_k5b_simultaneous_zero_events_terminate','test_stack_overflow_retains_unprocessed_energy','test_boundary_free_diagnostic_has_no_queries']:
        getattr(cases,name)(args)

def test_all_orders_and_actual_lookup_match_mf():
    from prototypes.mv_electron.gs_backend import coefficients
    from prototypes.mv_electron.backend_sampling import table_rates
    from prototypes.mv_electron.single_table import inverse_moments
    from prototypes.mv_electron.check_dcs_p1c import interpolation_midpoints
    g=production_table();b=McKinleyFeshbach();ts=g['single'][:,0]
    G,total=coefficients(b,2.,16);ref,t,_=moments(b,2.)
    np.testing.assert_allclose(G[:3],ref,rtol=1e-9,atol=1e-12)
    assert total==pytest.approx(t,rel=1e-12) and np.all(G[1:]>0)
    for T in np.r_[[.01,.02,.1,1.,2.,10.],interpolation_midpoints(g)]:
        G,total,_=moments(b,T);rate,G1=table_rates(T,g['T'],g['rates'])
        i=min(max(np.searchsorted(ts,T,side='right')-1,0),len(ts)-2);a=np.log(T/ts[i])/np.log(ts[i+1]/ts[i])
        q=(1-a)*g['single'][i,1:]+a*g['single'][i+1,1:]
        np.testing.assert_allclose([rate,G1],[total,G[1]],rtol=1e-4)
        np.testing.assert_allclose(rate*inverse_moments(g['u'],q),G[1:3],rtol=1e-4)
