"""Independent invariants for provider selection and protected P1 definitions."""
import hashlib,json,types
from pathlib import Path
import numpy as np
import pytest
from numba import njit
from prototypes.mv_electron.dcs import Dcslib,moments,inigrd
from prototypes.mv_electron.eedl import Eedl
from prototypes.mv_electron.gs_backend import coefficients
from prototypes.mv_electron.physics import ROOT
P=ROOT/'docs/validation/mv/p1b'

@njit
def forbidden_rates(T):
    if T>=0:raise RuntimeError('Rutherford called')
    return 0.,0.
@njit
def forbidden_elastic(T):
    if T>=0:raise RuntimeError('Rutherford called')
    return np.zeros(2),np.zeros(2),np.zeros(2),0.
@njit
def forbidden_single(eta,total):
    if len(eta)>0:raise RuntimeError('Rutherford called')
    return 0.
@njit
def forbidden_gs(T,K,ts,ks,us,ys):
    if T>=0:raise RuntimeError('Rutherford called')
    return 0.

def isolated(function):
    gl=function.py_func.__globals__.copy()
    gl.update(elastic_rates=forbidden_rates,elastic=forbidden_elastic,single_cached=forbidden_single,gs_mu=forbidden_gs)
    f=types.FunctionType(function.py_func.__code__,gl,function.py_func.__name__,function.py_func.__defaults__)
    return njit(f)

@pytest.mark.parametrize('backend',['dcslib','eedl'])
def test_no_rutherford_in_table_transport_and_reference(backend):
    from prototypes.mv_electron.transport import run
    from prototypes.mv_electron.reference import fixed
    file=P/backend/'data/gs.npz'
    # Selection isolation needs no expensive production GS table: arbitrary positive
    # provider arrays exercise compiled branches. This is not physics acceptance.
    ts=np.array([.01,20.]);ks=np.array([.0005,.2]);us=np.array([0.,1.]);ys=np.zeros((2,2,2));ys[:,:,1]=2
    rates=np.array([[100.,1.],[100.,1.]]);single=np.array([[0.,2.],[0.,2.]])
    if file.exists():
        g=np.load(file);ts,ks,us,ys=(g[x] for x in ['T','K','u','y']);rates=g['rates'];single=g['single']
    kw=dict(backend_rates=rates,backend_single=single)
    guarded=isolated(run)
    for mode in [0,1]:
        r=guarded(2.,10,20261007,.05,.05,np.array([0.,1.2]),.01,ts,ks,us,ys,mode=mode,**kw)
        assert not np.any(r[1][:,4:6])
    guarded_fixed=isolated(fixed)
    for condensed in [False,True]:assert np.all(np.isfinite(guarded_fixed(2.,.05,10,20261007,condensed,.05,ts,ks,us,ys,**kw)))

@pytest.mark.parametrize('z,expected',[(1,1.13312333e-20),(8,2.86423422e-19)])
def test_dcslib_independent_atomic_integrals(z,expected):
    b=Dcslib();G,total,_=moments(b,2,z=z);H,ref,_=moments(b,2,order=32,z=z)
    assert total==pytest.approx(expected,rel=1e-4)
    np.testing.assert_allclose(np.r_[G[1:],total],np.r_[H[1:],ref],rtol=1e-6)

def test_inigrd_and_no_header_renormalization():
    x=inigrd();assert len(x)==606 and x[0]==0 and x[-1]==pytest.approx(1)
    assert np.all(np.diff(x)>0)
    b=Dcslib();G,t,_=moments(b,2,z=1);e,a,h=b.atomic[1]
    assert abs(t/h[np.argmin(abs(e-2)),0]-1)>1e-3

@pytest.mark.parametrize('z',[1,8])
@pytest.mark.parametrize('t',[.02,.1,.256,1.,2.,10.])
def test_eedl_branches_and_file_interpolation(z,t):
    b=Eedl();mu,p,yc,eta,A,large,total=b.parameters(t,z)
    assert mu[-1]==.999999
    assert np.trapezoid(p,mu)==pytest.approx(1,abs=1e-6)
    assert A*(1/eta-1/(eta+yc))==pytest.approx(total-large,rel=1e-12,abs=1e-40)
    G,integral,_=moments(b,t,z=z)
    assert integral==pytest.approx(total,rel=1e-3)
    assert np.all(p>=0) and A>=0

def test_all_orders_come_from_effective_dcs():
    b=Dcslib();G,total=coefficients(b,2.,16);ref,t,_=moments(b,2.)
    np.testing.assert_allclose(G[:3],ref,rtol=1e-9,atol=1e-12)
    assert total==pytest.approx(t,rel=1e-12)
    assert np.all(G[1:]>0)

def test_p1_unchanged_and_analysis_fixed():
    saved=json.loads((P/'p1_initial_hashes.json').read_text())
    assert saved=={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'docs/validation/mv/p1').rglob('*') if f.is_file()}
    assert (ROOT/'prototypes/mv_electron/analysis.py').read_bytes()==(P/'baseline_code/analysis.py').read_bytes()
    assert (ROOT/'prototypes/mv_electron/physics.py').read_bytes()==(P/'baseline_code/physics.py').read_bytes()

@pytest.mark.parametrize('backend',['dcslib','eedl'])
def test_original_k5b_cases_with_selected_backend(backend,monkeypatch):
    # Run the original P1 assertions unchanged against each provider. This covers
    # scoring, hinges, parent restore, surfaces and explicit failure accounting.
    import functools
    from prototypes.mv_electron.tests import test_transport as cases
    from prototypes.mv_electron.transport import run
    g=np.load(P/backend/'data/gs.npz');args=tuple(g[x] for x in ['T','K','u','y'])
    monkeypatch.setattr(cases,'run',functools.partial(run,backend_rates=g['rates'],backend_single=g['single']))
    cases.test_k5_all_descendants(args)
    for thick in [.00001,.0008,.01,.1,1.2]:cases.test_k5b_energy_hinge_escape_final_cutoff(args,thick)
    for first,second in [(.003,.047),(.047,.003),(.02,.02),(0,.12)]:cases.test_k5b_analytic_bin_overlap(first,second)
    cases.test_k5b_faces_parallel_reverse_and_zero()
    for name in ['test_k5b_parent_restore_and_moller_settlement','test_limits_are_failure_not_local_deposition','test_k5b_final_hinge_residual_is_scored_on_track','test_k5b_simultaneous_zero_events_terminate','test_stack_overflow_retains_unprocessed_energy','test_boundary_free_diagnostic_has_no_queries']:
        getattr(cases,name)(args)

@pytest.mark.parametrize('backend',['dcslib','eedl'])
def test_saved_production_history_prefix_reproduces_with_current_code(backend):
    # Provenance bridge: output metadata/report additions during long calculations
    # must not change the physical history stream saved by the launch version.
    from prototypes.mv_electron.transport import run
    g=np.load(P/backend/'data/gs.npz');saved=np.load(P/backend/'runs/base_2.npz')
    r=run(2.,10000,20261007,.05,.05,np.array([0.,1.2]),.01,*(g[x] for x in ['T','K','u','y']),backend_rates=g['rates'],backend_single=g['single'])
    np.testing.assert_array_equal(r[1],saved['ledger'][:10000])

@pytest.mark.parametrize('backend',['dcslib','eedl'])
def test_converged_rate_single_lookup_matches_provider(backend):
    from prototypes.mv_electron.backend_sampling import table_rates
    from prototypes.mv_electron.sampling import bracket
    from prototypes.mv_electron.dcs import get_backend
    from prototypes.mv_electron.single_table import inverse_moments
    g=np.load(P/backend/'data/gs.npz');b=get_backend(backend)
    info=json.loads(str(g['lookup_diagnostics']))
    assert info['energy_refinement'][-1]['max_relative_error']<=1e-4
    assert info['native_probability_max_relative_error']<=2.5e-5
    ts=g['single'][:,0];u=g['u']
    for t in [.02,.1,.256,1.,2.,10.]:
        G,total,_=moments(b,t);r,G1=table_rates(t,g['T'],g['rates'])
        i=np.searchsorted(ts,t)-1;a=np.log(t/ts[i])/np.log(ts[i+1]/ts[i])
        q=(1-a)*g['single'][i,1:]+a*g['single'][i+1,1:]
        np.testing.assert_allclose([r,G1], [total,G[1]],rtol=1e-4)
        np.testing.assert_allclose(r*inverse_moments(u,q),G[1:3],rtol=1e-4)
