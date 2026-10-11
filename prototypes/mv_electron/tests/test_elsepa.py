"""P1e parser, interpolation, stop/freeze scope and actual transport dependency checks."""
from pathlib import Path
import json
import numpy as np
import pytest
from prototypes.mv_electron.p1e_common import WORK, OUT
from prototypes.mv_electron.generate_elsepa import input_text,energies,parse

def test_fixed_input_and_energy_grid():
    for q in (-1,1):
        lines=input_text(8,q,2.).splitlines();assert lines[-1]=='EV     2.000000E+06'
        kv={line[:6].strip():line[7:19].strip() for line in lines}
        assert kv['IHEF']=='1' and kv['MEXCH']==('1' if q==-1 else '0')
        assert kv['MCPOL']==kv['MABS']==kv['MUFFIN']=='0'
    es=energies();assert len(es)==83
    assert all(t in es for t in (.01,.02,.1,1.,2.,10.))

def test_freeze_excludes_plans_and_unrelated_docs():
    from prototypes.mv_electron.p1e_common import targets
    paths=targets()
    assert not any(p.startswith('docs/ai/plans/') for p in paths)
    assert 'CLAUDE.md' not in paths and 'docs/validation/mv/p1e/RESULTS.md' not in paths
    assert 'tests/test_transport.py' in paths

def test_atomic_parser_preserves_missing_series(tmp_path):
    # Independently use the unchanged bundled 606-row output with deliberately empty stdout.
    src=WORK/'source/elsepa-2020/test-run-output/dcs_1p000e03.dat'
    if not src.exists():pytest.skip('ELSEPA archive not extracted')
    a=parse(src,'');assert np.all(~a['series_available']) and np.all(np.isnan(a['series_xs']))
    assert a['mu'][0]==0 and a['mu'][-1]==1 and len(a['dcs'])==606

@pytest.fixture
def provider():
    from prototypes.mv_electron.elsepa import Elsepa
    if not all((WORK/'tables'/f'z{z}_q-1.npz').exists() for z in (1,8)):
        pytest.skip('ELSEPA tables absent; skip is not an S2/S4 pass')
    return Elsepa()

def test_range_and_identical_cubic_rules(provider):
    from scipy.interpolate import CubicSpline
    for t in (.009,20.01):
        with pytest.raises(ValueError,match='range'):provider.density(t,np.array([0.,1.]))
    e,a,_=provider.atomic[8];t=.1
    x=CubicSpline(np.log(e),np.log(a),axis=0,bc_type='natural')(np.log(t))
    expected=2*np.pi*np.exp(CubicSpline(provider.rmu,x,bc_type='natural')(np.array([.01,.5])))
    assert np.array_equal(provider.density(t,np.array([.02,1.]),8),expected)

def test_backend_does_not_read_dcslib(monkeypatch,provider):
    from prototypes.mv_electron import dcs
    monkeypatch.setattr(dcs,'read_dcslib',lambda *args:(_ for _ in ()).throw(AssertionError('dcslib read')))
    other=dcs.get_backend('elsepa');assert other.name=='elsepa'
    assert np.array_equal(other.density(2.,np.array([0.,1.])),provider.density(2.,np.array([0.,1.])))

def test_actual_elsepa_transport_and_k5b(monkeypatch):
    if not (WORK/'gs.npz').exists():pytest.skip('ELSEPA GS absent; S4/S5 not evaluated')
    from prototypes.mv_electron.run_p1e import table
    from prototypes.mv_electron.tests import test_transport as cases
    from prototypes.mv_electron import transport
    from prototypes.mv_electron.transport import run
    _,args,kw=table()
    def forbidden(*a,**k):raise AssertionError('screened Rutherford used')
    for name in ('elastic','elastic_rates','single_cached'):
        monkeypatch.setattr(transport,name,forbidden)
    # The Python transport loop resolves guarded globals, rather than an old JIT cache.
    for mode in (0,1):
        result=run.py_func(2.,100,20261011,.05,.025,np.array([0.,1.2]),.01,*args,mode=mode,**kw)
        cases.check(result,2.,100)
    original_read=Path.read_text
    def read(p,*a,**k):
        if 'dcslib' in p.parts:raise AssertionError('dcslib file read')
        return original_read(p,*a,**k)
    monkeypatch.setattr(Path,'read_text',read)
    def adopted(*pos,**kwargs):
        pos=list(pos);pos[2]=20261011;pos[4]=.025
        return run(*pos,**kwargs,**kw)
    monkeypatch.setattr(cases,'run',adopted)
    cases.test_k5_all_descendants(args)
    for thick in (.00001,.0008,.01,.1,1.2):cases.test_k5b_energy_hinge_escape_final_cutoff(args,thick)
    for first,second in ((.003,.047),(.047,.003),(.02,.02),(0,.12)):cases.test_k5b_analytic_bin_overlap(first,second)
    cases.test_k5b_faces_parallel_reverse_and_zero()
    for name in ('test_k5b_parent_restore_and_moller_settlement','test_limits_are_failure_not_local_deposition','test_k5b_final_hinge_residual_is_scored_on_track','test_k5b_simultaneous_zero_events_terminate','test_stack_overflow_retains_unprocessed_energy','test_boundary_free_diagnostic_has_no_queries'):getattr(cases,name)(args)
