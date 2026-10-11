"""P1f selection, bounds, warning thresholds, freeze scope and real dependency guards."""
from pathlib import Path
import json
import numpy as np
import pytest
from prototypes.mv_electron.p1f_common import WORK,OUT,provider,targets,table_hashes

def test_grid_and_frozen_scope():
    from prototypes.mv_electron.generate_elsepa_p1f import energies,input_text
    es=energies();assert len(es)==51 and es[0]==.01 and es[-1]==10
    assert all(t in es for t in (.02,.1,1.,2.))
    assert 'IHEF              0' in input_text(8,-1,2.)
    paths=targets();assert not any(p.startswith('docs/ai/plans/') for p in paths)
    assert 'CLAUDE.md' not in paths and 'docs/validation/mv/p1e/RESULTS.md' in paths
    assert 'tests/test_transport.py' in paths

@pytest.fixture
def p1f_provider():
    if not (OUT/'t1.json').exists():pytest.skip('P1f tables absent; skip is not a pass')
    return provider()

def test_explicit_model_hash_and_range(p1f_provider):
    from prototypes.mv_electron.elsepa import Elsepa
    assert p1f_provider.energy_max==10
    p1f_provider.validate_energy(10.)
    for t in (.009,10.0000001):
        with pytest.raises(ValueError,match='range'):p1f_provider.validate_energy(t)
    with pytest.raises(ValueError,match='IHEF'):Elsepa(WORK/'tables',expected_ihef=1,expected_hashes=table_hashes())
    with pytest.raises(ValueError,match='hash'):Elsepa(WORK/'tables',expected_ihef=0,expected_hashes={})

def test_entry_rejects_before_run(monkeypatch,tmp_path,p1f_provider):
    from prototypes.mv_electron import run_transport as entry
    def forbidden(*a,**kw):raise AssertionError('transport called outside energy range')
    monkeypatch.setattr(entry,'run',forbidden)
    for t in (.009,10.001):
        with pytest.raises(ValueError,match='range'):
            entry.execute('out_of_range',T=t,backend='elsepa',seed=20261013,gs_table=WORK/'gs.npz',elsepa_directory=WORK/'tables',expected_ihef=0,expected_hashes=table_hashes(),output_dir=tmp_path)

def guarded_p1f_run():
    # Match P1b's isolated JIT guards, retaining exactly P1f's three forbidden globals.
    import types
    from numba import njit
    from prototypes.mv_electron.transport import run
    from prototypes.mv_electron.tests.test_dcs_p1b import (
        forbidden_elastic, forbidden_rates, forbidden_single,
    )
    gl = run.py_func.__globals__.copy()
    gl.update(elastic=forbidden_elastic, elastic_rates=forbidden_rates,
              single_cached=forbidden_single)
    function = types.FunctionType(run.py_func.__code__, gl, run.py_func.__name__,
                                  run.py_func.__defaults__)
    return njit(function)


def test_k5b_guard_detects_rutherford():
    from prototypes.mv_electron.p1f_common import TABLES
    guarded = guarded_p1f_run()
    with np.load(TABLES['rutherford']) as g:
        args = tuple(g[x] for x in ('T', 'K', 'u', 'y'))
        for mode in (0, 1):
            with pytest.raises(RuntimeError, match='Rutherford called'):
                guarded(2., 100, 20261013, .05, .025, np.array([0., 1.2]),
                        .01, *args, mode=mode)


def test_actual_p1f_transport_and_k5b(monkeypatch):
    if not (WORK/'gs.npz').exists():pytest.skip('P1f GS absent; T4/T5 not evaluated')
    if not (OUT/'t2.json').exists() or json.loads((OUT/'t2.json').read_text())['status']!='pass':pytest.skip('T2 has not passed; T4/T5 forbidden')
    from prototypes.mv_electron.run_p1f import table
    from prototypes.mv_electron.tests import test_transport as cases
    _,args,kw=table()
    guarded=guarded_p1f_run()
    for mode in (0,1):
        result=guarded(2.,100,20261013,.05,.025,np.array([0.,1.2]),.01,*args,mode=mode,**kw)
        cases.check(result,2.,100)
    original_read=Path.read_text
    def read(p,*a,**k):
        if 'dcslib' in p.parts:raise AssertionError('dcslib file read')
        return original_read(p,*a,**k)
    monkeypatch.setattr(Path,'read_text',read)
    def adopted(*pos,**kwargs):
        pos=list(pos);pos[2]=20261013;pos[4]=.025
        return guarded(*pos,**kwargs,**kw)
    monkeypatch.setattr(cases,'run',adopted)
    cases.test_k5_all_descendants(args)
    for thick in (.00001,.0008,.01,.1,1.2):cases.test_k5b_energy_hinge_escape_final_cutoff(args,thick)
    for first,second in ((.003,.047),(.047,.003),(.02,.02),(0,.12)):cases.test_k5b_analytic_bin_overlap(first,second)
    cases.test_k5b_faces_parallel_reverse_and_zero()
    for name in ('test_k5b_parent_restore_and_moller_settlement','test_limits_are_failure_not_local_deposition','test_k5b_final_hinge_residual_is_scored_on_track','test_k5b_simultaneous_zero_events_terminate','test_stack_overflow_retains_unprocessed_energy','test_boundary_free_diagnostic_has_no_queries'):getattr(cases,name)(args)

def test_warning_policy_and_error_limit(monkeypatch):
    from prototypes.mv_electron import check_elsepa_p1f as check
    monkeypatch.setattr(check,'integrated',lambda *a,**kw:np.ones(3))
    warning='WARNING: relative differences are too large.\nThe dcs table is not consistent.'
    def row(total=0.,g1=0.,warning_text=warning,error=1e-7):
        a=dict(mu=np.linspace(0,1,5),dcs=np.ones(5),error=np.full(5,error),table_xs=np.ones(3),series_xs=1/(1+np.array([total,g1,0.])),warnings=np.array([warning_text]),notices=np.array([],dtype=str),convergence='[]')
        return check.inspect(a,8,-1,2.,'unit')
    assert row(.004,.0009)['status']=='pass'
    assert row(.0051,.0009)['status']=='fail'
    assert row(.004,.0011)['status']=='fail'
    assert row(warning_text='WARNING: Possible accumulation of round-off errors.')['status']=='fail'
    assert row(error=.00101)['status']=='fail'

def test_entry_accepts_exact_upper_endpoint(monkeypatch,tmp_path,p1f_provider):
    if not (WORK/'gs.npz').exists():pytest.skip('P1f GS not yet built')
    from prototypes.mv_electron import run_transport as entry
    def reached(*a,**kw):raise RuntimeError('validated 10 MeV reaches transport')
    monkeypatch.setattr(entry,'run',reached)
    with pytest.raises(RuntimeError,match='validated 10 MeV'):
        entry.execute('upper',T=10.,backend='elsepa',seed=20261013,gs_table=WORK/'gs.npz',elsepa_directory=WORK/'tables',expected_ihef=0,expected_hashes=table_hashes(),output_dir=tmp_path)
