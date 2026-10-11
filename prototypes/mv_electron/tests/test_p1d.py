"""Run every legacy K5b case with each actual P1d transport table."""
import numpy as np
import pytest
from prototypes.mv_electron.p1d_common import BACKENDS, table, status, aggregate

@pytest.mark.parametrize('backend', BACKENDS)
def test_k5b_actual_table(monkeypatch, backend):
    from prototypes.mv_electron.tests import test_transport as cases
    from prototypes.mv_electron.transport import run
    _, args, kw = table(backend)
    def adopted(*pos, **kwargs):
        pos = list(pos); pos[4] = .025
        return run(*pos, **kwargs, **kw)
    monkeypatch.setattr(cases, 'run', adopted)
    cases.test_k5_all_descendants(args)
    for thick in (.00001, .0008, .01, .1, 1.2):
        cases.test_k5b_energy_hinge_escape_final_cutoff(args, thick)
    for first, second in ((.003, .047), (.047, .003), (.02, .02), (0, .12)):
        cases.test_k5b_analytic_bin_overlap(first, second)
    cases.test_k5b_faces_parallel_reverse_and_zero()
    for name in ('test_k5b_parent_restore_and_moller_settlement', 'test_limits_are_failure_not_local_deposition',
                 'test_k5b_final_hinge_residual_is_scored_on_track', 'test_k5b_simultaneous_zero_events_terminate',
                 'test_stack_overflow_retains_unprocessed_energy', 'test_boundary_free_diagnostic_has_no_queries'):
        getattr(cases, name)(args)

def test_stop_and_statistics_states():
    assert status(dict(statistics_ok=False, **{'pass': False})) == 'statistics-insufficient'
    assert aggregate(['pass', 'statistics-insufficient']) == 'statistics-insufficient'
    assert aggregate(['pass', 'not-evaluated']) == 'not-evaluated'
    assert aggregate(['fail', 'statistics-insufficient']) == 'fail'

def test_reduction_same_comparator():
    from prototypes.mv_electron.run_checks_p1d import reduction
    ref = np.tile([1., 2., 1.], (4, 1))
    old = ref + .2; new = ref + .1
    r = reduction(old, new, ref, np.ones(3, bool))
    assert r['point'] == pytest.approx(.5)
    assert r['CI95'] == pytest.approx([.5, .5])
