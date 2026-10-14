"""Fast P5a checks; private arrays and build products stay in ignored work."""
from pathlib import Path
import json
import uuid

import numpy as np
import pytest

from prototypes.mv_electron.gpu import p5a


@pytest.fixture(scope='module')
def work():
    path = p5a.ROOT/'.p5a0_work/p5a'/('pytest_'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    return path


@pytest.fixture(scope='module')
def data():
    return p5a.load_inputs(full=True)


@pytest.fixture(scope='module')
def cpu(work, data):
    return p5a.CPU(work, data)


@pytest.mark.parametrize('name', ['collision', 'hard', 'coefficients', 'table_rates'])
def test_representative_boundaries(cpu, data, name):
    x = np.array([.01, .02, np.nextafter(.02, np.inf), .0200000041,
                  .03, .5, 1., np.nextafter(1., np.inf), 2., 10.])
    actual = cpu.deterministic(name, x)
    expected = p5a.reference(name, x, data)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('name', ['gs_mu_cached', 'sample_W'])
def test_rejection_consumption(cpu, data, name):
    x = np.linspace(.021, 10., 100)
    if name == 'gs_mu_cached':
        x = np.column_stack((x, np.resize(data['ks'], len(x))))
    expected, next_expected = p5a.reference(name, x, data, seed=20291940)
    actual, next_actual, calls = cpu.random(name, 20291940, x)
    assert expected.tobytes() == actual.tobytes()
    assert next_expected.tobytes() == next_actual.tobytes()
    assert calls >= len(x)


def test_mt_state_recurrence(cpu):
    from numba import njit

    @njit
    def expected():
        np.random.seed(20291940)
        return np.random.random(20000)

    actual = np.empty(20000)
    cpu.lib.p5a_rng(20291940, 0, len(actual), actual)
    assert actual.tobytes() == expected().tobytes()


def test_philox_history_mapping(work, data):
    from prototypes.mv_electron.gpu.p5a0 import philox_python
    cpu = p5a.CPU(work, data, rng='philox')
    seed, history = 20291960, 2**32+7
    actual = np.empty(1000)
    cpu.lib.p5a_rng(seed, history, len(actual), actual)
    triples = np.column_stack((np.full(1000, seed, np.uint64),
                               np.full(1000, history, np.uint64), np.arange(1000, dtype=np.uint64)))
    assert actual.tobytes() == philox_python(triples)[1].tobytes()


def test_scoring_all_bins(cpu, data):
    x = np.array([[0., 30., 1.], [30., 0., .5], [1., 1., .2],
                  [1., 1.+5e-16, .3], [1., 29., 0.]])
    for nb, dz in ((150, .2), (600, .05)):
        expected = p5a.reference('score_segment', x, data, nb=nb, dz=dz)
        assert expected.tobytes() == cpu.score('score_segment', x, nb, dz).tobytes()


def test_input_rejection(cpu, data):
    for name, x in (('collision', [0.]), ('hard', [-1.]), ('coefficients', [21.]),
                    ('table_rates', [0.]), ('sample_W', [.02]), ('gs_mu_cached', [[0., .025]])):
        with pytest.raises(ValueError):
            p5a.reference(name, x, data, seed=20291940)
        with pytest.raises(ValueError):
            if name in ('sample_W', 'gs_mu_cached'):
                cpu.random(name, 20291940, x)
            else:
                cpu.deterministic(name, x)


def test_mismatch_includes_signed_zero():
    d = p5a.mismatch(np.array([0., 1.]), np.array([-0., 1.]))
    assert d['mismatches'] == 1
    assert d['first']['index'] == [0]
    assert d['first']['reference_hex'] != d['first']['cpu_hex']


def test_independent_sem_keeps_bin_covariance():
    # Perfectly correlated bins: averaging their individual variances would
    # underestimate SEM. Form each batch's block mean first.
    a = np.repeat(np.array([0., 2., 4., 6.])[:, None], 150, axis=1)
    b = np.repeat(np.array([1., 3., 5., 7.])[:, None], 150, axis=1)
    ab, bb = p5a.blocks(a), p5a.blocks(b)
    d = p5a.difference_summary(ab, bb, 100.)
    assert np.array_equal(np.array(d['delta_pct']), np.full(5, -1.))
    assert np.allclose(d['SEM_pct'], np.sqrt(10/3))


def test_saved_configuration_refused(work, data):
    cpu = p5a.CPU(work, data, rng='philox')
    dest = work/'resume_config'
    p5a.batch_run(cpu, data, dest, [], 4, 500)
    config = json.loads((dest/'config.json').read_text())
    config['seed'] += 1
    (dest/'config.json').write_text(json.dumps(config))
    with pytest.raises(p5a.GateStop, match='configuration mismatch'):
        p5a.batch_run(cpu, data, dest, [], 4, 500)


def test_missing_saved_batch_refused(work, data):
    cpu = p5a.CPU(work, data, rng='philox')
    dest = work/'resume_missing'
    p5a.batch_run(cpu, data, dest, [], 4, 500)
    config = json.loads((dest/'config.json').read_text())
    identity = p5a.sha(json.dumps(config, sort_keys=True).encode())
    p5a.write_json(dest/'progress.json', dict(config_sha256=identity, completed=[0]))
    with pytest.raises(p5a.GateStop, match='missing previously'):
        p5a.batch_run(cpu, data, dest, [], 4, 500)


def test_incomplete_chain0_refuses_philox(work, data):
    # --skip-saved must never authorize chain1 or an overall P5a pass.
    cpu = p5a.CPU(work, data, rng='philox')
    dest = work/'incomplete_chain0'
    p5a.write_json(dest/'chain0.json', dict(checks={k: dict(status='not_run' if k == 'B3' else 'pass')
                                                  for k in p5a.CHECKS}))
    with pytest.raises(p5a.GateStop, match='incomplete or failed'):
        p5a.require_chain0(dest, cpu, data)


def test_cached_output_shape_refused():
    # A healthy numerical ledger is insufficient when a cached output has a
    # missing scoring bin. Validate shape/dtype before accepting a checkpoint.
    r = dict(batches=np.zeros((100, 149)), fine_batches=np.zeros((100, 600)),
             totals=np.zeros(6), maxima=np.zeros(3), violations=np.zeros(3, np.int64),
             counts=np.zeros(16, np.int64), maxdepth=np.array(0, np.int64),
             nspec=np.zeros((20, 10), np.int64), step_batches=np.zeros(100, np.int64), brems=np.zeros(10))
    with pytest.raises(p5a.GateStop, match='shape/dtype mismatch: batches'):
        p5a.validate_result(r, 20)
