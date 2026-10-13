"""Offline P4bc component checks; never execute workflow CLI commands.

Output tests write only tmp_path. External tables are opened read-only; missing
or inaccessible tables skip only the two transport checks.
"""
import numpy as np
import pytest
from prototypes.mv_electron import run_p4bc as r


@pytest.mark.parametrize('difference,sem,healthy,expected', [
    (.01, .001, True, '合格'), (-.01, .001, True, '合格'),
    (np.nextafter(.01, np.inf), .001, True, '不合格'),
    (.02, np.nextafter(.001, np.inf), True, '統計不足'),
    (.02, .002, False, '判定できない'),
    (np.nan, 0., True, '判定できない'), (0., np.inf, True, '判定できない'),
    (0., -.01, True, '判定できない'),
])
def test_classification(difference, sem, healthy, expected):
    assert r.classify(difference, sem, .01, .001, healthy) == expected


def test_c1_synthetic_hand_calculation():
    p = np.r_[np.ones(20), .5, np.zeros(20)]
    value = r.c1_statistics(p)
    assert value['difference'] == pytest.approx(0, abs=1e-17)
    assert value['sem'] == pytest.approx(.0005*np.sqrt(.25/2000))
    assert value['quadrature_bound'] == .00025
    assert value['status'] == '合格'


@pytest.mark.parametrize('index,value', [(0, .9), (40, .1), (10, np.nan),
                                         (20, np.inf), (12, -1), (12, 2)])
def test_c1_invalid_probabilities(index, value):
    p = np.linspace(1, 0, 41)
    p[index] = value
    assert r.c1_statistics(p)['status'] == '判定できない'


def point():
    return dict(ledger=np.zeros((2000, 7)), counts=np.zeros(7), seconds=0.)


@pytest.mark.parametrize('key,index,value', [
    ('counts', 5, 1), ('counts', 6, 1), ('counts', 1, 1), ('counts', 2, 1),
    ('ledger', (0, 4), 1), ('ledger', (0, 5), 1), ('ledger', (0, 1), 1),
    ('ledger', (0, 3), 2e-9), ('ledger', (0, 0), np.nan),
])
def test_failed_history(key, index, value):
    data = point()
    assert r.c1_point_healthy(data, 1.)
    data[key][index] = value
    healthy = r.c1_point_healthy(data, 1.)
    assert not healthy
    assert r.c1_statistics(np.linspace(1, 0, 41), healthy)['status'] == '判定できない'


@pytest.mark.parametrize('T0', [.1, 1., 10.])
@pytest.mark.parametrize('power', [0, -1])
def test_integral_analytic(T0, power):
    result = r.integrate_range(T0, lambda t: 2*t**power)
    exact = (T0-.01)/2 if power == 0 else (T0*T0-.01**2)/4
    assert result['range'] == pytest.approx(exact, rel=1e-7)
    assert result['healthy']


def test_hinge_constant_stopping():
    moments = r.hinge_moments(1., lambda t: 2.)
    assert moments['mean'] == pytest.approx(.99/2)
    assert moments['variance'] == 0.


def estar(rows=None):
    if rows is None:
        rows = ['0.01 1 2 3 0.001 0.1 0', '0.1 1 2 3 0.04 0.1 0',
                '1 1 2 3 0.5 0.1 0', '10 1 2 3 5 0.1 0']
    # These numbers are synthetic fixtures, not NIST reference data.
    return ('<html><pre>CSDA Range\nMeV g/cm^2\n'+'\n'.join(rows)+'</pre></html>').encode()


def test_estar_and_c2():
    ranges = r.parse_estar(estar())
    assert ranges == {.01: .001, .1: .04, 1.: .5, 10.: 5.}
    assert r.c2_statistics(1., dict(range=.499, healthy=True), ranges)['status'] == '合格'
    assert r.c2_statistics(1., dict(range=.5, healthy=True), ranges)['status'] == '不合格'
    assert r.c2_statistics(1., dict(range=.499, healthy=False), ranges)['status'] == '判定できない'
    assert r.c2_statistics(1., {})['status'] == '未実施'


@pytest.mark.parametrize('raw', [
    b'<pre>1 2 3 4 5</pre>', estar().replace(b'CSDA Range', b'Other table'),
    estar().replace(b'g/cm^2', b'cm'), estar().replace(b'0.1 1 2 3', b'0.2 1 2 3'),
    estar().replace(b'0.04', b'nan'), estar().replace(b'0.04', b'-1'),
    estar().replace(b'0.04 0.1 0', b'0.04'), estar()+estar(),
])
def test_estar_rejects_unverified_shapes(raw):
    with pytest.raises(ValueError, match='ESTAR response differs'):
        r.parse_estar(raw)


def synthetic_b1():
    return dict(kerma=np.full(100, 2.1), collision_kerma=np.full(100, 2.),
                totals=np.array([200., 200., 0., 0., 0.]),
                violations=np.zeros(2), counts=np.zeros(7), maxima=np.zeros(2),
                brems=np.array([0., 1., 0., 0., 0., 0., 0., 0., 3.]), seconds=0.)


def test_b1_normalization_and_sizing():
    value = r.b1_statistics(synthetic_b1(), 200, 1.)
    assert value['rho_col'] == 1.
    assert value['rho_tr'] == pytest.approx(1.05)
    assert value['tr_minus_col'] == pytest.approx(.05)
    assert value['continuous_brems_fraction'] == .01
    assert value['status'] == '合格'
    assert r.required_histories(0) == 100000
    assert r.required_histories(.002) == 600000
    assert r.required_histories(.006) > 5000000


@pytest.mark.parametrize('key,index,value', [
    ('totals', 0, 199.), ('totals', 2, .001), ('violations', 0, 1),
    ('counts', 5, 1), ('counts', 6, 1), ('kerma', 0, np.nan),
    ('collision_kerma', 0, 0), ('brems', 0, np.inf),
])
def test_b1_unhealthy(key, index, value):
    data = synthetic_b1()
    data[key][index] = value
    assert r.b1_statistics(data, 200, 1.)['status'] == '判定できない'


def test_b1_shape():
    data = synthetic_b1()
    data['kerma'] = np.ones((100, 1))
    assert r.b1_statistics(data, 200, 1.)['status'] == '判定できない'


def test_b2_blocks():
    data = synthetic_b1()
    data.update(fine=np.full((100, 600), 2.02), kerma=np.full((100, 600), 2.2),
                collision_kerma=np.full((100, 600), 2.))
    rows = r.b2_statistics(data)
    assert len(rows) == 6
    assert rows[0]['depth_cm'] == [0, 5]
    assert rows[-1]['depth_cm'] == [25, 30]
    for row in rows:
        assert row['ratio'] == pytest.approx(1.01)
        assert row['sem'] == pytest.approx(0., abs=1e-16)
        assert row['status'] == '報告のみ'
    data['collision_kerma'][:, :100] = 0
    assert r.b2_statistics(data)[0]['status'] == '判定できない'


def test_saved_no_overwrite_and_resume(tmp_path):
    path = tmp_path/'condition.npz'
    config = dict(n=200, seed=1, provenance=dict(source_hashes={'a': '123'}))
    continuous = synthetic_b1()
    result = r.saved(path, config, lambda: continuous)
    before = path.read_bytes()
    def must_not_run():
        pytest.fail('resumed calculation was executed again')
    resumed = r.saved(path, config, must_not_run)
    for key in continuous:
        np.testing.assert_array_equal(result[key], continuous[key])
        np.testing.assert_array_equal(resumed[key], continuous[key])
    assert r.b1_statistics(resumed, 200, 1.) == r.b1_statistics(continuous, 200, 1.)
    with pytest.raises(RuntimeError, match='mismatch'):
        r.saved(path, dict(config, seed=2), must_not_run)
    assert path.read_bytes() == before


@pytest.fixture(scope='module')
def inputs():
    from prototypes.mv_electron.p2_common import TABLES, ELSEPA
    from prototypes.mv_electron.photon_data_p2 import TABLE
    from prototypes.mv_electron import run_p3s
    files = [TABLES['elsepa'], TABLE]+[ELSEPA/'tables'/f'z{z}_q-1.npz' for z in (1, 8)]
    try:
        for path in files:
            with path.open('rb') as stream:
                stream.read(1)
    except OSError as exc:
        pytest.skip('required ELSEPA/photon tables unreadable: '+str(exc))
    # Hash mismatches and numerical errors fail; only unavailable files skip.
    return run_p3s.inputs('elsepa')


def test_b1_transport(inputs):
    on = r.execute_b1(1.25, 200, 20291791, inputs)
    off = r.execute_b1(1.25, 200, 20291791, inputs, score_kerma=False)
    assert on['totals'][0] == 200*1.25
    assert on['totals'][2] == 0.
    assert r.photon_healthy(on, 200, 1.25)
    for key in ('kerma', 'collision_kerma'):
        assert on[key].shape == (100,)
        assert np.all(np.isfinite(on[key])) and np.all(on[key] > 0)
    for key in ('totals', 'counts'):
        np.testing.assert_array_equal(on[key], off[key])


def test_c1_transport_endpoints(inputs):
    from prototypes.mv_electron.physics import stopping
    integral = r.integrate_range(1., lambda t: stopping(t, False))
    assert integral['healthy']
    for j, expected in [(0, 1.), (40, 0.)]:
        result = r.c1_point(1., integral['range']*(1+r.X[j]), 20291900+j, inputs)
        assert r.c1_point_healthy(result, 1.)
        assert np.mean(result['ledger'][:, 2] > 0) == expected


def fake_provenance():
    return dict(source_hashes={name: 'synthetic-hash' for name in r.SOURCE_FILES},
                table_hashes={'fixture': 'synthetic-table'},
                versions=dict(python='test', numpy='test', numba='test'))


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    """Exercise orchestration with deterministic fakes, never real CLI commands."""
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path/'mpl'))
    monkeypatch.setattr(r, 'provenance', lambda inputs: fake_provenance())
    monkeypatch.setattr(r, 'integrate_range', lambda energy, stopping:
                        dict(range=energy-.01, relative_change=0., healthy=True))
    monkeypatch.setattr(r, 'hinge_moments', lambda energy, stopping:
                        dict(mean=energy-.01, variance=0.))
    def fake_point(energy, length, seed, inputs):
        result = point()
        result['seconds'] = .01
        result['ledger'][:, 2] = float(length < energy-.01)
        return result
    def fake_b1(energy, n, seed, inputs):
        result = synthetic_b1()
        result['totals'][0:2] = n*energy
        result['collision_kerma'] = np.full(100, n*energy/100)
        result['kerma'] = np.full(100, n*energy*1.05/100)
        result['seconds'] = .2
        return result
    def fake_b2(inputs):
        result = synthetic_b1()
        return {**{key: result[key] for key in ('totals', 'violations', 'counts')},
                'fine': np.full((100, 600), 2.02), 'kerma': np.full((100, 600), 2.2),
                'collision_kerma': np.full((100, 600), 2.), 'seconds': .3}
    monkeypatch.setattr(r, 'c1_point', fake_point)
    monkeypatch.setattr(r, 'execute_b1', fake_b1)
    monkeypatch.setattr(r, 'execute_b2', fake_b2)
    return r.Workflow((), out=tmp_path/'out', cache=tmp_path/'cache')


def test_stage_flow_and_outputs(workflow, monkeypatch):
    lock = workflow.make_lock()
    assert len(lock['C1']) == len(lock['C2']) == 3
    assert all(row['status'] == '未実施' for row in lock['C2'])
    assert all(row['status'] == '報告のみ' for row in lock['pilot'])
    assert [row['seed'] for row in lock['runs']] == list(range(20291810, 20291815))
    assert {row['n'] for row in lock['runs']} == {100000}
    assert len(list(workflow.cache.glob('p4bc_C1_*.npz'))) == 123
    assert set(p.name for p in workflow.out.iterdir()) == {'lock_p4bc.json'}
    with pytest.raises(FileNotFoundError):
        workflow.analysis_data()
    workflow.produce()
    assert set(p.name for p in workflow.out.iterdir()) == {'lock_p4bc.json'}
    data = workflow.analysis_data()
    assert len(data['B1']) == 5 and len(data['B2']) == 6
    assert all(row['status'] == '合格' for row in data['B1'])
    assert data['provenance'] == fake_provenance()
    assert data['seconds'] > 0
    def no_calculation(*args, **kwargs):
        pytest.fail('completed production should reuse NPZ')
    monkeypatch.setattr(r, 'execute_b1', no_calculation)
    monkeypatch.setattr(r, 'execute_b2', no_calculation)
    original = {path: path.read_bytes() for path in workflow.cache.glob('*.npz')}
    workflow.produce()
    assert data == workflow.analysis_data()
    workflow.publish_analysis()
    assert set(p.name for p in workflow.out.iterdir()) == {
        'lock_p4bc.json', 'checks_p4bc.json', r.FIGURE, 'RESULTS_P4BC.md'}
    assert (workflow.out/r.FIGURE).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    report = (workflow.out/'RESULTS_P4BC.md').read_text()
    for phrase in ('0.025%', 'B3は未実施', 'P1の核', '総計算時間', '分散 cm²', '試しSEM', '未実施'):
        assert phrase in report
    for path, content in original.items():
        assert path.read_bytes() == content
    with pytest.raises(FileExistsError):
        workflow.make_lock()
    with pytest.raises(FileExistsError):
        workflow.publish_analysis()


def test_run_interruption_resumes_per_thickness(workflow, monkeypatch):
    original_point = r.c1_point
    calls = []
    def interrupted(energy, length, seed, inputs):
        calls.append(seed)
        if len(calls) == 4:
            raise RuntimeError('synthetic interruption')
        return original_point(energy, length, seed, inputs)
    monkeypatch.setattr(r, 'c1_point', interrupted)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        workflow.make_lock()
    assert not (workflow.out/'lock_p4bc.json').exists()
    paths = list(workflow.cache.glob('*.npz'))
    assert len(paths) == 3
    preserved = {p: p.read_bytes() for p in paths}
    resumed_calls = []
    def resume(energy, length, seed, inputs):
        resumed_calls.append(seed)
        return original_point(energy, length, seed, inputs)
    monkeypatch.setattr(r, 'c1_point', resume)
    lock = workflow.make_lock()
    assert len(resumed_calls) == 120
    assert resumed_calls[0] == 20291803
    assert all(p.read_bytes() == data for p, data in preserved.items())
    assert all(row['status'] == '合格' for row in lock['C1'])
    with np.load(paths[0], allow_pickle=False) as npz:
        config = __import__('json').loads(str(npz['config_json']))
        assert config['provenance'] == fake_provenance()
        assert config['settings']['mode'] == 2
        assert config['settings']['hard_on'] is False


def test_production_interruption_resume(workflow, monkeypatch):
    workflow.make_lock()
    original = r.execute_b1
    calls = []
    def interrupted(energy, n, seed, inputs):
        calls.append(seed)
        if len(calls) == 3:
            raise RuntimeError('production interrupted')
        return original(energy, n, seed, inputs)
    monkeypatch.setattr(r, 'execute_b1', interrupted)
    with pytest.raises(RuntimeError, match='production interrupted'):
        workflow.produce()
    paths = list(workflow.cache.glob('*B1_production*'))
    assert len(paths) == 2
    preserved = {p: p.read_bytes() for p in paths}
    calls.clear()
    def resumed(energy, n, seed, inputs):
        calls.append(seed)
        return original(energy, n, seed, inputs)
    monkeypatch.setattr(r, 'execute_b1', resumed)
    workflow.produce()
    assert calls == [20291812, 20291813, 20291814]
    assert all(p.read_bytes() == content for p, content in preserved.items())


def fetch_fixture():
    raw = estar()
    return raw, r.parse_estar(raw), dict(url=r.ESTAR_URL, fields=r.ESTAR_FIELDS,
        retrieved_utc='2026-10-13T00:00:00+00:00', sha256=r.hashlib.sha256(raw).hexdigest())


def test_fetch_persistence_and_c2(workflow):
    meta = r.fetch_outputs(workflow.out, request=fetch_fixture)
    assert (workflow.out/'estar_range_water.html').read_bytes() == estar()
    assert r.reference(workflow.out)[1]['metadata'] == meta
    lock = workflow.make_lock()
    assert all(row['status'] != '未実施' for row in lock['C2'])
    def never_request():
        pytest.fail('must check outputs before network request')
    with pytest.raises(FileExistsError):
        r.fetch_outputs(workflow.out, request=never_request)


@pytest.mark.parametrize('existing', ['fetch_p4bc.json', 'estar_range_water.html'])
def test_fetch_partial_outputs_are_not_overwritten(tmp_path, existing):
    (tmp_path/existing).write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        r.fetch_outputs(tmp_path, request=lambda: pytest.fail('network request forbidden'))
    assert (tmp_path/existing).read_bytes() == b'existing'


def test_fetch_rejects_changed_shape_without_outputs(tmp_path):
    def bad_response():
        raw, ranges, metadata = fetch_fixture()
        return b'<html>Error</html>', ranges, metadata
    with pytest.raises(ValueError, match='ESTAR response differs'):
        r.fetch_outputs(tmp_path, request=bad_response)
    assert not list(tmp_path.iterdir())


def test_reference_hash_tampering(tmp_path):
    raw, _, metadata = fetch_fixture()
    metadata['sha256'] = 'wrong'
    r.write_json_once(tmp_path/'fetch_p4bc.json', metadata)
    r.write_bytes_once(tmp_path/'estar_range_water.html', raw)
    with pytest.raises(RuntimeError, match='sha256 mismatch'):
        r.reference(tmp_path)


def test_missing_grid_point_is_c2_unperformed(tmp_path):
    raw, _, metadata = fetch_fixture()
    raw = raw.replace(b'0.1 1 2 3', b'0.2 1 2 3')
    metadata['sha256'] = r.hashlib.sha256(raw).hexdigest()
    r.write_bytes_once(tmp_path/'estar_range_water.html', raw)
    r.write_json_once(tmp_path/'fetch_p4bc.json', metadata)
    ranges, ref = r.reference(tmp_path)
    assert ranges is None and ref['status'] == '未実施'
    assert 'no interpolation' in ref['reason']


def test_changed_provenance_stops_before_checkpoint(workflow, monkeypatch):
    monkeypatch.setattr(r, 'provenance', lambda inputs: {'changed': True})
    with pytest.raises(RuntimeError, match='changed'):
        workflow.b1(0, 100000)
    assert not workflow.cache.exists()


def test_changed_provenance_during_calculation_not_saved(workflow, monkeypatch):
    def changed():
        monkeypatch.setattr(r, 'provenance', lambda inputs: {'changed': True})
        return synthetic_b1()
    with pytest.raises(RuntimeError, match='changed'):
        workflow.condition('test', workflow.config('test'), changed)
    assert not workflow.cache.exists()


@pytest.mark.parametrize('tamper', ['provenance', 'rules', 'seed', 'n'])
def test_lock_mismatch_refuses_production(workflow, tamper):
    # Make a deliberately inconsistent lock before its single exclusive write.
    pilots = [dict(E=e, n=100000, seed=20291790+i, status='報告のみ', sem=0.)
              for i, e in enumerate(r.B_ENERGIES)]
    lock = dict(provenance=workflow.current, rules=dict(r.RULES), C1=[{}]*3, C2=[{}]*3,
                pilot=pilots, runs=[dict(E=e, n=100000, seed=20291810+i)
                                   for i, e in enumerate(r.B_ENERGIES)])
    if tamper == 'provenance':
        lock['provenance'] = {'changed': True}
    elif tamper == 'rules':
        lock['rules']['fe'] = .1
    else:
        lock['runs'][0][tamper] = 123
    r.write_json_once(workflow.out/'lock_p4bc.json', lock)
    with pytest.raises(RuntimeError, match='mismatch|differs'):
        workflow.produce()
    assert not workflow.cache.exists()


@pytest.mark.parametrize('unhealthy', [False, True])
def test_pilot_stop_recorded_before_production(workflow, monkeypatch, unhealthy):
    original = r.execute_b1
    def bad_pilot(energy, n, seed, inputs):
        result = original(energy, n, seed, inputs)
        if unhealthy:
            result['counts'][5] = 1
        else:
            result['collision_kerma'] *= np.linspace(.8, 1.2, 100)
        return result
    monkeypatch.setattr(r, 'execute_b1', bad_pilot)
    lock = workflow.make_lock()
    assert (workflow.out/'lock_p4bc.json').exists()
    assert all(row['n'] is None if unhealthy else row['n'] > 5000000 for row in lock['runs'])
    with pytest.raises(RuntimeError, match='production forbidden'):
        workflow.produce()
    assert not list(workflow.cache.glob('*production*'))
    assert not (workflow.cache/'p4bc_B2.npz').exists()


def test_cli_dispatch_without_executing_commands(monkeypatch):
    for name in ('fetch', 'run', 'production', 'analyse'):
        monkeypatch.setattr(r, name, lambda name=name: name)
    for name in ('fetch', 'run', 'production', 'analyse'):
        assert r.main([name]) == name


def test_report_and_figure_refuse_partial_existing_output(workflow):
    workflow.out.mkdir()
    (workflow.out/r.FIGURE).write_bytes(b'prior figure')
    with pytest.raises(FileExistsError):
        workflow.publish_analysis()
    assert not (workflow.out/'checks_p4bc.json').exists()
    assert (workflow.out/r.FIGURE).read_bytes() == b'prior figure'
