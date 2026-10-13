"""Synthetic P3b-3 checks: no transport, EGS5 processes or external writes."""
import ast
import copy
import json
import math
from pathlib import Path
import struct
import numpy as np
import pytest
from prototypes.mv_electron import analysis_p3b3 as a, run_p3b3 as r


def curve(shift=0.):
    x = (np.arange(36)+.5)*.25
    return np.clip(1-(x-4-shift)/3, 0, 1)


def regions(N=4000000, seed=20291760):
    jitter = np.linspace(-.2, .2, 100)
    out = {key:np.ones(shape) for key, shape in a.SHAPES.items()}
    out['axis_coarse'] *= (1+jitter[:, None])
    for j, key in enumerate(('profile', 'profile_fine')):
        width = .25 if j == 0 else .125
        x = np.abs(-9+(np.arange(out[key].shape[-1])+.5)*width)
        p = np.clip(1-(x[None, :]-4-jitter[:, None]*.1)/3, 0, 1)
        out[key][:] = p[:, None, None, :]
    return dict(out, N=np.asarray(N), seed=np.asarray(seed))


def proto(N=4000000, seed=20291760):
    data = regions(N, seed)
    for key in r.kernel.KEYS:
        data.setdefault(key, np.zeros((100, 1)))
    data.update(region_ledger=np.zeros((100, 15), int), totals=np.zeros((100, 6)),
                counts=np.zeros((100, 16), int), violations=np.zeros((100, 3), int),
                brems=np.zeros((100, 10)), coarse=np.ones((100, 150)),
                fine=np.ones((100, 600)), nbatch=np.asarray(100),
                batch_seeds=np.asarray(r.batch_seeds(seed, 100)), wall_seconds=np.asarray(2.))
    data['totals'][:, :2] = [10., 9.]
    data['counts'][:, 0] = 23  # legitimate counters, including row 5
    data['brems'][:, 0] = 11
    data['batch_totals'] = data['totals'].copy()
    return data


def reference(N=4000000, seed=20291761):
    data = regions(N, seed)
    data.update(region_ledger=np.zeros(15, int), totals=np.array([1000., 900., 100., 0, 0, 0, 0, 0]),
                batchin=np.full(100, 10.), coarse=np.ones((100, 150)), fine=np.ones((100, 600)))
    summary = dict(status='合格', tests=dict(balance=True, callbacks=True), ledger=data['totals'].tolist(),
                   wall_seconds=2., CPU_seconds=1.)
    return data, summary


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path/'output')
    monkeypatch.setattr(r, 'CACHE', tmp_path/'cache')
    monkeypatch.setattr(r.kernel, 'fingerprint', lambda p: dict(synthetic='fixed'))
    monkeypatch.setattr(r.egs, 'egs_hashes', lambda: {})
    monkeypatch.setattr(r, 'spectrum', lambda p: (np.array([0., 1., 10.]), np.array([.5, 1.])))
    def forbidden(*args, **kwargs):
        raise AssertionError('transport/process attempted')
    monkeypatch.setattr(r, 'run_batches', forbidden)
    monkeypatch.setattr(r.egs, 'run_egs', forbidden)
    return tmp_path


def save_proto(name, N, seed):
    path = r.proto_path(name)
    r.io.save(path, **proto(N, seed))
    r.io.write(path.parent/'config.json', r.proto_config(N, seed))


def save_egs(name, N, seed, work=False):
    target = r.OUT/'egs5'/name
    data, summary = reference(N, seed)
    r.io.save(target/'result.npz', **data)
    r.io.write(target/'summary.json', summary)
    config = r.egs_config(N, seed, *r.spectrum(r.params()))
    r.io.write(target/'config.json', config)
    if work:
        r.io.write(r.CACHE/'egs5'/name/'config.json', config)
    return config


@pytest.fixture
def locked(env):
    save_proto('pilot_proto', r.PILOT_N, r.SEEDS['pilot_proto'])
    save_egs('pilot_egs', r.PILOT_N, r.SEEDS['pilot_egs'])
    lock = r.lock_content(r.read_proto('pilot_proto', r.PILOT_N, r.SEEDS['pilot_proto']),
                          r.read_egs('pilot_egs', r.PILOT_N, r.SEEDS['pilot_egs']))
    r.io.write(r.OUT/'pilot_lock_p3b3.json', lock)
    return lock


@pytest.mark.parametrize('code', ['proto', 'egs'])
def test_save_read_dose(env, code):
    if code == 'proto':
        save_proto('pilot_proto', 4000000, 20291760)
        data = r.read_proto('pilot_proto', 4000000, 20291760)
    else:
        save_egs('pilot_egs', 4000000, 20291761)
        data = r.read_egs('pilot_egs', 4000000, 20291761)
    converted = a.dose(data)
    assert set(converted) == set(a.SHAPES)
    for key in a.SHAPES:
        np.testing.assert_allclose(converted[key], data[key]*1.602176634e-10/(40000*a.VOLUMES[key]), rtol=1e-12)
    assert a.VOLUMES == dict(axis_coarse=1., axis_fine=.4, slab_coarse=.25, profile=.5, profile_fine=.25)
    assert 'coarse' not in converted


@pytest.mark.parametrize('mutation', ['missing', 'shape', 'nan', 'negative', 'N', 'nbatch'])
def test_input_rejected(mutation):
    data = regions()
    if mutation == 'missing': data.pop('profile')
    elif mutation == 'shape': data['axis_coarse'] = np.ones((100, 150))
    elif mutation == 'nan': data['axis_fine'][0, 0] = np.nan
    elif mutation == 'negative': data['profile'][0, 0, 0, 0] = -1
    elif mutation == 'N': data['N'] = 101
    else: data['nbatch'] = 99
    with pytest.raises(ValueError): a.dose(data)


@pytest.mark.parametrize('width', [.25, .125])
def test_fold_orientation(width):
    n = int(9/width)
    p = np.zeros((100, 2, 3, 2*n))
    x = -9+(np.arange(2*n)+.5)*width
    for j in range(2):
        p[:, j] = np.where(x >= 0, 10+20*j+abs(x), 20+20*j+abs(x))
    expected = 25+(np.arange(n)+.5)*width
    np.testing.assert_allclose(a.fold(p), np.broadcast_to(expected, (100, 3, n)), atol=1e-12)
    np.testing.assert_allclose(a.directions(p)[0, :, 0, 0], [10+width/2, 20+width/2, 30+width/2, 40+width/2])


def test_crossing_definition():
    assert a.crossing(curve()) == pytest.approx(5.5, abs=1e-12)
    assert a.crossing(13*curve()) == pytest.approx(5.5, abs=1e-12)
    assert a.crossing(curve(), fraction=.2) == pytest.approx(6.4, abs=1e-12)
    assert a.crossing(curve(), fraction=.8) == pytest.approx(4.6, abs=1e-12)
    assert a.crossing(np.ones(36)) is None
    assert a.crossing(np.zeros(36)) is None
    p = curve(); p[0] = 0
    assert a.crossing(p) is None
    p = curve(); p[20] = .5; p[21] = .4
    assert a.crossing(p) == pytest.approx(5.125, abs=1e-12)
    p[26] = 1  # recrossing must not change first crossing
    assert a.crossing(p) == pytest.approx(5.125, abs=1e-12)
    p[1] = np.inf
    assert a.crossing(p) is None


def test_bootstrap_amplitude_zero():
    for data in (np.tile(curve(), (100, 1)), np.linspace(.5, 2, 100)[:, None]*curve()):
        p = a.bootstrap_position(data)
        assert p == a.bootstrap_position(data)
        assert p['missing'] == 0
        assert p['position'] == pytest.approx(5.5, abs=1e-12)
        assert p['SEM'] == 0


def test_bootstrap_translation_theory():
    shifts = np.linspace(-.1, .1, 100)
    data = np.array([curve(s) for s in shifts])
    result = a.bootstrap_position(data)
    theory = shifts.std(ddof=1)/10
    assert result['position'] == pytest.approx(5.5, abs=1e-12)
    assert result['SEM'] == pytest.approx(theory, rel=.15)  # no further sqrt(100)
    rng = np.random.default_rng(a.BOOT_SEED)
    positions = [a.crossing(data[rng.integers(0, 100, 100)].mean(0)) for _ in range(2000)]
    assert result['SEM'] == np.std(positions, ddof=1)


def test_missing_bootstrap_and_j3_order():
    assert a.bootstrap_position(np.ones((100, 36)))['missing'] == 2000
    mixed = np.tile(curve(), (100, 1))
    mixed[50:, 0] = 0
    sampled = a.bootstrap_position(mixed)
    assert sampled['position'] is not None
    assert 0 < sampled['missing'] < 2000
    assert sampled['SEM'] is None
    left = a.dose(regions()); right = a.dose(regions())
    left['profile'].fill(1)
    comparison = a.compare(left, right, 1.)
    assert all(v['status'] == '判定できない' for v in comparison['J3'])
    assert a.classify(None, .2, .1) == '判定できない'


@pytest.mark.parametrize('tolerance', [1., .1])
def test_classification_boundaries(tolerance):
    label = '1%' if tolerance == 1 else '0.1 cm'
    assert a.classify(tolerance, tolerance/3, tolerance) == label+'以内'
    assert a.classify(-tolerance, 0, tolerance) == label+'以内'
    assert a.classify(np.nextafter(tolerance, np.inf), 0, tolerance) == label+'を超える'
    assert a.classify(0, np.nextafter(tolerance/3, np.inf), tolerance) == '統計不足'
    assert a.classify(tolerance*10, tolerance, tolerance) == '統計不足'


@pytest.mark.parametrize('negative', [False, True])
def test_correlated_bins_and_directions(negative):
    z = np.linspace(-.2, .2, 100)
    left = a.dose(regions()); right = a.dose(regions())
    p = np.empty(a.SHAPES['profile'])
    for direction, sign in enumerate((1, -1) if negative else (1, 1)):
        p[:, direction] = (1+sign*z)[:, None, None]
    # Also correlate opposite sides negatively, leaving a residual direction mean.
    if negative:
        p[:, 0, :, :36] = (1-.5*z)[:, None, None]
    left['profile'] = p
    right['profile'] = np.ones_like(p)
    left['axis_coarse'] = np.broadcast_to((1+z)[:, None], (100, 120)).copy()
    if negative:
        left['axis_coarse'][:, 20:30] = (1-.5*z)[:, None]
    right['axis_coarse'] = np.ones((100, 120))
    result = a.compare(left, right, 1.)
    expected = a.difference(left['axis_coarse'][:, 20:40].mean(1), np.ones(100), 100)
    assert result['J1'][0]['SEM'] == expected['SEM']
    expected = a.difference(a.fold(p)[:, 0, :16].mean(1), np.ones(100), 100)
    assert result['J2'][0]['SEM'] == expected['SEM']
    if not negative:
        assert result['J1'][0]['SEM'] == pytest.approx(100*z.std(ddof=1)/10)


def test_difference_hand_calculation():
    x = np.arange(100, dtype=float)
    out = a.difference(x, 2*x+5, 3)
    assert out['delta'] == pytest.approx((-x.mean()-5)*3)
    assert out['SEM'] == pytest.approx(math.sqrt(5)*x.std(ddof=1)/10*3)
    assert out['upper_abs_95'] == abs(out['delta'])+1.96*out['SEM']


def test_dref_smoothing_first_peak():
    x = np.arange(120, dtype=float)
    assert a.dref(np.tile(x, (100, 1))) == np.convolve(x, np.ones(5)/5, mode='same').max()
    for value in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError): a.dref(np.full((100, 120), value))
    left = a.dose(regions()); left['axis_fine'].fill(1)
    report = a.reports(left, left, 1)
    assert report['peaks']['proto'] == pytest.approx(.25)
    assert [row['interval_cm'] for row in report['intervals']] == [[0, 2], [2, 3], [3, 5]]
    assert all('status' not in row for row in report['intervals']+report['slab_blocks'])


def pilot_stats(dose_sem=.6, position_sem=.02):
    return dict(J1=[dict(SEM=dose_sem)]*5, J2=[dict(SEM=dose_sem)]*3,
                J3=[dict(SEM=position_sem)]*3)


def test_size_rule():
    assert a.main_size(pilot_stats()) == dict(N_main=20000000, status='合格')
    assert a.main_size(pilot_stats(.01, .001))['N_main'] == 2000000
    assert a.main_size(pilot_stats(0, .1))['N_main'] == 54000000
    assert a.main_size(pilot_stats(2, .01))['status'] == '相談'
    for s in (0, np.nan, np.inf, -1):
        with pytest.raises(ValueError, match='外挿できない'): a.main_size(pilot_stats(s, s))


def sym_data(bias=0):
    z = np.linspace(-.1, .1, 100); z -= z.mean()
    p = np.ones(a.SHAPES['profile'])
    p[:, 0, :, 36:] += (z+bias)[:, None, None]
    p[:, 0, :, :36] -= z[:, None, None]
    p[:, 1, :, 36:] += .5*z[:, None, None]
    p[:, 1, :, :36] -= .2*z[:, None, None]
    return p


def test_symmetry_detection():
    p = sym_data(); result = a.symmetry(p)
    assert result['status'] == '偏りは検出されず'
    row = result['contrasts'][0]
    expected = 2*np.linspace(-.1, .1, 100).std(ddof=1)/10
    assert row['SEM'] == pytest.approx(expected)
    assert row['detection_percent'] == pytest.approx(100*3.61*expected)
    assert a.symmetry(sym_data(5*expected))['status'] == '偏りを検出'
    assert a.symmetry(np.ones_like(p))['status'] == '判定できない（SEM 0）'
    p = np.ones_like(p); p[:, 0, :, 36:] = 2
    assert a.symmetry(p)['status'] == '偏りを検出'


def test_symmetry_threshold_equal(monkeypatch):
    monkeypatch.setattr(a, 'mean_sem', lambda x: (np.full((3, 3), 2.77), np.ones((3, 3))))
    assert a.symmetry(sym_data())['status'] == '偏りを検出'


@pytest.mark.parametrize('mutation', ['region', 'failure', 'balance', 'brems'])
def test_one_unhealthy_proto_batch(mutation):
    data = proto()
    r.validate(data, 'proto', 4000000, 20291760)
    if mutation == 'region': data['region_ledger'][77, 2] = 1
    elif mutation == 'failure': data['counts'][77, 5] = 1
    elif mutation == 'balance': data['violations'][77, 0] = 1
    else: data['brems'][77, 2] = 1
    with pytest.raises(RuntimeError): r.validate(data, 'proto', 4000000, 20291760)


@pytest.mark.parametrize('mutation', ['region', 'status', 'test', 'missing_tests'])
def test_unhealthy_egs(mutation):
    data, summary = reference()
    r.validate(data, 'egs', 4000000, 20291761, summary)
    if mutation == 'region': data['region_ledger'][2] = 1
    elif mutation == 'status': summary['status'] = '不合格'
    elif mutation == 'test': summary['tests']['balance'] = False
    else: summary['tests'] = {}
    with pytest.raises(RuntimeError): r.validate(data, 'egs', 4000000, 20291761, summary)


def checkpoint(config):
    def record(data):
        return struct.pack('=i', len(data))+data+struct.pack('=i', len(data))
    state = struct.pack('=i', config['N']//100)+bytes(607596)
    extra = bytes(8*320*16*100+4*16+4*100+8*5*1000+4)
    return b''.join(record(x) for x in (r.egs.checkpoint_header(config), state, extra))


@pytest.mark.parametrize('state', ['new', 'resume', 'complete', 'no_checkpoint', 'partial', 'mismatch', 'bad_checkpoint'])
def test_egs_dispatch(env, monkeypatch, state):
    name = 'pilot_egs'; N = 4000000; seed = 20291761
    work = r.CACHE/'egs5'/name; dest = r.OUT/'egs5'/name
    config = r.egs_config(N, seed, *r.spectrum(r.params()))
    calls = []
    if state in ('resume', 'no_checkpoint', 'bad_checkpoint'):
        r.io.write(work/'config.json', config); r.io.write(dest/'config.json', config)
        if state != 'no_checkpoint':
            (work/'checkpoint.bin').write_bytes(checkpoint(config) if state == 'resume' else b'bad')
    elif state in ('complete', 'partial', 'mismatch'):
        save_egs(name, N, seed)
        if state == 'partial':
            # Build partial output in a separate destination, without deletion.
            r.OUT = env/'partial'; dest = r.OUT/'egs5'/name
            r.io.save(dest/'result.npz', **reference(N, seed)[0])
        elif state == 'mismatch':
            r.io.write(dest/'config.json', dict(config, seed=0))
    def launch(*args, **kw):
        calls.append((args, kw)); save_egs(name, N, seed, work=work.exists())
    monkeypatch.setattr(r.egs, 'run_egs', launch)
    if state in ('no_checkpoint', 'partial', 'mismatch', 'bad_checkpoint'):
        with pytest.raises((RuntimeError, FileNotFoundError)): r.run_reference(name, N, seed)
        assert calls == []
    else:
        r.run_reference(name, N, seed)
        assert len(calls) == (0 if state == 'complete' else 1)
        if calls:
            assert calls[0][1].get('resume', False) == (state == 'resume')
            assert calls[0][0][:3] == (name, N, seed)


def test_proto_runner_and_roundtrip(env, monkeypatch):
    calls = []
    def run(work, key, p, N, seed, workers):
        calls.append((key, N, seed, workers))
        data = proto(N, seed)
        r.io.save(work/'result.npz', **{k:v for k, v in data.items() if k != 'batch_totals'})
        return work/'result.npz'
    monkeypatch.setattr(r, 'run_batches', run)
    data = r.run_proto('pilot_proto', 4000000, 20291760)
    assert calls == [('p3b_field', 4000000, 20291760, 2)]
    assert np.array_equal(data['totals'], data['batch_totals'])
    with np.load(r.proto_path('pilot_proto'), allow_pickle=False) as f:
        assert all(f[k].dtype != object for k in f.files)
    r.run_proto('pilot_proto', 4000000, 20291760)
    assert len(calls) == 1


def test_saved_proto_requires_batch_totals(env):
    save_proto('pilot_proto', 4000000, 20291760)
    data = proto(); data.pop('batch_totals')
    r.io.save(r.proto_path('pilot_proto'), **data)
    with pytest.raises(RuntimeError, match='batch_totals missing'):
        r.read_proto('pilot_proto', 4000000, 20291760)


@pytest.mark.parametrize('entry', ['production', 'analyse'])
def test_missing_lock(env, entry):
    with pytest.raises(RuntimeError, match='lock missing'): getattr(r, entry)()


@pytest.mark.parametrize('entry', ['production', 'analyse'])
@pytest.mark.parametrize('mutation', ['failed', 'hash', 'fingerprint', 'egs_sources', 'pilot_result', 'summary',
                                    'N_main', 'statistics', 'Dref', 'seeds', 'rules', 'regions'])
def test_lock_refuses_changes(locked, entry, mutation):
    path = r.OUT/'pilot_lock_p3b3.json'; lock = copy.deepcopy(locked)
    if mutation == 'failed': lock['status'] = '相談'; lock['N_main'] = 162000000
    elif mutation == 'hash': lock['fixed']['sources']['analysis_p3b3.py'] = 'changed'
    elif mutation == 'fingerprint': lock['fixed']['fingerprint']['synthetic'] = 'changed'
    elif mutation == 'egs_sources': lock['fixed']['egs_sources']['prototypes/mv_electron/run_p3b2.py'] = 'changed'
    elif mutation == 'pilot_result':
        data = proto(); data['axis_coarse'][0, 0] += 1
        r.io.save(r.proto_path('pilot_proto'), **data)
    elif mutation == 'summary':
        summary_path = r.OUT/'egs5/pilot_egs/summary.json'
        summary = r.read_json(summary_path); summary['CPU_seconds'] += 1
        r.io.write(summary_path, summary)
    elif mutation == 'N_main': lock['N_main'] += 2000000
    elif mutation == 'statistics': lock['statistics']['J1'][0]['SEM'] += .01
    elif mutation == 'Dref': lock['Dref'] *= 2
    elif mutation == 'seeds': lock['fixed']['seeds']['main_egs'] += 1
    elif mutation == 'rules': lock['fixed']['rules']['margin'] = 2.25
    else: lock['fixed']['regions'][0] = 1
    r.io.write(path, lock)
    with pytest.raises(RuntimeError): getattr(r, entry)()


def test_lock_roundtrip_and_existing_pilot(locked):
    assert r.require_lock() == locked
    assert r.pilot() == locked


def test_production_sequence_and_fixed_counts(locked, monkeypatch):
    calls = []
    monkeypatch.setattr(r, 'run_proto', lambda name, N, seed: calls.append(('proto', name, N, seed)))
    monkeypatch.setattr(r, 'run_reference', lambda name, N, seed, **kw:
                        r.read_egs(name, N, seed) if name == 'pilot_egs' else calls.append(('egs', name, N, seed)))
    assert r.production() == dict(N_main=locked['N_main'], status='完了')
    assert calls == [('proto', 'main_proto', locked['N_main'], 20291762),
                     ('egs', 'main_egs', locked['N_main'], 20291763)]


def test_pilot_sequence_and_reference_label(env, monkeypatch):
    calls = []
    def run_proto(name, N, seed):
        calls.append(('proto', N, seed)); save_proto(name, N, seed)
        return r.read_proto(name, N, seed)
    def run_egs(name, N, seed):
        calls.append(('egs', N, seed)); save_egs(name, N, seed)
        return r.read_egs(name, N, seed)
    monkeypatch.setattr(r, 'run_proto', run_proto); monkeypatch.setattr(r, 'run_reference', run_egs)
    lock = r.pilot()
    assert calls == [('proto', 4000000, 20291760), ('egs', 4000000, 20291761)]
    assert '参考' in lock['pilot_label']
    assert r.read_json(r.OUT/'pilot_lock_p3b3.json') == lock


def test_consultation_blocks_production(locked):
    lock = copy.deepcopy(locked); lock['status'] = '相談'; lock['N_main'] = 162000000
    r.io.write(r.OUT/'pilot_lock_p3b3.json', lock)
    with pytest.raises(RuntimeError, match='相談'): r.production()


def test_analyse_requires_both_results(locked, monkeypatch):
    called = []
    monkeypatch.setattr(a, 'compare', lambda *args: called.append(True))
    # The lock itself recomputes pilot statistics; isolate its verified result here.
    monkeypatch.setattr(r, 'require_lock', lambda: locked)
    save_proto('main_proto', locked['N_main'], r.SEEDS['main_proto'])
    with pytest.raises(RuntimeError, match='both production'): r.analyse()
    assert called == []


def test_analyse_order_outputs_and_symmetry_warning(locked, monkeypatch):
    N = locked['N_main']
    save_proto('main_proto', N, r.SEEDS['main_proto'])
    save_egs('main_egs', N, r.SEEDS['main_egs'])
    order = []; original_compare = a.compare; original_sym = a.symmetry
    def compare(*args):
        order.append('compare'); return original_compare(*args)
    def symmetry(*args):
        order.append('symmetry'); value = original_sym(*args)
        value['contrasts'][0].update(status='偏りを検出', z=5.)
        return value
    monkeypatch.setattr(a, 'compare', compare); monkeypatch.setattr(a, 'symmetry', symmetry)
    # Verify plotting separately, so the analysis ordering assertion is unambiguous.
    plotted = []
    monkeypatch.setattr(r, 'figures', lambda *args: plotted.append(True))
    result = r.analyse()
    assert order == ['compare', 'symmetry', 'symmetry', 'compare', 'compare']
    assert plotted == [True]
    assert len(result['conclusion_prefix']) == 2
    assert sum(len(v) for v in result['comparisons'].values()) == 11
    assert len(result['input_hashes']) == 4
    assert result['ledgers']['proto']['deposited_over_incoming'] == .9
    assert result['ledgers']['egs']['deposited_over_incoming'] == .9
    assert (r.OUT/'analysis_p3b3.json').is_file()


def test_figures(env):
    r.OUT.mkdir(parents=True)
    data = a.dose(regions())
    r.figures(data, data, a.dref(data['axis_coarse']), a.compare(data, data, a.dref(data['axis_coarse'])))
    assert all((r.OUT/name).stat().st_size > 1000 for name in
               ('pdd_p3b3.png', 'profiles_p3b3.png', 'symmetry_p3b3.png'))


def test_no_deletion_commands():
    forbidden = {'unlink', 'rmdir', 'rmtree', 'remove', 'system', 'Popen', 'run'}
    for module in (a, r):
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden


def test_cache_environment():
    tree = ast.parse(Path(r.__file__).read_text())
    assert 'CHATCARLO_P3B_CACHE' in Path(r.__file__).read_text()
    assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'OUT' for t in n.targets)
               for n in tree.body)
