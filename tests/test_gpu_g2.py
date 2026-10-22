"""G2 CPU acceptance: exact coefficients/traversals/moments and ABI regression."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pytest
from chatcarlo.gpu import bundle, core, g2_tables
from chatcarlo.gpu.g2_core import Core
from chatcarlo.scene import load_scene
from chatcarlo.tally import VoxelGrid, accumulate_track_length_multi, standard_error
from chatcarlo.transport import run_transport

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/validation/gpu/g2'


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp('g2')
    scene = load_scene(str(DOC/'scene_lead_slab.yaml'))
    bundle.create(scene, None, path, g2_tally=True)
    arrays, _, _, settings, _ = bundle.load(path)
    driver = core.Core(arrays, build_dir=path/'build')
    return driver, arrays, settings['material_names']


def probe(driver, arrays, cfg, x, which, out, h10=None, count=None):
    x = np.ascontiguousarray(x, float)
    cfg = np.ascontiguousarray(cfg, float)
    fn = driver.lib.g2_probe
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn(which, driver.nm, arrays['g2_table'].ctypes.data, cfg.ctypes.data,
       x.ctypes.data, len(x), out.ctypes.data,
       h10.ctypes.data if h10 is not None else None,
       count.ctypes.data if count is not None else None)


def historical():
    spec = importlib.util.spec_from_file_location('g2_old', ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/run_chatcarlo.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_gate0a_coefficients(setup):
    from chatcarlo.materials import mu_en_rho, _XAAMDI_FILES, _muen_interpolator
    from chatcarlo.dose_coefficients import h_star_10_per_fluence, _table
    driver, arrays, names = setup
    cfg = g2_tables.config(arrays)
    for m, name in enumerate(names):
        knots = np.exp(_muen_interpolator(_XAAMDI_FILES[name]).x)
        knots = knots[(knots >= 1) & (knots <= 150)]
        ee, _ = _table()
        knots = np.r_[knots, ee[(ee >= 1) & (ee <= 150)], 10., 1., 150.]
        # Both sides of every absorption edge AND every tabulated knot.
        e = np.r_[np.geomspace(1, 150, 200000), knots, np.nextafter(knots, 0), np.nextafter(knots, np.inf)]
        e = e[(e >= 1) & (e <= 150)]
        x = np.zeros((len(e), 10)); x[:, 0] = e; x[:, 1] = m
        out = np.zeros((len(e), 2))
        probe(driver, arrays, cfg, x, 0, out)
        np.testing.assert_allclose(out[:, 0], mu_en_rho(name, e), rtol=1e-12, atol=0)
        np.testing.assert_allclose(out[:, 1], h_star_10_per_fluence(e), rtol=1e-12, atol=0)


def test_gate0a_overlap_million(setup):
    driver, arrays, _ = setup
    rng = np.random.default_rng(720)
    a, b = rng.uniform(-30, 30, (2, 1000000, 3))
    a[:8] = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-2, 0, 0], [0, 0, 0], [1, 1, 1]]
    b[:8] = [[0, 0, 0], [2, 2, 2], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [2, 0, 0], [1e-13, 0, 0], [1, 1, -1]]
    lo, hi = np.full(3, -1.), np.ones(3)
    x = np.zeros((len(a), 10)); x[:, :3] = a; x[:, 3:6] = b
    out = np.zeros(len(a))
    probe(driver, arrays, np.r_[lo, hi], x, 2, out)
    expected = historical().old._segment_box_overlap_cm(a, b, lo, hi)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def segments():
    # Direction components use the production 1e-12 cutoff. The external top
    # face belongs outside the grid; the bottom face belongs to cell zero.
    rows = []
    for o, d, length in [
        ([-1, .5, .5], [1, 0, 0], 5), ([4, .5, .5], [-1, 0, 0], 5),
        ([0, 0, 0], [1, 1, 1], 3), ([3, 3, 3], [-1, -1, -1], 3),
        ([0, 1, 1], [1, 0, 0], 3), ([0, 3, 1], [1, 0, 0], 3),
        ([0, 0, 1], [1, 0, 0], 3), ([.5, .5, .5], [1, 0, 0], 0),
        ([.5, .5, .5], [1, 0, 0], .25),  # detector-truncated
        ([1-5e-8, .5, .5], [1, 0, 0], .5+5e-8),
        ([1-2e-7, .5, .5], [1, 0, 0], .5+2e-7),
        ([.5, 1-5e-8, .5], [1, 5e-13, 0], 2),
        ([.5, 1-5e-8, .5], [1, 2e-12, 0], 2),
        ([.5, .5, .5], [1, .999999999, 0], 2)]:
        rows.append([*o, *d, length, 0, 0, 0])
    return np.array(rows, float)


def test_gate0b_grid_and_boundary_rules(setup):
    from chatcarlo.materials import mu_en_rho, density
    from chatcarlo.dose_coefficients import h_star_10_per_fluence
    driver, arrays, names = setup
    x = segments()
    for m, name in enumerate(names):
        grid = VoxelGrid(np.zeros(3), (3, 3, 3), 1.)
        e = np.linspace(10, 150, len(x))
        x[:, 7] = e*mu_en_rho(name, e)*density(name)
        x[:, 8] = h_star_10_per_fluence(e)
        expected_k = np.zeros(grid.shape); expected_h = np.zeros(grid.shape)
        accumulate_track_length_multi(((expected_k, x[:, 7]), (expected_h, x[:, 8])),
                                      grid, x[:, :3], x[:, 3:6], x[:, 6])
        k = np.zeros(grid.shape); h = np.zeros(grid.shape); count = np.zeros(grid.shape)
        probe(driver, arrays, g2_tables.config(arrays, grid), x, 1, k, h, count)
        for actual, expected in ((k, expected_k), (h, expected_h)):
            np.testing.assert_array_equal(actual == 0, expected == 0)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
        assert np.all(count == np.floor(count))
    # Each segment separately prevents cancellation/hiding wrong cell selection.
    for row in segments():
        g = VoxelGrid(np.zeros(3), (3, 3, 3), 1.)
        a = np.zeros(g.shape); h = np.zeros(g.shape)
        one = row[None].copy(); one[:, 7:9] = 1
        accumulate_track_length_multi(((g.kerma_keV, np.ones(1)),), g, one[:, :3], one[:, 3:6], one[:, 6])
        probe(driver, arrays, g2_tables.config(arrays, g), one, 1, a, h)
        np.testing.assert_allclose(a, g.kerma_keV, rtol=1e-12, atol=0)


def test_gate0c_grid_moments():
    # Independent unequal batches (including a zero-contribution history).
    histories = np.array([2., 3., 0., 7., 1.])
    batches = [(2, histories[:2].sum()), (2, histories[2:4].sum()), (1, histories[4:].sum())]
    for track in (True, False):
        grid = VoxelGrid(np.zeros(3), (1, 1, 1), 1., track)
        for n, value in batches:
            grid.kerma_keV += value
            grid.h10_track_pSv_cm3 += 2*value
            grid.end_batch(n)
        assert grid.kerma_keV.item() == 13
        if not track:
            assert grid.n_batches == grid.n_histories == 0 and grid.kerma_sum2 is None
            continue
        q = 5**2/2+7**2/2+1
        sem = np.sqrt((q-13**2/5)/(3-1)/5)
        assert grid.kerma_sum2.item() == q
        assert grid.h10_sum2.item() == 4*q
        assert grid.n_batches_hit.item() == 3
        assert grid.n_histories == 5 and grid.n_batches == 3
        np.testing.assert_allclose(standard_error(grid.kerma_keV, grid.kerma_sum2, 3, 5), sem)
        np.testing.assert_allclose(grid.kerma_relative_error(), sem/(13/5))
    one = VoxelGrid(np.zeros(3), (1, 1, 1), 1., True)
    one.end_batch(1)
    assert one.n_batches_hit.item() == 0 and np.isnan(one.kerma_relative_error()).all()


def test_gate0c_box_history_moments(setup):
    driver, _, _ = setup
    # Two segments 2+3 in the same history MUST be squared after summation.
    x = np.array([[2+3., 1.], [0., 0.], [7., 2.], [1., 0.]])
    s = np.zeros(2); q = np.zeros(2); count = np.zeros(2)
    fn = driver.lib.g2_moment_probe
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn(2, x.ctypes.data, len(x), s.ctypes.data, q.ctypes.data, count.ctypes.data)
    np.testing.assert_array_equal(s, [13, 3])
    np.testing.assert_array_equal(q, [75, 5])
    np.testing.assert_array_equal(count, [3, 2])
    expected_sem = np.sqrt((np.array([75, 5])-np.array([13, 3])**2/4)/3/4)
    np.testing.assert_allclose(standard_error(s, q, 4, 4), expected_sem)


def test_null_tally_fixed_g1(tmp_path):
    from chatcarlo.source import sample_source_photons
    scene = load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    bundle.create(scene, None, tmp_path)
    a, *_ = bundle.load(tmp_path)
    driver = core.Core(a)
    o, d, e = sample_source_photons(scene.raw['source'], 128, np.random.default_rng(713))
    r = driver.run(o, d, e, seed=712, record=True)
    expected = dict(deposits='d424085911c924d0b70c2678da6caf91991a0f5a1ad5753d2c9e5cdbbae36e06',
                    endings='46d27f6b8fc6ff1a71c1acb951457a9a9c49c26b0c5cdc39e4c55391a3d9b9c9',
                    scatter_sum='6602cf082886b358e25715c49ab10047eb8871d8c2ea5e3340dcc0b76065f69e',
                    n_fluorescence='af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc',
                    d='5a28e04a585df9951c0e807e6f949474b10402fb3bac1a5e3672e33df292d071',
                    i='8391c9b32ff975bf86558de646d1111834b311f1fabb66acc3df3da98e9b108c')
    assert {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in r.items()} == expected


@pytest.mark.parametrize('track', [True, False])
@pytest.mark.parametrize('resolution', [2., 5.])
def test_api_grid(track, resolution):
    from chatcarlo.geometry import Geometry
    scene = load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    geom = Geometry(scene.raw['geometry'])
    expected = VoxelGrid.from_bbox(geom.tally_bbox_min, geom.tally_bbox_max, resolution)
    r = run_transport(scene, n_histories=17, seed=725, batch_size=8, engine='gpu',
                      dose_grid=True, grid_resolution_cm=resolution, track_uncertainty=track)
    g = r.grid
    assert g.shape == expected.shape and g.voxel_size_cm == resolution
    np.testing.assert_array_equal(g.origin_cm, expected.origin_cm)
    assert np.sum(g.kerma_keV) > 0 and np.sum(g.h10_track_pSv_cm3) > 0
    if track:
        assert g.n_batches == 3 and g.n_histories == 17
        for key in ('kerma_sum2', 'h10_sum2', 'n_batches_hit'):
            assert getattr(g, key).shape == g.shape
    else:
        assert g.kerma_sum2 is None and g.n_batches == 0


def test_g1_abi_and_cuda_source():
    assert core.CPU_INPUTS == {0:'data', 1:'offsets', 2:'me', 3:'ng', 4:'nm', 6:'seed', 7:'start', 8:'n', 9:'origins', 10:'directions', 11:'energies', 12:'fluorescence', 13:'record', 18:'trace_capacity'}
    assert core.CUDA_INPUTS == {0:'data', 1:'offsets', 2:'me', 3:'ng', 4:'nm', 5:'centers', 6:'halves', 7:'materials', 8:'nb', 9:'bg', 10:'world', 11:'detector', 12:'seed', 13:'start', 14:'n', 15:'origins', 16:'directions', 17:'energies', 18:'fluorescence', 19:'record', 24:'trace_capacity'}
    source = core.cuda_source()
    assert 'g2_histories_kernel' in source and 'g1_histories_kernel' in source
    assert '#include "' not in source
    # g2 introduces no host-only includes/constants; the existing G1 infinity
    # definition already provides the NVRTC intrinsic before all uses.
    g2 = (core.HERE/'g2_tally.h').read_text()
    assert 'INFINITY' not in g2 and '<cmath>' not in g2
    assert source.index('#define INFINITY') < source.index('g2_grid_segment')


@pytest.mark.parametrize('distance', [0., .25, 1.])
def test_detector_truncates(setup, distance):
    from chatcarlo.detector import DetectorPlane
    driver, arrays, _ = setup
    a = {k: v.copy() for k, v in arrays.items()}
    a['centers'] = np.zeros((1, 3)); a['halves'] = np.ones((1, 3))
    a['materials'] = np.zeros(1, np.int64); a['background'] = np.array(0)
    a['world'] = np.r_[np.zeros(3), np.ones(3)]
    plane = DetectorPlane([distance, 0, 0], [-1, 0, 0], [0, 1, 0], (2, 2), (1, 1))
    a['detector'] = np.r_[plane.center_cm, plane.normal, plane.u_axis, plane.v_axis, plane.size_cm, plane.shape]
    g = VoxelGrid(np.array([-1., -1., -1.]), (4, 4, 4), .5)
    c = Core(a, grid=g)
    r = c.run(np.zeros((1, 3)), np.array([[1., 0, 0]]), np.array([60.]), seed=731, record=True)
    assert r['endings'][2] == 1
    from chatcarlo.dose_coefficients import h_star_10_per_fluence
    np.testing.assert_allclose(r['h10'].sum(), distance*h_star_10_per_fluence(60.).item(), rtol=1e-12, atol=0)
    xs = g.voxel_centers()[:, 0]
    assert np.all(r['h10'][xs > max(distance, 0)] == 0)


def test_missing_xaamdi_and_default_margin(tmp_path):
    from chatcarlo.materials import _XAAMDI_FILES
    scene = load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    scene.raw['geometry'][0]['material'] = 'Fe'
    assert 'fe' not in _XAAMDI_FILES
    a = bundle.create(scene, None, tmp_path/'a')
    b = bundle.create(scene, None, tmp_path/'b', bbox_margin_cm=50.)
    assert a == b
    assert 'g2_tables.npz' not in a['files']
    arrays, *_ = bundle.load(tmp_path/'a')
    r = core.Core(arrays).run(np.array([[0., 0., 15.]]), np.array([[0., 1., 0.]]), np.array([60.]), seed=732)
    assert r['endings'].sum() == 1


def test_colab_g2_rejects_unmatched_runner(tmp_path, monkeypatch):
    from chatcarlo.gpu.colab_driver import g2_gate
    scene = load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    manifest = bundle.create(scene, None, tmp_path/'bundle')
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(runner='run_gates.py', runner_sha256='0'*64, gate=2)))
    def forbidden(*args, **kwargs):
        raise AssertionError('must not execute')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='SHA256'):
        g2_gate(tmp_path/'bundle', path, tmp_path/'out')


def test_egs5_preparation_and_boundaries(tmp_path):
    runner = importlib.util.spec_from_file_location('g2_egs5', DOC/'run_egs5_60kev.py')
    m = importlib.util.module_from_spec(runner); runner.loader.exec_module(m)
    m.prepare(tmp_path)
    for name, nboxes, seed in (('pdd', 47, 3101), ('thin', 1, 3102), ('phantom', 1, 3103)):
        arrays, *_ = bundle.load(tmp_path/name)
        with np.load(tmp_path/name/'g2_tables.npz') as z:
            assert z['boxes'].shape == (nboxes, 6) and z['seed'] == seed


def test_gate0c_box_multiple_segments_and_zero_histories(setup):
    from chatcarlo.materials import mu_en_rho, density
    driver, arrays, names = setup
    water = names.index('water')
    boxes = [('one', np.zeros(3), np.ones(3))]
    cfg = g2_tables.config(arrays, boxes=boxes, water=water)
    x = np.array([[0, .5, .5, 1, 0, 0, .25, 60, water, 0],
                  [.25, .5, .5, 1, 0, 0, .5, 60, water, 0],
                  [0, .5, .5, 1, 0, 0, .5, 60, water, 0]], float)
    ids = np.array([0, 0, 2], np.int64)  # histories 1 and 3 contribute zero
    s, q, count = np.zeros(1), np.zeros(1), np.zeros(1)
    fn = driver.lib.g2_box_history_probe
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn(driver.nm, arrays['g2_table'].ctypes.data, cfg.ctypes.data, x.ctypes.data,
       ids.ctypes.data, len(x), 4, s.ctypes.data, q.ctypes.data, count.ctypes.data)
    kw = 60*mu_en_rho('water', 60).item()*density('water')
    np.testing.assert_allclose(s, 1.25*kw, rtol=1e-12)
    np.testing.assert_allclose(q, (.75**2+.5**2)*kw**2, rtol=1e-12)
    assert count.item() == 2


def test_detector_interaction_tie(setup):
    from chatcarlo.detector import DetectorPlane
    driver, arrays, _ = setup
    a = {k: v.copy() for k, v in arrays.items()}
    fn = driver.lib.g2_first_distance
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_double]
    fn.restype = ctypes.c_double
    seed = 733
    distance = fn(a['data'].ctypes.data, a['offsets'].ctypes.data, driver.me, driver.ng,
                  driver.nm, seed, 1, 60.)
    a['centers'] = np.zeros((1, 3)); a['halves'] = np.full((1, 3), 2*distance)
    a['materials'] = np.ones(1, np.int64)
    a['world'] = np.r_[np.zeros(3), np.full(3, 2*distance)]
    plane = DetectorPlane([distance, 0, 0], [-1, 0, 0], [0, 1, 0], (2, 2), (1, 1))
    a['detector'] = np.r_[plane.center_cm, plane.normal, plane.u_axis, plane.v_axis, plane.size_cm, plane.shape]
    g = VoxelGrid(np.array([0., -1., -1.]), (2, 2, 2), distance)
    r = Core(a, grid=g).run(np.zeros((1, 3)), np.array([[1., 0, 0]]), np.array([60.]), seed=seed, record=True)
    assert r['i'][0, 0] == 0 and r['endings'][2] == 1
    from chatcarlo.dose_coefficients import h_star_10_per_fluence
    np.testing.assert_allclose(r['h10'].sum(), distance*h_star_10_per_fluence(60).item(), rtol=1e-12)


def test_imported_comparison_reproduces_historical_numbers(tmp_path):
    old = ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07'
    spec = importlib.util.spec_from_file_location('g2_comparison_runner', DOC/'run_egs5_60kev.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    bsf = json.loads((old/'chatcarlo/bsf.json').read_text())['results']
    for name in ('thin', 'phantom'):
        (tmp_path/(name+'.json')).write_text(json.dumps(dict(results=bsf[name])))
    (tmp_path/'pdd.json').write_bytes((old/'chatcarlo/pdd.json').read_bytes())
    runner.compare(tmp_path)
    expected = json.loads((old/'comparison.json').read_text())
    actual = json.loads((tmp_path/'comparison.json').read_text())
    for key in ('BSF_w', 'PDD_OCR', 'independent_analytic_references'):
        assert actual[key] == expected[key]
    import csv
    with (old/'comparison.csv').open() as a, (tmp_path/'comparison.csv').open() as b:
        assert list(csv.DictReader(a)) == list(csv.DictReader(b))


@pytest.fixture
def gate_runner():
    spec = importlib.util.spec_from_file_location('g2_gate_runner', DOC/'run_gates.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def grid_pairs():
    c = np.tile([1., 1., 0.], (10, 1))
    return c, c.copy(), np.full(c.shape, 100000.), np.full(c.shape, 100000.)


def test_gate2_grid_redistribution(gate_runner):
    c, g, kc, kg = grid_pairs()
    g[:, 0] += 1e-8
    g[:, 1] -= 1e-8
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == 'pass'
    assert all(b['conservation_pass'] for b in report['batches'])


@pytest.mark.parametrize('relative, expected', [(0.9e-6, 'pass'), (1.1e-6, 'fail')])
def test_gate2_grid_relative_width(gate_runner, relative, expected):
    c, g, kc, kg = grid_pairs()
    # Balance the perturbation so this tests the voxel width independently.
    g[:, 0] += relative
    g[:, 1] -= relative
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == expected
    assert all(b['conservation_pass'] for b in report['batches'])


@pytest.mark.parametrize('value, expected, rescued', [(0., 'pass', 0),
                                                     (1e-9, 'pass', 1),
                                                     (2e-9, 'fail', 0)])
@pytest.mark.parametrize('swap', [False, True])
def test_gate2_grid_one_sided_zero(gate_runner, value, expected, rescued, swap):
    c, g, kc, kg = grid_pairs()
    g[:, 0] -= value
    g[:, 2] = value
    if swap:
        c, g = g, c
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == expected
    for b in report['batches']:
        assert b['one_sided_zero_count'] == int(value != 0)
        assert b['one_sided_zero_rescued_count'] == rescued
        assert b['both_zero_count'] == int(value == 0)
        assert b['hit_mismatch_indices'] == ([2] if value else [])
        assert b['hit_mismatch_failed_indices'] == ([2] if expected == 'fail' else [])


@pytest.mark.parametrize('slot', range(4))
@pytest.mark.parametrize('invalid', ['nan', 'inf', '-inf', 'shape', 'batches'])
def test_gate2_grid_rejects_invalid(gate_runner, slot, invalid):
    args = list(grid_pairs())
    if invalid == 'shape':
        args[slot] = args[slot][:, :2]
    elif invalid == 'batches':
        args[slot] = args[slot][:9]
    else:
        args[slot][0, 0] = float(invalid)
    with pytest.raises(ValueError):
        gate_runner.grid_paired_judge(*args)


@pytest.mark.parametrize('cancel_batches', [False, True])
def test_gate2_grid_never_pools_batches(gate_runner, cancel_batches):
    c, g, kc, kg = grid_pairs()
    g[0, :2] += [2e-6, -2e-6]
    if cancel_batches:
        g[1, :2] -= [2e-6, -2e-6]
        np.testing.assert_array_equal(g.sum(axis=0), c.sum(axis=0))
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == 'fail'
    assert [i for i, b in enumerate(report['batches']) if b['status'] == 'fail'] == (
        [0, 1] if cancel_batches else [0])


def test_gate2_grid_conservation(gate_runner):
    c, g, kc, kg = grid_pairs()
    g[0, 0] += 1e-8  # Within voxel tolerance, outside total rounding allowance.
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == 'fail'
    b = report['batches'][0]
    assert b['failed_indices'] == []
    assert not b['conservation_pass']
    assert b['total_rounding_bound'] == pytest.approx(
        (kc[0].sum()+kg[0].sum()+20)*2**-52*(abs(c[0])+abs(g[0])).sum())


def test_gate2_grid_integer_diagnostics(gate_runner):
    c, g, kc, kg = grid_pairs()
    kg[0, 0] += 1
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['status'] == 'pass'
    assert report['batches'][0]['addition_mismatch_indices'] == [0]
    assert report['batches'][0]['addition_mismatch_failed_indices'] == []
    g[0, :2] += [2e-6, -2e-6]
    report = gate_runner.grid_paired_judge(c, g, kc, kg)
    assert report['batches'][0]['addition_mismatch_failed_indices'] == [0]


@pytest.mark.parametrize('scene, nboxes, fault, expected', [
    ('lead', 0, None, True), ('pdd', 47, None, True),
    ('pdd', 47, 'kerma', False), ('pdd', 47, 'h10', False),
    ('pdd', 47, 'conservation', False),
    ('pdd', 47, 'box_sum', False), ('pdd', 47, 'box_sum2', False),
    ('pdd', 0, None, False), ('pdd', 46, None, False),
    ('lead', 1, None, False), ('pdd', 47, 'histories', False),
    ('pdd', 47, 'integers', True),
])
def test_gate2_integration(gate_runner, tmp_path, monkeypatch, scene, nboxes, fault, expected):
    from chatcarlo.gpu import g2_core, source_host, validation
    np.savez(tmp_path/'g2_tables.npz', grid_origin=np.zeros(3), grid_shape=[3, 1, 1],
             grid_resolution=2., boxes=np.zeros((nboxes, 6)), water=0)
    names = ['water', 'lead'] if scene == 'lead' else ['water']
    manifest = {'files': {'historical': 'unchanged'}}
    monkeypatch.setattr(bundle, 'load', lambda path: ({}, {}, {},
                        dict(material_names=names, fluorescence=True), manifest))
    calls, draws, box_judgements = [], [], []

    def sample(src, n, rng, spec):
        assert n == 100000
        draws.append(rng.random())
        return np.zeros((1, 3)), np.ones((1, 3)), np.ones(1)

    class FakeCore:
        def __init__(self, arrays, device, **kwargs):
            self.device = device
            assert kwargs['count_additions']
            assert len(kwargs['boxes']) == nboxes

        def run(self, o, d, e, *, seed, start, fluorescence, record):
            calls.append((self.device, seed, start))
            assert fluorescence and record
            r = dict(kerma=np.array([1., 1., 0.]), h10=np.array([2., 2., 0.]),
                     grid_additions=np.full(3, 100000.), box_sum=np.ones(nboxes),
                     box_sum2=np.ones(nboxes), box_additions=np.full(nboxes, 100000.),
                     endings=np.array([100000, 0, 0]), i=np.zeros((1, 2), int),
                     d=np.zeros((1, 2)))
            if self.device == 'cuda':
                # Old paired_judge rejects this constant balanced difference;
                # the approved voxel criterion must accept it.
                for q in ('kerma', 'h10'):
                    r[q][:2] += [1e-8, -1e-8]
                if fault in ('kerma', 'h10') and start == 0:
                    r[fault][:2] += [1e-4, -1e-4]
                elif fault == 'conservation':
                    r['kerma'][0] += 1e-8
                elif fault in ('box_sum', 'box_sum2'):
                    r[fault][0] += .001
                elif fault == 'histories':
                    r['endings'][0] -= 1
                elif fault == 'integers':
                    r['grid_additions'][0] += 1
                    r['box_additions'][0] += 1
                    r['kerma'][2] = 1e-10
                    r['h10'][2] = 1e-10
                    r['i'][0, 0] = 1
            return r

    paired = validation.paired_judge
    def box_judge(c, g, bounds):
        assert np.shape(c) == (10, nboxes)
        box_judgements.append(True)
        return paired(c, g, bounds)

    monkeypatch.setattr(source_host, 'sample_source_photons', sample)
    monkeypatch.setattr(g2_core, 'Core', FakeCore)
    monkeypatch.setattr(validation, 'paired_judge', box_judge)
    assert gate_runner.gate2(tmp_path, tmp_path) == expected
    report = json.loads((tmp_path/'gate2.json').read_text())
    assert report['status'] == ('pass' if expected else 'fail')
    assert report['criterion_version'] == 'v2-addendum2'
    assert (report['seed'], report['source_seed']) == (3300, 3301)
    assert (report['n_histories'], report['batch_size'], report['n_batches']) == (1000000, 100000, 10)
    assert report['n_boxes'] == nboxes
    assert report['bundle'] == manifest
    assert len(box_judgements) == 2
    assert calls == [(device, 3300, b*100000) for b in range(10) for device in ('cpu', 'cuda')]
    np.testing.assert_array_equal(draws, np.random.default_rng(3301).random(10))
    assert set(report['quantities']) == {'kerma', 'h10', 'box_sum', 'box_sum2'}
    if fault in ('kerma', 'h10', 'box_sum', 'box_sum2'):
        assert report['quantities'][fault]['status'] == 'fail'
        assert all(r['status'] == 'pass' for q, r in report['quantities'].items() if q != fault)
    if fault == 'integers':
        assert report['integer_differences'][0]['diverged_histories'] == [0]
        assert report['integer_differences'][0]['box_additions'] == [0]
        b = report['quantities']['kerma']['batches'][0]
        assert b['one_sided_zero_rescued_count'] == 1
        assert b['addition_mismatch_indices'] == [0]
        assert b['hit_mismatch_indices'] == [2]
    # Existing artifacts' data keys stay usable for reference-only rejudging.
    with np.load(tmp_path/'paired_00.npz') as z:
        for device in ('cpu', 'gpu'):
            for q in ('kerma', 'h10', 'grid_additions', 'box_sum', 'box_sum2', 'box_additions'):
                assert device+'_'+q in z
