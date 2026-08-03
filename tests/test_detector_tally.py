import numpy as np
import pytest

from chatcarlo.detector import CAT_PRIMARY, CAT_SINGLE, CAT_MULTIPLE, CAT_FLUOR, DetectorPlane, DetectorTally, classify, rebin_area_preserving, rebin_counts
from chatcarlo.geometry import Geometry
from chatcarlo.materials import linear_mu
from chatcarlo.scene import validate_scene
from chatcarlo.tally import VoxelGrid
from chatcarlo.transport import run_transport, transport_photons


def _plane():
    return DetectorPlane(np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]), (10., 10.), (10, 10))


def _geom(plane):
    return Geometry([{"shape": "box", "material": "air", "center": [0, 0, 10], "size_cm": [1, 1, 1]}], detector_plane=plane)


def test_terminal_detection_and_energy_moments():
    plane, tally = _plane(), DetectorTally(_plane())
    r = transport_photons(np.array([[0., 0., 5.], [8., 0., 5.]]), np.array([[0., 0., -1.], [0., 0., -1.]]), np.array([60., 60.]), _geom(plane), np.random.default_rng(4), detector_tally=tally)
    assert np.array_equal(r.detected, [True, False])
    assert np.array_equal(r.absorbed | r.detected | r.escaped, [True, True])
    assert not np.any(r.detected & r.escaped)
    assert tally.photon_count.sum() == 1 and tally.energy_sum_keV.sum() == 60 and tally.energy_sum2_keV2.sum() == 3600
    assert tally.category_fluence[CAT_PRIMARY].sum() == 60 / plane.pixel_area_cm2
    assert tally.photon_count[5, 5] == 1
    assert np.array_equal(tally.total_fluence(), tally.category_fluence.sum(axis=0))


def test_category_truth_table():
    assert np.array_equal(classify(np.array([0, 1, 2, 9]), np.array([False, False, False, True])),
                          [CAT_PRIMARY, CAT_SINGLE, CAT_MULTIPLE, CAT_FLUOR])


def test_detector_shortens_track_before_grid_scoring():
    plane = _plane(); geom = _geom(plane)
    grid = VoxelGrid.from_bbox(geom.tally_bbox_min, geom.tally_bbox_max, 1.0)
    result = transport_photons(np.array([[0., 0., 5.]]), np.array([[0., 0., -1.]]), np.array([60.]),
                               geom, np.random.default_rng(7), grid=grid, detector_tally=DetectorTally(plane))
    assert result.detected[0]
    centers_z = grid.origin_cm[2] + (np.arange(grid.shape[2]) + .5) * grid.voxel_size_cm
    assert grid.kerma_keV[:, :, centers_z < 0].sum() == 0


def test_batch_statistics_and_off_errors():
    p = _plane(); t = DetectorTally(p, track_uncertainty=True, roi=((0, 10), (0, 10)))
    for e in (10., 20., 30.):
        t.category_fluence[0, 5, 5] += e
        t.end_batch(10)
    expected_q = (10**2 + 20**2 + 30**2) / 10
    assert t.category_sum2[0, 5, 5] == expected_q and t.n_batches == 3 and t.n_histories == 30
    assert np.isfinite(t.category_relative_error()[0, 5, 5])
    off = DetectorTally(p, roi=((0, 10), (0, 10)))
    for fn in (off.category_relative_error, off.total_relative_error, off.stpr_sem):
        with pytest.raises(ValueError): fn()


def test_stpr_sem_includes_batch_covariance():
    p = _plane(); t = DetectorTally(p, track_uncertainty=True, roi=((0, 10), (0, 10)))
    ps, ss = np.array([10., 20., 10.]), np.array([5., 30., 10.])
    for primary, scatter in zip(ps, ss):
        t.category_fluence[0, 5, 5] += primary
        t.category_fluence[1, 5, 5] += scatter
        t.end_batch(10)
    n, m, P, S = 30., 3., ps.sum(), ss.sum()
    vp = ((ps @ ps / 10 - P**2 / n) / (m - 1)) / n
    vs = ((ss @ ss / 10 - S**2 / n) / (m - 1)) / n
    cov = ((ps @ ss / 10 - P*S / n) / (m - 1)) / n
    expected = np.sqrt((S / P)**2 * (vs / (S/n)**2 + vp / (P/n)**2 - 2*cov / ((S/n)*(P/n))))
    assert np.isclose(t.stpr_sem(), expected, rtol=1e-12)


def test_rebinning_is_area_preserving_and_counts_sum():
    a = np.arange(16.).reshape(4, 4)
    assert np.array_equal(rebin_area_preserving(a, 2), np.array([[2.5, 4.5], [10.5, 12.5]]))
    assert np.array_equal(rebin_counts(a, 2), np.array([[10., 18.], [42., 50.]]))
    with pytest.raises(ValueError): rebin_counts(a, 3)


def _scene(material="water", thickness=8.0):
    scene = validate_scene({
        "source": {"spectrum": [{"energy_keV": 60., "weight": 1.}],
                   "position": [0., -20., 0.], "direction": [0., 1., 0.],
                   "field": {"shape": "parallel", "size_cm": [12., 12.]}},
        "geometry": [{"shape": "box", "material": material, "center": [0., 0., 0.],
                      "size_cm": [20., thickness, 20.]}],
    })
    assert scene.ok
    return scene


def _detector(y=10., shape=(8, 8)):
    return DetectorPlane(np.array([0., y, 0.]), np.array([0., -1., 0.]),
                         np.array([1., 0., 0.]), (12., 12.), shape)


def _assert_batch_result_equal(a, b):
    for name in ("n_scatter", "absorbed", "escaped", "final_energy"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.energy_deposited == b.energy_deposited
    assert a.n_fluorescence == b.n_fluorescence


class _InlineFuture:
    def __init__(self, value): self.value = value
    def result(self): return self.value


class _InlineProcessPool:
    """workers=2の集約経路を、OS semaphore不要の同期executorで回す。"""
    def __init__(self, max_workers): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def submit(self, fn, *args, **kwargs): return _InlineFuture(fn(*args, **kwargs))


def test_detector_none_is_bit_identical_for_low_and_high_level_transport():
    """D-1: detector未指定は既存輸送の乱数列・既存フィールドを変えない。"""
    geom = Geometry(_scene().raw["geometry"])
    pos = np.tile(np.array([0., -15., 0.]), (2000, 1))
    direction = np.tile(np.array([0., 1., 0.]), (2000, 1))
    energy = np.full(2000, 60.)
    a = transport_photons(pos.copy(), direction.copy(), energy.copy(), geom, np.random.default_rng(31))
    b = transport_photons(pos.copy(), direction.copy(), energy.copy(), geom, np.random.default_rng(31), detector_tally=None)
    _assert_batch_result_equal(a, b)

    kwargs = dict(n_histories=6000, seed=31, dose_grid=True, grid_resolution_cm=4., batch_size=1000)
    r0 = run_transport(_scene(), **kwargs)
    r1 = run_transport(_scene(), detector=None, **kwargs)
    for name in ("energy_deposited_MeV", "fraction_absorbed", "fraction_escaped",
                 "mean_scatter_events", "n_fluorescence", "n_batches",
                 "energy_deposited_sem_MeV", "energy_deposited_rel_err"):
        assert getattr(r0, name) == getattr(r1, name), name
    assert np.array_equal(r0.grid.kerma_keV, r1.grid.kerma_keV)
    assert np.array_equal(r0.grid.h10_track_pSv_cm3, r1.grid.h10_track_pSv_cm3)


def test_kernel_rejects_detector_combination():
    """D-3: kernel経路へ検出器が黙って渡らない。"""
    with pytest.raises(ValueError, match="kernel engine.*detector"):
        run_transport(_scene(), n_histories=1, engine="kernel", detector=_detector())


def test_primary_image_ratio_matches_beer_lambert_per_pixel():
    """B-1: 水スラブ像/空気ベースライン像をピクセルごとに exp(-μt) と比較する。"""
    thickness, n = 2.0, 128_000
    detector = _detector()
    water = run_transport(_scene("water", thickness), n_histories=n, seed=7, batch_size=2000,
                          detector=detector, track_uncertainty=True)
    baseline = run_transport(_scene("air", thickness), n_histories=n, seed=7, batch_size=2000,
                             detector=detector, track_uncertainty=True)
    w = water.detector.category_fluence[CAT_PRIMARY]
    b = baseline.detector.category_fluence[CAT_PRIMARY]
    ratio = w / b
    expected = np.exp(-linear_mu("water", 60.) * thickness)
    # baselineの各pixel history数を二項試行数として保守的に扱う。airの寄与は
    # 0.5%平均誤差の許容内であり、ここでは登録どおり water の μ を用いる。
    # 2 runは各batchで輸送乱数を共有するため、後続batchの線源位置は独立に
    # サンプルされる。したがって透過の二項揺らぎに加え、分母像のpixel population
    # の揺らぎもデルタ法で結合する。
    b_count = baseline.detector.photon_count
    sigma = np.sqrt(expected * (1 - expected) / b_count
                    + expected**2 * (1 - b_count / n) / b_count)
    assert np.all(np.abs(ratio - expected) <= 3 * sigma)
    assert abs(float(ratio.mean()) - expected) / expected < 0.005


@pytest.mark.parametrize("fluorescence", [False, True])
def test_detector_energy_balance_is_exact_with_and_without_fluorescence(fluorescence):
    """B-3: 入射=衝突沈着+検出器終端+未検出脱出（grid estimatorは混ぜない）。"""
    raw = _scene("lead", 0.4).raw
    raw["physics"] = {"fluorescence": fluorescence}
    scene = validate_scene(raw); assert scene.ok
    n, e = 25_000, 90.
    plane = _detector(y=10.)
    geom = Geometry(scene.raw["geometry"], detector_plane=plane)
    result = transport_photons(np.tile([0., -15., 0.], (n, 1)), np.tile([0., 1., 0.], (n, 1)),
                               np.full(n, e), geom, np.random.default_rng(44),
                               fluorescence_enabled=fluorescence, detector_tally=DetectorTally(plane))
    accounted = (sum(result.energy_deposited.values()) + result.final_energy[result.detected].sum()
                 + result.final_energy[result.escaped].sum())
    assert abs(accounted - n * e) / (n * e) < 1e-9


def test_detector_statistics_on_off_and_partial_batch_are_consistent():
    """C-1/C-3: 統計の有無は総量不変、最終500 historyバッチも正しく記録される。"""
    plane = _detector()
    kwargs = dict(n_histories=2500, seed=99, batch_size=1000, detector=plane, dose_grid=True,
                  grid_resolution_cm=4.)
    on = run_transport(_scene(), track_uncertainty=True, **kwargs)
    off = run_transport(_scene(), track_uncertainty=False, **kwargs)
    for name in ("category_fluence", "photon_count", "energy_sum_keV", "energy_sum2_keV2"):
        assert np.array_equal(getattr(on.detector, name), getattr(off.detector, name)), name
    assert on.energy_deposited_MeV == off.energy_deposited_MeV
    assert np.array_equal(on.grid.kerma_keV, off.grid.kerma_keV)
    assert on.detector.n_batches == 3 and on.detector.n_histories == 2500
    assert on.n_batches == 3


def test_parallel_detector_category_totals_are_statistically_equivalent(monkeypatch):
    """D-2: workers変更では各カテゴリ総和が結合3σ以内（同一workersは既存の再現契約）。"""
    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlineProcessPool)
    plane = _detector()
    kwargs = dict(n_histories=30_000, seed=6, batch_size=1000, detector=plane, track_uncertainty=True)
    serial = run_transport(_scene(), n_workers=1, **kwargs).detector
    parallel = run_transport(_scene(), n_workers=2, **kwargs).detector
    for cat in range(4):
        x, y = serial.category_fluence[cat].sum(), parallel.category_fluence[cat].sum()
        rx, ry = serial.category_relative_error()[cat], parallel.category_relative_error()[cat]
        sx = np.sqrt(np.sum((serial.category_fluence[cat] * rx) ** 2))
        sy = np.sqrt(np.sum((parallel.category_fluence[cat] * ry) ** 2))
        if sx + sy > 0:
            assert abs(x - y) <= 3 * np.hypot(sx, sy), f"category={cat}"
