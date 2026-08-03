import numpy as np
import pytest

from chatcarlo.detector import DetectorPlane, DetectorTally
from chatcarlo.geometry import Geometry
from chatcarlo.materials import linear_mu
from chatcarlo.scene import validate_scene
from chatcarlo.transport import _run_worker, run_transport, transport_photons


class _InlineFuture:
    def __init__(self, value): self.value = value
    def result(self): return self.value


class _InlineProcessPool:
    def __init__(self, max_workers): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def submit(self, fn, *args, **kwargs): return _InlineFuture(fn(*args, **kwargs))


def _plane():
    return DetectorPlane(np.array([0., 0., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]), (4., 4.), (4, 4))


def test_front_back_and_plane_origin_conventions():
    p = _plane()
    o = np.array([[0., 0., 5.], [0., 0., -5.], [0., 0., 0.], [0., 0., 0.]])
    d = np.array([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.], [0., 0., 1.]])
    t, iu, iv, ok = p.intersect_segments(o, d, np.full(4, 10.))
    assert np.array_equal(ok, [True, False, True, False])
    assert t[0] == 5 and t[2] == 0 and np.all(iu[ok] == 2) and np.all(iv[ok] == 2)
    assert np.all(t[~ok] == np.inf) and np.all(iu[~ok] == -1)


def test_half_open_pixel_boundaries_and_parallel_ray():
    p = _plane()
    o = np.array([[-2., -2., 1.], [2., 0., 1.], [0., 0., 1.]])
    d = np.array([[0., 0., -1.], [0., 0., -1.], [1., 0., 0.]])
    _, iu, iv, ok = p.intersect_segments(o, d, np.full(3, 10.))
    assert ok[0] and (iu[0], iv[0]) == (0, 0)
    assert not ok[1] and not ok[2]


def test_geometry_detector_expands_world_not_tally_bbox():
    geoms = [{"shape": "box", "material": "water", "center": [0, 0, 0], "size_cm": [2, 2, 2]}]
    far = DetectorPlane(np.array([0., 0., 200.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]), (4., 4.), (4, 4))
    plain, detected = Geometry(geoms), Geometry(geoms, detector_plane=far)
    assert np.array_equal(plain.bbox_min, plain.tally_bbox_min)
    assert np.array_equal(plain.bbox_max, plain.tally_bbox_max)
    assert np.array_equal(plain.tally_bbox_min, detected.tally_bbox_min)
    assert np.array_equal(plain.tally_bbox_max, detected.tally_bbox_max)
    assert np.all(detected.bbox_min <= plain.bbox_min) and np.all(detected.bbox_max >= plain.bbox_max)


def test_invalid_axes_rejected():
    with pytest.raises(ValueError):
        DetectorPlane(np.zeros(3), np.array([0., 0., 2.]), np.array([1., 0., 0.]), (1, 1), (1, 1))


def test_worker_uses_tally_bbox_not_detector_world_bbox():
    raw = {"source": {"spectrum": [{"energy_keV": 60., "weight": 1.}], "position": [0, 0, 10], "direction": [0, 0, -1],
                      "field": {"shape": "parallel", "size_cm": [1, 1]}},
           "geometry": [{"shape": "box", "material": "air", "center": [0, 0, 0], "size_cm": [2, 2, 2]}]}
    scene = validate_scene(raw); assert not scene.errors
    far = DetectorPlane(np.array([0., 0., -200.]), np.array([0., 0., 1.]), np.array([1., 0., 0.]), (4., 4), (4, 4))
    result = _run_worker(scene.raw, 10, np.random.SeedSequence(1), 10, True, 5.0, None, False, far)
    expected = Geometry(scene.raw["geometry"]).tally_bbox_max - Geometry(scene.raw["geometry"]).tally_bbox_min
    assert result["kerma_keV"].shape == tuple(np.ceil(expected / 5.0).astype(int))


def test_detector_wins_ties_with_interaction_and_material_boundary():
    """A-4: t_det == ds では相互作用・境界のどちらよりも検出を優先する。"""
    # 相互作用タイは tau=-log(random) を固定して水中5 cm先に置く。
    class FixedRng:
        def random(self, n): return np.full(n, np.exp(-float(linear_mu("water", 60.)[0]) * 5.0))
    p = DetectorPlane(np.array([0., 0., 0.]), np.array([0., 1., 0.]), np.array([1., 0., 0.]), (4., 4.), (4, 4))
    water = Geometry([{"shape": "box", "material": "water", "center": [0., 2.5, 0.], "size_cm": [4., 10., 4.]}], detector_plane=p)
    r = transport_photons(np.array([[0., 5., 0.]]), np.array([[0., -1., 0.]]), np.array([60.]), water, FixedRng(), detector_tally=DetectorTally(p))
    assert r.detected[0] and not r.absorbed[0]
    # 水スラブの出口面に検出器を重ねる境界タイ。
    p_boundary = DetectorPlane(np.array([0., -5., 0.]), np.array([0., 1., 0.]), np.array([1., 0., 0.]), (4., 4.), (4, 4))
    boundary = Geometry([{"shape": "box", "material": "water", "center": [0., 0., 0.], "size_cm": [4., 10., 4.]}], detector_plane=p_boundary)
    class NoInteractionRng:
        def random(self, n): return np.full(n, np.exp(-float(linear_mu("water", 60.)[0]) * 20.0))
    r = transport_photons(np.array([[0., 4.999999, 0.]]), np.array([[0., -1., 0.]]), np.array([60.]), boundary, NoInteractionRng(), detector_tally=DetectorTally(p_boundary))
    assert r.detected[0] and not r.escaped[0]


def test_divergent_beam_known_projection_lands_in_expected_pixels():
    """A-8(b): 点源からの投影 x=SID/SOD*x_source が detector pixel と一致する。"""
    p = DetectorPlane(np.array([0., 10., 0.]), np.array([0., -1., 0.]), np.array([1., 0., 0.]), (20., 20.), (20, 20))
    geom = Geometry([{"shape": "box", "material": "air", "center": [0., 0., 0.], "size_cm": [1., 1., 1.]}], detector_plane=p)
    origin = np.array([0., -10., 0.])
    targets = np.array([[2., 10., 0.], [-4., 10., 3.]])
    dirs = targets - origin; dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    tally = DetectorTally(p)
    r = transport_photons(np.tile(origin, (2, 1)), dirs, np.full(2, 60.), geom, np.random.default_rng(2), detector_tally=tally)
    assert np.all(r.detected)
    for x, _, z in targets:
        iu, iv = int(np.floor(x + 10)), int(np.floor(z + 10))
        assert tally.photon_count[iu, iv] == 1


def test_detector_does_not_change_grid_metadata_in_serial_or_parallel(monkeypatch):
    """A-7(c): worker側もtally bboxを使い、検出器距離でgridを膨張させない。"""
    import concurrent.futures
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _InlineProcessPool)
    raw = {"source": {"spectrum": [{"energy_keV": 60., "weight": 1.}], "position": [0, -10, 0], "direction": [0, 1, 0], "field": {"shape": "parallel", "size_cm": [1, 1]}}, "geometry": [{"shape": "box", "material": "water", "center": [0, 0, 0], "size_cm": [2, 2, 2]}]}
    scene = validate_scene(raw); assert scene.ok
    far = DetectorPlane(np.array([0., 200., 0.]), np.array([0., -1., 0.]), np.array([1., 0., 0.]), (4., 4.), (4, 4))
    for workers in (1, 2):
        plain = run_transport(scene, n_histories=200, seed=4, n_workers=workers, dose_grid=True, grid_resolution_cm=1.)
        detected = run_transport(scene, n_histories=200, seed=4, n_workers=workers, dose_grid=True, grid_resolution_cm=1., detector=far)
        assert plain.grid.shape == detected.grid.shape
        assert np.array_equal(plain.grid.origin_cm, detected.grid.origin_cm)
        assert plain.grid.voxel_size_cm == detected.grid.voxel_size_cm
