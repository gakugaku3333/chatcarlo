import numpy as np
import pytest

import chatcarlo.kernel as kernel
import chatcarlo.tally as tally
import chatcarlo.tally_njit as tally_njit
from chatcarlo.tally import VoxelGrid


def _unit(value):
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value)


def _compare(grid, origin, direction, length, wa=None, wb=None, initial=0.0):
    origin = np.asarray(origin, dtype=float).reshape(-1, 3)
    direction = np.asarray(direction, dtype=float).reshape(-1, 3)
    length = np.asarray(length, dtype=float).reshape(-1)
    n = len(length)
    wa = np.ones(n) if wa is None else np.asarray(wa, dtype=float)
    wb = np.full(n, 3.25) if wb is None else np.asarray(wb, dtype=float)
    ref_a = np.full(grid.shape, initial)
    ref_b = np.full(grid.shape, initial)
    got_a = np.full(grid.shape, initial)
    got_b = np.full(grid.shape, initial)
    tally.accumulate_track_length_multi(
        ((ref_a, wa), (ref_b, wb)), grid, origin, direction, length)
    tally_njit.accumulate_track_length_multi_njit(
        ((got_a, wa), (got_b, wb)), grid, origin, direction, length)
    assert np.array_equal(got_a, ref_a)
    assert np.array_equal(got_b, ref_b)


def test_boundary_cases():
    grid = VoxelGrid(np.array([-2.0, -3.0, -4.0]), (7, 6, 5), 1.0)
    cases = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            direction = np.zeros(3)
            direction[axis] = sign
            cases.append((np.zeros(3), direction, 20.0))
    cases.extend([
        ([-9, -1, -2], [1, 0, 0], 20),
        ([-9, 20, 20], [1, 0, 0], 3),
        ([0.2, -0.7, -1.1], _unit([1, 2, 3]), 0.2),
        ([-2, -2, -3], [1, 0, 0], 7),
        ([-8, -2, -3], [1, 0, 0], 6),
        ([0, -2.5, -3.5], [1, 0, 0], 3),
        ([-1.5, -2.5, -3.5], [1, 0, 0], 1.5),
        ([-2, -2.5, -3.5], [0, 1, 0], 8),
        ([5, -2.5, -3.5], [0, 1, 0], 8),
        ([-2, -3, -3.5], _unit([1, 1, 0]), 12),
        ([-2, -3, -4], _unit([1, 1, 1]), 12),
        ([-1.5, -2.5, -3.5], [0, 0, 1], 8),
        ([-1.5, -2.5, -3.5], _unit([0.9e-12, 1, 0]), 8),
        ([-1.5, -2.5, -3.5], _unit([1.1e-12, 1, 0]), 8),
        ([0, 0, 0], [1, 0, 0], 0),
        ([0, 0, 0], [1, 0, 0], 1e-13),
        ([-1.8, -2.8, -3.8], _unit([1, .1, .1]), .4),
    ])
    for origin, direction, length in cases:
        _compare(grid, [origin], [direction], [length])

    long_grid = VoxelGrid(np.zeros(3), (1200, 2, 2), 0.25)
    _compare(long_grid, [[-1., .1, .1]], [[1., 0., 0.]], [302.])


def test_endpoint_regression():
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.0)
    origin = [[-3.037619819265908, 0.0, 0.0]]
    direction = [[0.018394208153120428, 0.6110258451767133,
                  -0.7913969103000736]]
    _compare(grid, origin, direction, [1.5015538205527557],
             [1.9965502538144548], [0.5335686225843084])


def test_boundary_near_representative_fuzz():
    rng = np.random.default_rng(20260729)
    grid = VoxelGrid(np.zeros(3), (8, 8, 8), 1.0)
    offsets = (0.0, .99 * tally._EPS_PLANE, 1.01 * tally._EPS_PLANE)
    for axis in range(3):
        for endpoint_kind in ("start", "end"):
            for sign in (-1.0, 1.0):
                for offset in offsets:
                    plane = 2 + axis
                    point = rng.uniform(1.2, 6.8, 3)
                    point[axis] = plane + sign * offset
                    raw = rng.uniform(.2, 1.0, 3)
                    raw[axis] *= sign
                    direction = _unit(raw)
                    length = .07 if offset == 0 else 3.2
                    origin = point if endpoint_kind == "start" else point - direction * length
                    _compare(grid, [origin], [direction], [length])

    for delta in (0.0, 1e-12, -1e-12, 3e-12, -3e-12):
        origin = np.array([.2, .3, .4])
        raw = np.array([.8, .7 / (1.0 + delta), .6])
        direction = _unit(raw)
        _compare(grid, [origin], [direction], [4.0])


def test_eps_plane_is_read_at_call_time(monkeypatch):
    grid = VoxelGrid(np.zeros(3), (2, 1, 1), 1.0)
    origin = np.array([[0.5, .5, .5]])
    direction = np.array([[1., 0., 0.]])
    length = np.array([.5 + 5e-7])
    weights = np.array([1.])

    default = np.zeros(grid.shape)
    tally_njit.accumulate_track_length_multi_njit(
        ((default, weights), (np.zeros(grid.shape), weights)),
        grid, origin, direction, length)
    monkeypatch.setattr(tally, "_EPS_PLANE", 1e-6)
    changed = np.zeros(grid.shape)
    changed_b = np.zeros(grid.shape)
    tally_njit.accumulate_track_length_multi_njit(
        ((changed, weights), (changed_b, weights)),
        grid, origin, direction, length)
    ref = np.zeros(grid.shape)
    ref_b = np.zeros(grid.shape)
    tally.accumulate_track_length_multi(
        ((ref, weights), (ref_b, weights)), grid, origin, direction, length)
    assert np.array_equal(changed, ref)
    assert np.array_equal(changed_b, ref_b)
    assert not np.array_equal(default, changed)


def test_intersection_scratch_overflow_is_guarded():
    with pytest.raises(ValueError, match="scratch"):
        tally_njit._enumerate_intersections(
            np.array([.1, .1, .1]), _unit([1, 1, 1]), 0., 2.,
            np.zeros(3), np.array([3, 3, 3]), 1., tally._EPS_PLANE,
            np.empty(2))


def test_empty_and_nonzero_accumulation():
    grid = VoxelGrid(np.zeros(3), (2, 2, 2), 1.0)
    _compare(grid, np.empty((0, 3)), np.empty((0, 3)), np.empty(0), initial=4.)
    _compare(grid, [[-1., .5, .5]], [[1., 0., 0.]], [2.], [2.], [7.], initial=4.)


@pytest.mark.parametrize("pair_count", [1, 3])
def test_invalid_pair_count(pair_count):
    grid = VoxelGrid(np.zeros(3), (2, 2, 2), 1.)
    pair = (np.zeros(grid.shape), np.empty(0))
    with pytest.raises(ValueError):
        tally_njit.accumulate_track_length_multi_njit(
            tuple(pair for _ in range(pair_count)), grid,
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0))


@pytest.mark.parametrize("bad", ["target", "weight", "origin", "direction", "length"])
def test_invalid_shapes(bad):
    grid = VoxelGrid(np.zeros(3), (2, 2, 2), 1.)
    target = np.zeros(grid.shape)
    weight = np.ones(1)
    origin = np.zeros((1, 3))
    direction = np.ones((1, 3))
    length = np.ones(1)
    if bad == "target":
        target = np.zeros((1, 2))
    elif bad == "weight":
        weight = np.ones((1, 1))
    elif bad == "origin":
        origin = np.zeros((1, 2))
    elif bad == "direction":
        direction = np.ones((1, 2))
    else:
        length = np.ones((1, 1))
    with pytest.raises(ValueError):
        tally_njit.accumulate_track_length_multi_njit(
            ((target, weight), (np.zeros(grid.shape), np.ones(1))),
            grid, origin, direction, length)


def _water_case():
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.)
    tables = kernel.bake_scene_materials(["water", "air"])
    geom = kernel.bake_box_scene(
        [{"center": (0., 0., 0.), "size_cm": (10., 100., 100.),
          "material": "water"}],
        background="air", tables=tables, bbox_margin_cm=.01)
    return tables, geom, grid


@pytest.mark.parametrize("use_njit", [False, True])
def test_kernel_selects_requested_accumulator(monkeypatch, use_njit):
    tables, geom, grid = _water_case()
    calls = {"numpy": 0, "njit": 0}
    numpy_impl = tally.accumulate_track_length_multi
    njit_impl = tally_njit.accumulate_track_length_multi_njit

    def numpy_spy(*args):
        calls["numpy"] += 1
        return numpy_impl(*args)

    def njit_spy(*args):
        calls["njit"] += 1
        return njit_impl(*args)

    monkeypatch.setattr(tally, "accumulate_track_length_multi", numpy_spy)
    monkeypatch.setattr(tally_njit, "accumulate_track_length_multi_njit", njit_spy)
    kernel.run_batch_with_tally(
        tables, geom, 60., (-5.01, 0., 0.), (1., 0., 0.), 20, 7, grid,
        n_chunks=1, use_njit_dda=use_njit)
    assert calls == ({"numpy": 0, "njit": 1} if use_njit
                     else {"numpy": 1, "njit": 0})


def test_run_dose_grid_propagates_variant_to_every_batch(monkeypatch):
    tables, geom, grid = _water_case()
    seen = []
    original = kernel.run_batch_with_tally

    def spy(*args, **kwargs):
        seen.append(kwargs["use_njit_dda"])
        return original(*args, **kwargs)

    monkeypatch.setattr(kernel, "run_batch_with_tally", spy)
    kernel.run_dose_grid(
        tables, geom, 60., (-5.01, 0., 0.), (1., 0., 0.), 125, 9, grid,
        batch_size=60, n_chunks=1, use_njit_dda=False)
    assert seen == [False, False, False]
