"""njitスカラーDDAの前提検証（本番コードへ組み込まない独立プロトタイプ）。

方向ベクトルは単位長であること。length_cmはその方向に沿う幾何学的長さ。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np
from numba import njit

from chatcarlo.scene import load_scene
from chatcarlo.tally import VoxelGrid, _EPS_PLANE, accumulate_track_length_multi
import chatcarlo.transport as transport_mod
import chatcarlo.kernel as kernel_mod

PARALLEL_EPS = 1e-12
ATOL = 2e-10
RTOL = 2e-12


@njit(cache=True)
def dda_njit(origin, direction, length_cm, weights_a, weights_b,
             lo, shape, h, target_a, target_b):
    """参照実装と同じ交差面列挙・中点分類を1区間ずつスカラー実行する。"""
    hi0 = lo[0] + shape[0] * h
    hi1 = lo[1] + shape[1] * h
    hi2 = lo[2] + shape[2] * h
    for s in range(origin.shape[0]):
        enter = 0.0
        leave = length_cm[s]
        valid = True
        for k in range(3):
            o = origin[s, k]
            d = direction[s, k]
            lk = lo[k]
            hk = hi0 if k == 0 else (hi1 if k == 1 else hi2)
            if abs(d) < PARALLEL_EPS:
                if o < lk or o > hk:
                    valid = False
                    break
            else:
                ta = (lk - o) / d
                tb = (hk - o) / d
                if ta > tb:
                    tmp = ta
                    ta = tb
                    tb = tmp
                if ta > enter:
                    enter = ta
                if tb < leave:
                    leave = tb
        if not valid or leave <= enter:
            continue

        # 両端点と、端点から_EPS_PLANE以内を除いた全軸の面交点を列挙する。
        ts = np.empty(shape[0] + shape[1] + shape[2] + 5, dtype=np.float64)
        n_t = 2
        ts[0] = enter
        ts[1] = leave
        for k in range(3):
            d = direction[s, k]
            if abs(d) < PARALLEL_EPS:
                continue
            p_enter = (origin[s, k] + enter * d - lo[k]) / h
            p_leave = (origin[s, k] + leave * d - lo[k]) / h
            p_small = min(p_enter, p_leave)
            p_big = max(p_enter, p_leave)
            m_lo = int(np.ceil(p_small + _EPS_PLANE))
            m_hi = int(np.floor(p_big - _EPS_PLANE))
            for m in range(m_lo, m_hi + 1):
                ts[n_t] = (lo[k] + m * h - origin[s, k]) / d
                n_t += 1

        # 小配列なので挿入ソートで十分。近接時刻は統合せず、厳密な同着だけが
        # 隣接差0となって下で落ちる。
        for i in range(1, n_t):
            value = ts[i]
            j = i - 1
            while j >= 0 and ts[j] > value:
                ts[j + 1] = ts[j]
                j -= 1
            ts[j + 1] = value

        for i in range(n_t - 1):
            dl = ts[i + 1] - ts[i]
            if dl <= 0.0:
                continue
            mid = (ts[i] + ts[i + 1]) * 0.5
            ix = int(np.floor((origin[s, 0] + direction[s, 0] * mid - lo[0]) / h))
            iy = int(np.floor((origin[s, 1] + direction[s, 1] * mid - lo[1]) / h))
            iz = int(np.floor((origin[s, 2] + direction[s, 2] * mid - lo[2]) / h))
            if ix < 0 or ix >= shape[0] or iy < 0 or iy >= shape[1] or iz < 0 or iz >= shape[2]:
                continue
            target_a[ix, iy, iz] += weights_a[s] * dl
            target_b[ix, iy, iz] += weights_b[s] * dl


def _run_pair(grid, o, d, ds, wa, wb):
    ra = np.zeros(grid.shape)
    rb = np.zeros(grid.shape)
    na = np.zeros(grid.shape)
    nb = np.zeros(grid.shape)
    accumulate_track_length_multi(((ra, wa), (rb, wb)), grid, o, d, ds)
    dda_njit(o, d, ds, wa, wb, grid.origin_cm, np.asarray(grid.shape, dtype=np.int64),
             grid.voxel_size_cm, na, nb)
    return ra, rb, na, nb


def _compare(name, grid, o, d, ds, wa, wb, strict=False):
    ra, rb, na, nb = _run_pair(grid, o, d, ds, wa, wb)
    ok = True
    for label, ref, got in (("kerma", ra, na), ("H10", rb, nb)):
        abs_err = float(np.max(np.abs(ref - got)))
        denom = np.maximum(np.abs(ref), ATOL)
        rel_err = float(np.max(np.abs(ref - got) / denom))
        nz_equal = np.array_equal(np.flatnonzero(ref), np.flatnonzero(got))
        tol_a = 5e-12 if strict else ATOL
        tol_r = 5e-12 if strict else RTOL
        same = np.allclose(ref, got, rtol=tol_r, atol=tol_a) and nz_equal
        conserved = np.isclose(ref.sum(), got.sum(), rtol=tol_r, atol=tol_a)
        ok &= same and conserved
        print(f"{name}/{label}: voxel={'PASS' if same else 'FAIL'} "
              f"nonzero={'PASS' if nz_equal else 'FAIL'} sum={'PASS' if conserved else 'FAIL'} "
              f"max_abs={abs_err:.3e} max_rel={rel_err:.3e}")
    if not ok:
        raise AssertionError(name)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def boundary_cases():
    g = VoxelGrid(np.array([-2.0, -3.0, -4.0]), (7, 6, 5), 1.0)
    cases = []
    for k in range(3):
        for sign in (-1.0, 1.0):
            d = np.zeros(3); d[k] = sign
            cases.append((f"axis{k}_{sign:+g}", np.zeros(3), d, 20.0))
    cases += [
        ("outside-through", [-9, -1, -2], [1, 0, 0], 20),
        ("outside-only", [-9, 20, 20], [1, 0, 0], 3),
        ("inside-only", [0.2, -0.7, -1.1], _unit([1, 2, 3]), 0.2),
        ("outer-start", [-2, -2, -3], [1, 0, 0], 7),
        ("outer-end", [-8, -2, -3], [1, 0, 0], 6),
        ("inner-start", [0, -2.5, -3.5], [1, 0, 0], 3),
        ("inner-end", [-1.5, -2.5, -3.5], [1, 0, 0], 1.5),
        ("face-parallel", [-2, -2.5, -3.5], [0, 1, 0], 8),
        ("hi-face-parallel", [5, -2.5, -3.5], [0, 1, 0], 8),
        ("edge", [-2, -3, -3.5], _unit([1, 1, 0]), 12),
        ("vertex", [-2, -3, -4], _unit([1, 1, 1]), 12),
        ("zero-components", [-1.5, -2.5, -3.5], [0, 0, 1], 8),
        ("below-parallel-eps", [-1.5, -2.5, -3.5], _unit([0.9e-12, 1, 0]), 8),
        ("above-parallel-eps", [-1.5, -2.5, -3.5], _unit([1.1e-12, 1, 0]), 8),
        ("zero-length", [0, 0, 0], [1, 0, 0], 0),
        ("tiny", [0, 0, 0], [1, 0, 0], 1e-13),
        ("one-voxel", [-1.8, -2.8, -3.8], _unit([1, .1, .1]), .4),
    ]
    for name, o, d, ds in cases:
        _compare(name, g, np.array([o], float), np.array([d], float),
                 np.array([ds], float), np.array([3.25]), np.array([1e6]), strict=True)
    long_g = VoxelGrid(np.zeros(3), (1200, 2, 2), 0.25)
    _compare("long-1200", long_g, np.array([[-1., .1, .1]]), np.array([[1., 0., 0.]]),
             np.array([302.]), np.array([1.]), np.array([1e8]), strict=True)
    o = np.repeat([[-1.5, -2.5, -3.5]], 4, axis=0)
    d = np.repeat([[1., 0., 0.]], 4, axis=0)
    _compare("same-voxel-multiweight", g, o, d, np.array([.2, .3, .4, .5]),
             np.array([0., 1e-12, 1., 1e9]), np.array([1e8, 0., 3., 1e-8]))


def endpoint_regression():
    """water60_freeで判明した終点近傍面の完全精度回帰ケース。"""
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.0)
    origin = np.array([[-3.037619819265908, 0.0, 0.0]])
    direction = np.array([[
        0.018394208153120428,
        0.6110258451767133,
        -0.7913969103000736,
    ]])
    length = np.array([1.5015538205527557])
    wa = np.array([1.9965502538144548])
    wb = np.array([0.5335686225843084])
    ra, rb, na, nb = _run_pair(grid, origin, direction, length, wa, wb)
    assert np.array_equal(np.flatnonzero(ra), np.flatnonzero(na))
    assert np.array_equal(np.flatnonzero(rb), np.flatnonzero(nb))
    assert np.allclose(ra, na, atol=5e-12, rtol=5e-12)
    assert np.allclose(rb, nb, atol=5e-12, rtol=5e-12)
    assert na[1, 25, 24] == 0.0
    assert nb[1, 25, 24] == 0.0
    print("water60_free_endpoint_near_plane: PASS; voxel(1,25,24)=0")


def _compare_one_silent(grid, origin, direction, length):
    o = np.asarray(origin, dtype=float).reshape(1, 3)
    d = np.asarray(direction, dtype=float).reshape(1, 3)
    ds = np.array([length], dtype=float)
    wa = np.array([3.25])
    wb = np.array([1e6])
    ra, rb, na, nb = _run_pair(grid, o, d, ds, wa, wb)
    for ref, got in ((ra, na), (rb, nb)):
        assert np.array_equal(np.flatnonzero(ref), np.flatnonzero(got))
        assert np.allclose(ref, got, atol=5e-12, rtol=5e-12)


def boundary_near_fuzz():
    """端点・クリップ点・近接同着を、集約せず区間単位で参照比較する。"""
    rng = np.random.default_rng(20260729)
    grid = VoxelGrid(np.zeros(3), (8, 8, 8), 1.0)
    offsets = np.array([0.0, 0.5 * _EPS_PLANE, 0.99 * _EPS_PLANE,
                        1.01 * _EPS_PLANE, 2.0 * _EPS_PLANE])
    checked = 0

    # 3軸×始終点×正負方向×5水準の各組合せにつき50区間。
    for axis in range(3):
        for endpoint_kind in ("start", "end"):
            for sign in (-1.0, 1.0):
                for offset in offsets:
                    for repetition in range(50):
                        side = -1.0 if repetition % 2 == 0 else 1.0
                        plane = 1 + repetition % 6
                        point = rng.uniform(1.2, 6.8, 3)
                        point[axis] = plane + side * offset
                        raw = rng.uniform(0.2, 1.0, 3)
                        raw[axis] *= sign
                        raw[(axis + 1) % 3] *= -1.0 if repetition % 3 == 0 else 1.0
                        direction = _unit(raw)
                        # 短い初回境界ケースと長い走査を同じ組合せに含める。
                        length = 0.02 + 0.005 * repetition if repetition < 10 else 1.0 + 0.13 * repetition
                        origin = point if endpoint_kind == "start" else point - direction * length
                        actual = origin if endpoint_kind == "start" else origin + direction * length
                        plane_distance = abs((actual[axis] - grid.origin_cm[axis])
                                             / grid.voxel_size_cm - plane)
                        assert np.isclose(plane_distance, offset, rtol=0.0, atol=2e-15)
                        _compare_one_silent(grid, origin, direction, length)
                        checked += 1

    # AABBクリップでできるenter/exitが、別軸の内部面近傍に来る斜め区間。
    clipped_checked = 0
    for axis in range(3):
        clip_axis = (axis + 1) % 3
        for at_exit in (False, True):
            for offset in offsets:
                for repetition in range(50):
                    side = -1.0 if repetition % 2 == 0 else 1.0
                    plane = 1 + repetition % 6
                    direction = _unit([0.47 + 0.01 * repetition,
                                       0.73 - 0.003 * repetition, 0.39])
                    if at_exit:
                        direction = -direction
                        boundary = 8.0
                    else:
                        boundary = 0.0
                    t_clip = 1.0 / abs(direction[clip_axis])
                    origin = rng.uniform(1.0, 7.0, 3)
                    origin[clip_axis] = boundary - direction[clip_axis] * t_clip
                    origin[axis] = plane + side * offset - direction[axis] * t_clip
                    length = t_clip + 3.0
                    actual = origin + direction * t_clip
                    plane_distance = abs(actual[axis] - plane)
                    assert np.isclose(plane_distance, offset, rtol=0.0, atol=2e-15)
                    _compare_one_silent(grid, origin, direction, length)
                    clipped_checked += 1

    # 2軸・3軸の厳密同着、および旧tol=2e-12の内外にある近接同着。
    tie_checked = 0
    for delta in (0.0, 1e-12, -1e-12, 3e-12, -3e-12):
        for three_axes in (False, True):
            for repetition in range(50):
                origin = np.array([0.2, 0.3, 0.4])
                tx = 1.0
                ty = tx + delta
                tz = tx if delta == 0.0 else tx - delta
                raw = np.array([(1.0 - origin[0]) / tx,
                                (1.0 - origin[1]) / ty,
                                (1.0 - origin[2]) / tz if three_axes else 0.31])
                norm = np.linalg.norm(raw)
                direction = raw / norm
                arrivals = ((1.0 - origin) / direction)
                assert arrivals[1] == arrivals[0] if delta == 0.0 else arrivals[1] != arrivals[0]
                if three_axes and delta == 0.0:
                    assert arrivals[2] == arrivals[0]
                _compare_one_silent(grid, origin, direction, 4.0 * norm)
                tie_checked += 1

    print(f"boundary-near-fuzz: PASS endpoint={checked} clipped={clipped_checked} "
          f"near-tie={tie_checked} (interval-by-interval)")


def fuzz():
    rng = np.random.default_rng(20260728)
    g = VoxelGrid(np.array([-5.3, -7.1, 2.2]), (17, 11, 23), 0.7)
    n = 6000
    o = rng.uniform([-10, -12, -3], [12, 8, 24], (n, 3))
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    ds = rng.uniform(0, 35, n)
    wa = 10 ** rng.uniform(-5, 5, n)
    wb = rng.choice([0., 1e-9, 1., 1e7], n)
    _compare("random-fuzz-6000", g, o, d, ds, wa, wb)


def chest_segments(n=4000):
    """chest_roomを実際に輸送し、既存タリー呼出しへ渡る散乱区間を捕捉する。"""
    captured = []
    original = transport_mod.accumulate_track_length_multi
    def capture(pairs, grid, o, d, ds):
        captured.append((o.copy(), d.copy(), ds.copy(),
                         pairs[0][1].copy(), pairs[1][1].copy()))
    transport_mod.accumulate_track_length_multi = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            transport_mod.run_transport(load_scene("examples/chest_room.yaml"), n_histories=n,
                                        seed=20260728, batch_size=n, dose_grid=True,
                                        grid_resolution_cm=2.0, n_workers=1)
    finally:
        transport_mod.accumulate_track_length_multi = original
    o = np.concatenate([x[0] for x in captured])
    d = np.concatenate([x[1] for x in captured])
    ds = np.concatenate([x[2] for x in captured])
    wa = np.concatenate([x[3] for x in captured])
    wb = np.concatenate([x[4] for x in captured])
    scene = load_scene("examples/chest_room.yaml")
    # run_transportと同じscene bboxから2 cmグリッドを作る。
    from chatcarlo.geometry import Geometry
    geom = Geometry(scene.raw["geometry"])
    grid = VoxelGrid.from_bbox(geom.bbox_min, geom.bbox_max, 2.0)
    return grid, o, d, ds, wa, wb


def water_segments(n=200_000):
    """固定seedでwater60_freeのkernelが実際に吐く(o,d,ds,e,mat)由来入力。"""
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.0)
    tables = kernel_mod.bake_scene_materials(["water", "air"])
    boxes = [{"center": (0., 0., 0.), "size_cm": (10., 100., 100.),
              "material": "water"}]
    geom = kernel_mod.bake_box_scene(boxes, background="air", tables=tables,
                                     bbox_margin_cm=.01)
    captured = []
    original = kernel_mod.accumulate_track_length_multi
    def capture(pairs, ignored_grid, o, d, ds):
        captured.append((o.copy(), d.copy(), ds.copy(),
                         pairs[0][1].copy(), pairs[1][1].copy()))
    kernel_mod.accumulate_track_length_multi = capture
    try:
        kernel_mod.run_dose_grid(
            tables, geom, 60., (-5.01, 0., 0.), (1., 0., 0.),
            n, 20260728, grid, batch_size=200_000, n_chunks=8,
            fluorescence_enabled=True, max_segments_per_history=16)
    finally:
        kernel_mod.accumulate_track_length_multi = original
    return (grid, *(np.concatenate([x[k] for x in captured]) for k in range(5)))


def integrated_water_time():
    """既存kernel+DDA統合経路を同じ条件で1回計測する（判定には使わない）。"""
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), 2.0)
    tables = kernel_mod.bake_scene_materials(["water", "air"])
    boxes = [{"center": (0., 0., 0.), "size_cm": (10., 100., 100.),
              "material": "water"}]
    geom = kernel_mod.bake_box_scene(boxes, background="air", tables=tables,
                                     bbox_margin_cm=.01)
    t0 = time.perf_counter()
    kernel_mod.run_dose_grid(
        tables, geom, 60., (-5.01, 0., 0.), (1., 0., 0.),
        200_000, 20260728, grid, batch_size=200_000, n_chunks=8,
        fluorescence_enabled=True, max_segments_per_history=16)
    return time.perf_counter() - t0


def correctness():
    print(f"tolerances: single atol=5e-12 rtol=5e-12; aggregate atol={ATOL:g} rtol={RTOL:g}")
    boundary_cases()
    endpoint_regression()
    boundary_near_fuzz()
    fuzz()
    g, o, d, ds, wa, wb = chest_segments()
    _compare("chest_room-derived", g, o, d, ds, wa, wb)
    print("CORRECTNESS: PASS")


def timing(scenario):
    data = water_segments() if scenario == "water60_free" else chest_segments(20_000)
    g, o, d, ds, wa, wb = data
    # 両方をウォームアップ。入力生成・コンパイルは計測外。
    _run_pair(g, o[:10], d[:10], ds[:10], wa[:10], wb[:10])
    ref_times, njit_times = [], []
    for rep in range(8):
        order = ("numpy", "njit") if rep % 2 == 0 else ("njit", "numpy")
        row = {}
        for arm in order:
            a = np.zeros(g.shape); b = np.zeros(g.shape)
            t0 = time.perf_counter()
            if arm == "numpy":
                accumulate_track_length_multi(((a, wa), (b, wb)), g, o, d, ds)
            else:
                dda_njit(o, d, ds, wa, wb, g.origin_cm, np.asarray(g.shape, dtype=np.int64),
                         g.voxel_size_cm, a, b)
            row[arm] = time.perf_counter() - t0
        ref_times.append(row["numpy"]); njit_times.append(row["njit"])
        print(f"rep{rep} order={'/'.join(order)} numpy={row['numpy']:.6f}s njit={row['njit']:.6f}s")
    mr = float(np.median(ref_times)); mn = float(np.median(njit_times))
    speed = mr / mn
    pred = .538 / (.136 + .402 / speed)
    gate_a = mn < mr
    gate_b = pred >= 3.0
    print(f"scenario={scenario} segments={len(o)} median_numpy={mr:.6f}s "
          f"median_njit={mn:.6f}s dda_speedup={speed:.3f}x")
    print(f"gate_A={'PASS' if gate_a else 'FAIL'}")
    if scenario == "water60_free":
        integrated = integrated_water_time()
        print(f"integrated_reproduction={integrated:.6f}s "
              "(kernel+existing DDA, one run, sanity only)")
        print("registered_reference: total=0.538s dda=0.402s residual=0.136s")
        print(f"amdahl_predicted_end_to_end={pred:.3f}x gate_B={'PASS' if gate_b else 'FAIL'}")
        print(f"decision={'PROCEED' if gate_a and gate_b else 'DO_NOT_PROCEED'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=("correctness", "timing"))
    p.add_argument("--scenario", choices=("water60_free", "chest_room"))
    args = p.parse_args()
    if args.mode == "correctness":
        correctness()
    else:
        if args.scenario is None:
            p.error("--scenario is required for timing")
        timing(args.scenario)


if __name__ == "__main__":
    main()
