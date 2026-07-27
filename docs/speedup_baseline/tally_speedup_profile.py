"""docs/plan_tally_speedup.md Step 0: _segment_grid_traversal(_accumulate)の
フェーズ別プロファイル。

回帰条件（chest_room.yaml, --dose-grid, res=2cm, n=2e5, batch=2e5, single
worker）で、AABBクリップ/軸別counts/ラグド構築+gather/ソート/中点+voxel_index/
np.add.atの6区間に分けて壁時間を計測する。関数を丸ごとコピーして計装した版に
差し替える（chatcarlo/tally.py自体は変更しない）。

実行:
    PYTHONPATH=. .venv/bin/python docs/speedup_baseline/tally_speedup_profile.py
"""
from __future__ import annotations

import time
import collections

import numpy as np

import chatcarlo.tally as tally_mod
from chatcarlo.tally import _EPS_PLANE, _argsort_within_segment

PHASE_TOTALS = collections.defaultdict(float)
CALL_COUNT = 0
TOTAL_INTERSECTIONS = 0


def instrumented_segment_grid_traversal_accumulate(grid, origin, direction, length_cm,
                                                    target_weight_pairs):
    global CALL_COUNT, TOTAL_INTERSECTIONS
    CALL_COUNT += 1
    t0 = time.perf_counter()

    n = origin.shape[0]
    h = grid.voxel_size_cm
    lo = grid.origin_cm
    shape = np.asarray(grid.shape)
    hi = lo + shape * h

    t_enter = np.zeros(n)
    t_exit = length_cm.astype(float).copy()
    for k in range(3):
        dk = direction[:, k]
        ok = origin[:, k]
        parallel = np.abs(dk) < 1e-12
        dk_safe = np.where(parallel, 1.0, dk)
        ta = (lo[k] - ok) / dk_safe
        tb = (hi[k] - ok) / dk_safe
        tmin_k = np.where(parallel, -np.inf, np.minimum(ta, tb))
        tmax_k = np.where(parallel, np.inf, np.maximum(ta, tb))
        outside_slab = parallel & ((ok < lo[k]) | (ok > hi[k]))
        tmin_k = np.where(outside_slab, np.inf, tmin_k)
        tmax_k = np.where(outside_slab, -np.inf, tmax_k)
        t_enter = np.maximum(t_enter, tmin_k)
        t_exit = np.minimum(t_exit, tmax_k)

    active = np.where(t_exit > t_enter)[0]
    t1 = time.perf_counter()
    PHASE_TOTALS["1_aabb_clip"] += t1 - t0
    if len(active) == 0:
        return

    o = origin[active]
    d = direction[active]
    t_enter = t_enter[active]
    t_exit = t_exit[active]
    m = len(active)

    counts = np.zeros((3, m), dtype=np.int64)
    m_lo_all = np.zeros((3, m), dtype=np.int64)
    d_safe_all = np.zeros((3, m))
    for k in range(3):
        dk = d[:, k]
        parallel = np.abs(dk) < 1e-12
        dk_safe = np.where(parallel, 1.0, dk)
        x_enter = o[:, k] + t_enter * dk
        x_exit = o[:, k] + t_exit * dk
        p_enter = (x_enter - lo[k]) / h
        p_exit = (x_exit - lo[k]) / h
        p_small = np.minimum(p_enter, p_exit)
        p_big = np.maximum(p_enter, p_exit)
        m_lo_k = np.ceil(p_small + _EPS_PLANE).astype(np.int64)
        m_hi_k = np.floor(p_big - _EPS_PLANE).astype(np.int64)
        cnt_k = np.clip(m_hi_k - m_lo_k + 1, 0, None)
        cnt_k = np.where(parallel, 0, cnt_k)
        counts[k] = cnt_k
        m_lo_all[k] = m_lo_k
        d_safe_all[k] = dk_safe

    t2 = time.perf_counter()
    PHASE_TOTALS["2_axis_counts"] += t2 - t1

    t_parts = [t_enter, t_exit]
    seg_parts = [np.arange(m), np.arange(m)]
    for k in range(3):
        cnt_k = counts[k]
        total_k = int(cnt_k.sum())
        if total_k == 0:
            continue
        seg_id_k = np.repeat(np.arange(m), cnt_k)
        starts_k = np.cumsum(cnt_k) - cnt_k
        offset_k = np.arange(total_k) - np.repeat(starts_k, cnt_k)
        m_idx_k = m_lo_all[k][seg_id_k] + offset_k
        t_k = (lo[k] + m_idx_k * h - o[seg_id_k, k]) / d_safe_all[k][seg_id_k]
        t_parts.append(t_k)
        seg_parts.append(seg_id_k)

    all_t = np.concatenate(t_parts)
    all_seg = np.concatenate(seg_parts)
    TOTAL_INTERSECTIONS += len(all_t)

    t3 = time.perf_counter()
    PHASE_TOTALS["3_ragged_build_gather"] += t3 - t2

    order = _argsort_within_segment(all_t, all_seg, t_enter, t_exit)
    sorted_t = all_t[order]
    sorted_seg = all_seg[order]

    t4 = time.perf_counter()
    PHASE_TOTALS["4_sort"] += t4 - t3

    same_seg = sorted_seg[1:] == sorted_seg[:-1]
    overlap = np.where(same_seg, sorted_t[1:] - sorted_t[:-1], 0.0)
    keep = overlap > 0
    if not np.any(keep):
        t5 = time.perf_counter()
        PHASE_TOTALS["5_midpoint_voxelindex"] += t5 - t4
        return

    seg_for_interval = sorted_seg[:-1][keep]
    mid_t = ((sorted_t[:-1] + sorted_t[1:]) / 2.0)[keep]
    overlap = overlap[keep]

    points = o[seg_for_interval] + d[seg_for_interval] * mid_t[:, None]
    idx, in_grid = grid.voxel_index(points)
    idx, overlap, seg_for_interval = idx[in_grid], overlap[in_grid], seg_for_interval[in_grid]

    t5 = time.perf_counter()
    PHASE_TOTALS["5_midpoint_voxelindex"] += t5 - t4

    seg_id_final = active[seg_for_interval]
    flat_idx = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.shape)
    for target, weight_per_cm in target_weight_pairs:
        weight_flat = weight_per_cm[seg_id_final] * overlap
        np.add.at(target.reshape(-1), flat_idx, weight_flat)
    t6 = time.perf_counter()
    PHASE_TOTALS["6_add_at"] += t6 - t5


def main():
    tally_mod._segment_grid_traversal_accumulate = instrumented_segment_grid_traversal_accumulate
    import chatcarlo.transport as transport_mod
    transport_mod.accumulate_track_length_multi.__globals__["_segment_grid_traversal_accumulate"] = \
        instrumented_segment_grid_traversal_accumulate

    from chatcarlo.scene import load_scene

    scene = load_scene("examples/chest_room.yaml")

    t_start = time.perf_counter()
    result = transport_mod.run_transport(
        scene, n_histories=200_000, seed=42, batch_size=200_000,
        dose_grid=True, grid_resolution_cm=2.0, n_workers=1,
        track_uncertainty=True,
    )
    t_end = time.perf_counter()

    print(f"total wall time: {t_end - t_start:.3f}s")
    print(f"traversal calls: {CALL_COUNT}, total intersections (all_t len summed): {TOTAL_INTERSECTIONS}")
    print()
    total_instrumented = sum(PHASE_TOTALS.values())
    for name, secs in sorted(PHASE_TOTALS.items()):
        pct = 100 * secs / total_instrumented if total_instrumented > 0 else 0
        print(f"  {name:28s} {secs:8.3f}s  {pct:5.1f}%")
    print(f"  {'(sum of phases)':28s} {total_instrumented:8.3f}s")
    print(f"kerma max: {result.grid.kerma_keV.max():.6e}")


if __name__ == "__main__":
    main()
