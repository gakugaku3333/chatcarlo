"""njitスカラーDDAをkernel実経路へ一時統合するend-to-end A/B測定。

本番コードは変更せず、計測中だけkernel.accumulate_track_length_multiを
モンキーパッチする。end-to-endはkernel.run_dose_grid全体を指す。
"""
from __future__ import annotations

import argparse
import contextlib
import resource
import time
from dataclasses import dataclass

import numpy as np

import chatcarlo.kernel as kernel_mod
from chatcarlo.tally import VoxelGrid
from docs.speedup_baseline.dda_njit_prototype_benchmark import ATOL, RTOL, dda_njit


N_HISTORIES = 200_000
SEED = 20260728
N_CHUNKS = 8
RESOLUTION_CM = 2.0


def make_water_case():
    grid = VoxelGrid(np.array([-5.01, -50.01, -50.01]), (6, 51, 51), RESOLUTION_CM)
    tables = kernel_mod.bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0),
              "material": "water"}]
    geom = kernel_mod.bake_box_scene(
        boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    return tables, geom, grid


def _check_contract(pairs, grid, origin, direction, length_cm):
    assert len(pairs) == 2, "dda_njit adapter requires exactly two targets"
    n_segments = len(length_cm)
    assert origin.shape == (n_segments, 3)
    assert direction.shape == (n_segments, 3)
    for target, weights in pairs:
        assert target.shape == grid.shape
        assert len(weights) == n_segments


def njit_adapter(pairs, grid, origin, direction, length_cm):
    """2-target版dda_njitを既存関数と同じ積算セマンティクスで呼ぶ。"""
    _check_contract(pairs, grid, origin, direction, length_cm)
    (target_a, weights_a), (target_b, weights_b) = pairs
    dda_njit(
        origin, direction, length_cm, weights_a, weights_b,
        grid.origin_cm, np.asarray(grid.shape, dtype=np.int64),
        grid.voxel_size_cm, target_a, target_b)


@contextlib.contextmanager
def patched_accumulator(replacement):
    original = kernel_mod.accumulate_track_length_multi
    kernel_mod.accumulate_track_length_multi = replacement
    try:
        yield
    finally:
        kernel_mod.accumulate_track_length_multi = original


def run_arm(accumulator, batch_size):
    tables, geom, grid = make_water_case()
    with patched_accumulator(accumulator):
        result = kernel_mod.run_dose_grid(
            tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
            N_HISTORIES, SEED, grid, batch_size=batch_size, n_chunks=N_CHUNKS,
            fluorescence_enabled=True, max_segments_per_history=16)
    return result, grid


def compare_grid(label, ref, got):
    all_ok = True
    for name, a, b in (
            ("kerma_keV", ref.kerma_keV, got.kerma_keV),
            ("h10_track_pSv_cm3", ref.h10_track_pSv_cm3, got.h10_track_pSv_cm3)):
        nz_equal = np.array_equal(np.flatnonzero(a), np.flatnonzero(b))
        abs_err = float(np.max(np.abs(a - b)))
        denom = np.maximum(np.abs(a), ATOL)
        rel_err = float(np.max(np.abs(a - b) / denom))
        values_ok = np.allclose(a, b, atol=ATOL, rtol=RTOL)
        ok = nz_equal and values_ok
        all_ok &= ok
        print(f"{label}/{name}: {'PASS' if ok else 'FAIL'} "
              f"nonzero={'PASS' if nz_equal else 'FAIL'} "
              f"max_abs={abs_err:.17g} max_rel={rel_err:.17g}")
    if not all_ok:
        raise AssertionError(f"{label}: grid mismatch")


def compare_batch_result(label, ref, got):
    fields = ("n_scatter", "absorbed", "escaped", "final_energy",
              "energy_deposited", "n_fluorescence")
    outcomes = []
    for field in fields:
        same = np.array_equal(getattr(ref, field), getattr(got, field))
        outcomes.append(same)
        print(f"{label}/KernelBatchResult.{field}: {'PASS' if same else 'FAIL'}")
    if not all(outcomes):
        raise AssertionError(f"{label}: KernelBatchResult mismatch")


def adapter_unit_checks():
    grid = VoxelGrid(np.zeros(3), (2, 2, 2), 1.0)
    a = np.full(grid.shape, 3.0)
    b = np.full(grid.shape, 5.0)
    origin = np.array([[-1.0, 0.5, 0.5]])
    direction = np.array([[1.0, 0.0, 0.0]])
    length = np.array([2.0])
    njit_adapter(((a, np.array([2.0])), (b, np.array([7.0]))),
                 grid, origin, direction, length)
    assert a[0, 0, 0] == 5.0 and b[0, 0, 0] == 12.0
    assert np.all(a[1:] == 3.0) and np.all(b[1:] == 5.0)
    print("adapter accumulation into nonzero targets: PASS")

    empty_o = np.empty((0, 3))
    empty_d = np.empty((0, 3))
    empty = np.empty(0)
    before_a, before_b = a.copy(), b.copy()
    njit_adapter(((a, empty), (b, empty)), grid, empty_o, empty_d, empty)
    assert np.array_equal(a, before_a) and np.array_equal(b, before_b)
    print("adapter empty input: PASS")

    original = kernel_mod.accumulate_track_length_multi
    try:
        with patched_accumulator(njit_adapter):
            kernel_mod.accumulate_track_length_multi(
                ((a, empty),), grid, empty_o, empty_d, empty)
    except AssertionError:
        pass
    else:
        raise AssertionError("intentional adapter exception was not raised")
    assert kernel_mod.accumulate_track_length_multi is original
    print("monkey-patch restoration after intentional exception: PASS")


def correctness():
    print("mode: correctness")
    adapter_unit_checks()
    for label, batch_size in (("one_batch_200000", 200_000),
                              ("remainder_batches_60000", 60_000)):
        ref_result, ref_grid = run_arm(kernel_mod.accumulate_track_length_multi, batch_size)
        got_result, got_grid = run_arm(njit_adapter, batch_size)
        compare_grid(label, ref_grid, got_grid)
        compare_batch_result(label, ref_result, got_result)
    print("correctness overall: PASS")


@dataclass
class TimedAccumulator:
    function: object
    elapsed_s: float = 0.0
    calls: int = 0

    def __call__(self, pairs, grid, origin, direction, length_cm):
        start = time.perf_counter()
        try:
            return self.function(pairs, grid, origin, direction, length_cm)
        finally:
            self.elapsed_s += time.perf_counter() - start
            self.calls += 1


def timed_arm(kind, n_histories=N_HISTORIES):
    tables, geom, grid = make_water_case()
    function = (kernel_mod.accumulate_track_length_multi
                if kind == "A" else njit_adapter)
    timer = TimedAccumulator(function)
    start = time.perf_counter()
    with patched_accumulator(timer):
        kernel_mod.run_dose_grid(
            tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
            n_histories, SEED, grid, batch_size=200_000, n_chunks=N_CHUNKS,
            fluorescence_enabled=True, max_segments_per_history=16)
    e2e = time.perf_counter() - start
    return e2e, timer.elapsed_s, timer.calls


def stats(values):
    return min(values), float(np.median(values)), max(values)


def timing():
    print("mode: timing")
    print("end-to-end definition: kernel.run_dose_grid overall; excludes CLI startup, "
          "scene loading, and bake_scene_materials")
    print("timing-wrapper overhead: one perf_counter pair per DDA call "
          "(one call per batch), expected negligible")
    print("warmup: A then B, n_histories=2000")
    timed_arm("A", 2_000)
    timed_arm("B", 2_000)

    rows = []
    orders = (("A", "B"), ("B", "A")) * 4
    for repetition, order in enumerate(orders, 1):
        row = {"repetition": repetition, "order": "".join(order)}
        for arm in order:
            e2e, dda, calls = timed_arm(arm)
            row[f"E2E_{arm}"] = e2e
            row[f"DDA_{arm}"] = dda
            row[f"residual_{arm}"] = e2e - dda
            row[f"calls_{arm}"] = calls
        row["speedup"] = row["E2E_A"] / row["E2E_B"]
        rows.append(row)
        print(
            f"rep={repetition} order={row['order']} "
            f"E2E_A={row['E2E_A']:.9f} DDA_in_A={row['DDA_A']:.9f} "
            f"residual_A={row['residual_A']:.9f} calls_A={row['calls_A']} "
            f"E2E_B={row['E2E_B']:.9f} DDA_in_B={row['DDA_B']:.9f} "
            f"residual_B={row['residual_B']:.9f} calls_B={row['calls_B']} "
            f"speedup={row['speedup']:.6f}")

    e2e_a = [r["E2E_A"] for r in rows]
    e2e_b = [r["E2E_B"] for r in rows]
    dda_a = [r["DDA_A"] for r in rows]
    dda_b = [r["DDA_B"] for r in rows]
    residual_a = [r["residual_A"] for r in rows]
    residual_b = [r["residual_B"] for r in rows]
    for label, values in (("E2E_A", e2e_a), ("DDA_in_A", dda_a),
                          ("residual_A", residual_a), ("E2E_B", e2e_b),
                          ("DDA_in_B", dda_b), ("residual_B", residual_b)):
        lo, median, hi = stats(values)
        print(f"{label}: min={lo:.9f} median={median:.9f} max={hi:.9f}")

    median_e2e_a = float(np.median(e2e_a))
    median_e2e_b = float(np.median(e2e_b))
    median_dda_a = float(np.median(dda_a))
    median_dda_b = float(np.median(dda_b))
    median_residual_a = float(np.median(residual_a))
    median_residual_b = float(np.median(residual_b))
    measured = median_e2e_a / median_e2e_b
    dda_speedup = median_dda_a / median_dda_b
    measured_prediction = median_e2e_a / (
        median_residual_a + median_dda_a / dda_speedup)
    all_faster = all(r["E2E_B"] < r["E2E_A"] for r in rows)
    achieved = all_faster and measured > 1.0
    residual_change = median_residual_b - median_residual_a
    residual_change_pct = residual_change / median_residual_a * 100.0

    print("per-repetition speedups: " +
          ", ".join(f"{r['speedup']:.6f}" for r in rows))
    print(f"S_E2E_measured={measured:.6f}")
    print("fixed_prediction=2.721000")
    print(f"same-run decomposition prediction={measured_prediction:.6f}")
    print(f"measured/fixed_prediction={measured / 2.721:.6f}")
    print(f"median E2E_A vs reported 0.538 s: {median_e2e_a:.9f} / 0.538000000")
    print(f"median residual change B-A={residual_change:.9f}s "
          f"({residual_change_pct:.3f}% of residual_A)")
    print(f"all 8 repetitions E2E_B < E2E_A: {all_faster}")
    print(f"adoption gate: {'高速化を達成' if achieved else '高速化を確認できず'}")
    print("post-replacement bottleneck candidate: residual_B (transport, segment-buffer "
          "allocation/concatenation, and weight calculation collectively); no further "
          "profiling is in scope")
    print(f"process ru_maxrss reference={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
    print("memory note: process-lifetime cumulative peak only; no arm-wise claim")
    print("chest_room end-to-end: not measurable because kernel supports boxes only")
    if not achieved:
        raise AssertionError("adoption gate failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("correctness", "timing"), required=True)
    args = parser.parse_args()
    if args.mode == "correctness":
        correctness()
    else:
        timing()


if __name__ == "__main__":
    main()
