"""本番kernel経路のuse_njit_dda引数によるend-to-end A/B測定。"""
from __future__ import annotations

import argparse
import resource
import time

import numpy as np

import chatcarlo.kernel as kernel_mod
from chatcarlo.tally import VoxelGrid


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


def run_arm(use_njit_dda, batch_size):
    tables, geom, grid = make_water_case()
    result = kernel_mod.run_dose_grid(
        tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
        N_HISTORIES, SEED, grid, batch_size=batch_size, n_chunks=N_CHUNKS,
        fluorescence_enabled=True, max_segments_per_history=16,
        use_njit_dda=use_njit_dda)
    return result, grid


def compare_grid(label, ref, got):
    all_ok = True
    for name, a, b in (
            ("kerma_keV", ref.kerma_keV, got.kerma_keV),
            ("h10_track_pSv_cm3", ref.h10_track_pSv_cm3, got.h10_track_pSv_cm3)):
        nz_equal = np.array_equal(np.flatnonzero(a), np.flatnonzero(b))
        abs_err = float(np.max(np.abs(a - b)))
        rel_err = 0.0 if abs_err == 0.0 else float("inf")
        values_ok = np.array_equal(a, b)
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


def correctness():
    print("mode: correctness")
    for label, batch_size in (("one_batch_200000", 200_000),
                              ("remainder_batches_60000", 60_000)):
        ref_result, ref_grid = run_arm(False, batch_size)
        got_result, got_grid = run_arm(True, batch_size)
        compare_grid(label, ref_grid, got_grid)
        compare_batch_result(label, ref_result, got_result)
    print("correctness overall: PASS")


def timed_arm(kind, n_histories=N_HISTORIES):
    tables, geom, grid = make_water_case()
    start = time.perf_counter()
    kernel_mod.run_dose_grid(
        tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0),
        n_histories, SEED, grid, batch_size=200_000, n_chunks=N_CHUNKS,
        fluorescence_enabled=True, max_segments_per_history=16,
        use_njit_dda=(kind == "B"))
    e2e = time.perf_counter() - start
    return e2e


def stats(values):
    return min(values), float(np.median(values)), max(values)


def timing():
    print("mode: timing")
    print("end-to-end definition: kernel.run_dose_grid overall; excludes CLI startup, "
          "scene loading, and bake_scene_materials")
    print("A=False (numpy reference), B=True (production njit path)")
    print("warmup: A then B, n_histories=2000")
    timed_arm("A", 2_000)
    timed_arm("B", 2_000)

    rows = []
    orders = (("A", "B"), ("B", "A")) * 4
    for repetition, order in enumerate(orders, 1):
        row = {"repetition": repetition, "order": "".join(order)}
        for arm in order:
            e2e = timed_arm(arm)
            row[f"E2E_{arm}"] = e2e
        row["speedup"] = row["E2E_A"] / row["E2E_B"]
        rows.append(row)
        print(
            f"rep={repetition} order={row['order']} "
            f"E2E_A={row['E2E_A']:.9f} "
            f"E2E_B={row['E2E_B']:.9f} "
            f"speedup={row['speedup']:.6f}")

    e2e_a = [r["E2E_A"] for r in rows]
    e2e_b = [r["E2E_B"] for r in rows]
    for label, values in (("E2E_A", e2e_a), ("E2E_B", e2e_b)):
        lo, median, hi = stats(values)
        print(f"{label}: min={lo:.9f} median={median:.9f} max={hi:.9f}")

    median_e2e_a = float(np.median(e2e_a))
    median_e2e_b = float(np.median(e2e_b))
    measured = median_e2e_a / median_e2e_b
    all_faster = all(r["E2E_B"] < r["E2E_A"] for r in rows)
    achieved = all_faster and measured > 1.0

    print("per-repetition speedups: " +
          ", ".join(f"{r['speedup']:.6f}" for r in rows))
    print(f"S_E2E_measured={measured:.6f}")
    print("fixed_prediction=2.721000")
    print(f"measured/fixed_prediction={measured / 2.721:.6f}")
    print(f"median E2E_A vs reported 0.538 s: {median_e2e_a:.9f} / 0.538000000")
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
