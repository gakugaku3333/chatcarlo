"""Profile the non-DDA remainder of kernel.run_batch_with_tally.

Run with:
    .venv/bin/python -m docs.speedup_baseline.residual_b_breakdown_profile
"""
from __future__ import annotations

import time

import numpy as np

import chatcarlo.kernel as kernel_mod
from chatcarlo.dose_coefficients import h_star_10_per_fluence
from chatcarlo.materials import density, material_groups, mu_en_rho
from chatcarlo.tally_njit import accumulate_track_length_multi_njit
from docs.speedup_baseline.dda_njit_e2e_benchmark import (
    N_CHUNKS,
    N_HISTORIES,
    RESOLUTION_CM,
    SEED,
    make_water_case,
)


N_REPETITIONS = 8
WARMUP_HISTORIES = 2_000
MAX_SEGMENTS_PER_HISTORY = 16

REPORTED_E2E_B = 0.199722438
REPORTED_RESIDUAL_B = 0.128791854
REPORTED_DDA_IN_B = 0.070733292
LARGE_DIVERGENCE_REL = 0.10


def _legacy_weight_calculation(tables, seg_mat_all, seg_e_all):
    """Comparison-only reproduction of the pre-fast-path production code."""
    mat_names = np.array(tables.material_names, dtype=object)[seg_mat_all]
    mu_en_linear = np.zeros(len(seg_e_all))
    for name, mask in material_groups(mat_names):
        mu_en_linear[mask] = (
            mu_en_rho(name, seg_e_all[mask]) * density(name)
        )
    return (
        seg_e_all * mu_en_linear,
        h_star_10_per_fluence(seg_e_all),
    )


def profile_once(n_histories: int):
    """Reproduce run_batch_with_tally's njit path with timers between steps."""
    tables, geom, grid = make_water_case()
    energy0_kev = 60.0
    origin = (-5.01, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0)

    t0 = time.perf_counter()
    chunk_seeds, chunk_offsets, chunk_counts = kernel_mod._chunk_plan(
        n_histories, N_CHUNKS, SEED
    )
    t1 = time.perf_counter()

    max_elem = tables.zs.shape[1]
    n_chunks_actual = len(chunk_seeds)
    seg_capacity_per_chunk = (
        int(chunk_counts.max()) * MAX_SEGMENTS_PER_HISTORY
    )
    seg_o = np.zeros((n_chunks_actual, seg_capacity_per_chunk, 3))
    seg_d = np.zeros((n_chunks_actual, seg_capacity_per_chunk, 3))
    seg_ds = np.zeros((n_chunks_actual, seg_capacity_per_chunk))
    seg_e = np.zeros((n_chunks_actual, seg_capacity_per_chunk))
    seg_mat = np.zeros(
        (n_chunks_actual, seg_capacity_per_chunk), dtype=np.int64
    )
    t2 = time.perf_counter()

    (
        _n_scatter,
        _absorbed,
        _escaped,
        _final_energy,
        _energy_deposited,
        _n_fluorescence,
        seg_count_per_chunk,
        seg_overflow_per_chunk,
    ) = kernel_mod._run_batch_scalar_tally(
        n_histories,
        n_chunks_actual,
        chunk_seeds,
        chunk_offsets,
        chunk_counts,
        energy0_kev,
        origin[0],
        origin[1],
        origin[2],
        direction[0],
        direction[1],
        direction[2],
        geom.n_boxes,
        geom.box_center,
        geom.box_half,
        geom.box_material,
        geom.background_material,
        geom.world_center,
        geom.world_half,
        tables.n_elem,
        tables.zs,
        tables.fracs,
        tables.log_e,
        tables.step,
        tables.n_grid,
        tables.density_g_cm3,
        tables.photo,
        tables.compt,
        tables.rayl,
        tables.incoh_q,
        tables.incoh_s,
        tables.rayl_x,
        tables.rayl_a,
        tables.k_edge,
        tables.k_omega,
        tables.k_frac,
        tables.k_line_e,
        tables.k_line_p,
        tables.n_lines,
        max_elem,
        True,
        seg_o,
        seg_d,
        seg_ds,
        seg_e,
        seg_mat,
        seg_capacity_per_chunk,
    )
    t3 = time.perf_counter()

    if np.any(seg_overflow_per_chunk):
        raise ValueError(
            "tally segment buffer overflowed during residual_B profiling"
        )
    t4 = time.perf_counter()

    seg_o_parts, seg_d_parts = [], []
    seg_ds_parts, seg_e_parts, seg_mat_parts = [], [], []
    for c in range(n_chunks_actual):
        cnt = int(seg_count_per_chunk[c])
        if cnt == 0:
            continue
        seg_o_parts.append(seg_o[c, :cnt])
        seg_d_parts.append(seg_d[c, :cnt])
        seg_ds_parts.append(seg_ds[c, :cnt])
        seg_e_parts.append(seg_e[c, :cnt])
        seg_mat_parts.append(seg_mat[c, :cnt])

    if not seg_o_parts:
        raise RuntimeError("profile case unexpectedly produced no flight segments")

    seg_o_all = np.concatenate(seg_o_parts)
    seg_d_all = np.concatenate(seg_d_parts)
    seg_ds_all = np.concatenate(seg_ds_parts)
    seg_e_all = np.concatenate(seg_e_parts)
    seg_mat_all = np.concatenate(seg_mat_parts)
    t5 = time.perf_counter()

    kerma_weights, h10_weights = kernel_mod._compute_tally_weights(
        tables, seg_mat_all, seg_e_all)
    t6 = time.perf_counter()

    accumulate_track_length_multi_njit(
        (
            (grid.kerma_keV, kerma_weights),
            (grid.h10_track_pSv_cm3, h10_weights),
        ),
        grid,
        seg_o_all,
        seg_d_all,
        seg_ds_all,
    )
    t7 = time.perf_counter()

    chunk_plan = t1 - t0
    buffer_allocation = t2 - t1
    transport = t3 - t2
    overflow_check = t4 - t3
    concatenate = t5 - t4
    weight_calculation = t6 - t5
    dda_accumulator = t7 - t6
    step_1_2 = chunk_plan + buffer_allocation
    residual = (
        step_1_2 + overflow_check + concatenate + weight_calculation
    )
    non_dda_including_transport = residual + transport
    total = non_dda_including_transport + dda_accumulator

    timings = {
        "chunk_plan": chunk_plan,
        "buffer_allocation": buffer_allocation,
        "step_1+2": step_1_2,
        "transport": transport,
        "overflow_check": overflow_check,
        "concatenate": concatenate,
        "weight_calculation": weight_calculation,
        "dda_accumulator": dda_accumulator,
        "residual_B_reconstructed": residual,
        "non_DDA_including_transport": non_dda_including_transport,
        "total_E2E_B_equivalent": total,
    }
    return timings, (tables, seg_mat_all, seg_e_all)


def benchmark_weight_calculation(tables, seg_mat_all, seg_e_all) -> None:
    """Compare old/new weight calculations on one fixed transport output."""
    for function in (_legacy_weight_calculation, kernel_mod._compute_tally_weights):
        for _ in range(2):
            weights = function(tables, seg_mat_all, seg_e_all)
            assert np.isfinite(weights[0].sum() + weights[1].sum())

    rows = []
    for repetition, order in enumerate((("legacy", "new"), ("new", "legacy")) * 4, 1):
        row = {}
        for arm in order:
            function = (_legacy_weight_calculation if arm == "legacy"
                        else kernel_mod._compute_tally_weights)
            start = time.perf_counter()
            weights = function(tables, seg_mat_all, seg_e_all)
            row[arm] = time.perf_counter() - start
            row[f"{arm}_checksum"] = float(weights[0].sum() + weights[1].sum())
        if row["legacy_checksum"] != row["new_checksum"]:
            raise AssertionError("legacy/new weight checksum mismatch")
        row["ratio"] = row["legacy"] / row["new"]
        rows.append(row)
        print(
            f"weight_rep={repetition} order={'/'.join(order)} "
            f"legacy={row['legacy']:.9f} new={row['new']:.9f} "
            f"legacy/new={row['ratio']:.6f}"
        )

    print("weight calculation AB/BA summary (seconds)")
    for arm in ("legacy", "new"):
        lo, median, hi = stats([row[arm] for row in rows])
        print(f"{arm}: min={lo:.9f} median={median:.9f} max={hi:.9f}")
    ratios = [row["ratio"] for row in rows]
    print("per-repetition legacy/new ratios: "
          + ", ".join(f"{ratio:.6f}" for ratio in ratios))
    ratio_median = float(np.median(ratios))
    print(f"median legacy/new ratio={ratio_median:.6f}")
    print(f"performance gate (>=1.20): {'PASS' if ratio_median >= 1.20 else 'FAIL'}")

    component_rows = []
    for _ in range(N_REPETITIONS):
        start = time.perf_counter()
        names = np.array(tables.material_names, dtype=object)[seg_mat_all]
        after_object = time.perf_counter()
        unique_names = sorted(set(names.tolist()))
        after_legacy_group = time.perf_counter()
        codes = np.unique(seg_mat_all)
        sorted((int(code) for code in codes),
               key=lambda code: tables.material_names[code])
        after_new_group = time.perf_counter()
        mu_en_linear = np.zeros(len(seg_e_all))
        for name in unique_names:
            mask = names == name
            mu_en_linear[mask] = (
                mu_en_rho(name, seg_e_all[mask]) * density(name))
        seg_e_all * mu_en_linear
        h_star_10_per_fluence(seg_e_all)
        after_interpolation = time.perf_counter()
        component_rows.append((
            after_object - start,
            after_legacy_group - after_object,
            after_new_group - after_legacy_group,
            after_interpolation - after_new_group,
        ))
    print("exploratory component medians (np.unique is included in new grouping)")
    for index, label in enumerate((
            "object_array", "legacy_set_grouping", "new_unique_grouping",
            "interpolation_and_mask_fill")):
        print(f"{label}={np.median([row[index] for row in component_rows]):.9f}")


def stats(values: list[float]) -> tuple[float, float, float]:
    return min(values), float(np.median(values)), max(values)


def print_comparison(
    label: str, measured: float, reported: float
) -> None:
    difference = measured - reported
    relative_error = difference / reported
    marker = (
        " LARGE_DIVERGENCE"
        if abs(relative_error) > LARGE_DIVERGENCE_REL
        else ""
    )
    print(
        f"{label}: reconstructed_median={measured:.9f} "
        f"reported_median={reported:.9f} difference={difference:+.9f} "
        f"relative_error={relative_error:+.3%}{marker}"
    )


def main() -> None:
    print("residual_B breakdown profile")
    print(
        f"condition: water60_free N={N_HISTORIES} "
        f"resolution_cm={RESOLUTION_CM} batch_size={N_HISTORIES} "
        f"n_chunks={N_CHUNKS} seed={SEED}"
    )
    print(
        "timing boundary: direct reconstruction of "
        "kernel.run_batch_with_tally(use_njit_dda=True); "
        "make_water_case is outside the timer"
    )
    print(
        "step 4 overflow_check is measured separately and included in "
        "residual_B_reconstructed"
    )
    print(
        f"warmup: one full path with n_histories={WARMUP_HISTORIES}; "
        "output discarded"
    )
    profile_once(WARMUP_HISTORIES)

    rows = []
    fixed_inputs = None
    for repetition in range(1, N_REPETITIONS + 1):
        row, inputs = profile_once(N_HISTORIES)
        if fixed_inputs is None:
            fixed_inputs = inputs
        rows.append(row)
        print(
            f"rep={repetition} "
            + " ".join(f"{name}={value:.9f}" for name, value in row.items())
        )

    display_order = (
        "chunk_plan",
        "buffer_allocation",
        "step_1+2",
        "transport",
        "overflow_check",
        "concatenate",
        "weight_calculation",
        "dda_accumulator",
        "residual_B_reconstructed",
        "non_DDA_including_transport",
        "total_E2E_B_equivalent",
    )
    summary = {}
    print("\nsummary (seconds, 8 repetitions)")
    print(f"{'component':30s} {'min':>12s} {'median':>12s} {'max':>12s}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12}")
    for name in display_order:
        values = [row[name] for row in rows]
        lo, median, hi = stats(values)
        summary[name] = (lo, median, hi)
        print(f"{name:30s} {lo:12.9f} {median:12.9f} {hi:12.9f}")

    print("\ncomparison with previously reported medians")
    print_comparison(
        "E2E_B",
        summary["total_E2E_B_equivalent"][1],
        REPORTED_E2E_B,
    )
    print_comparison(
        "residual_B",
        summary["residual_B_reconstructed"][1],
        REPORTED_RESIDUAL_B,
    )
    print(
        "definition note: requested residual_B_reconstructed excludes transport, "
        "whereas the previously reported residual_B=E2E_B-DDA_in_B includes it"
    )
    print_comparison(
        "reported-compatible non_DDA_including_transport",
        summary["non_DDA_including_transport"][1],
        REPORTED_RESIDUAL_B,
    )
    print_comparison(
        "DDA_in_B",
        summary["dda_accumulator"][1],
        REPORTED_DDA_IN_B,
    )
    if any(
        abs(measured[1] - reported) / reported > LARGE_DIVERGENCE_REL
        for measured, reported in (
            (summary["total_E2E_B_equivalent"], REPORTED_E2E_B),
            (summary["residual_B_reconstructed"], REPORTED_RESIDUAL_B),
            (summary["dda_accumulator"], REPORTED_DDA_IN_B),
        )
    ):
        print(
            "warning: at least one reconstructed median differs from the "
            "reported value by more than 10%; the measurement boundary uses "
            "direct step calls without the prior timing monkey-patch"
        )
    else:
        print("divergence assessment: all median relative errors are within 10%")

    candidates = (
        "transport",
        "buffer_allocation",
        "concatenate",
        "weight_calculation",
    )
    bottleneck = max(candidates, key=lambda name: summary[name][1])
    print(f"次のボトルネック候補: {bottleneck}")
    benchmark_weight_calculation(*fixed_inputs)


if __name__ == "__main__":
    main()
