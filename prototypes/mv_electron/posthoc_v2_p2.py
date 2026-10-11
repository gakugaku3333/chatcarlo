"""Read-only P2 posthoc analysis under the approved 2026-10-11 criteria.

Run with ``.venv/bin/python -m prototypes.mv_electron.posthoc_v2_p2``.
``--check`` additionally checks the saved artifacts against the in-memory result.
No transport/reference modules are imported and no P2 calculations are rerun.
"""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "docs/validation/mv/p2"
OUT = P2 / "posthoc_v2"
REVIEW = ROOT / "docs/ai/plans/reviews/2026-10-11-p2b-review-1.md"
ENERGIES = (1.25, 2.0, 6.0, 10.0)
BACKENDS = ("dcslib", "mf", "elsepa")
TOLERANCE = 0.01
SEM_LIMIT = TOLERANCE / 3
SEED = 20261011
REPLICATES = 1000
EXPECTED_BLOCKS = {
    "1.25": [[13, 37], [38, 62], [63, 87], [88, 112], [113, 149]],
    "2.0": [[15, 39], [40, 64], [65, 89], [90, 114], [115, 149]],
    "6.0": [[29, 53], [54, 78], [79, 103], [104, 128], [129, 149]],
    "10.0": [[38, 62], [63, 87], [88, 112], [113, 149]],
}
WITHIN = "1%以内"
EXCEEDS = "1%を超える"
INSUFFICIENT = "統計不足"
ALL_WITHIN = "ブロック平均が1%以内"
ANY_EXCEEDS = "1%を超えるブロックあり"
STEP_STATUS = {
    ALL_WITHIN: "半分の刻みとの差が許容内",
    INSUFFICIENT: INSUFFICIENT,
    ANY_EXCEEDS: "許容を超える刻み依存を検出",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(path):
    return path.relative_to(ROOT).as_posix()


def load(path):
    """Only the two dose arrays and their fixed-run metadata are needed."""
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k].copy() for k in ("coarse", "fine", "E", "N", "seed")}
    for key, shape in (("coarse", (100, 150)), ("fine", (100, 600))):
        require(result[key].shape == shape, f"Unexpected {key} shape: {portable(path)}")
        require(np.all(np.isfinite(result[key])), f"Nonfinite {key}: {portable(path)}")
    require(int(result["N"]) > 0 and int(result["N"]) % 100 == 0,
            f"Unequal batch histories: {portable(path)}")
    return result


def blocks_from_deep(deep):
    require(deep == list(range(deep[0], deep[-1] + 1)), "Deep bins are not contiguous")
    chunks = [deep[i:i + 25] for i in range(0, len(deep), 25)]
    if len(chunks) > 1 and len(chunks[-1]) < 13:
        chunks[-2].extend(chunks.pop())
    return [[chunk[0], chunk[-1]] for chunk in chunks]


def mean_sem(batches):
    return batches.mean(axis=0), batches.std(axis=0, ddof=1) / np.sqrt(len(batches))


def difference(left, right, dref):
    lm, ls = mean_sem(left)
    rm, rs = mean_sem(right)
    return (lm - rm) / dref, np.hypot(ls, rs) / dref


def block_status(delta, sem):
    if sem > SEM_LIMIT:
        return INSUFFICIENT
    return WITHIN if abs(delta) <= TOLERANCE else EXCEEDS


def aggregate(states):
    if EXCEEDS in states:
        return ANY_EXCEEDS
    if INSUFFICIENT in states:
        return INSUFFICIENT
    require(bool(states) and all(s == WITHIN for s in states), "Invalid block states")
    return ALL_WITHIN


def first_crossing(curve, threshold, spacing=0.05):
    """First threshold reached on the rising side, between fine-bin centres.

    No extrapolation to the surface is made if the first bin is already above
    threshold. Exact equality at a centre counts, including a plateau.
    """
    hits = np.flatnonzero((curve[:-1] < threshold) & (curve[1:] >= threshold))
    equal = np.flatnonzero(curve == threshold)
    positions = []
    if len(hits):
        i = int(hits[0])
        positions.append((i + 0.5 + (threshold - curve[i]) /
                          (curve[i + 1] - curve[i])) * spacing)
    if len(equal):
        positions.append((int(equal[0]) + 0.5) * spacing)
    return float(min(positions)) if positions else None


def crossings_for_mean(batches, dref):
    curve = batches.mean(axis=0)
    return [first_crossing(curve, level * dref) for level in (0.5, 0.8)]


def bootstrap_curve(fine, dref, seed_sequence):
    rng = np.random.default_rng(seed_sequence)
    samples = []
    for _ in range(REPLICATES):
        indices = rng.integers(0, 100, 100)
        samples.append(crossings_for_mean(fine[indices], dref))
    return {"mean_curve_depths_cm": crossings_for_mean(fine, dref),
            "replicate_depths_cm": samples}


def crossing_difference(left, right, level_index):
    lpoint = left["mean_curve_depths_cm"][level_index]
    rpoint = right["mean_curve_depths_cm"][level_index]
    samples = [l[level_index] - r[level_index]
               for l, r in zip(left["replicate_depths_cm"], right["replicate_depths_cm"])
               if l[level_index] is not None and r[level_index] is not None]
    total = len(left["replicate_depths_cm"])
    require(total == len(right["replicate_depths_cm"]), "Bootstrap count mismatch")
    return {"delta_cm": None if lpoint is None or rpoint is None else lpoint - rpoint,
            "SEM_cm": float(np.std(samples, ddof=1)) if len(samples) >= 2 else None,
            "valid_replicates": len(samples), "invalid_replicates": total - len(samples)}


def regional_summary(left, right, bins, dref):
    delta, sem = difference(left[:, bins].mean(axis=1), right[:, bins].mean(axis=1), dref)
    per_bin, _ = difference(left[:, bins], right[:, bins], dref)
    index = int(np.argmax(abs(per_bin)))
    return {"delta_pct_Dref": float(delta * 100), "SEM_pct_Dref": float(sem * 100),
            "max_abs_bin_pct_Dref": float(abs(per_bin[index]) * 100),
            "max_abs_bin_index": bins[index]}


def deep_diagnostics(left, right, deep, blocks, dref):
    delta, sem = difference(left, right, dref)
    variance = sem ** 2
    i = deep[int(np.argmax(abs(delta[deep])))]
    lm, _ = mean_sem(left)
    rm, _ = mean_sem(right)
    require(lm[deep].mean() != 0 and rm[deep].mean() != 0, "Zero deep shape denominator")
    shape = lm / lm[deep].mean() - rm / rm[deep].mean()
    return {
        "bin_indices": deep,
        "d_i_over_Dref": delta[deep].tolist(),
        "v_i_over_Dref_squared": variance[deep].tolist(),
        "systematic_mean_square_point_estimate": float(np.mean(delta[deep] ** 2) -
                                                       np.mean(variance[deep])),
        "systematic_mean_square_unit": "ratio to Dref^2 (dimensionless)",
        "systematic_mean_square_note": "負値を保持。上限は算出しない。Kawrakow & Fippelの方法そのものではない。",
        "max_abs_bin_pct_Dref": float(abs(delta[i]) * 100),
        "max_abs_bin_index": i,
        "max_abs_bin_signed_z": float(delta[i] / sem[i]) if sem[i] > 0 else None,
        "max_abs_bin_z_note": "合成分散が0なら算出不能（null）",
        "shape_block_means": [float(shape[a:b + 1].mean()) for a, b in blocks],
        "shape_unit": "dimensionless difference of deep-mean-normalized curves; not % Dref",
        "shape_note": "一律の倍率の成分を除いた差。SEMは算出しない。",
    }


def comparison_specs(lock):
    specs = []
    for j, energy in enumerate(ENERGIES):
        ref = f"egs5/main_ref_{j}_0/result.npz"
        for backend in BACKENDS:
            specs.append({"id": f"a_{backend}_{j}", "kind": "a", "energy_MeV": energy,
                          "backend": backend, "parameter": None,
                          "left": f"runs/main_{backend}_{j}/result.npz", "right": ref})
        specs.append({"id": f"b_{j}", "kind": "b", "energy_MeV": energy,
                      "backend": None, "parameter": None,
                      "left": f"egs5/main_ref_{j}_4/result.npz", "right": ref})
        if j in (1, 3):
            for backend in BACKENDS:
                for parameter in ("fe", "k"):
                    specs.append({"id": f"c_{backend}_{j}_{parameter}", "kind": "c",
                                  "energy_MeV": energy, "backend": backend, "parameter": parameter,
                                  "left": f"runs/main_{backend}_{j}_{parameter}/result.npz",
                                  "right": f"runs/main_{backend}_{j}/result.npz"})
    require(set(s["left"].split("/")[1] for s in specs if s["kind"] != "b") == set(lock["runs"]),
            "Prototype run list differs from the fixed lock")
    return specs


def compare(spec, curves, window, blocks, bootstraps):
    left, right = curves[spec["left"]], curves[spec["right"]]
    dref = window["Dref"]
    rows = []
    for a, b in blocks:
        delta, sem = difference(left["coarse"][:, a:b + 1].mean(axis=1),
                                right["coarse"][:, a:b + 1].mean(axis=1), dref)
        rows.append({"bins_inclusive": [a, b], "depth_edges_cm": [a * 0.2, (b + 1) * 0.2],
                     "delta_pct_Dref": float(delta * 100), "SEM_pct_Dref": float(sem * 100),
                     "reference_abs_delta_plus_1_96_SEM_pct_Dref": float((abs(delta) + 1.96 * sem) * 100),
                     "precision_target_met": bool(sem <= SEM_LIMIT),
                     "status": block_status(delta, sem)})
    result = dict(spec)
    result.update(Dref_Gy_cm2=dref, blocks=rows, status=aggregate([r["status"] for r in rows]),
                  precision_target_met=all(r["precision_target_met"] for r in rows),
                  regions={k: regional_summary(left[array], right[array], window[bins], dref)
                           for k, array, bins in (("buildup", "fine", "build"),
                                                  ("window", "coarse", "near"),
                                                  ("deep", "coarse", "deep"))},
                  deep_diagnostics=deep_diagnostics(left["coarse"], right["coarse"],
                                                    window["deep"], blocks, dref),
                  threshold_depth_differences={str(level): crossing_difference(
                      bootstraps[spec["left"]], bootstraps[spec["right"]], i)
                      for i, level in enumerate((0.5, 0.8))})
    return result


def annotate(comparisons):
    sensitivity = {c["energy_MeV"]: c for c in comparisons if c["kind"] == "b"}
    steps = [c for c in comparisons if c["kind"] == "c"]
    for c in comparisons:
        c["notes"] = []
        c["verdict_held"] = False
        c["reported_status"] = STEP_STATUS[c["status"]] if c["kind"] == "c" else c["status"]
        if c["kind"] == "a":
            s = sensitivity[c["energy_MeV"]]["status"]
            if s != ALL_WITHIN:
                c["notes"].append("設定間の感度が許容を超える" if s == ANY_EXCEEDS
                                  else "設定間の感度を確認できない")
                c["verdict_held"] = True
                c["reported_status"] = "判定保留（" + c["status"] + "）"
            backend_steps = [v for v in steps if v["backend"] == c["backend"]]
            exceeded = [v for v in backend_steps if v["status"] == ANY_EXCEEDS]
            c["step_comparison_ids"] = [v["id"] for v in backend_steps]
            c["numerically_stable_result"] = False if exceeded else None
            if exceeded:
                c["notes"].append("許容を超える刻み依存を検出: " +
                                  ", ".join(v["id"] for v in exceeded) +
                                  "。このbackendの(a)は数値的に安定した結果として扱わない。")
        elif c["kind"] == "b":
            c["notes"].append("モデルの設定への感度。基準の不確かさを定量したものではない。")
        else:
            c["notes"].append("1回の半減への安定性。収束の証明ではない。")


def synthetic_checks():
    """Deterministic fixtures; no random draws outside the A-9 bootstrap."""
    passed = []

    def check(name, condition):
        require(bool(condition), "Synthetic check failed: " + name)
        passed.append(name)

    zero = np.zeros((100, 4))
    for value in (0.004, -0.006):
        delta, sem = difference(zero + value, zero, 1.0)
        check(f"known_signed_difference_{value}", np.allclose(delta, value, atol=1e-17, rtol=0)
              and np.all(sem < 1e-17))
    correlated = np.repeat(np.arange(100, dtype=float)[:, None], 4, axis=1)
    _, actual = difference(correlated.mean(axis=1), zero.mean(axis=1), 1.0)
    independent = np.sqrt(np.sum(mean_sem(correlated)[1] ** 2)) / 4
    check("fully_correlated_batch_SEM", np.isclose(actual, mean_sem(correlated)[1][0])
          and np.isclose(actual, 2 * independent))
    check("SEM_exact_boundary", block_status(0, SEM_LIMIT) == WITHIN)
    check("positive_delta_exact_boundary", block_status(TOLERANCE, 0) == WITHIN)
    check("negative_delta_exact_boundary", block_status(-TOLERANCE, SEM_LIMIT) == WITHIN)
    check("SEM_just_above", block_status(0, np.nextafter(SEM_LIMIT, np.inf)) == INSUFFICIENT)
    check("delta_just_above", block_status(np.nextafter(TOLERANCE, np.inf), 0) == EXCEEDS)
    check("statistics_before_difference", block_status(0.02, 0.004) == INSUFFICIENT)
    for states, expected in (([WITHIN, WITHIN], ALL_WITHIN),
                             ([WITHIN, INSUFFICIENT], INSUFFICIENT),
                             ([INSUFFICIENT, EXCEEDS, WITHIN], ANY_EXCEEDS)):
        check("aggregate_" + expected, aggregate(states) == expected)
    for state, expected in ((ALL_WITHIN, "半分の刻みとの差が許容内"),
                            (INSUFFICIENT, "統計不足"),
                            (ANY_EXCEEDS, "許容を超える刻み依存を検出")):
        rows = [{"kind": "c", "status": state}]
        annotate(rows)
        check("step_" + expected, rows[0]["reported_status"] == expected)
    for state, note in ((ANY_EXCEEDS, "設定間の感度が許容を超える"),
                        (INSUFFICIENT, "設定間の感度を確認できない"), (ALL_WITHIN, None)):
        rows = [{"kind": "a", "status": ALL_WITHIN, "energy_MeV": 2.0, "backend": "mf"},
                {"kind": "b", "status": state, "energy_MeV": 2.0}]
        annotate(rows)
        check("sensitivity_annotation_" + state,
              rows[0]["verdict_held"] == (note is not None) and
              (note in rows[0]["notes"] if note else not rows[0]["notes"]))
    rows = [{"kind": "a", "status": ALL_WITHIN, "energy_MeV": 6.0, "backend": "mf"},
            {"kind": "b", "status": ALL_WITHIN, "energy_MeV": 6.0},
            {"kind": "c", "status": ANY_EXCEEDS, "backend": "mf", "id": "c_mf_1_k"}]
    annotate(rows)
    check("step_exceedance_backend_annotation", rows[0]["numerically_stable_result"] is False
          and bool(rows[0]["notes"]))
    noisy = correlated * 0.0001 + 1
    diagnostic = deep_diagnostics(noisy, noisy, list(range(4)), [[0, 3]], 1.0)
    check("negative_systematic_mean_square", diagnostic["systematic_mean_square_point_estimate"] < 0)
    check("zero_shape_for_identical_curves", diagnostic["shape_block_means"] == [0.0])
    scaled = deep_diagnostics(noisy * 2, noisy, list(range(4)), [[0, 3]], 1.0)
    check("shape_removes_uniform_scale", abs(scaled["shape_block_means"][0]) < 1e-15)
    for energy, expected in EXPECTED_BLOCKS.items():
        check("fixed_blocks_" + energy,
              blocks_from_deep(list(range(expected[0][0], 150))) == expected)
    x = (np.arange(10) + 0.5) * 0.05
    check("linear_50_crossing", np.isclose(first_crossing(2 * x, 0.5), 0.25))
    check("linear_80_crossing", np.isclose(first_crossing(2 * x, 0.8), 0.4))
    check("multiple_crossings_first", np.isclose(first_crossing(np.array([0., 1., 0., 1.]), 0.5), 0.05))
    check("exact_centre_and_plateau", first_crossing(np.array([0., 0.5, 0.5, 1.]), 0.5) == 0.07500000000000001)
    check("no_crossing", first_crossing(np.zeros(10), 0.5) is None)
    check("no_surface_extrapolation", first_crossing(np.ones(10), 0.5) is None)
    repeated = np.repeat((2 * x)[None, :], 100, axis=0)
    check("sampled_batch_mean_crossing", np.allclose(crossings_for_mean(repeated[np.arange(100)[::-1]], 1), [0.25, 0.4]))
    left = {"mean_curve_depths_cm": [1., None], "replicate_depths_cm": [[1., None], [3., None], [None, None]]}
    right = {"mean_curve_depths_cm": [0., 2.], "replicate_depths_cm": [[0., 2.], [0., 2.], [0., 2.]]}
    diff = crossing_difference(left, right, 0)
    check("bootstrap_difference_SEM_ddof1", diff["valid_replicates"] == 2
          and diff["invalid_replicates"] == 1 and np.isclose(diff["SEM_cm"], np.sqrt(2)))
    check("bootstrap_no_valid_null", crossing_difference(left, right, 1) ==
          {"delta_cm": None, "SEM_cm": None, "valid_replicates": 0, "invalid_replicates": 3})
    left["replicate_depths_cm"][1][0] = None
    diff = crossing_difference(left, right, 0)
    check("bootstrap_one_valid_null", diff["SEM_cm"] is None
          and diff["valid_replicates"] == 1 and diff["invalid_replicates"] == 2)
    return {"count": len(passed), "passed": passed}


def reconcile(comparisons):
    """Compare only the identically defined quantities, before any output."""
    table = {}
    for line in REVIEW.read_text().splitlines():
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) == 5 and parts[0] in ("1.25", "2", "6", "10"):
            table[float(parts[0])] = dict(zip((*BACKENDS, "sensitivity"), map(float, parts[1:])))
    require(len(table) == 4, "Independent review table could not be read")
    trial = json.loads((OUT / "trial_before_criteria.json").read_text())
    trial_rows = {(r["E"], r["pair"]): r for r in trial["rows"]}
    review_diffs, trial_diffs = [], []
    for c in comparisons:
        if c["kind"] not in ("a", "b"):
            continue
        name = c["backend"] if c["kind"] == "a" else "sensitivity"
        value = max(b["reference_abs_delta_plus_1_96_SEM_pct_Dref"] for b in c["blocks"])
        error = abs(value - table[c["energy_MeV"]][name])
        review_diffs.append({"id": c["id"], "absolute_difference_percentage_points": error})
        require(error <= 0.002, f"Review reconciliation outside tolerance: {c['id']}")
        pair = (c["backend"] if c["kind"] == "a" else "EGS5 sensitivity") + " - EGS5 main"
        r = trial_rows[c["energy_MeV"], pair]
        for region, key in (("buildup", "build"), ("window", "peak"), ("deep", "deep")):
            for quantity, trial_key in (("delta_pct_Dref", "mean"), ("SEM_pct_Dref", "SEM")):
                error = abs(c["regions"][region][quantity] - r[key + "_" + trial_key + "_pct"])
                trial_diffs.append({"id": c["id"], "region": region, "quantity": quantity,
                                    "absolute_difference_percentage_points": error})
                require(error <= 1e-9, f"Trial reconciliation outside tolerance: {c['id']} {region} {quantity}")
    require(len(review_diffs) == 16 and len(trial_diffs) == 96, "Reconciliation count mismatch")
    return {"review": {"comparisons": 16, "tolerance_percentage_points": 0.002,
                       "max_difference_percentage_points": max(r["absolute_difference_percentage_points"] for r in review_diffs),
                       "differences": review_diffs},
            "trial": {"quantities": 96, "tolerance_percentage_points": 1e-9,
                      "max_difference_percentage_points": max(r["absolute_difference_percentage_points"] for r in trial_diffs),
                      "differences": trial_diffs}}


def analyze():
    lock_path = P2 / "pilot_lock.json"
    lock = json.loads(lock_path.read_text())
    order = [f"egs5/main_ref_{j}_{mode}/result.npz" for j in range(4) for mode in (0, 4)]
    order += [f"runs/{name}/result.npz" for name in lock["runs"]]
    require(len(order) == len(set(order)) == 32, "Expected exactly 32 dose inputs")
    source_paths = [lock_path] + [P2 / name for name in order]
    hashes = {portable(p): sha(p) for p in source_paths}
    trial_path = OUT / "trial_before_criteria.json"
    reconciliation_hashes = {portable(p): sha(p) for p in (trial_path, REVIEW)}
    blocks = {e: blocks_from_deep(w["deep"]) for e, w in lock["windows"].items()}
    require(blocks == EXPECTED_BLOCKS, "Fixed blocks disagree with A-2; stop")
    curves = {name: load(P2 / name) for name in order}
    for name, curve in curves.items():
        config = (lock["reference_runs"] if name.startswith("egs5/") else lock["runs"])[name.split("/")[1]]
        for key in ("E", "N", "seed"):
            require(curve[key].item() == config[key], f"Run metadata mismatch: {name} {key}")
    seeds = np.random.SeedSequence(SEED).spawn(32)
    bootstraps = {name: bootstrap_curve(curves[name]["fine"],
                                       lock["windows"][str(float(curves[name]["E"]))]["Dref"], seed)
                  for name, seed in zip(order, seeds)}
    specs = comparison_specs(lock)
    comparisons = [compare(s, curves, lock["windows"][str(s["energy_MeV"])],
                           blocks[str(s["energy_MeV"])], bootstraps) for s in specs]
    require(Counter(c["kind"] for c in comparisons) == {"a": 12, "b": 4, "c": 12},
            "Expected 12/4/12 comparisons")
    annotate(comparisons)
    checks = {"synthetic": synthetic_checks(), "reconciliation": reconcile(comparisons)}
    require(hashes == {portable(p): sha(p) for p in source_paths}, "Input sha256 changed during analysis")
    require(reconciliation_hashes == {portable(p): sha(p) for p in (trial_path, REVIEW)},
            "Reconciliation sources changed during analysis")
    checks["input_hashes_unchanged"] = True
    result = {
        "analysis": "探索的な事後解析。P2の正式な判定は計画0.3のまま。",
        "criteria": {"tolerance_over_Dref": TOLERANCE, "SEM_limit_over_Dref": SEM_LIMIT,
                     "rule": "SEM <= 1%/3, then abs(delta) <= 1%; 1.96 SEM is reference only",
                     "normalization": "fixed pilot Dref; its estimation uncertainty is not propagated",
                     "multiple_comparisons": "超える側は多重比較の補正をしていない。",
                     "scope": "期待線量の深部ブロック平均の差。ブロック内の打ち消し合いは検出できない。"},
        "input_sha256": hashes,
        "reconciliation_source_sha256": reconciliation_hashes,
        "regions": {e: {"Dref_Gy_cm2": w["Dref"], "buildup_fine_bins": w["build"],
                        "window_coarse_bins": w["near"], "deep_coarse_bins": w["deep"]}
                    for e, w in lock["windows"].items()},
        "blocks_coarse_bins_inclusive": blocks,
        "comparisons": comparisons,
        "bootstrap": {"seed": SEED, "replicates_per_curve": REPLICATES,
                      "batch_resample": "rng.integers(0,100,100); mean of resampled batches",
                      "thresholds_over_fixed_Dref": [0.5, 0.8], "fine_bin_spacing_cm": 0.05,
                      "order": order, "curves": bootstraps,
                      "SEM_definition": "sample SD (ddof=1) of paired independent-curve depth differences; <2 valid -> null"},
        "checks": checks,
        "status_counts": {kind: dict(Counter(c["reported_status"] for c in comparisons if c["kind"] == kind))
                          for kind in ("a", "b", "c")},
    }
    return result, curves


def fmt(value, signed=False):
    return "算出不能" if value is None else format(value, "+.6f" if signed else ".6f")


def report(result):
    lines = [
        "# P2 保存結果の基準v2による事後解析", "",
        "これは探索的な事後解析である。P2の事前登録の正式な判定は計画0.3のまま変更しない。"
        "Q0〜Q4は旧判定を維持、Q5は記録、Q6は判定保留、Q7は統計不足である。"
        "Q6の内訳はdcslib・elsepaの全エネルギーが未評価、mfの1.25・6・10 MeVが判定保留、2 MeVが未評価。", "",
        "基準v2は文献調査と保存結果の探索的解析の両方を踏まえた再設計である。"
        "P2の計画を書いたClaudeはコード間比較の先行報告を調べずに基準を決めており、計画時のリサーチ不足を経緯として記録する。"
        "計画0.4の試算と骨子レビューの値を見たあとに基準を決めた。深部は通りそうでビルドアップは境界に近いと分かったあとの選択であり、"
        "結果から独立した変更とは主張しない。将来の新規計算では計算前に固定する。"
        "TG-105の表I等は比較結果の要約であり、5 cmブロック・深部だけ・1%という一式の判定法を規定していない。"
        "旧基準の全項目が誤りだったとは主張しない。", "",
        "判定する量は固定した深部ブロックの平均線量の差だけであり、ビルドアップを含むP2の元の問いより狭い。"
        "ここでいう系統差は期待線量のブロック平均のコード間差である。"
        "ブロック内部の打ち消し合い（2 cmが+2%、残りが−1.33%など）は検出できない。"
        "差は固定したDrefに対する割合で、倍率で合わせず、Drefの推定誤差は伝播しない。", "",
        "この方式の性質: SEMがちょうど1%/3のとき、「1%以内」と出る確率は、"
        "真の差が0.5%で約93%、0.8%で約73%、1.0%で約50%、1.3%で約18%。"
        "真の差が1%をわずかに超えていても、半分近い確率で「1%以内」と出る。"
        "SEM上限の1%/3は許容の3分の1という経験則であり、文献の規定ではない。"
        "|Δ|+1.96 SEMは参考値で、判定には使わない。超える側の多重比較補正は行っていない。", "",
        "表の線量差・SEM・参考値・bin最大差はすべて % Dref。±はSEM（95%区間の半幅ではない）。"
        "形の差は深部平均で規格化した曲線どうしの無次元差で、% Drefとは別の尺度である。", "",
    ]
    for kind, title in (("a", "(a) 試作 − EGS5主（S＋）"),
                        ("b", "(b) EGS5感度 − EGS5主"),
                        ("c", "(c) 半分の刻み − 基本の計算")):
        lines += [f"## {title}", ""]
        if kind == "b":
            lines += ["モデルの設定への感度であり、基準の不確かさを定量したものではない。", ""]
        if kind == "c":
            lines += ["1回の半減への安定性であり、収束の証明ではない。feは損失刻み、kは多重散乱刻み。", ""]
        lines += ["| MeV | backend | パラメータ | 比較の状態 | 精度目標 | 最大参考値 | 注記 |",
                  "|---|---|---|---|---|---:|---|"]
        for c in result["comparisons"]:
            if c["kind"] != kind:
                continue
            reference = max(b["reference_abs_delta_plus_1_96_SEM_pct_Dref"] for b in c["blocks"])
            lines.append(f"| {c['energy_MeV']:g} | {c['backend'] or 'EGS5'} | {c['parameter'] or '—'} | "
                         f"{c['reported_status']} | {'達成' if c['precision_target_met'] else '未達'} | "
                         f"{fmt(reference)} | {' '.join(c['notes']) or '—'} |")
        lines += ["", "| 比較ID | 2 mm bin（両端含む） | Δ | SEM | 精度目標 | ブロック状態 | 絶対差+1.96 SEM（参考） | 形の差（無次元、% Drefとは別） |",
                  "|---|---|---:|---:|---|---|---:|---:|"]
        for c in result["comparisons"]:
            if c["kind"] == kind:
                for b, shape in zip(c["blocks"], c["deep_diagnostics"]["shape_block_means"]):
                    a, end = b["bins_inclusive"]
                    lines.append(f"| {c['id']} | {a}–{end} | {fmt(b['delta_pct_Dref'], True)} | "
                                 f"{fmt(b['SEM_pct_Dref'])} | {'達成' if b['precision_target_met'] else '未達'} | {b['status']} | "
                                 f"{fmt(b['reference_abs_delta_plus_1_96_SEM_pct_Dref'])} | {shape:+.8g} |")
        lines.append("")
    lines += ["## ビルドアップ・窓（記述のみ、判定なし）", "",
              "| 比較ID | 領域 | 単純平均差 ± SEM (% Dref) | bin最大絶対差 (% Dref) |",
              "|---|---|---:|---:|"]
    for c in result["comparisons"]:
        for key, label in (("buildup", "ビルドアップ"), ("window", "窓")):
            r = c["regions"][key]
            lines.append(f"| {c['id']} | {label} | {fmt(r['delta_pct_Dref'], True)} ± {fmt(r['SEM_pct_Dref'])} | {fmt(r['max_abs_bin_pct_Dref'])} |")
    lines += ["", "固定Drefの50%・80%に表面側から最初に達する深さの差。"
              "0.5 mm bin中心間で線形補間し、外挿はしない。SEMは曲線ごとに独立な1000回のバッチブートストラップの差の標本標準偏差。", "",
              "| 比較ID | 固定Drefの割合 | 深さ差 ± SEM (cm) | 有効回数 | 無効回数 |",
              "|---|---|---:|---:|---:|"]
    for c in result["comparisons"]:
        for level, r in c["threshold_depth_differences"].items():
            lines.append(f"| {c['id']} | {float(level) * 100:g}% | {fmt(r['delta_cm'], True)} ± {fmt(r['SEM_cm'])} | "
                         f"{r['valid_replicates']} | {r['invalid_replicates']} |")
    lines += ["", "## 深部の補助量（判定に使わない）", "",
              "二乗平均の点推定は mean(d_i²) − mean(v_i)。負値を保持し、上限は出さない。"
              "Kawrakow & Fippelの方法そのものではない。単位はDref²に対する比。zは最大絶対差のbinの符号つきd_i/√v_i。", "",
              "| 比較ID | 二乗平均点推定 (Dref²比) | bin最大絶対差 (% Dref) | bin | z | 深部全体平均差 ± SEM (% Dref) |",
              "|---|---:|---:|---:|---:|---:|"]
    for c in result["comparisons"]:
        d, r = c["deep_diagnostics"], c["regions"]["deep"]
        lines.append(f"| {c['id']} | {d['systematic_mean_square_point_estimate']:+.9g} | "
                     f"{fmt(d['max_abs_bin_pct_Dref'])} | {d['max_abs_bin_index']} | {fmt(d['max_abs_bin_signed_z'], True)} | "
                     f"{fmt(r['delta_pct_Dref'], True)} ± {fmt(r['SEM_pct_Dref'])} |")
    checks = result["checks"]
    lines += ["", "## 数値照合", "",
              f"合成データ検査: {checks['synthetic']['count']}項目。独立レビュー16比較の最大差: "
              f"{checks['reconciliation']['review']['max_difference_percentage_points']:.12g}ポイント（上限0.002）。"
              f"既存試算96量の最大差: {checks['reconciliation']['trial']['max_difference_percentage_points']:.12g} "
              "% Dref（上限1×10⁻⁹）。入力32ファイルとpilot_lock.jsonの実行前後のsha256は一致。", "",
              "入力・照合資料のsha256、全ブートストラップ交点、binごとの補助量はposthoc_v2.jsonに記録。", ""]
    return "\n".join(lines)


def plots(result, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    images = {}
    for energy in ENERGIES:
        comparisons = [c for c in result["comparisons"] if c["energy_MeV"] == energy]
        fig, axes = plt.subplots((len(comparisons) + 1) // 2, 2,
                                 figsize=(13, 3.2 * ((len(comparisons) + 1) // 2)),
                                 sharex=True, squeeze=False)
        w = result["regions"][str(energy)]
        for ax, c in zip(axes.flat, comparisons):
            left, right = curves[c["left"]], curves[c["right"]]
            dref = c["Dref_Gy_cm2"]
            delta, _ = difference(left["coarse"], right["coarse"], dref)
            ax.plot((np.arange(150) + 0.5) * 0.2, delta * 100, color="C0", alpha=0.3, lw=0.8)
            bins = w["buildup_fine_bins"]
            fine_delta, _ = difference(left["fine"], right["fine"], dref)
            ax.plot((np.asarray(bins) + 0.5) * 0.05, fine_delta[bins] * 100,
                    color="C1", alpha=0.5, lw=0.8)
            ax.axvspan(bins[0] * 0.05, (bins[-1] + 1) * 0.05, color="C1", alpha=0.1)
            near = w["window_coarse_bins"]
            ax.axvspan(near[0] * 0.2, (near[-1] + 1) * 0.2, color="C2", alpha=0.1)
            for b in c["blocks"]:
                lo, hi = b["depth_edges_cm"]
                ax.errorbar((lo + hi) / 2, b["delta_pct_Dref"],
                            xerr=(hi - lo) / 2, yerr=1.96 * b["SEM_pct_Dref"],
                            color="C0", lw=1.6, capsize=3)
            for value in (-1, 1):
                ax.axhline(value, color="black", ls="--", lw=0.8)
            ax.axhline(0, color="grey", lw=0.5)
            ax.set(title=c["id"], ylabel="Difference (% fixed Dref)", xlim=(0, 30))
            ax.set_xlabel("Depth (cm)")
            ax.grid(alpha=0.15)
        for ax in list(axes.flat)[len(comparisons):]:
            ax.set_visible(False)
        legend = [Line2D([], [], color="C0", alpha=0.3, label="Per-bin difference"),
                  Line2D([], [], color="C0", marker="o", label="Block mean +/- 1.96 SEM (reference)"),
                  Line2D([], [], color="black", ls="--", label="+/- 1% fixed Dref"),
                  Patch(color="C1", alpha=0.15, label="Buildup: outside verdict"),
                  Patch(color="C2", alpha=0.15, label="Window: outside verdict")]
        fig.legend(handles=legend, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.975))
        fig.suptitle(f"P2 exploratory posthoc: {energy:g} MeV", y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        data = BytesIO()
        fig.savefig(data, format="png", dpi=140)
        plt.close(fig)
        images[f"posthoc_blocks_{energy:g}MeV.png"] = data.getvalue()
    return images


def json_bytes(result):
    return (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        result, curves = analyze()
        # Render from the exact JSON representation, rather than separate values.
        encoded = json_bytes(result)
        document = report(json.loads(encoded))
        require("合格" not in document and "検証された" not in document, "Prohibited standalone verdict language")
        require(str(ROOT) not in encoded.decode() + document, "Absolute user path in artifacts")
        if args.check:
            require((OUT / "posthoc_v2.json").read_bytes() == encoded,
                    "Saved JSON is not byte-identical to the repeated analysis")
            require((OUT / "RESULTS_POSTHOC.md").read_text() == document, "Saved report differs from JSON rendering")
            for energy in ENERGIES:
                require((OUT / f"posthoc_blocks_{energy:g}MeV.png").is_file(), "Missing plot")
        else:
            images = plots(result, curves)
            # All analysis, fixtures, reconciliations and rendering finish before writing.
            require(all(sha(ROOT / p) == h for p, h in result["input_sha256"].items()),
                    "Inputs changed before artifact write")
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "posthoc_v2.json").write_bytes(encoded)
            (OUT / "RESULTS_POSTHOC.md").write_text(document)
            for name, data in images.items():
                (OUT / name).write_bytes(data)
        print(json.dumps({"synthetic_checks": result["checks"]["synthetic"]["count"],
                          "review_max_difference_percentage_points": result["checks"]["reconciliation"]["review"]["max_difference_percentage_points"],
                          "trial_max_difference_percentage_points": result["checks"]["reconciliation"]["trial"]["max_difference_percentage_points"],
                          "input_hashes_unchanged": True, "status_counts": result["status_counts"],
                          "saved_artifacts_checked": args.check}, ensure_ascii=False))
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        # Do not create or update RESULTS_POSTHOC.md after a failed reconciliation.
        print("停止: " + str(exc).replace(str(ROOT), "."))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
