"""Current production transport, historical exact-box estimator, isolated outputs.

Run from the repository root with PYTHONPATH=. .venv/bin/python <this file>.
The old scripts are imported for their beam/bin/overlap definitions only;
their main functions and hard-coded historical output paths are never called.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from chatcarlo.geometry import Geometry
from chatcarlo.materials import density, linear_mu, mu_en_rho
from chatcarlo.tally import relative_error, standard_error
from chatcarlo.trajectory import TrajectoryRecorder
from chatcarlo.transport import transport_photons

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "historical_pdd_definitions", HERE.parent / "run_chatcarlo_pdd60.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(depth):
    return Geometry([{"name": "water", "shape": "box", "material": "water",
                      "center": [0., 0., depth / 2], "size_cm": [30., 30., depth]}],
                    bbox_margin_cm=0.01)


def score(depth, bins, n_total, batch_size, seed):
    rng = np.random.default_rng(seed)
    geom = geometry(depth)
    sums = np.zeros(len(bins))
    sums2 = np.zeros(len(bins))
    low_energy_sums = np.zeros(len(bins))
    diagnostics = {"n_histories": n_total, "seed": seed, "batch_size": batch_size,
                   "depth_cm": depth, "n_absorbed": 0, "n_escaped": 0,
                   "n_primary_transmitted": 0, "n_air_interactions": 0,
                   "n_water_segments_below_10keV": 0,
                   "n_segments_below_1keV": 0, "deposited_keV": 0.,
                   "escaped_keV": 0., "air_track_length_cm": 0.,
                   "minimum_segment_energy_keV": 60.}
    started = time.perf_counter()
    for done in range(0, n_total, batch_size):
        n = min(batch_size, n_total - done)
        pos, direction, energy = old._sample_parallel_beam(n, rng)
        recorder = TrajectoryRecorder()
        result = transport_photons(pos, direction, energy, geom, rng,
                                   recorder=recorder, fluorescence_enabled=True)
        starts = np.concatenate(recorder.starts)
        ends = np.concatenate(recorder.ends)
        energies = np.concatenate(recorder.energies)
        ids = np.concatenate(recorder.photon_ids)
        materials = np.concatenate(recorder.materials)
        events = np.concatenate(recorder.events)
        water = materials == "water"
        # Score only water segments. Do not apply water coefficients to an
        # exterior-air segment that merely touches a scoring face.
        mu_en_linear = mu_en_rho("water", energies) * density("water")
        for i, (_, lo, hi) in enumerate(bins):
            length = old._segment_box_overlap_cm(starts, ends, lo, hi)
            hit = (length > 0) & water
            local = np.zeros(n)
            contribution = length[hit] * energies[hit] * mu_en_linear[hit]
            np.add.at(local, ids[hit], contribution)
            sums[i] += local.sum()
            sums2[i] += np.dot(local, local)
            low_energy_sums[i] += contribution[energies[hit] < 10.].sum()
        diagnostics["n_absorbed"] += int(result.absorbed.sum())
        diagnostics["n_escaped"] += int(result.escaped.sum())
        diagnostics["n_primary_transmitted"] += int(
            (result.escaped & (result.n_scatter == 0)).sum())
        diagnostics["n_air_interactions"] += int(((materials == "air") &
            np.isin(events, ["photoelectric", "compton", "rayleigh", "fluorescence"])).sum())
        diagnostics["n_water_segments_below_10keV"] += int((water & (energies < 10.)).sum())
        diagnostics["n_segments_below_1keV"] += int((energies < 1.).sum())
        diagnostics["deposited_keV"] += sum(result.energy_deposited.values())
        diagnostics["escaped_keV"] += float(result.final_energy[result.escaped].sum())
        diagnostics["air_track_length_cm"] += float(
            np.linalg.norm(ends - starts, axis=1)[materials == "air"].sum())
        diagnostics["minimum_segment_energy_keV"] = min(
            diagnostics["minimum_segment_energy_keV"], float(energies.min()))
        print(f"depth={depth:g} seed={seed}: {done+n}/{n_total}", flush=True)
    diagnostics["wall_seconds"] = time.perf_counter() - started
    diagnostics["energy_balance_relative_residual"] = (
        diagnostics["deposited_keV"] + diagnostics["escaped_keV"] - 60*n_total) / (60*n_total)
    assert diagnostics["n_absorbed"] + diagnostics["n_escaped"] == n_total
    assert abs(diagnostics["energy_balance_relative_residual"]) < 1e-10
    rows = {}
    for i, (name, lo, hi) in enumerate(bins):
        mass = float(np.prod(hi-lo)) * density("water")
        factor = old.KEV_PER_G_TO_GY / mass
        mean = float(sums[i] / n_total)
        sem = float(standard_error(sums[i], sums2[i], n_total, n_total))
        rows[name] = {"lo_cm": lo.tolist(), "hi_cm": hi.tolist(), "mass_g": mass,
                      "sum_keV": float(sums[i]), "sum2_keV2": float(sums2[i]),
                      "mean_Gy_per_history": mean*factor,
                      "sem_Gy_per_history": sem*factor,
                      "rel_err": float(relative_error(sums[i], sums2[i], n_total, n_total)),
                      "below_10keV_score_fraction": float(low_energy_sums[i]/sums[i]) if sums[i] else 0.}
    return {"diagnostics": diagnostics, "bins": rows}


def stable(result):
    # Timing is the only field not expected to reproduce bit-for-bit.
    result = json.loads(json.dumps(result))
    result["diagnostics"].pop("wall_seconds")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bsf", "pdd", "repro"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg_hash = sha(HERE / "PREREGISTRATION.md")
    bins = old._build_bins()
    surface = [("surface", np.array([-5., -5., 0.]), np.array([5., 5., .2]))]
    manifest = {"code_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": prereg_hash, "runner_sha256": sha(__file__),
        "historical_definitions_sha256": sha(HERE.parent / "run_chatcarlo_pdd60.py"),
        "mode": args.mode, "water_density_g_cm3": density("water"),
        "bbox_margin_cm": .01, "source_z_cm": -.0001,
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "xraylib"]},
        "production_source_sha256": {name: sha(ROOT / "chatcarlo" / name) for name in
            ["transport.py", "physics.py", "materials.py", "geometry.py", "tally.py", "trajectory.py"]}}
    assert density("water") == 1.
    if args.mode == "bsf":
        thin = score(.2, surface, 500_000, 25_000, 1)
        phantom = score(20., surface, 500_000, 25_000, 2)
        numerator = phantom["bins"]["surface"]
        denominator = thin["bins"]["surface"]
        ratio = numerator["mean_Gy_per_history"] / denominator["mean_Gy_per_history"]
        relsem = np.hypot(numerator["rel_err"], denominator["rel_err"])
        result = {"thin": thin, "phantom": phantom, "BSF_w": ratio,
                  "sem_BSF_w": float(ratio*relsem), "rel_sem_BSF_w": float(relsem)}
    elif args.mode == "pdd":
        result = score(20., bins, 5_000_000, 100_000, 1)
    else:
        result = {}
        for name, depth, selection, seed in [("thin", .2, surface, 1),
                                            ("surface", 20., surface, 2),
                                            ("pdd", 20., bins, 1)]:
            a = score(depth, selection, 10_000, 10_000, seed)
            b = score(depth, selection, 10_000, 10_000, seed)
            assert stable(a) == stable(b), name
            result[name] = {"bit_identical": True, "first": a, "second": b}
    mu = float(np.asarray(linear_mu("water", 60.)).item())
    mu_air = float(np.asarray(linear_mu("air", 60.)).item())
    # Central 2x2 column samples 4/100 of histories, then divide by 4g.
    primary_surface_gy = 60.*float(np.asarray(mu_en_rho("water", 60.)).item())*(1-np.exp(-mu))/mu/100.*old.KEV_PER_G_TO_GY
    result["analytic_references"] = {"mu_water_per_cm": mu,
        "mu_air_per_cm": mu_air, "primary_transmission_20cm_water_only": float(np.exp(-mu*20.)),
        "primary_transmission_20cm_with_source_air_gap": float(np.exp(-mu*20.-mu_air*.0001)),
        "source_air_gap_attenuation_fraction": float(-np.expm1(-mu_air*.0001)),
        "primary_surface_pdd_Gy_per_history_water_only": float(primary_surface_gy)}
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    assert sha(HERE / "PREREGISTRATION.md") == prereg_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"manifest": manifest, "results": result},
                                      ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
