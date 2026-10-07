"""Recalculate every decision from isolated ChatCarlo JSON and fresh EGS5 logs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xraylib

HERE = Path(__file__).resolve().parent
GY_PER_KEV_G = 1.602176634e-13
BONFERRONI_Z = 3.27307836404
NUMBER = r"[+\-0-9.EeDd]+"


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cc(mode):
    data = json.loads((HERE / "chatcarlo" / f"{mode}.json").read_text())
    assert data["manifest"]["preregistration_sha256"] == sha(HERE / "PREREGISTRATION.md")
    assert data["manifest"]["water_density_g_cm3"] == 1.
    return data["results"]


def egs_text(mode, n):
    folder = HERE / "egs5" / mode
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["history_count"] == n
    assert meta["run_exit_code"] == meta["build_exit_code"] == 0
    assert meta["physics"]["rho"] == 1.
    assert meta["preregistration_sha256_before"] == meta["preregistration_sha256_after"] == sha(HERE / "PREREGISTRATION.md")
    assert meta["output_sha256"] == sha(folder / "egs5job.out")
    out = (folder / "egs5job.out").read_text()
    assert re.search(rf"ncase=\s*{n}\b", out)
    assert "OPTION NOT REQUESTED" not in out
    return out


def bsf_energy(text):
    mean = number(re.search(rf"Mean energy deposited per history.*?=\s*({NUMBER})", text).group(1))
    sem = number(re.search(rf"Standard error of the mean.*?=\s*({NUMBER})", text).group(1))
    return mean, sem


def criterion(relative_percent, z, eligible, limit):
    if not eligible:
        return "insufficient_statistics"
    return "pass" if abs(relative_percent) < 2. and z < limit else "fail"


def main():
    bsf = cc("bsf")
    thin, thin_sem = bsf_energy(egs_text("bsf_thinslab", 8_000_000))
    phantom, phantom_sem = bsf_energy(egs_text("bsf_phantom", 8_000_000))
    ratio = phantom/thin
    ratio_sem = ratio*math.hypot(phantom_sem/phantom, thin_sem/thin)
    diff = 100*(ratio-bsf["BSF_w"])/bsf["BSF_w"]
    z = abs(ratio-bsf["BSF_w"])/math.hypot(ratio_sem, bsf["sem_BSF_w"])
    combined_rel = math.hypot(ratio_sem/ratio, bsf["sem_BSF_w"]/bsf["BSF_w"])
    bsf_compare = {"chatcarlo": bsf["BSF_w"], "chatcarlo_sem": bsf["sem_BSF_w"],
                   "egs5": ratio, "egs5_sem": ratio_sem, "relative_difference_pct": diff,
                   "z": z, "combined_relative_sem": combined_rel,
                   "status": criterion(diff, z, combined_rel < .01, 2.),
                   "egs5_thin_mean_MeV": thin, "egs5_thin_sem_MeV": thin_sem,
                   "egs5_phantom_mean_MeV": phantom, "egs5_phantom_sem_MeV": phantom_sem}
    pdd = cc("pdd")
    pat = re.compile(rf"^\s*(\S+)\s+mean\(MeV\)=\s*({NUMBER})\s+sem\(MeV\)=\s*({NUMBER})", re.M)
    egs_bins = {name: (number(mean), number(sem)) for name, mean, sem in
                pat.findall(egs_text("pdd_phantom", 100_000_000))}
    assert len(egs_bins) == len(pdd["bins"]) == 47
    assert set(egs_bins) == set(pdd["bins"])
    rows = []
    for name, bin_cc in pdd["bins"].items():
        volume = math.prod(b-a for a, b in zip(bin_cc["lo_cm"], bin_cc["hi_cm"]))
        mean_mev, sem_mev = egs_bins[name]
        factor = 1000.*GY_PER_KEV_G/(volume*1.)
        egs, egs_sem = mean_mev*factor, sem_mev*factor
        mean_cc, sem_cc = bin_cc["mean_Gy_per_history"], bin_cc["sem_Gy_per_history"]
        relpct = 100.*(egs-mean_cc)/mean_cc
        z = abs(egs-mean_cc)/math.hypot(egs_sem, sem_cc)
        eligible = egs_sem/egs < .01 and sem_cc/mean_cc < .01
        rows.append({"bin": name, "chatcarlo_Gy_history": mean_cc,
                     "chatcarlo_SEM_Gy_history": sem_cc, "chatcarlo_rel_SEM_pct": 100*sem_cc/mean_cc,
                     "egs5_Gy_history": egs, "egs5_SEM_Gy_history": egs_sem,
                     "egs5_rel_SEM_pct": 100*egs_sem/egs, "relative_difference_pct": relpct,
                     "z": z, "historical_2sigma": criterion(relpct, z, eligible, 2.),
                     "bonferroni_auxiliary": criterion(relpct, z, eligible, BONFERRONI_Z)})
    counts = {method: {status: sum(r[method] == status for r in rows) for status in
                      ["pass", "fail", "insufficient_statistics"]} for method in
                      ["historical_2sigma", "bonferroni_auxiliary"]}
    pdd_summary = {"counts": counts, "average_relative_difference_pct": float(np.mean(
        [r["relative_difference_pct"] for r in rows])),
        "pdd_average_relative_difference_pct": float(np.mean([r["relative_difference_pct"] for r in rows[:15]])),
        "ocr_average_relative_difference_pct": float(np.mean([r["relative_difference_pct"] for r in rows[15:]])),
        "maximum_absolute_relative_difference_pct": max(abs(r["relative_difference_pct"]) for r in rows),
        "maximum_z": max(r["z"] for r in rows), "bonferroni_z": BONFERRONI_Z}
    with (HERE / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Independent attenuation: call xraylib directly, not ChatCarlo's interpolated
    # transport table. Read the published NIST mu_en value at the exact 60keV knot.
    nist = np.loadtxt(HERE.parents[2] / "chatcarlo/data/nist_xaamdi/water.csv", delimiter=",")
    muen60 = float(nist[nist[:, 0] == 60., 2].item())
    direct_mu = xraylib.CS_Total_CP("H2O", 60.)
    transmission = math.exp(-direct_mu*20.)
    counted = pdd["diagnostics"]["n_primary_transmitted"]
    n = pdd["diagnostics"]["n_histories"]
    observed = counted/n
    transmission_sem = math.sqrt(observed*(1-observed)/n)
    independent = {"mu_water_direct_xraylib_per_cm": direct_mu,
        "mu_en_water_NIST_60keV_cm2_g": muen60,
        "primary_transmission_20cm_water_only": transmission,
        "chatcarlo_primary_transmission_observed": observed,
        "chatcarlo_primary_transmission_sem": transmission_sem,
        "primary_transmission_z_vs_water_only": abs(observed-transmission)/transmission_sem,
        "primary_surface_pdd_Gy_per_history": 60.*muen60*(1-math.exp(-direct_mu))/direct_mu/100.*GY_PER_KEV_G,
        "scope": "Primary-only analytic anchor; not an exact scattered-dose solution"}
    summary = {"preregistration_sha256": sha(HERE / "PREREGISTRATION.md"),
               "comparison_script_sha256": sha(Path(__file__)),
               "source_sha256": {str(path.relative_to(HERE)): sha(path) for path in
                    [HERE / "chatcarlo/bsf.json", HERE / "chatcarlo/pdd.json",
                     HERE / "egs5/bsf_thinslab/egs5job.out", HERE / "egs5/bsf_phantom/egs5job.out",
                     HERE / "egs5/pdd_phantom/egs5job.out"]},
               "BSF_w": bsf_compare, "PDD_OCR": pdd_summary,
               "independent_analytic_references": independent}
    (HERE / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    plot(rows, pdd)
    print(json.dumps(summary, indent=2))


def plot(rows, pdd):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]},
                             constrained_layout=True)
    for column, (title, selection) in enumerate([
            ("Central PDD", rows[:15]), ("Lateral: z=0-1 cm", rows[15:31]),
            ("Lateral: z=9-10 cm", rows[31:])]):
        x = []
        for row in selection:
            item = pdd["bins"][row["bin"]]
            axis = 2 if column == 0 else 0
            x.append((item["lo_cm"][axis]+item["hi_cm"][axis])/2)
        axes[0, column].errorbar(x, [r["chatcarlo_Gy_history"]*1e15 for r in selection],
            yerr=[r["chatcarlo_SEM_Gy_history"]*1e15 for r in selection], fmt="o-", label="ChatCarlo")
        axes[0, column].errorbar(x, [r["egs5_Gy_history"]*1e15 for r in selection],
            yerr=[r["egs5_SEM_Gy_history"]*1e15 for r in selection], fmt="s--", label="EGS5")
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("Kerma [1e-15 Gy/history]")
        axes[0, column].legend()
        axes[1, column].plot(x, [r["relative_difference_pct"] for r in selection], "o-")
        axes[1, column].axhline(0, color="grey", linewidth=.8)
        axes[1, column].axhline(2, color="red", linestyle=":")
        axes[1, column].axhline(-2, color="red", linestyle=":")
        axes[1, column].set_ylabel("(EGS5-CC)/CC [%]")
        axes[1, column].set_xlabel("Depth [cm]" if column == 0 else "x [cm]")
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    fig.suptitle("60 keV water / 2026-10-07 revalidation / error bars: 1 SEM")
    fig.savefig(HERE / "pdd_ocr_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
