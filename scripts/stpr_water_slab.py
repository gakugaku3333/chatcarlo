#!/usr/bin/env python3
"""Phase 0 STPR study: ideal detector behind a water slab.

The 0 cm condition deliberately uses an equal-size air slab.  It is therefore
not exactly zero: air itself weakly scatters, so the registered criterion is
STPR(0 cm) < 0.01 rather than strict zero.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatcarlo.detector import CAT_MULTIPLE, CAT_SINGLE, DetectorPlane
from chatcarlo.scene import validate_scene
from chatcarlo.transport import run_transport

THICKNESSES_CM = (0, 5, 10, 15, 20, 25, 30)
KVPS = (60, 80, 100, 120)
SLAB_WIDTH_CM = 40.0
DETECTOR_SIZE_CM = 43.0
DETECTOR_SHAPE = (256, 256)
AIR_GAP_CM = 5.0
SID_CM = 180.0
ROI = ((98, 158), (98, 158))
N_HISTORIES = 2_000_000
BATCH_SIZE = 100_000
SEED = 42


def make_scene(thickness, kvp):
    # Entrance face is y=-t/2.  SOD = SID - gap - thickness by registration.
    sod = SID_CM - AIR_GAP_CM - thickness
    source_y = -thickness / 2 - sod
    return validate_scene({
        "source": {"kvp": float(kvp), "filtration_mm_al": 2.5,
                   "position": [0, source_y, 0], "direction": [0, 1, 0],
                   "field": {"shape": "cone", "diameter_cm": DETECTOR_SIZE_CM * math.sqrt(2), "sid_cm": SID_CM}},
        "geometry": [{"name": "slab", "shape": "box", "material": "air" if thickness == 0 else "water",
                      "center": [0, 0, 0], "size_cm": [SLAB_WIDTH_CM, max(float(thickness), 1e-9), SLAB_WIDTH_CM]}],
    })


def detector_for(thickness):
    return DetectorPlane(np.array([0., thickness / 2 + AIR_GAP_CM, 0.]), np.array([0., -1., 0.]),
                         np.array([1., 0., 0.]), (DETECTOR_SIZE_CM, DETECTOR_SIZE_CM), DETECTOR_SHAPE)


def _two_sigma_status(higher, lower):
    """事前登録E-2(a)(b): 差が結合SEMの2σ未満なら結論を保留する。"""
    delta = higher["stpr"] - lower["stpr"]
    combined_sem = math.hypot(higher["stpr_sem"], lower["stpr_sem"])
    threshold = 2.0 * combined_sem
    if abs(delta) < threshold:
        status = "判定保留"
    elif delta > 0:
        status = "確認"
    else:
        status = "不合格"
    return delta, combined_sem, status


def e2_judgments(rows):
    """E-2(a)(b)の判定を、後処理で再現できる行形式にする。"""
    by_condition = {(r["kvp"], r["thickness_cm"]): r for r in rows}
    judgments = []
    for kvp in KVPS:
        for low_t, high_t in zip(THICKNESSES_CM, THICKNESSES_CM[1:]):
            low, high = by_condition[kvp, low_t], by_condition[kvp, high_t]
            delta, sem, status = _two_sigma_status(high, low)
            judgments.append({"criterion": "E-2a_thickness_monotonicity", "thickness_cm": f"{low_t}->{high_t}",
                              "kvp_pair": str(kvp), "delta_stpr": delta, "combined_sem": sem, "status": status})
    for thickness in (t for t in THICKNESSES_CM if t >= 20):
        for low_kvp, high_kvp in ((100, 120), (80, 100)):
            low, high = by_condition[low_kvp, thickness], by_condition[high_kvp, thickness]
            delta, sem, status = _two_sigma_status(high, low)
            judgments.append({"criterion": "E-2b_kvp_order", "thickness_cm": thickness,
                              "kvp_pair": f"{low_kvp}->{high_kvp}", "delta_stpr": delta,
                              "combined_sem": sem, "status": status})
    return judgments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roi_cm = (ROI[0][1] - ROI[0][0]) * DETECTOR_SIZE_CM / DETECTOR_SHAPE[0]
    print(f"thicknesses={THICKNESSES_CM} cm; kVp={KVPS}; detector={DETECTOR_SIZE_CM} cm/{DETECTOR_SHAPE}; air_gap={AIR_GAP_CM} cm")
    print(f"ROI={ROI} = {roi_cm:.3f} cm square; n_histories={N_HISTORIES}; batch_size={BATCH_SIZE}; seed={SEED}")
    rows = []
    for kvp in KVPS:
        for thickness in THICKNESSES_CM:
            result = run_transport(make_scene(thickness, kvp), n_histories=N_HISTORIES, seed=SEED,
                                   batch_size=BATCH_SIZE, detector=detector_for(thickness), detector_roi=ROI)
            tally = result.detector
            rows.append({"kvp": kvp, "thickness_cm": thickness, "stpr": tally.stpr(), "stpr_sem": tally.stpr_sem(),
                         "single_fluence": tally.category_fluence[CAT_SINGLE].sum(),
                         "multiple_fluence": tally.category_fluence[CAT_MULTIPLE].sum()})
            print(f"{kvp:3d} kV {thickness:2d} cm: STPR={rows[-1]['stpr']:.6g} ± {rows[-1]['stpr_sem']:.3g}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    judgments = e2_judgments(rows)
    judgment_out = out.with_name(f"{out.stem}_judgments.csv")
    with judgment_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=judgments[0].keys()); writer.writeheader(); writer.writerows(judgments)
    for row in judgments:
        print(f"{row['criterion']} {row['thickness_cm']} {row['kvp_pair']}: {row['status']} "
              f"(Δ={row['delta_stpr']:.6g}, combined SEM={row['combined_sem']:.3g})")
    print(f"E-2 judgments: {judgment_out}")


if __name__ == "__main__":
    main()
