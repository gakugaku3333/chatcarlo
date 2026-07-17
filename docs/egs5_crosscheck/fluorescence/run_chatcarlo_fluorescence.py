"""K殻蛍光X線EGS5相互検証: ChatCarlo側の実行スクリプト。

幾何・線質・許容基準はPREREGISTRATION.md参照。100 keV単色鉛筆ビーム->
鉛スラブ(厚さ0.05cm)、蛍光ON/OFFの2条件、n=1,000,000、seed=1。
transport_photonsを直接叩く（source.pyのmAs校正等は不要な単純ケースのため）。
"""
from __future__ import annotations

import json

import numpy as np

from chatcarlo.geometry import Geometry
from chatcarlo.materials import linear_mu
from chatcarlo.transport import transport_photons

ENERGY_KEV = 100.0
THICKNESS_CM = 0.05
N = 1_000_000
SEED = 1
PEAK_BAND = (72.0, 86.0)


def _slab_arrays(n, seed):
    geometry = Geometry([{
        "name": "slab", "shape": "box", "material": "lead",
        "center": [0.0, 0.0, 0.0],
        "size_cm": [THICKNESS_CM, 100.0, 100.0],
    }])
    rng = np.random.default_rng(seed)
    pos = np.tile(np.array([-THICKNESS_CM / 2 - 5.0, 0.0, 0.0]), (n, 1))
    dirv = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    energy = np.full(n, ENERGY_KEV)
    return pos, dirv, energy, geometry, rng


def run(fluorescence_enabled: bool) -> dict:
    pos, dirv, energy, geometry, rng = _slab_arrays(N, SEED)
    result = transport_photons(pos, dirv, energy, geometry, rng,
                                fluorescence_enabled=fluorescence_enabled)
    e_in_total = N * ENERGY_KEV
    e_escaped = float(np.sum(result.final_energy[result.escaped]))
    e_deposited = sum(result.energy_deposited.values())

    escaped_e = result.final_energy[result.escaped]
    in_peak = np.mean((escaped_e >= PEAK_BAND[0]) & (escaped_e <= PEAK_BAND[1]))
    n_escaped = int(np.sum(result.escaped))
    n_uncollided = int(np.sum(result.escaped & (result.n_scatter == 0)))

    transmitted_energy_fraction = e_escaped / e_in_total
    # 二項近似の統計誤差（光子ごとのエネルギーは一様でないため厳密ではないが、
    # オーダー評価として妥当。厳密な誤差伝播はA項の主要変動源=脱出するか否かの
    # 二値なので、平均脱出エネルギーで重みづけた近似式を使う）
    p_escape = n_escaped / N
    stderr_p_escape = np.sqrt(p_escape * (1 - p_escape) / N)
    mean_escaped_e = e_escaped / n_escaped if n_escaped > 0 else 0.0
    stderr_transmitted_energy_fraction = stderr_p_escape * mean_escaped_e / ENERGY_KEV

    return {
        "fluorescence_enabled": fluorescence_enabled,
        "n_histories": N,
        "seed": SEED,
        "energy_keV": ENERGY_KEV,
        "thickness_cm": THICKNESS_CM,
        "linear_mu_per_cm": float(linear_mu("lead", ENERGY_KEV)[0]),
        "n_fluorescence": int(result.n_fluorescence),
        "n_escaped": n_escaped,
        "uncollided_transmission": n_uncollided / N,
        "uncollided_transmission_stderr": float(np.sqrt(
            (n_uncollided / N) * (1 - n_uncollided / N) / N)),
        "transmitted_energy_fraction": transmitted_energy_fraction,
        "transmitted_energy_fraction_stderr": float(stderr_transmitted_energy_fraction),
        "peak_band_fraction_of_escaped": float(in_peak),
    }


if __name__ == "__main__":
    results = {}
    for fluor in (False, True):
        label = "on" if fluor else "off"
        r = run(fluor)
        results[label] = r
        print(f"--- fluorescence={label} ---")
        for k, v in r.items():
            print(f"  {k}: {v}")

    delta = (results["on"]["transmitted_energy_fraction"]
              - results["off"]["transmitted_energy_fraction"])
    results["delta_transmitted_energy_fraction_on_minus_off"] = delta
    print(f"\nΔ(透過エネルギー割合, ON-OFF) = {delta:.6f}")

    with open("docs/egs5_crosscheck/fluorescence/chatcarlo_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n書き出し: docs/egs5_crosscheck/fluorescence/chatcarlo_results.json")
