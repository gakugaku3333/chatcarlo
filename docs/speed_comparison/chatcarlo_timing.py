"""docs/plan_speed_comparison_egs5.md 指標A/B計測スクリプト（ChatCarlo側）。

EGS5相互検証済みの3診断領域スラブシナリオ（water20kev/water60_free/water150kev、
いずれも鉛筆ビーム+均質水スラブ1枚 — ジオメトリーエンジンの速度差が混入しにくい
最小構成）について、transport_photons本体のみの壁時計時間を計測する
（scene.yaml/CLIは経由しない — tallyなし、断面積テーブル構築等の一回限りの
オーバーヘッドは初回呼び出しで吸収させ、計測対象からは除外）。

インターリーブA/B測定の規律（docs/speedup_baseline/tally_speedup_timing.pyと同じ）:
このスクリプト自体はChatCarlo単体の計測用。EGS5側との突き合わせは、両方のプロセスを
交互に手動実行し中央値を取る運用とする（Phase 2で実施）。

使い方:
    PYTHONPATH=. .venv/bin/python docs/speed_comparison/chatcarlo_timing.py \\
        --scenario water60_free --n 1e5,1e6 --repeats 3

出力: シナリオ・N・反復ごとに wall秒・histories/sec・一次透過率（±統計誤差）。
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from chatcarlo.geometry import Geometry
from chatcarlo.materials import linear_mu
from chatcarlo.transport import transport_photons

# egs5_crosscheck/water*/run_chatcarlo_water*.py と同一の幾何・線質パラメータ。
# bbox_margin_cmを極小にする理由はそちらのコメント参照（EGS5のtutor5パターン
# ＝スラブ外は真空、に合わせるため）。
SCENARIOS = {
    "water20kev": {"material": "water", "thickness_cm": 1.5, "energy_kev": 20.0},
    "water60_free": {"material": "water", "thickness_cm": 10.0, "energy_kev": 60.0},
    "water150kev": {"material": "water", "thickness_cm": 10.0, "energy_kev": 150.0},
}


def run_once(scenario: str, n: int, seed: int) -> dict:
    p = SCENARIOS[scenario]
    material, thickness_cm, energy_kev = p["material"], p["thickness_cm"], p["energy_kev"]

    geom = Geometry([{
        "name": "slab", "shape": "box", "material": material,
        "center": [0.0, 0.0, 0.0],
        "size_cm": [thickness_cm, 100.0, 100.0],
    }], bbox_margin_cm=0.01)
    rng = np.random.default_rng(seed)
    pos = np.tile(np.array([-thickness_cm / 2 - 0.01, 0.0, 0.0]), (n, 1))
    dirv = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    energy = np.full(n, energy_kev)

    t0 = time.perf_counter()
    result = transport_photons(pos, dirv, energy, geom, rng)
    dt = time.perf_counter() - t0

    uncollided = np.sum(result.escaped & (result.n_scatter == 0)) / n
    mu = float(linear_mu(material, np.array([energy_kev]))[0])
    expected = math.exp(-mu * thickness_cm)
    stderr = math.sqrt(uncollided * (1 - uncollided) / n)

    return {
        "scenario": scenario, "n": n, "seed": seed, "wall_s": dt,
        "histories_per_s": n / dt, "uncollided_frac": uncollided,
        "stderr": stderr, "beer_lambert": expected,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    p.add_argument("--n", default="1e5,1e6", help="comma-separated history counts")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed0", type=int, default=1)
    args = p.parse_args()

    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    n_values = [int(float(x)) for x in args.n.split(",")]

    for scenario in scenarios:
        for n in n_values:
            rows = [run_once(scenario, n, args.seed0 + rep) for rep in range(args.repeats)]
            walls = sorted(r["wall_s"] for r in rows)
            median_wall = walls[len(walls) // 2]
            median_rate = n / median_wall
            print(f"[{scenario}] n={n:.0f}  wall(median of {args.repeats})={median_wall:.4f}s"
                  f"  rate={median_rate:.0f} histories/s  all_wall={[f'{w:.4f}' for w in walls]}")
            r0 = rows[0]
            print(f"    一次透過率(seed={r0['seed']})={r0['uncollided_frac']*100:.4f}%"
                  f"  (±{r0['stderr']*100:.4f}pp)  Beer-Lambert={r0['beer_lambert']*100:.4f}%")


if __name__ == "__main__":
    main()
