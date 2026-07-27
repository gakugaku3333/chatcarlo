"""Phase B-1bの検証戦略・層1: `chatcarlo.kernel`（Numba per-historyカーネル）と
`chatcarlo.transport.transport_photons`（ベクトル化参照実装）の統計的クロスチェック。

両実装はビット一致しない（乱数アルゴリズムがカーネル=レガシーMT19937・
参照実装=PCG64で根本的に異なる、docs/plan_chatcarlo_speedup_post_egs5.md
「Phase Bの検証戦略」参照）ため、統計的な一致で検証する。

**事前登録（結果を見る前にここに固定する。lessons_learnedの教訓
「計測後に基準を決めるな」に従う）**:

- 主指標: 一次透過率（無散乱透過率）。両実装は独立なMC推定量なので、
  結合標準誤差 sqrt(sem_kernel^2 + sem_ref^2) に対して**差が4σ以内**を合格
  基準とする（EGS5相互検証で使った「2σ」より緩いのは、ここではEGS5という
  外部コードとの比較ではなく同じxraylibデータを共有する2つの自己実装間の
  比較であり、有意水準を3シナリオ×複数指標に配分する多重比較の余裕を
  持たせるため——単一指標なら2σでも十分通る想定)。
- 副指標（参考、合否のゲートにはしない）: fraction_absorbed・fraction_escaped・
  mean_scatter_events・蛍光放出率。大きく外れていないか目視確認する。
- N=2,000,000（両実装とも）。シナリオはEGS5相互検証で確立済みの3本
  （water20kev/water60_free/water150kev）、実運用と同じ条件
  （background="air"、fluorescence_enabled=True、bbox_margin_cm=0.01）。

**層3（EGS5再相互検証）についての重要な注記——命名の罠**: シナリオ名
"water60_free"は`docs/speed_comparison/chatcarlo_timing.py`から引き継いだ
名前だが、この名前がEGS5クロスチェック側で指す`water60_free/`ディレクトリは
**IBOUND=0（自由電子Klein-Nishinaコンプトン）**で実行されたものであり、
ChatCarlo/カーネルの束縛コンプトン（EPDL、xraylib.CS_Compt）とは物理モデルが
異なる（[docs/egs5_crosscheck/RESULTS.md](../egs5_crosscheck/RESULTS.md)参照:
IBOUND=0では相対4.7%・8.6σの既知の乖離がある——実装バグではなく物理モデル差）。
物理モデルを揃えた検証済み比較は`water60_bound/`（IBOUND=1, 12.70%）であり、
層3では**この値を使う**（"water60_free"という名前に反してEGS5側は
`water60_bound`の数値を参照する——紛らわしいが、既存ファイル名を変更する
理由が薄いためこの注記で対応する）。water20kev/water150kevはEGS5側も
最初からIBOUND=1で実行済みなのでそのまま使える。
"""
from __future__ import annotations

import math

import numpy as np

from chatcarlo.geometry import Geometry
from chatcarlo.kernel import bake_box_scene, bake_scene_materials, run_batch
from chatcarlo.transport import transport_photons

SCENARIOS = {
    "water20kev": {"thickness_cm": 1.5, "energy_kev": 20.0},
    "water60_free": {"thickness_cm": 10.0, "energy_kev": 60.0},
    "water150kev": {"thickness_cm": 10.0, "energy_kev": 150.0},
}
N = 2_000_000
SIGMA_GATE = 4.0


def run_reference(thickness_cm: float, energy_kev: float, n: int, seed: int) -> dict:
    geom = Geometry([{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0.0, 0.0, 0.0], "size_cm": [thickness_cm, 100.0, 100.0],
    }], bbox_margin_cm=0.01)
    rng = np.random.default_rng(seed)
    pos = np.tile(np.array([-thickness_cm / 2 - 0.01, 0.0, 0.0]), (n, 1))
    dirv = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    energy = np.full(n, energy_kev)
    result = transport_photons(pos, dirv, energy, geom, rng, fluorescence_enabled=True)
    uncollided = np.sum(result.escaped & (result.n_scatter == 0)) / n
    return {
        "uncollided_frac": uncollided,
        "fraction_absorbed": np.sum(result.absorbed) / n,
        "fraction_escaped": np.sum(result.escaped) / n,
        "mean_scatter": np.mean(result.n_scatter),
        "n_fluorescence": result.n_fluorescence,
    }


def run_kernel(thickness_cm: float, energy_kev: float, n: int, seed: int) -> dict:
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (thickness_cm, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    origin = (-thickness_cm / 2 - 0.01, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0)
    r = run_batch(tables, geom, energy_kev, origin, direction, n, seed=seed, n_chunks=8,
                  fluorescence_enabled=True)
    uncollided = np.sum(r.escaped & (r.n_scatter == 0)) / n
    return {
        "uncollided_frac": uncollided,
        "fraction_absorbed": np.sum(r.absorbed) / n,
        "fraction_escaped": np.sum(r.escaped) / n,
        "mean_scatter": np.mean(r.n_scatter),
        "n_fluorescence": int(np.sum(r.n_fluorescence)),
    }


# 層3: 既存のEGS5相互検証で確立済みの一次透過率（n=500,000、いずれもIBOUND=1
# 束縛コンプトンでChatCarloと物理モデルを揃えた条件、docs/egs5_crosscheck/*/RESULTS.md
# に事前登録・監査済み）。EGS5を再実行するのではなく、この既に確立された数値に対して
# カーネルを再判定する（計画書「Phase Bの検証戦略」層3、水スラブ+蛍光は無散乱透過率に
# 影響しないため既存の合格基準をそのまま転用できる）。**注意**: water60_freeの欄は
# EGS5側`water60_bound/`（IBOUND=1）の12.70%を使う——`water60_free/`自体は
# IBOUND=0（自由電子コンプトン）で物理モデルが異なり、直接比較すると既知の理由
# （docs/egs5_crosscheck/RESULTS.md）で約4.7%・8.6σ乖離する、これはカーネルの
# バグではない。既存の合格基準（相互差2%以内かつ2σ以内）をそのまま適用する。
EGS5_ESTABLISHED = {
    "water20kev": (0.2976, 500_000),
    "water60_free": (0.1270, 500_000),  # = water60_bound (IBOUND=1) の値
    "water150kev": (0.2226, 500_000),
}
EGS5_REL_GATE = 0.02
EGS5_SIGMA_GATE = 2.0


def main() -> None:
    print(f"事前登録(層1): 主指標=一次透過率, 合格基準=結合{SIGMA_GATE}σ以内, N={N:,}")
    print(f"事前登録(層3): 既存EGS5確立値との相互差{EGS5_REL_GATE*100:.0f}%以内かつ"
          f"{EGS5_SIGMA_GATE}σ以内（docs/egs5_crosscheck/*/RESULTS.mdの既存基準を転用）\n")
    all_pass = True
    for name, p in SCENARIOS.items():
        ref = run_reference(p["thickness_cm"], p["energy_kev"], N, seed=101)
        ker = run_kernel(p["thickness_cm"], p["energy_kev"], N, seed=101)

        p_ref = ref["uncollided_frac"]
        p_ker = ker["uncollided_frac"]
        sem_ref = math.sqrt(p_ref * (1 - p_ref) / N)
        sem_ker = math.sqrt(p_ker * (1 - p_ker) / N)
        combined = math.sqrt(sem_ref ** 2 + sem_ker ** 2)
        diff_sigma = (p_ker - p_ref) / combined
        verdict = "PASS" if abs(diff_sigma) <= SIGMA_GATE else "FAIL"
        all_pass &= (verdict == "PASS")

        print(f"[{name}] 層1(主指標=一次透過率): kernel={p_ker*100:.4f}%  ref={p_ref*100:.4f}%  "
              f"combined_sem=±{combined*100:.4f}pp  diff_sigma={diff_sigma:+.2f}  {verdict}")
        print(f"    absorbed: kernel={ker['fraction_absorbed']:.4f} ref={ref['fraction_absorbed']:.4f}"
              f"  escaped: kernel={ker['fraction_escaped']:.4f} ref={ref['fraction_escaped']:.4f}"
              f"  mean_scatter: kernel={ker['mean_scatter']:.4f} ref={ref['mean_scatter']:.4f}"
              f"  fluor_rate: kernel={ker['n_fluorescence']/N:.5f} ref={ref['n_fluorescence']/N:.5f}")

        p_egs5, n_egs5 = EGS5_ESTABLISHED[name]
        sem_egs5 = math.sqrt(p_egs5 * (1 - p_egs5) / n_egs5)
        combined3 = math.sqrt(sem_egs5 ** 2 + sem_ker ** 2)
        diff3 = p_ker - p_egs5
        rel3 = diff3 / p_egs5 * 100
        sigma3 = diff3 / combined3
        verdict3 = "PASS" if abs(rel3) <= EGS5_REL_GATE * 100 and abs(sigma3) <= EGS5_SIGMA_GATE else "FAIL"
        all_pass &= (verdict3 == "PASS")
        print(f"    層3(EGS5確立値): kernel={p_ker*100:.4f}%  EGS5={p_egs5*100:.4f}%  "
              f"diff={diff3*100:+.4f}pp  rel={rel3:+.3f}%  sigma={sigma3:+.2f}  {verdict3}")
        print()

    print("総合判定:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
