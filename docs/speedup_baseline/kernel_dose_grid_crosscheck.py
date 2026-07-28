"""Phase B-2の検証: `chatcarlo.kernel.run_dose_grid`（カーネルのtrack-lengthタリー、
既存の監査済み`tally.accumulate_track_length_multi`を再利用する設計(b)）と
`chatcarlo.transport.transport_photons`のdose_grid経路（参照実装）の統計的クロスチェック。

**事前登録（結果を見る前にここに固定する）**:
- 比較量: グリッド合計カーマ[keV]・グリッド合計H*(10)飛程積分[pSv・cm³]
  （ボクセルごとの値ではなく、まず全体量で大きなロジック誤りがないかを見る——
  B-2は既存の監査済みDDAをそのまま再利用するため、ボクセル空間分配自体の
  正しさは既に担保済みで、ここで検証すべきは主に「カーネルが正しい区間
  (o, d, ds, e, mat)を吐き出せているか」）。
- 統計誤差の見積もり: 両実装ともREPS回の独立反復（異なるseed）を行い、
  標本平均・標本標準誤差(std/sqrt(REPS))で評価する（バッチ統計機構を
  カーネル側に配線するのはB-2のスコープ外——計画書参照）。
- 合格基準: 結合標準誤差の4σ以内（`kernel_crosscheck.py`の層1と同じ基準を
  踏襲、複数量・複数シナリオへの多重比較の余裕を持たせる）。
- シナリオ: water60_free相当（水10cm・60 keV・鉛筆ビーム）1本。resolution=2cm。
  N=400,000×REPS=6（両実装とも）。
"""
from __future__ import annotations

import math

import numpy as np

from chatcarlo.geometry import Geometry
from chatcarlo.kernel import bake_box_scene, bake_scene_materials, run_dose_grid
from chatcarlo.tally import VoxelGrid
from chatcarlo.transport import transport_photons

THICKNESS_CM = 10.0
ENERGY_KEV = 60.0
N_PER_REP = 400_000
REPS = 6
RESOLUTION_CM = 2.0
SIGMA_GATE = 4.0


def _bbox():
    margin = 0.01
    lo = np.array([-THICKNESS_CM / 2 - margin, -50.0 - margin, -50.0 - margin])
    hi = np.array([THICKNESS_CM / 2 + margin, 50.0 + margin, 50.0 + margin])
    return lo, hi


def run_reference(seed: int) -> tuple[float, float]:
    geom = Geometry([{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0.0, 0.0, 0.0], "size_cm": [THICKNESS_CM, 100.0, 100.0],
    }], bbox_margin_cm=0.01)
    lo, hi = _bbox()
    grid = VoxelGrid.from_bbox(lo, hi, RESOLUTION_CM)
    rng = np.random.default_rng(seed)
    pos = np.tile(np.array([-THICKNESS_CM / 2 - 0.01, 0.0, 0.0]), (N_PER_REP, 1))
    dirv = np.tile(np.array([1.0, 0.0, 0.0]), (N_PER_REP, 1))
    energy = np.full(N_PER_REP, ENERGY_KEV)
    transport_photons(pos, dirv, energy, geom, rng, grid=grid, fluorescence_enabled=True)
    return float(grid.kerma_keV.sum()), float(grid.h10_track_pSv_cm3.sum())


def run_kernel(seed: int) -> tuple[float, float]:
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (THICKNESS_CM, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    lo, hi = _bbox()
    grid = VoxelGrid.from_bbox(lo, hi, RESOLUTION_CM)
    origin = (-THICKNESS_CM / 2 - 0.01, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0)
    run_dose_grid(tables, geom, ENERGY_KEV, origin, direction, N_PER_REP, seed, grid,
                   batch_size=200_000, n_chunks=8, fluorescence_enabled=True,
                   max_segments_per_history=16)
    return float(grid.kerma_keV.sum()), float(grid.h10_track_pSv_cm3.sum())


def main() -> None:
    print(f"事前登録: 合格基準=結合{SIGMA_GATE}σ以内, N={N_PER_REP:,}×{REPS}反復, "
          f"resolution={RESOLUTION_CM}cm\n")

    ref_kerma, ref_h10 = [], []
    ker_kerma, ker_h10 = [], []
    for rep in range(REPS):
        rk, rh = run_reference(seed=200 + rep)
        kk, kh = run_kernel(seed=200 + rep)
        ref_kerma.append(rk)
        ref_h10.append(rh)
        ker_kerma.append(kk)
        ker_h10.append(kh)
        print(f"  rep{rep}: ref_kerma={rk:,.1f}  kernel_kerma={kk:,.1f}  "
              f"ref_h10={rh:,.1f}  kernel_h10={kh:,.1f}")

    def _compare(name, ref_vals, ker_vals):
        ref_vals = np.array(ref_vals)
        ker_vals = np.array(ker_vals)
        ref_mean, ref_sem = ref_vals.mean(), ref_vals.std(ddof=1) / math.sqrt(len(ref_vals))
        ker_mean, ker_sem = ker_vals.mean(), ker_vals.std(ddof=1) / math.sqrt(len(ker_vals))
        combined = math.sqrt(ref_sem ** 2 + ker_sem ** 2)
        diff = ker_mean - ref_mean
        sigma = diff / combined if combined > 0 else float("nan")
        rel = diff / ref_mean * 100
        verdict = "PASS" if abs(sigma) <= SIGMA_GATE else "FAIL"
        print(f"\n[{name}] ref={ref_mean:,.1f}±{ref_sem:,.1f}  kernel={ker_mean:,.1f}±{ker_sem:,.1f}  "
              f"diff={diff:+,.1f} ({rel:+.3f}%)  sigma={sigma:+.2f}  {verdict}")
        return verdict == "PASS"

    p1 = _compare("グリッド合計カーマ[keV]", ref_kerma, ker_kerma)
    p2 = _compare("グリッド合計H*(10)飛程積分[pSv・cm3]", ref_h10, ker_h10)
    print("\n総合判定:", "PASS" if (p1 and p2) else "FAIL")


if __name__ == "__main__":
    main()
