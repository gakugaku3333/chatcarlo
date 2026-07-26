"""解像度を細かくすると最大値が増大する現象の再調査（事前登録済み仮説）。

CLAUDE.md「Known sharp edges」に記載の仮説H1（max_substepsクランプ由来の
極値統計）を、accumulate_track_lengthを解析的重なり長方式（乱数不使用・
分散ゼロ）に置き換えたあとで検証する。もう一つの仮説H2（線源近傍は
1/r²発散なのでボクセル平均が解像度依存で増大するのは物理的に正しい挙動）
と対比する。

事前登録:
  シナリオA「表面隣接空気」（非特異点、鉛筆ビームが水スラブに入射する直前の
  空気ボクセル、docs/plan.../test_dose_diagnostics.pyの
  test_dose_max_voxel_on_water_slab_surface_triggers_background_warningと同一幾何）:
    H1が正しければ解像度を細かくしても最大値はほぼ収束する（旧実装のサブステップ
    クランプが原因だったため、原因が消えれば増大も消えるはず）。
    増大が残るならH1は誤りで、CLAUDE.mdの記述を訂正する必要がある。

  シナリオB「線源近傍空気」（特異点、点線源の1/r²発散域、物体から離れた
  ビーム軸上の点）:
    H2が正しければ解像度を細かくするほど最大値は増大し続ける（ボクセル平均が
    真の1/r²ピークへ近づくだけの物理的に正しい挙動）。これはnear_source_air_warning
    が既にカバーしている想定内の現象。

判定: シナリオAで収束・シナリオBで増大継続、ならH1もH2も正しく整合的。
      シナリオAでも増大が続くなら、解析的重なり長方式でも説明できない
      別の要因が残っていることになり、CLAUDE.mdの記述を「原因未確定」に
      訂正する必要がある。
"""
from __future__ import annotations

import numpy as np

from chatcarlo.diagnostics import (background_medium_warning, dose_map_Gy,
                                    max_voxel_position_cm, near_source_air_warning)
from chatcarlo.geometry import Geometry
from chatcarlo.tally import VoxelGrid
from chatcarlo.transport import _run_batches

RESOLUTIONS = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
N_HISTORIES = 800_000
BATCH_SIZE = 20_000
SEED = 100


def _report_cell(label, grid, geometry, data, rel_err, iy):
    """y層iy内でx/z方向に最大の値を持つセルを1件報告する（ビーム軸中心を自動追跡）。"""
    layer = data[:, iy, :]
    ix, iz = np.unravel_index(int(np.argmax(layer)), layer.shape)
    idx = (ix, iy, iz)
    pos = grid.origin_cm + (np.asarray(idx, dtype=float) + 0.5) * grid.voxel_size_cm
    material = str(geometry.material_at(pos[None, :])[0])
    r = float(rel_err[idx])
    n_hit = int(grid.n_batches_hit[idx])
    print(f"  [{label:8s}] value={data[idx]:.6e}  pos={pos}  material={material}  "
          f"R={r:.4f}  hit={n_hit}/{grid.n_batches}")


def run_scenario_boundary(name, src, geoms, box_lo, box_hi, quantity):
    """境界(表面)をまたぐ箱: 境界の直前(旧材料)・直後(新材料)を固定追跡する。

    箱はy方向で境界がちょうど中央(shape[1]//2)に来るよう設計してあるので、
    「境界直前レイヤーiy=shape[1]//2-1」「境界直後レイヤーiy=shape[1]//2」は
    解像度に関わらず物理的に同じ位置を指す固定参照点になる（argmaxの位置が
    解像度ごとに air<->water で飛び移る問題を避ける、advisor指摘対応）。
    """
    print(f"\n=== シナリオ{name} ===")
    geometry = Geometry(geoms)
    for res in RESOLUTIONS:
        grid = VoxelGrid.from_bbox(np.array(box_lo), np.array(box_hi), res, track_uncertainty=True)
        rng = np.random.default_rng(SEED)
        _run_batches(src, geometry, rng, N_HISTORIES, BATCH_SIZE, grid,
                     fluorescence_enabled=True, track_uncertainty=True)
        if quantity == "dose":
            data = dose_map_Gy(grid, geometry) / N_HISTORIES
            rel_err = grid.kerma_relative_error()
        else:
            data = grid.h10_map_pSv() / N_HISTORIES
            rel_err = grid.h10_relative_error()

        assert grid.shape[1] % 2 == 0, "箱のy分割数は偶数である必要がある(境界を中央に置く設計)"
        iy_before = grid.shape[1] // 2 - 1
        iy_after = grid.shape[1] // 2
        print(f"res={res:6.3f}cm  shape={grid.shape}")
        _report_cell("境界直前", grid, geometry, data, rel_err, iy_before)
        _report_cell("境界直後", grid, geometry, data, rel_err, iy_after)


def run_scenario_singular(name, src, geoms, box_lo, box_hi, quantity):
    """特異点近傍: 単純にargmaxを追う(単調減衰なので同じ物理位置=線源に一番近い
    セルを指し続けるはずで、シナリオAのような位置の飛び移りは起きない)。"""
    print(f"\n=== シナリオ{name} ===")
    geometry = Geometry(geoms)
    for res in RESOLUTIONS:
        grid = VoxelGrid.from_bbox(np.array(box_lo), np.array(box_hi), res, track_uncertainty=True)
        rng = np.random.default_rng(SEED)
        _run_batches(src, geometry, rng, N_HISTORIES, BATCH_SIZE, grid,
                     fluorescence_enabled=True, track_uncertainty=True)
        if quantity == "dose":
            data = dose_map_Gy(grid, geometry) / N_HISTORIES
            rel_err = grid.kerma_relative_error()
        else:
            data = grid.h10_map_pSv() / N_HISTORIES
            rel_err = grid.h10_relative_error()

        pos = max_voxel_position_cm(grid, data)
        idx = np.unravel_index(int(np.argmax(data)), data.shape)
        material = str(geometry.material_at(pos[None, :])[0])
        r = float(rel_err[idx])
        n_hit = int(grid.n_batches_hit[idx])
        print(f"res={res:6.3f}cm  max={data[idx]:.6e}  pos={pos}  material={material}  "
              f"R={r:.4f}  hit={n_hit}/{grid.n_batches}  shape={grid.shape}")


def main():
    # --- シナリオA: 表面隣接空気（非特異点） ---
    src_a = {
        "kvp": 120, "position": [0, -20, 140], "direction": [0, 1, 0],
        "field": {"size_cm": [2, 2], "sid_cm": 20}, "filtration_mm_al": 2.5, "mas": 4.0,
    }
    geoms_a = [{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0, 0, 140], "size_cm": [30, 30, 30],
    }]
    # y=-15が表面。ちょうど境界をまたぐ薄い箱（水側2cm・空気側2cm）。
    run_scenario_boundary("A（表面隣接空気）", src_a, geoms_a,
                           box_lo=[-2.0, -17.0, 138.0], box_hi=[2.0, -13.0, 142.0], quantity="dose")

    # --- シナリオB: 線源近傍空気（1/r²特異点） ---
    src_b = {
        "kvp": 120, "position": [0, -180, 140], "direction": [0, 1, 0],
        "field": {"size_cm": [35, 43], "sid_cm": 180}, "filtration_mm_al": 2.5, "mas": 4.0,
    }
    geoms_b = [{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0, 0, 140], "size_cm": [30, 30, 30],
    }]
    # 線源から5cm(y=-175)の空気中。最寄り物体(スラブ、y=-15)までは160cmあるので
    # near_source_air_warningの「シーン内物体より線源に近い」条件を満たす。
    run_scenario_singular("B（線源近傍空気, 5cm)", src_b, geoms_b,
                           box_lo=[-2.0, -177.0, 138.0], box_hi=[2.0, -173.0, 142.0], quantity="h10")


if __name__ == "__main__":
    main()
