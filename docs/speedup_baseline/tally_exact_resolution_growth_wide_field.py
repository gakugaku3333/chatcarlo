"""機構検証(advisor指摘2): ビーム幅を変えたときプラトーに転じる解像度
（knee）がビーム幅に比例して動くかどうかの確認。

narrow(field=[2,2]cm, SID=20)ではy=-15での実効幅≈0.5cm、kneeは
res=0.25→0.125cmの間で観測された（成長率×3.9→×1.04）。
wide(field=[8,8]cm)なら実効幅は4倍の≈2cmになるはずなので、機構が正しければ
kneeも4倍粗い解像度側（res=1→0.5cm付近）へ移動するはず。動かなければ
「voxel<ビーム幅で収束する」という説明そのものが誤り。
"""
from __future__ import annotations

import numpy as np

from chatcarlo.diagnostics import dose_map_Gy
from chatcarlo.geometry import Geometry
from chatcarlo.tally import VoxelGrid
from chatcarlo.transport import _run_batches

N_HISTORIES = 800_000
BATCH_SIZE = 20_000
SEED = 100
RESOLUTIONS = [4.0, 2.0, 1.0, 0.5, 0.25]


def main():
    src = {
        "kvp": 120, "position": [0, -20, 140], "direction": [0, 1, 0],
        "field": {"size_cm": [8, 8], "sid_cm": 20}, "filtration_mm_al": 2.5, "mas": 4.0,
    }
    geoms = [{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0, 0, 140], "size_cm": [30, 30, 30],
    }]
    geometry = Geometry(geoms)
    # ビーム幅が広がった分、箱もx/zを広げる(±6cm)。y方向は前と同じ境界追跡設計。
    box_lo, box_hi = [-6.0, -17.0, 134.0], [6.0, -13.0, 146.0]

    print("=== シナリオA'（表面隣接空気、field=[8,8]cm、実効幅~2cm） ===")
    for res in RESOLUTIONS:
        grid = VoxelGrid.from_bbox(np.array(box_lo), np.array(box_hi), res, track_uncertainty=True)
        rng = np.random.default_rng(SEED)
        _run_batches(src, geometry, rng, N_HISTORIES, BATCH_SIZE, grid,
                     fluorescence_enabled=True, track_uncertainty=True)
        data = dose_map_Gy(grid, geometry) / N_HISTORIES
        rel_err = grid.kerma_relative_error()
        iy_before = grid.shape[1] // 2 - 1
        iy_after = grid.shape[1] // 2
        for label, iy in [("境界直前(air)", iy_before), ("境界直後(water)", iy_after)]:
            layer = data[:, iy, :]
            ix, iz = np.unravel_index(int(np.argmax(layer)), layer.shape)
            idx = (ix, iy, iz)
            print(f"res={res:6.3f}cm [{label}] value={data[idx]:.6e}  R={rel_err[idx]:.4f}  "
                  f"hit={int(grid.n_batches_hit[idx])}/{grid.n_batches}")


if __name__ == "__main__":
    main()
