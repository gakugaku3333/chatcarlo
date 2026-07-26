"""旧実装（サブステップ+層化乱数点、max_substeps=40クランプあり）で同じ
シナリオAを再現し、新しい解析的重なり長方式の結果（resolution_growth_result.txt）
と比較するための対照実験。chatcarlo.tally.accumulate_track_lengthを一時的に
旧アルゴリズムへ差し替える（本体は変更しない、比較専用）。
"""
from __future__ import annotations

import numpy as np

import chatcarlo.tally as tally_mod
from chatcarlo.diagnostics import dose_map_Gy
from chatcarlo.geometry import Geometry
from chatcarlo.tally import VoxelGrid
from chatcarlo.transport import _run_batches

_persistent_rng = np.random.default_rng(777)


def _old_accumulate_track_length(target, grid, origin, direction, length_cm, weight_per_cm,
                                  substep_cm=None, max_substeps=40):
    n = origin.shape[0]
    if n == 0:
        return
    if substep_cm is None:
        substep_cm = grid.voxel_size_cm / 2.0
    nsub = np.clip(np.ceil(length_cm / substep_cm).astype(int), 1, max_substeps)
    max_n = int(nsub.max())
    j = np.arange(max_n)
    frac = (j[None, :] + _persistent_rng.random((n, max_n))) / nsub[:, None]
    valid = j[None, :] < nsub[:, None]
    points = (origin[:, None, :] + direction[:, None, :]
              * (length_cm[:, None] * frac)[:, :, None])
    sub_weight = weight_per_cm * (length_cm / nsub)
    points_flat = points.reshape(-1, 3)
    weight_flat = np.broadcast_to(sub_weight[:, None], (n, max_n)).reshape(-1)
    valid_flat = valid.reshape(-1)
    idx, in_grid = grid.voxel_index(points_flat)
    keep = valid_flat & in_grid
    idx, weight_flat = idx[keep], weight_flat[keep]
    if len(idx) == 0:
        return
    flat_idx = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.shape)
    np.add.at(target.reshape(-1), flat_idx, weight_flat)


tally_mod.accumulate_track_length = _old_accumulate_track_length
import chatcarlo.transport as transport_mod
transport_mod.accumulate_track_length = _old_accumulate_track_length

N_HISTORIES = 800_000
BATCH_SIZE = 20_000
SEED = 100
RESOLUTIONS = [0.5, 0.25, 0.125, 0.0625]


def main():
    src = {
        "kvp": 120, "position": [0, -20, 140], "direction": [0, 1, 0],
        "field": {"size_cm": [2, 2], "sid_cm": 20}, "filtration_mm_al": 2.5, "mas": 4.0,
    }
    geoms = [{
        "name": "slab", "shape": "box", "material": "water",
        "center": [0, 0, 140], "size_cm": [30, 30, 30],
    }]
    geometry = Geometry(geoms)
    box_lo, box_hi = [-2.0, -17.0, 138.0], [2.0, -13.0, 142.0]

    print("=== 旧実装(サブステップ+層化乱数点, max_substeps=40) シナリオA ===")
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
