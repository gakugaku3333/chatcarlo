"""依頼 D: 履歴数 N を段階的に変えて平面検出器の画像を作る（本番 chatcarlo、NumPy 輸送）。

使い方（リポジトリの最上位で。chatcarlo を読み込むため PYTHONPATH=. が要る）:
  PYTHONPATH=. .venv/bin/python docs/jsmp133/request_d_history_noise/run_history_noise.py --n 1e4 --seed 101 --workers 1 --out DIR
条件は PREREGISTRATION.md に事前記録してある。ここでは計算と保存だけを行い、解析は analyze_history_noise.py。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chatcarlo.detector import DetectorPlane
from chatcarlo.scene import load_scene
from chatcarlo.transport import run_transport

HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene_bone_softtissue.yaml"
# 検出器: z=0、20 cm 角、128x128 画素（画素 0.156 cm）。法線 +z（下向きに進む光子を受ける）。
DET = dict(center_cm=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], u_axis=[1.0, 0.0, 0.0],
           size_cm=(20.0, 20.0), shape=(128, 128))
N_BATCHES = 10          # 段階全体でのバッチ数の目安（R は補助の指標）
MAX_BATCH = 200_000    # 既定の batch_size を上限にする（8 ワーカー同時のメモリのため）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="A", help="同じ N の独立な繰り返しの記号（A/B）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    n = int(round(a.n))
    scene = load_scene(str(SCENE))
    plane = DetectorPlane(np.array(DET["center_cm"]), np.array(DET["normal"]), np.array(DET["u_axis"]),
                          DET["size_cm"], DET["shape"])
    batch = int(min(MAX_BATCH, max(1, -(-n // N_BATCHES))))
    t0 = time.time()
    res = run_transport(scene, n_histories=n, seed=a.seed, batch_size=batch, n_workers=a.workers,
                        track_uncertainty=True, detector=plane)
    dt = time.time() - t0
    tally = res.detector
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / f"N{n:09d}_{a.tag}"
    np.savez_compressed(f"{stem}.npz", category_fluence=tally.category_fluence, photon_count=tally.photon_count,
                        energy_sum_keV=tally.energy_sum_keV, energy_sum2_keV2=tally.energy_sum2_keV2,
                        total_rel_err=tally.total_relative_error(), n_batches_hit=tally.n_batches_hit,
                        n_histories=tally.n_histories, n_batches=tally.n_batches, pixel_area_cm2=plane.pixel_area_cm2)
    meta = dict(n_histories=n, tag=a.tag, seed=a.seed, workers=a.workers, batch_size=batch, n_batches=tally.n_batches,
                wall_s=dt, detected=float(tally.photon_count.sum()))
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
