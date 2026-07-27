"""docs/plan_tally_speedup.md: 回帰条件（またはbatch-size平坦性確認）の壁時間・
ピークメモリ計測スクリプト。git worktreeで別コミットを退避してこのスクリプトを
両方の作業ツリーで交互に実行することで、インターリーブA/B比較ができる。

使い方:
    # 現在のツリーで計測
    PYTHONPATH=. .venv/bin/python docs/speedup_baseline/tally_speedup_timing.py

    # 別コミットと比較する場合（例: 旧サブステップ方式0157179 vs 現行）
    git worktree add --detach /tmp/wt-old 0157179
    PYTHONPATH=/tmp/wt-old .venv/bin/python /tmp/wt-old/docs/speedup_baseline/tally_speedup_timing.py
    PYTHONPATH=. .venv/bin/python docs/speedup_baseline/tally_speedup_timing.py
    # ↑を交互に3反復以上（アーム順は固定しない）

    # n_histories/batch_size/resolutionを変えて平坦性を見る場合
    PYTHONPATH=. .venv/bin/python docs/speedup_baseline/tally_speedup_timing.py --n 50000 --batch 200000
"""
from __future__ import annotations

import argparse
import resource
import time

from chatcarlo.scene import load_scene
from chatcarlo.transport import run_transport


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="examples/chest_room.yaml")
    p.add_argument("--n", type=float, default=200_000)
    p.add_argument("--batch", type=int, default=200_000)
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    scene = load_scene(args.scene)
    t0 = time.perf_counter()
    result = run_transport(scene, n_histories=int(args.n), seed=args.seed,
                            batch_size=args.batch, dose_grid=True,
                            grid_resolution_cm=args.resolution, n_workers=1)
    dt = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    per_history_us = dt / args.n * 1e6
    print(f"wall={dt:.3f}s  peak_rss={peak_gb:.2f}GB  per_history={per_history_us:.1f}us "
          f"(n={args.n:.0f}, batch={args.batch}, res={args.resolution}cm)")
    print(f"kerma_sum={result.grid.kerma_keV.sum():.6f}")


if __name__ == "__main__":
    main()
