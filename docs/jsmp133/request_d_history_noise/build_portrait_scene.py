"""依頼 D: レントゲン博士の肖像から「凹凸のあるアルミ板」のシーンを作る（chatcarlo/ は変えない）。

肖像の明るい所（顔・背景）ほど厚く、暗い所（ひげ・上着）ほど薄くする。X 線をよく吸収する所が
白く写る表示（フルエンスが大きいほど黒）にすると、肖像の正像に見える。
同じ段階が横に並んだ画素を 1 つの箱にまとめ、さらに上下の行で同じ範囲・同じ段階の箱を縦にまとめる。

使い方（リポジトリの最上位で）:
  .venv/bin/python docs/jsmp133/request_d_history_noise/build_portrait_scene.py \
      --src ~/.cache/chatcarlo/jsmp133_request_d/source_image/roentgen_hanfstaengl_1280.jpg \
      --nx 42 --ny 56 --levels 6 --out docs/jsmp133/request_d_history_noise/scene_portrait.yaml
元画像はリポジトリに入れない（出典は PREREGISTRATION.md）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

# 元画像（1280x1648）から頭と肩を切り出す範囲 [左, 上, 右, 下]（画素）。縦横比 3:4。
CROP = (120, 160, 1160, 1547)
PIX_CM = 0.3            # 被写体の 1 画素の大きさ（板の上で）
BASE_MM = 1.0           # 枠を含む一様な土台の厚さ
STEP_MM = 1.5           # 1 段階あたりの厚さ
PLATE_Z0 = 1.0          # 土台の下面の高さ [cm]（検出面は z=0）
FRAME_CM = 22.0         # 土台（枠）の一辺
# 照射野は肖像の範囲だけ（検出面 SID 100 cm で、肖像の影 ±6.42×±8.56 cm を少しだけ覆う）。
# 枠の外に光子を飛ばす計算時間を省くため（ユーザーの指示 2026-10-21）。
FIELD_CM = [12.9, 17.2]


def level_map(src: Path, nx: int, ny: int, levels: int) -> np.ndarray:
    im = Image.open(src).convert("L").crop(CROP)
    im = im.filter(ImageFilter.GaussianBlur(radius=im.width / nx * 0.5))
    a = np.asarray(im.resize((nx, ny), Image.LANCZOS), dtype=float)
    lo, hi = np.percentile(a, [2, 98])
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return np.minimum((a * levels).astype(int), levels - 1)  # 0..levels-1、行 0 が上


def boxes_from_levels(L: np.ndarray):
    """同じ段階の横の並びを箱に、さらに縦に同じ範囲の箱をまとめる。段階 0 は箱なし。"""
    ny, nx = L.shape
    open_runs = {}  # (x0, x1, level) -> [row_start, row_end]
    done = []
    for r in range(ny):
        runs = []
        c = 0
        while c < nx:
            lv = L[r, c]
            c1 = c
            while c1 + 1 < nx and L[r, c1 + 1] == lv:
                c1 += 1
            if lv > 0:
                runs.append((c, c1 + 1, int(lv)))
            c = c1 + 1
        new_open = {}
        for key in runs:
            if key in open_runs:
                open_runs[key][1] = r + 1
                new_open[key] = open_runs.pop(key)
            else:
                new_open[key] = [r, r + 1]
        done += [(k, v) for k, v in open_runs.items()]
        open_runs = new_open
    done += [(k, v) for k, v in open_runs.items()]
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--nx", type=int, default=42)
    ap.add_argument("--ny", type=int, default=56)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels-out", type=Path, help="段階の地図（npy）の保存先")
    a = ap.parse_args()
    L = level_map(a.src.expanduser(), a.nx, a.ny, a.levels)
    if a.levels_out:
        np.save(a.levels_out, L)
    boxes = boxes_from_levels(L)
    w, h = a.nx * PIX_CM, a.ny * PIX_CM
    base_top = PLATE_Z0 + BASE_MM / 10
    geom = [{"name": "frame_base", "shape": "box", "material": "aluminum",
             "size_cm": [FRAME_CM, FRAME_CM, BASE_MM / 10],
             "center": [0.0, 0.0, round(PLATE_Z0 + BASE_MM / 20, 6)]}]
    for (x0, x1, lv), (r0, r1) in sorted(boxes, key=lambda b: (b[1][0], b[0][0])):
        t = lv * STEP_MM / 10
        cx = -w / 2 + (x0 + x1) / 2 * PIX_CM
        cy = h / 2 - (r0 + r1) / 2 * PIX_CM   # 行 0（画像の上）を +y に置く
        geom.append({"shape": "box", "material": "aluminum",
                     "size_cm": [round((x1 - x0) * PIX_CM, 6), round((r1 - r0) * PIX_CM, 6), round(t, 6)],
                     "center": [round(cx, 6), round(cy, 6), round(base_top + t / 2, 6)]})
    scene = {
        "source": {"type": "xray_tube", "kvp": 70, "filtration_mm_al": 2.5, "position": [0, 0, 100],
                   "direction": [0, 0, -1], "field": {"shape": "rect", "size_cm": FIELD_CM, "sid_cm": 100}},
        "geometry": geom,
        "physics": {"fluorescence": True},
    }
    header = (f"# 依頼 D（JSMP133 依頼 2）: レントゲン博士の肖像（Hanfstaengl 撮影、パブリックドメイン）を\n"
              f"# アルミの凹凸板にしたもの。build_portrait_scene.py で自動生成（手で編集しない）。\n"
              f"# {a.nx}x{a.ny} 画素（{PIX_CM} cm）、{a.levels} 段階、土台 {BASE_MM} mm + 段階あたり {STEP_MM} mm、"
              f"箱 {len(geom)} 個（土台を含む）\n")
    a.out.write_text(header + yaml.safe_dump(scene, sort_keys=False, allow_unicode=True))
    print(f"boxes={len(geom)} levels_hist={np.bincount(L.ravel(), minlength=a.levels).tolist()}")


if __name__ == "__main__":
    main()
