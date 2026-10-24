"""依頼 2（対話の入口版）の後処理: 入口が出した検出器 NPZ から、共通の濃淡で並べた PNG と均一領域のノイズの表を作る。

要件（requirements.md）どおり:
- 画像の量は photon_count（画素に入ってきた光子の数）。入射光子 1 個あたりに正規化し、全コマで同じ濃淡目盛り。
  目盛りの範囲は最大の段階（10⁸）の照射野内の画素の 1〜99 パーセンタイル（計算前に決めた規則）。
- ノイズは依頼 D の CV_plane と同じ定義: ROI 左 [13:29, 88:110]・右 [73:89, 88:110]（[iu, iv]）で、
  左右それぞれ 1 次の平面 a + b·u + c·v を最小二乗で引き、sqrt(残差の二乗和 / (704 − 6)) / ROI の平均。
- 相当する空気カーマは SpekPy（70 kV、2.5 mmAl、陽極角 12°、100 cm）で、フルエンス = N / 照射野面積（12.9×17.2 cm²）。
使い方: .venv/bin/python runs/xray_portrait_noise_20261024/make_frames.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STEPS = ['1e4', '1e5', '1e6', '1e7', '1e8']
ROIS = [(slice(13, 29), slice(88, 110)), (slice(73, 89), slice(88, 110))]
FIELD_CM2 = 12.9 * 17.2


def cv_plane(img, rois=ROIS):
    resid, values = [], []
    for su, sv in rois:
        block = img[su, sv].astype(float)
        u, v = np.meshgrid(np.arange(block.shape[0]), np.arange(block.shape[1]), indexing='ij')
        a = np.column_stack([np.ones(block.size), u.ravel(), v.ravel()])
        coef, *_ = np.linalg.lstsq(a, block.ravel(), rcond=None)
        resid.append(block.ravel() - a @ coef)
        values.append(block.ravel())
    resid, values = np.concatenate(resid), np.concatenate(values)
    return float(np.sqrt((resid ** 2).sum() / (values.size - 3 * len(rois))) / values.mean()), float(values.mean())


def kerma_per_photon_uGy():
    import spekpy as sp
    s = sp.Spek(kvp=70, th=12)
    s.filter('Al', 2.5)
    return float(s.get_kerma() / s.get_flu())  # µGy per (photon/cm²) at 100 cm


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Hiragino Sans', 'DejaVu Sans']
    k = kerma_per_photon_uGy()
    frames, rows = [], []
    for step in STEPS:
        with np.load(HERE / f'n{step}' / 'detector.npz') as z:
            count = z['photon_count'].astype(float)
            n = int(z['n_histories'])
        cv, mean = cv_plane(count)
        # 監査（ステージ A）の所見 1: 右の ROI は照射野の端まで x 方向 0.60 cm。左右を別にも出して縁の影響を確かめる
        cv_left, cv_right = (cv_plane(count, [r])[0] for r in ROIS)
        frames.append((step, n, count / n))
        rows.append(dict(N=n, detected=float(count.sum()), roi_mean_count=mean, cv_plane=cv,
                         cv_plane_left=cv_left, cv_plane_right=cv_right,
                         poisson_cv=1 / np.sqrt(mean), air_kerma_uGy=n / FIELD_CM2 * k))
    top = frames[-1][2]
    # 照射野（検出面で ±6.45 × ±8.6 cm）の内側の画素だけで範囲を決める（端から半画素内側）。
    # ステージ B 監査の所見 1: top > 0 では散乱線で照射野の外も入っていた
    nu, nv = top.shape
    uc = (np.arange(nu) + .5) * 15 / nu - 7.5
    vc = (np.arange(nv) + .5) * 18 / nv - 9
    inside = (np.abs(uc)[:, None] <= 6.45 - 7.5 / nu) & (np.abs(vc)[None, :] <= 8.6 - 9 / nv)
    lit = top[inside]
    vmin, vmax = np.percentile(lit, 1), np.percentile(lit, 99)
    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 4.4))
    for ax, (step, n, img), row in zip(axes, frames, rows):
        ax.imshow(img.T, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(f'N = {n:.0e}\n空気カーマ {row["air_kerma_uGy"]:.2g} µGy\nノイズ {100 * row["cv_plane"]:.3g}%', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('入射光子数（線量に相当）を増やすと X 線画像のざらつきが減る（70 kV、理想的な光子計数型の検出器、共通の濃淡）', fontsize=11)
    fig.text(.5, .01, '画素値は入射光子 1 個あたりの検出光子数（多いほど黒、X 線写真と同じ向き）。'
             'ノイズ = 背景の均一領域（アルミ 8.5 mm）で、平面を引いた残差の標準偏差 ÷ 平均（10⁴ は画素の大半が 0）。\n'
             '空気カーマは検出面・被写体なしの換算（SpekPy）で、10⁸ でも約 0.004 mAs（一般撮影の約 1/1000）。'
             '理想的な光子計数型の検出器で、検出器の効率・電気的ノイズは含まない。', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .93))
    fig.savefig(HERE / 'frames_common_scale.png', dpi=150)
    slope = np.polyfit(np.log10([r['N'] for r in rows]), np.log10([r['cv_plane'] for r in rows]), 1)[0]
    out = dict(rows=rows, slope_cv_vs_N=float(slope), gray_range_per_photon=[float(vmin), float(vmax)],
               kerma_uGy_per_photon_per_cm2=k)
    (HERE / 'frames_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print('| N | 検出光子 | ROI 平均（画素あたり） | ノイズ CV_plane | 左 | 右 | 1/√平均 | 空気カーマ µGy |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        print(f"| {r['N']:.0e} | {r['detected']:,.0f} | {r['roi_mean_count']:.4g} | {100*r['cv_plane']:.3g}% | "
              f"{100*r['cv_plane_left']:.3g}% | {100*r['cv_plane_right']:.3g}% | {100*r['poisson_cv']:.3g}% | {r['air_kerma_uGy']:.2g} |")
    print(f'傾き（log CV vs log N）: {slope:.3f}')


if __name__ == '__main__':
    main()
