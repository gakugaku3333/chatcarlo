"""依頼 D の解析: PREREGISTRATION.md の定義どおりに相対誤差を出し、図を作る。

使い方（リポジトリの最上位で）:
  .venv/bin/python docs/jsmp133/request_d_history_noise/analyze_history_noise.py \
      --data ~/.cache/chatcarlo/jsmp133_request_d/main
出力: このフォルダの results.json、relerr_vs_N.png、history_noise_A4.jpg、history_noise_panels.png、
      history_noise_9frames.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NS = [round(10 ** (4 + 0.5 * k)) for k in range(9)]
SHOWN = [10**4, 10**5, 10**6, 10**7, 10**8]
# 改訂版の ROI（PREREGISTRATION.md）: 肖像の背景（アルミ 8.5 mm）の左上と右上、合わせて 704 画素
ROIS = [(slice(13, 29), slice(88, 110)), (slice(73, 89), slice(88, 110))]
SIZE_U, SIZE_V, NU, NV = 15.0, 18.0, 100, 120
DU = SIZE_U / NU  # = SIZE_V / NV = 0.15 cm
FIELD_HALF = (6.3, 8.4)  # 濃淡の窓を決める範囲（検出面で |x|<=6.3, |y|<=8.4）


def roi_vals(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[r].ravel() for r in ROIS])
PLT_FONT = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Arial Unicode MS", "DejaVu Sans"]


def load(data: Path, n: int, tag: str):
    z = np.load(data / f"N{n:09d}_{tag}.npz")
    return {k: z[k] for k in z.files}


def image(d) -> np.ndarray:
    return d["category_fluence"].sum(axis=0) / float(d["n_histories"])  # Ψ/N [keV cm^-2 / history]


def cv_plane(img: np.ndarray) -> float:
    ss, npix, total = 0.0, 0, 0.0
    for su, sv in ROIS:
        r = img[su, sv]
        iu, iv = np.meshgrid(np.arange(su.start, su.stop), np.arange(sv.start, sv.stop), indexing="ij")
        x = -SIZE_U / 2 + (iu + 0.5) * DU
        y = -SIZE_V / 2 + (iv + 0.5) * DU
        A = np.column_stack([np.ones(r.size), x.ravel(), y.ravel()])
        coef, *_ = np.linalg.lstsq(A, r.ravel(), rcond=None)
        ss += float(((r.ravel() - A @ coef) ** 2).sum())
        npix += r.size
        total += float(r.sum())
    return float(np.sqrt(ss / (npix - 6)) / (total / npix))


def cv_pred(d) -> float:
    cnt = roi_vals(d["photon_count"])
    nsum = cnt.sum()
    lam = cnt.mean()
    e1 = roi_vals(d["energy_sum_keV"]).sum() / nsum
    e2 = roi_vals(d["energy_sum2_keV2"]).sum() / nsum
    return float(np.sqrt(e2 / e1**2 / lam)), float(lam)


def fit(ns, cvs):
    x, y = np.log10(ns), np.log10(cvs)
    A = np.column_stack([np.ones_like(x), x])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = (resid**2).sum() / (len(x) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[1]), float(np.sqrt(cov[1, 1])), float(coef[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    a = ap.parse_args()
    data = a.data.expanduser()
    plt.rcParams["font.family"] = PLT_FONT

    rows = []
    imgs = {}
    for n in NS:
        dA, dB = load(data, n, "A"), load(data, n, "B")
        IA, IB = image(dA), image(dB)
        imgs[n] = (IA, IB)
        diff = roi_vals(IA - IB)
        mean = roi_vals((IA + IB) / 2).mean()
        cvd = float(diff.std(ddof=1) / np.sqrt(2) / mean)
        cvp, lam = cv_pred(dA)
        R = roi_vals(dA["total_rel_err"])
        rows.append(dict(N=n, cv_diff=cvd, cv_plane=cv_plane(IA), cv_pred=cvp, lambda_per_pixel=lam,
                         ratio_diff_over_pred=cvd / cvp,
                         median_batch_R=float(np.nanmedian(np.where(np.isfinite(R), R, np.nan))) if np.isfinite(R).any() else None,
                         median_batches_hit=float(np.median(roi_vals(dA["n_batches_hit"]))),
                         n_batches=int(dA["n_batches"]),
                         detected_A=float(dA["photon_count"].sum()),
                         roi_mean_psi_per_history=float(mean)))
    ns = np.array([r["N"] for r in rows], float)
    s_d, se_d, b_d = fit(ns, [r["cv_diff"] for r in rows])
    s_p, se_p, b_p = fit(ns, [r["cv_plane"] for r in rows])
    out = dict(rows=rows,
               fit_cv_diff=dict(slope=s_d, slope_se=se_d, intercept=b_d, pass_slope=bool(-0.55 <= s_d <= -0.45)),
               fit_cv_plane=dict(slope=s_p, slope_se=se_p, intercept=b_p))
    out["ratio_in_0.8_1.2"] = [bool(0.8 <= r["ratio_diff_over_pred"] <= 1.2) for r in rows]

    # 共通の濃淡目盛り: N=1e8 の (A+B)/2、照射野の内側（|x|<=6.3, |y|<=8.4）の 1〜99 パーセンタイル
    ref = (imgs[10**8][0] + imgs[10**8][1]) / 2
    xc = -SIZE_U / 2 + (np.arange(NU) + 0.5) * DU
    yc = -SIZE_V / 2 + (np.arange(NV) + 0.5) * DU
    inside = (np.abs(xc)[:, None] <= FIELD_HALF[0]) & (np.abs(yc)[None, :] <= FIELD_HALF[1])
    vmin, vmax = np.percentile(ref[inside], [1, 99])
    out["display_window_psi_per_history"] = [float(vmin), float(vmax)]
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # --- 相対誤差のグラフ ---
    def relerr_axes(ax, fs=9):
        ax.loglog(ns, [r["cv_diff"] for r in rows], "o", color="#1f5fa8", ms=6, label="計算値（2本の差、主）")
        ax.loglog(ns, [r["cv_plane"] for r in rows], "s", mfc="none", color="#c0392b", ms=5, label="計算値（1本、平面を除去）")
        ax.loglog(ns, [r["cv_pred"] for r in rows], "--", color="#555", lw=1, label="複合ポアソンの予測")
        ref_line = rows[0]["cv_diff"] * (ns / ns[0]) ** -0.5
        ax.loglog(ns, ref_line, ":", color="k", lw=1.2, label="傾き −1/2（1/√N）")
        ax.set_xlabel("モンテカルロの履歴数 N", fontsize=fs)
        ax.set_ylabel("均一領域の相対誤差（画素ごと）", fontsize=fs)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(labelsize=fs - 1)
        ax.legend(fontsize=fs - 1.5, loc="lower left")
        ax.text(0.97, 0.95, f"当てはめた傾き {s_d:.3f} ± {se_d:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=fs)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    relerr_axes(ax)
    fig.tight_layout()
    fig.savefig(HERE / "relerr_vs_N.png")
    plt.close(fig)

    # --- A4 横の 1 枚 ---
    ext = [-SIZE_U / 2, SIZE_U / 2, -SIZE_V / 2, SIZE_V / 2]
    fig = plt.figure(figsize=(11.69, 8.27), dpi=200)
    fig.suptitle("モンテカルロの履歴数 N を増やすと、計算の統計誤差（ざらつき）が約 1/√N で減る",
                 fontsize=15, y=0.97)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15], left=0.05, right=0.97, top=0.88, bottom=0.08,
                          hspace=0.35, wspace=0.08)
    for i, n in enumerate(SHOWN):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(imgs[n][0].T, origin="lower", extent=ext, cmap="gray_r", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        for su, sv in ROIS:
            x0, y0 = -SIZE_U / 2 + su.start * DU, -SIZE_V / 2 + sv.start * DU
            ax.add_patch(plt.Rectangle((x0, y0), (su.stop - su.start) * DU, (sv.stop - sv.start) * DU,
                                       fill=False, ec="#e67e22", lw=1))
        r = next(r for r in rows if r["N"] == n)
        ax.set_title(f"N = 10$^{{{int(round(np.log10(n)))}}}$\n相対誤差 {100 * r['cv_diff']:.1f}%", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 0:2])
    relerr_axes(ax, fs=10)
    tx = fig.add_subplot(gs[1, 2:5]); tx.axis("off")
    txt = (
        "計算: ChatCarlo の GPU エンジン（Colab の T4）。10⁸ 履歴の 1 本は約 4 分。\n"
        "条件: 70 kVp・2.5 mmAl、SID 100 cm。被写体はレントゲン博士の肖像\n"
        "（1900 年前後、パブリックドメイン）を 6 段階の厚さ（1〜8.5 mm）のアルミ板に\n"
        "したもの。照射野は肖像の範囲だけ。\n"
        "検出面 15×18 cm・画素 1.5 mm の理想的な平面検出器（エネルギーフルエンス）。\n"
        "5 枚は履歴数あたりに正規化し、同じ濃淡目盛り（N=10⁸ の 1〜99 パーセンタイル）で表示。\n"
        "橙の枠が均一領域（背景のアルミ 8.5 mm、704 画素）。相対誤差 = 画素値の標準偏差 / 平均。\n"
        "主の値は、種だけ変えた 2 本の差から求めた（決まった形の濃淡を打ち消すため）。\n\n"
        "【区別】ここで変えたのはモンテカルロの履歴数（計算の統計誤差）であり、\n"
        "撮影の mAs（実際の X 線画像の量子ノイズ）ではない。画像のざらつきは計算の\n"
        "ばらつきで、重みつきの履歴を実際に検出された光子とは見なしていない。\n"
        "同じ撮影条件のまま N を増やすと、計算結果は真の期待値に近づく。"
    )
    tx.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10.5, linespacing=1.6)
    fig.savefig(HERE / "history_noise_A4.jpg", dpi=200, pil_kwargs={"quality": 92})
    plt.close(fig)

    # --- テンプレートに貼る素材（画像 5 枚の横並び、目盛りつき） ---
    fig, axs = plt.subplots(1, 5, figsize=(10, 2.5), dpi=300)
    for ax, n in zip(axs, SHOWN):
        im = ax.imshow(imgs[n][0].T, origin="lower", extent=ext, cmap="gray_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"N = 10$^{{{int(round(np.log10(n)))}}}$", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(HERE / "history_noise_panels.png")
    plt.close(fig)
    # --- 発表用の動画: 本計算の 9 段階を 1 コマ 1 秒ずつ（独立な計算を並べたもの、累積ではない） ---
    from matplotlib.animation import FFMpegWriter
    fig = plt.figure(figsize=(6.4, 7.2), dpi=150)
    ax = fig.add_axes([0.08, 0.17, 0.84, 0.75])
    cap = fig.text(0.5, 0.06, "", ha="center", va="center", fontsize=12)
    fig.text(0.5, 0.015, "モンテカルロの履歴数（計算の統計誤差）を変えた独立な計算。撮影の mAs（量子ノイズ）ではない。",
             ha="center", fontsize=7.5, color="#444")
    im = ax.imshow(imgs[NS[0]][0].T, origin="lower", extent=ext, cmap="gray_r", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("", fontsize=14)
    writer = FFMpegWriter(fps=1, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, HERE / "history_noise_9frames.mp4", dpi=150):
        for n in NS:
            r = next(r for r in rows if r["N"] == n)
            im.set_data(imgs[n][0].T)
            title.set_text(f"履歴数 N = {n:,}")
            cap.set_text(f"均一領域の相対誤差 {100 * r['cv_diff']:.2f}%")
            writer.grab_frame()
        writer.grab_frame()  # 最後のコマを 2 秒見せる
    plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, ensure_ascii=False))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
