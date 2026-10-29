"""Supporting Data (JSMP133) の 3 つの図（列）を作る。
 fig_method.png : ① 方法（障壁 2 つ → AI 対話 → 3D 確認 → Colab GPU → 任意の形式で出力）
 fig_result.png : ② EGS5 との照合 3 条件（60 keV 水 PDD・OCR、10 MV 積層 PDD）
 fig_apps.png   : ③ 対話で実施した 2 件の依頼（X 線画像 5 段階、前立腺 I-125 シード）
数値は docs/jsmp133/region_analysis/build_fragment.py の関数で 100M 版のデータから計算し、抄録 v5 の数値と照合する。
再利用した素材は assets/ にある（3D は前立腺シードの承認用プレビュー HTML を headless Chrome で撮影して切り出したもの、
線量比マップは runs/prostate_seed_7field/p9c_dose_slices_prostate_600.png の切り出し、距離別の値は assets/seed_radial_600.txt）。
usage: MPLCONFIGDIR=/private/tmp/jsmp-mpl .venv/bin/python docs/jsmp133/supporting_data/build_figures.py
"""
from pathlib import Path
import os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import cm, colors
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AS = OUT / 'assets'
os.chdir(ROOT)

# ---- 列のサイズ（インチ）。build_pptx.py も同じ値を使う ----
W_COL, H_COL = 3.27, 4.30

src = Path('docs/jsmp133/region_analysis/build_fragment.py').read_text().split('R10, R4, K60, ST =')[0]
src = src.replace('runs/diag_water_60kev_gpu/dose.npz', 'runs/diag_water_60kev_gpu/add50M_20261026/dose_100M.npz')
ns = {}
exec(compile(src, 'build_fragment_defs', 'exec'), ns)
K60, ST = ns['kev60'](), ns['stack']()
for p in K60['prof']:
    p['inner'] = p['x'] > .5          # 評価に使う |x|=1〜8 cm（中心 0〜1 cm は規格化に使う）
din = np.concatenate([p['d'][p['inner']] for p in K60['prof']])
pc = lambda a: 100 * abs(np.asarray(a))
stats = dict(pdd60=(pc(K60['dp']).mean(), pc(K60['dp']).max()), ocr60=(pc(din).mean(), pc(din).max()),
             stack=(pc(ST['dp']).mean(), pc(ST['dp']).max()))
r2 = {k: (round(float(v[0]), 2), round(float(v[1]), 1)) for k, v in stats.items()}
assert r2 == dict(pdd60=(0.16, 0.6), ocr60=(0.39, 1.3), stack=(0.17, 1.3)), r2   # 抄録 v5

plt.rcParams.update({'font.family': ['Hiragino Sans', 'DejaVu Sans'], 'font.size': 6.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .5,
                     'xtick.major.width': .5, 'ytick.major.width': .5})
C_SYS, C_EGS, C_BAND, C_RED, C_NAVY, C_HEAD = '#1f6fb4', '#c2410c', '#9ca3af', '#b42318', '#1f3a5f', '#4b5563'


def new_fig():
    fig = plt.figure(figsize=(W_COL, H_COL))
    return fig


def inch_axes(fig, x, y, w, h, **kw):
    """左下を (x, y) インチとする軸。"""
    return fig.add_axes([x / W_COL, y / H_COL, w / W_COL, h / H_COL], **kw)


def header(fig, text):
    a = inch_axes(fig, 0, H_COL - .24, W_COL, .24)
    a.axis('off'); a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0,rounding_size=.12', fc=C_HEAD, ec='none', transform=a.transAxes))
    a.text(.5, .5, text, color='white', ha='center', va='center', fontsize=8.2, fontweight='bold', transform=a.transAxes)


# ---------------- ② EGS5 との照合 ----------------
def result_fig():
    fig = new_fig()
    header(fig, '② 精度の根拠：EGS5 との比較（各 1.0×10⁸ 光子）')
    top = H_COL - .24
    blk = (top - .02) / 3                          # 1 条件の高さ
    gaps = dict(title=.17, xlab=.27, diff=.34)
    ov = blk - gaps['title'] - gaps['xlab'] - gaps['diff'] - .02
    axl = .50; axw = W_COL - axl - .08

    def block(i, title, stat):
        y0 = top - .02 - blk * (i + 1)
        b = inch_axes(fig, axl, y0 + gaps['xlab'], axw, gaps['diff'])
        a = inch_axes(fig, axl, y0 + gaps['xlab'] + gaps['diff'] + .07, axw, ov - .05)
        a.tick_params(labelbottom=False, length=2, pad=1.5); b.tick_params(length=2, pad=1.5)
        fig.text(.02 / W_COL * 1, (y0 + blk - .02) / H_COL, title, fontsize=7.4, fontweight='bold', va='top', ha='left')
        fig.text((W_COL - .06) / W_COL, (y0 + blk - .025) / H_COL, stat, fontsize=7, color=C_NAVY, va='top', ha='right')
        a.set_ylabel('相対線量 (%)', labelpad=1.5); b.set_ylabel('差 (%)', labelpad=1.5)
        b.set_ylim(-2.2, 2.2); b.set_yticks([-2, 0, 2])
        return a, b

    # (a) 60 keV PDD
    a, b = block(0, '60 keV 水 PDD', f"差 平均 {stats['pdd60'][0]:.2f}%・最大 {stats['pdd60'][1]:.1f}%")
    a.plot(K60['z'], 100 * K60['pe'], 'o', color=C_EGS, ms=2.3, label='EGS5')
    a.plot(K60['z'], 100 * K60['pg'], 'x', color=C_SYS, ms=3, mew=.7, label='本システム')
    b.fill_between(K60['z'], -100 * K60['sp'], 100 * K60['sp'], color=C_BAND, alpha=.4, lw=0)
    b.plot(K60['z'], 100 * K60['dp'], '.-', color=C_SYS, ms=2.5, lw=.5); b.axhline(0, color='k', lw=.4)
    b.set_xlabel('深さ (cm)', labelpad=1)
    a.legend(frameon=False, fontsize=6, loc='upper right', handletextpad=.2, borderaxespad=0, labelspacing=.2)
    a.set_ylim(bottom=0)

    # (b) 60 keV OCR（深さ 0〜1 cm と 9〜10 cm）
    a, b = block(1, '60 keV 水 OCR', f"差 平均 {stats['ocr60'][0]:.2f}%・最大 {stats['ocr60'][1]:.1f}%")
    for p, (me, mg) in zip(K60['prof'], [('o', 'x'), ('s', '+')]):
        a.plot(p['x'], 100 * p['e'], me, color=C_EGS, ms=2.2, mfc='none' if me == 's' else C_EGS)
        a.plot(p['x'], 100 * p['g'], mg, color=C_SYS, ms=3, mew=.7)
        b.plot(p['x'][p['inner']], 100 * p['d'][p['inner']], me, color=C_SYS, ms=2.5, mfc='none' if me == 's' else C_SYS)
    for t in (a, b):
        t.axvline(5, color='#6b7280', lw=.5, ls=':'); t.axvspan(5, 8, color='#fde68a', alpha=.45, lw=0); t.set_xlim(0, 8)
    a.text(6.5, 60, '照射野外', fontsize=5.8, color='#92400e', ha='center')
    a.text(.03, .08, '○× 深さ 0–1 cm\n□+ 深さ 9–10 cm', transform=a.transAxes, fontsize=5.4, color='#374151', linespacing=1.3)
    b.axhline(0, color='k', lw=.4); b.set_xlabel('中心からの距離 (cm)', labelpad=1)
    a.set_ylim(bottom=0)

    # (c) 10 MV 積層 PDD
    a, b = block(2, '10 MV 積層 PDD', f"差 平均 {stats['stack'][0]:.2f}%・最大 {stats['stack'][1]:.1f}%")
    a.plot(ST['z'], 100 * ST['pe'], '-', color=C_EGS, lw=.9, label='EGS5')
    a.plot(ST['z'], 100 * ST['pg'], '--', color=C_SYS, lw=.9, label='本システム')
    b.fill_between(ST['z'], -100 * ST['sp'], 100 * ST['sp'], color=C_BAND, alpha=.4, lw=0)
    b.plot(ST['z'], 100 * ST['dp'], '-', color=C_SYS, lw=.5); b.axhline(0, color='k', lw=.4)
    ed, layer = ST['edges'], ST['layer']
    order = ['水', '肺', '水', '骨', '水', '鉛']
    cuts = ed[1:][np.diff(np.r_[layer, -1]) != 0]
    starts = np.r_[ed[0], cuts[:-1]]
    assert len(cuts) == len(order), (len(cuts), order)
    for s0, s1, nm in zip(starts, cuts, order):
        for t in (a, b):
            t.axvline(s1, color='#d1d5db', lw=.5)
        a.text((s0 + s1) / 2, .04, nm, transform=a.get_xaxis_transform(), fontsize=5.6, ha='center', va='bottom', color='#374151')
    b.set_xlabel('深さ (cm)', labelpad=1)
    a.set_ylim(0, 150)
    a.legend(frameon=False, fontsize=6, loc='lower left', bbox_to_anchor=(.2, .20), handletextpad=.4, borderaxespad=0, labelspacing=.2)
    fig.savefig(OUT / 'fig_result.png', dpi=300)
    plt.close(fig)


# ---------------- ① 方法 ----------------
def method_fig():
    fig = new_fig()
    header(fig, '① 方法：対話から GPU 計算まで')
    ax = inch_axes(fig, 0, 0, W_COL, H_COL - .27)
    ax.set_xlim(0, W_COL); ax.set_ylim(0, H_COL - .27); ax.axis('off')

    def box(y0, h, fc, ec, lw=.9):
        ax.add_patch(FancyBboxPatch((.02, y0), W_COL - .04, h, boxstyle='round,pad=0,rounding_size=.06', fc=fc, ec=ec, lw=lw))

    def chip(y_top, text):
        ax.text(W_COL - .09, y_top - .03, text, ha='right', va='top', fontsize=5.8, color=C_RED, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', fc='#fef3f2', ec=C_RED, lw=.6))

    def arrow(y_from, y_to):
        ax.add_patch(FancyArrowPatch((W_COL / 2, y_from), (W_COL / 2, y_to), arrowstyle='-|>', mutation_scale=6, lw=.8, color='#374151'))

    top = H_COL - .27
    items = []
    y = top - .02
    # 1
    h1 = .36; box(y - h1, h1, '#ffffff', C_SYS)
    ax.text(W_COL / 2, y - h1 / 2, '計算したい内容を自然な言葉で話す（使用者）', ha='center', va='center', fontsize=7.2); y1 = y - h1
    arrow(y1, y1 - .12); y = y1 - .12
    # 2
    h2 = .56; box(y - h2, h2, '#eef4fb', C_SYS)
    ax.text(W_COL / 2, y - h2 / 2 - .06, 'AI エージェントが目的の聞き取り・方法の検討・\n出力形式の確認を支援（Agent Skills）', ha='center', va='center', fontsize=7.2, linespacing=1.3)
    chip(y, '障壁①：入力形式の理解 を AI が支援'); y2 = y - h2
    arrow(y2, y2 - .12); y = y2 - .12
    # 3
    h3 = .36; box(y - h3, h3, '#eef4fb', C_SYS)
    ax.text(W_COL / 2, y - h3 / 2, '合意した条件から計算の入力を自動生成', ha='center', va='center', fontsize=7.2); y3 = y - h3
    arrow(y3, y3 - .12); y = y3 - .12
    # 4: 3D 確認
    h4 = 1.22; box(y - h4, h4, '#fffbeb', '#d97706')
    ax.text(W_COL / 2, y - .12, '使用者が 3D 表示でジオメトリを確認・承認', ha='center', va='center', fontsize=7.2, fontweight='bold')
    img = Image.open(AS / 'seed3d_crop.png'); iw = 1.42; ih = iw * img.size[1] / img.size[0]
    # 画像は fig 座標で置く（ax は上の余白 .27 を除いた座標系）
    iax = fig.add_axes([.09 / W_COL, (y - .23 - ih) / H_COL, iw / W_COL, ih / H_COL])
    iax.imshow(img); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_visible(True); s.set_color('#d97706'); s.set_linewidth(.6)
    ax.text(.09 + iw + .08, y - .27, '前立腺（紫：シード 58 本）、\n直腸（橙）、体（青）、7 門の\nビーム（赤線）の位置関係を\n3D で回転して確認する\n（例：前立腺シード 7 門）', ha='left', va='top', fontsize=6.3, color='#374151', linespacing=1.35)
    y4 = y - h4
    arrow(y4, y4 - .12); y = y4 - .12
    # 5
    h5 = .56; box(y - h5, h5, '#e8f5ec', '#2f855a')
    ax.text(W_COL / 2, y - h5 / 2 - .06, 'Google Colaboratory の GPU で計算\n（光子・電子・陽電子、最大 10 MeV）', ha='center', va='center', fontsize=7.2, linespacing=1.3)
    chip(y, '障壁②：計算資源 をクラウド GPU で解消'); y5 = y - h5
    arrow(y5, y5 - .12); y = y5 - .12
    # 6
    h6 = y - .02; box(.02, h6, '#ffffff', C_SYS)
    ax.text(W_COL / 2, .02 + h6 / 2 + .08, '結果を任意の形式で出力', ha='center', va='center', fontsize=7.6, fontweight='bold')
    ax.text(W_COL / 2, .02 + h6 / 2 - .1, '（例：PDD・OCR の表と図、線量マップ、X 線画像）', ha='center', va='center', fontsize=6.3, color='#374151')
    fig.savefig(OUT / 'fig_method.png', dpi=300)
    plt.close(fig)


# ---------------- ③ 適用例 ----------------
def apps_fig():
    fig = new_fig()
    header(fig, '③ 対話で実施した 2 件の依頼')
    y = H_COL - .24 - .04
    # --- X 線画像 ---
    fig.text(.02 / W_COL, y / H_COL, 'X 線画像：入射光子数を 10⁴ から 10⁸ まで 5 段階', fontsize=7.2, fontweight='bold', va='top')
    y -= .2
    steps = ['1e4', '1e5', '1e6', '1e7', '1e8']
    base = ROOT / 'docs/jsmp133/request_2_entry'
    frames = []
    for st in steps:
        with np.load(base / f'n{st}' / 'detector.npz') as z:
            frames.append((z['photon_count'].astype(float), int(z['n_histories'])))
    top = frames[-1][0] / frames[-1][1]
    nu, nv = top.shape
    uc = (np.arange(nu) + .5) * 15 / nu - 7.5; vc = (np.arange(nv) + .5) * 18 / nv - 9
    inside = (np.abs(uc)[:, None] <= 6.45 - 7.5 / nu) & (np.abs(vc)[None, :] <= 8.6 - 9 / nv)
    vmin, vmax = np.percentile(top[inside], 1), np.percentile(top[inside], 99)    # make_frames.py と同じ規則
    import json
    rows = json.loads((base / 'frames_summary.json').read_text())['rows']
    fw = (W_COL - .04 - 4 * .05) / 5; fh = fw * 18 / 15
    for i, ((cnt, n), row) in enumerate(zip(frames, rows)):
        x = .02 + i * (fw + .05)
        a = fig.add_axes([x / W_COL, (y - fh) / H_COL, fw / W_COL, fh / H_COL])
        a.imshow((cnt / n).T, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax, aspect='auto')
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_linewidth(.4)
        exp = int(round(np.log10(n)))
        fig.text((x + fw / 2) / W_COL, (y - fh - .03) / H_COL, f'10{"⁴⁵⁶⁷⁸"[exp - 4]}', ha='center', va='top', fontsize=6.6, fontweight='bold')
        cv = 100 * row['cv_plane']
        fig.text((x + fw / 2) / W_COL, (y - fh - .15) / H_COL, f'{cv:.3g}%' if cv >= 10 else f'{cv:.2g}%', ha='center', va='top', fontsize=6, color=C_NAVY)
    y = y - fh - .15 - .13
    fig.text(.02 / W_COL, y / H_COL, '数字は背景のノイズ（光子数とともに低下）。\n70 kV・理想的な光子計数型の検出器・共通の濃淡', fontsize=5.8, color='#4b5563', va='top', linespacing=1.3)
    y -= .5
    # --- シード ---
    fig.text(.02 / W_COL, y / H_COL, '前立腺 I-125 シード 58 本の線量への影響（10 MV・7 門）', fontsize=7.2, fontweight='bold', va='top')
    y -= .22
    mw = 1.58
    img = Image.open(AS / 'seed_ratio_axial.png'); mh = mw * img.size[1] / img.size[0]
    ax_ = fig.add_axes([.02 / W_COL, (y - mh) / H_COL, mw / W_COL, mh / H_COL])
    ax_.imshow(img); ax_.set_xticks([]); ax_.set_yticks([])
    for sp in ax_.spines.values():
        sp.set_visible(True); sp.set_linewidth(.4)
    cax = fig.add_axes([.02 / W_COL, (y - mh - .13) / H_COL, mw / W_COL, .05 / H_COL])
    cb = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(.8, 1.2), cmap='RdBu_r'), cax=cax, ticks=[.9, 1.0, 1.1], orientation='horizontal')
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(.3)
    fig.text((.02 + mw / 2) / W_COL, (y - mh - .27) / H_COL, '線量の比（シードあり／なし）、横断面', ha='center', va='top', fontsize=5.8)
    txt = (AS / 'seed_radial_600.txt').read_text().splitlines()
    pts = [re.match(r'([\d.]+) mm\s+([+-][\d.]+) ± ([\d.]+) %', t) for t in txt]
    pts = [(float(m.group(1)), float(m.group(2)), float(m.group(3))) for m in pts if m]
    assert len(pts) == 12 and abs(pts[0][1] - 3.34) < .005, pts[0]                  # 抄録 v5 の 3.3%
    cx0 = .02 + mw + .42
    ca = fig.add_axes([cx0 / W_COL, (y - mh) / H_COL, (W_COL - cx0 - .08) / W_COL, mh / H_COL])
    d, v, e = zip(*pts)
    ca.errorbar(d, v, yerr=e, fmt='o-', ms=2.2, lw=.7, capsize=1.2, elinewidth=.5, color=C_SYS)
    ca.axhline(0, color='k', lw=.4); ca.set_xlim(.4, 3.6)
    ca.set_xlabel('シードの軸からの距離 (mm)', labelpad=1, fontsize=5.8); ca.set_ylabel('線量の増加 (%)', labelpad=1, fontsize=5.8)
    ca.tick_params(length=1.5, pad=1, labelsize=5.5)
    ca.text(.97, .93, '最大 +3.3%', transform=ca.transAxes, ha='right', va='top', fontsize=6.2, color=C_NAVY, fontweight='bold')
    fig.text((cx0 + (W_COL - cx0 - .08) / 2) / W_COL, (y - mh - .27) / H_COL, '全シードの平均（600 バッチ）', ha='center', va='top', fontsize=5.8)
    y = y - mh - .5
    fig.text(.02 / W_COL, y / H_COL, '前立腺全体の平均線量は +0.5%。シード近傍の線量は EGS5 と\n未照合（研究・教育用）。', fontsize=5.8, color='#4b5563', va='top', linespacing=1.3)
    fig.savefig(OUT / 'fig_apps.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    result_fig(); method_fig(); apps_fig()
    print('ok', r2)
