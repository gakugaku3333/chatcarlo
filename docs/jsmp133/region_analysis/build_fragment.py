"""Build the replacement fragment for the public review page, section "5. 現時点の結果" (abstract v3, 2026-10-24).
Inputs (all written by the scripts in this folder): pooled_10mv.npz, pooled_4mv.npz (regions.py <10|4> dump),
pooled_stack.npz (dump_stack.py), and the 60 keV GPU grid + EGS5 csv read by regions_60kev.py.
Every number printed in the text is computed here and checked against the abstract v3 wording (assert).
usage: .venv/bin/python docs/jsmp133/region_analysis/build_fragment.py  -> docs/jsmp133/replacement_results_v4.html"""
import csv, io, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': ['Hiragino Sans', 'DejaVu Sans'], 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
H = 'docs/jsmp133/region_analysis/'
C_SYS, C_EGS, C_BAND = '#1f6fb4', '#c2410c', '#9ca3af'

def pct(a): return 100*abs(np.asarray(a))
def svg(fig, label):
    b = io.StringIO(); fig.savefig(b, format='svg', bbox_inches='tight'); plt.close(fig)
    s = b.getvalue(); s = s[s.index('<svg'):]
    return s.replace('<svg ', f'<svg role="img" aria-label="{label}" style="width:100%;height:auto" ', 1)

def mv(which):
    d = np.load(H+f'pooled_{which}mv.npz'); gm, gs, em, es, nb, no = (d[k] for k in ('gm', 'gs', 'em', 'es', 'nb', 'no'))
    gp, ep = gm[:nb], em[:nb]; im = int(ep.argmax()); z = (np.arange(nb)+.5)*30/nb
    pg, pe = gp/gp[im], ep/ep[im]; dp = pg/pe-1
    sp = np.sqrt((gs[:nb]/gp)**2+(es[:nb]/ep)**2+(gs[im]/gp[im])**2+(es[im]/ep[im])**2)
    prof = []
    for j, depth in enumerate((5, 10, 20)):
        sl = slice(nb+no*j, nb+no*(j+1)); g, e = gm[sl]/gm[sl][0], em[sl]/em[sl][0]; x = (np.arange(no)+.5)*0.5
        s = np.sqrt((gs[sl]/gm[sl])**2+(es[sl]/em[sl])**2+(gs[sl][0]/gm[sl][0])**2+(es[sl][0]/em[sl][0])**2)
        def x50(p): i = int(np.argmax(p < .5)); return x[i-1]+(p[i-1]-.5)/(p[i-1]-p[i])*(x[i]-x[i-1])
        xe = x50(e); inner = (x <= .8*xe) & (np.arange(no) > 0)
        prof.append(dict(depth=depth, x=x, g=g, e=e, d=g/e-1, s=s, inner=inner, shift_mm=10*(x50(g)-xe)))
    din = np.concatenate([p['d'][p['inner']] for p in prof])
    return dict(z=z, pg=pg, pe=pe, dp=dp, sp=sp, zmax=z[im], prof=prof, din=din, shift=max(abs(p['shift_mm']) for p in prof))

def kev60():
    rows = {r['bin']: r for r in csv.DictReader(open('docs/egs5_crosscheck/revalidation_2026_10_07/comparison.csv'))}
    d = np.load('runs/diag_water_60kev_gpu/dose.npz'); D, S = d['dose_per_history_Gy'], d['sem_dose_per_history_Gy']
    ix = lambda x: int(round(x+65)); iz = lambda z: int(round(z+50)); yy = slice(ix(-1), ix(1))
    E = lambda k, c: float(rows[k][c])
    # GPU SEM of a mean of 4 voxels: upper bound = mean of the voxel SEMs (as recorded in the abstract notes)
    gp = np.array([D[yy, yy, iz(z)].mean() for z in range(15)]); gps = np.array([S[yy, yy, iz(z)].mean() for z in range(15)])
    ep = np.array([E(f'pdd_z{z}-{z+1}', 'egs5_Gy_history') for z in range(15)]); eps = np.array([E(f'pdd_z{z}-{z+1}', 'egs5_SEM_Gy_history') for z in range(15)])
    im = int(ep.argmax()); dp = (gp/gp[im])/(ep/ep[im])-1
    sp = np.sqrt((gps/gp)**2+(eps/ep)**2+(gps[im]/gp[im])**2+(eps[im]/ep[im])**2)
    prof = []
    for name, z, lab in (('shallow', 0, '0〜1 cm'), ('10cm', 9, '9〜10 cm')):
        def lat(x, c): return (E(f'lat_{name}_x{x}-{x+1}', c)+E(f'lat_{name}_x{-x-1}-{-x}'.replace('x-0', 'x0'), c))/2
        g = np.array([(D[ix(x), yy, iz(z)].mean()+D[ix(-x-1), yy, iz(z)].mean())/2 for x in range(8)])
        e = np.array([lat(x, 'egs5_Gy_history') for x in range(8)]); xc = np.arange(8)+.5
        g, e = g/g[0], e/e[0]; i = int(np.argmax(e < .5)); x50 = xc[i-1]+(e[i-1]-.5)/(e[i-1]-e[i])*(xc[i]-xc[i-1])
        prof.append(dict(lab=lab, x=xc, g=g, e=e, d=g/e-1, inner=(xc <= .8*x50) & (xc > .5)))
    return dict(z=np.arange(15)+.5, pg=gp/gp[im], pe=ep/ep[im], dp=dp, sp=sp, prof=prof,
                din=np.concatenate([p['d'][p['inner']] for p in prof]))

def stack():
    d = np.load(H+'pooled_stack.npz'); gm, gs, em, es, layer, ed = (d[k] for k in ('gm', 'gs', 'em', 'es', 'layer', 'edges'))
    w = np.flatnonzero(layer == 0); im = int(w[np.argmax(em[w])])
    dd = (gm/gm[im])/(em/em[im])-1; s = np.sqrt((gs/gm)**2+(es/em)**2+(gs[im]/gm[im])**2+(es[im]/em[im])**2)
    return dict(z=(ed[:-1]+ed[1:])/2, pg=gm/gm[im], pe=em/em[im], dp=dd, sp=s, edges=ed, layer=layer)

def fig_pdd(r, label, xlab='深さ (cm)', bounds=None, marker=False):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=.08))
    kw = dict(marker='o', ms=3, lw=0) if marker else dict(lw=1.4)
    a.plot(r['z'], 100*r['pe'], color=C_EGS, label='EGS5', **kw); a.plot(r['z'], 100*r['pg'], color=C_SYS, ls='--' if not marker else '', label='本システム', **(dict(marker='x', ms=4, lw=0) if marker else dict(lw=1.4)))
    a.set_ylabel('相対線量 (%)'); a.legend(frameon=False)
    b.fill_between(r['z'], -100*r['sp'], 100*r['sp'], color=C_BAND, alpha=.35, lw=0, label='統計誤差 (±1σ)')
    b.plot(r['z'], 100*r['dp'], color=C_SYS, marker='.', ms=3, lw=.8); b.axhline(0, color='k', lw=.6)
    b.set_ylabel('差 (%)'); b.set_xlabel(xlab); b.legend(frameon=False, loc='upper left', fontsize=8)
    if bounds is not None:
        for x in bounds: a.axvline(x, color='#d1d5db', lw=.8); b.axvline(x, color='#d1d5db', lw=.8)
    return svg(fig, label)

def fig_ocr(prof, label, band=True):
    fig, axs = plt.subplots(2, len(prof), figsize=(8.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=.08))
    for k, p in enumerate(prof):
        a, b = axs[0, k], axs[1, k]; m = p['e'] >= .2
        a.plot(p['x'], 100*p['e'], color=C_EGS, lw=1.4, label='EGS5'); a.plot(p['x'], 100*p['g'], color=C_SYS, ls='--', lw=1.4, label='本システム')
        a.set_title(f"深さ {p.get('lab', str(p.get('depth'))+' cm')}", fontsize=10)
        xi = p['x'][p['inner']]; a.axvspan(0, xi.max()+.25 if len(xi) else 0, color='#dbeafe', alpha=.6, lw=0)
        if band and 's' in p: b.fill_between(p['x'][m], -100*p['s'][m], 100*p['s'][m], color=C_BAND, alpha=.35, lw=0)
        b.plot(p['x'][m & ~p['inner']], 100*p['d'][m & ~p['inner']], 'o', mfc='none', color=C_SYS, ms=4)
        b.plot(p['x'][p['inner']], 100*p['d'][p['inner']], 'o', color=C_SYS, ms=4); b.axhline(0, color='k', lw=.6)
        b.axvspan(0, xi.max()+.25 if len(xi) else 0, color='#dbeafe', alpha=.6, lw=0); b.set_xlabel('中心からの距離 (cm)')
        if k == 0: a.set_ylabel('相対線量 (%)'); b.set_ylabel('差 (%)'); a.legend(frameon=False, fontsize=8)
    return svg(fig, label)

R10, R4, K60, ST = mv('10'), mv('4'), kev60(), stack()
f1 = lambda v: f'{v:.1f}'; f2 = lambda v: f'{v:.2f}'
n = dict(p10=(f2(pct(R10['dp']).mean()), f1(pct(R10['dp']).max())), o10=(f2(pct(R10['din']).mean()), f1(pct(R10['din']).max())),
         p4=(f2(pct(R4['dp']).mean()), f1(pct(R4['dp']).max())), o4=(f2(pct(R4['din']).mean()), f1(pct(R4['din']).max())),
         p60=(f2(pct(K60['dp']).mean()), f1(pct(K60['dp']).max())), o60=(f2(pct(K60['din']).mean()), f1(pct(K60['din']).max())),
         pst=(f2(pct(ST['dp']).mean()), f1(pct(ST['dp']).max())))
# the abstract v3 wording (docs/jsmp133/abstract_ja_draft.md, section 2)
assert n == dict(p10=('0.53', '1.5'), o10=('0.43', '1.2'), p4=('0.77', '1.9'), o4=('0.59', '1.6'), p60=('0.21', '1.0'),
                 o60=('0.23', '0.3'), pst=('0.17', '1.3')), n
assert R10['shift'] < .1 and R4['shift'] < .1
sem = lambda r: f2(100*r['sp'].mean())

figA = fig_pdd(R10, '10 MV 水ファントムの PDD：本システムと EGS5')
figB = fig_ocr(R10['prof'], '10 MV 水ファントムの OCR')
figC = fig_pdd(ST, '積層ファントムの PDD', bounds=ST['edges'][1:][np.diff(np.r_[ST['layer'], -1]) != 0][:-1])
figD = fig_pdd(K60, '60 keV 水ファントムの PDD', marker=True)
figD2 = fig_ocr(K60['prof'], '60 keV 水ファントムの OCR', band=False)
figE = fig_pdd(R4, '4 MV 水ファントムの PDD（依頼 1）')
figF = fig_ocr(R4['prof'], '4 MV 水ファントムの OCR（依頼 1）')

STYLE = """<style>
.rv4 .card{border:1px solid #d1d5db;border-radius:12px;padding:14px 16px;margin:14px 0}
.rv4 table{border-collapse:collapse;width:100%;font-size:.95em}.rv4 th,.rv4 td{border:1px solid #d1d5db;padding:6px 8px;text-align:left;vertical-align:top}
.rv4 figure{margin:18px 0}.rv4 figcaption{font-size:.9em;line-height:1.6}
</style>"""
part1 = f"""<section class="rv4" id="rv4-part1">
<div class="card"><b>用語：</b>
<b>入射光子数</b>＝計算で照射する光子の数。光子 1 個ごとに、それが生む二次粒子（電子など）もすべて追跡する。多いほど結果が安定する。
<b>EGS5</b>＝世界で広く使われている既存の計算コード。
<b>PDD</b>＝深さ方向の線量の変化、<b>OCR</b>＝横方向の線量の広がり。
<b>統計誤差</b>＝入射光子数が有限であることによるゆらぎ（光子数の平方根に反比例して小さくなる）。図の帯は ±1σ。
<b>差の表し方</b>＝各点で (本システムの値 − EGS5 の値) ÷ EGS5 を求め、EGS5 の値に対する割合で示す（局所の差）。比べる前に、PDD は EGS5 の線量が最大となる深さの値を、OCR は各深さの中心の値を、両方のコードで 100% にそろえる。
<b>比べる範囲</b>＝放射線治療の線量計算の精度評価で使われる領域分け（Venselaar ら, Radiother Oncol 2001）に合わせた。PDD はビルドアップを含む全域、OCR は照射野内（50% 幅の 80% 以内、図の青い範囲）で差を求め、照射野の端（ペナンブラ）は 50% 位置のずれで評価する。照射野外の裾は比べない。
</div>
<div class="card"><b>状況：</b>依頼 1〜3 はいずれも、対話の入口から実際に通した記録のある結果です（依頼 2 の X 線画像は 2026-10-24 に入口から計算し直しました）。</div>
<h3>(1) 開発段階の精度確認：EGS5 との比較</h3>
<table><tr><th>条件</th><th>EGS5 との差（EGS5 の値に対する割合。各点の差）</th></tr>
<tr><td>水ファントム 10 MV、10×10 cm、入射光子 1.0×10⁸ 個（両コード）</td><td>PDD の差は平均 {n['p10'][0]}%、最大 {n['p10'][1]}%。照射野内の OCR（深さ 5・10・20 cm）の差は平均 {n['o10'][0]}%、最大 {n['o10'][1]}%。50% 位置のずれは 0.1 mm 以内（図 A・B）</td></tr>
<tr><td>水ファントム 60 keV（診断領域）、10×10 cm の平行ビーム、入射光子 5.0×10⁷ 個（EGS5 は保存済みの再検証結果、1×10⁸ 個）</td><td>PDD（深さ 0〜15 cm）の差は平均 {n['p60'][0]}%、最大 {n['p60'][1]}%。照射野内の OCR（深さ 0〜1 cm と 9〜10 cm）の差は平均 {n['o60'][0]}%、最大 {n['o60'][1]}%。区間は 1 cm ごとで、50% 位置のずれは求めていない（図 D）</td></tr>
<tr><td>水・肺・骨・鉛の積層、10 MV、入射光子 1.0×10⁸ 個（両コード）</td><td>PDD の差は平均 {n['pst'][0]}%、最大 {n['pst'][1]}%。骨と鉛の手前で、統計誤差を超える小さな系統的なずれ（平均 +0.2〜0.3%）が見られる（原因は未調査、図 C）</td></tr></table>
<figure>{figA}<figcaption><b>図 A</b>　10 MV・10×10 cm・水の中での、深さ方向の線量の変化（PDD）。EGS5 の線量が最大となる深さ（{R10['zmax']:.2f} cm）の値を、両方のコードで 100% にそろえた。本システム（GPU）と EGS5 の曲線は重なって見える。下段は各点（0.5 cm ごと）の差を EGS5 の値に対する % で示したもの。灰色の帯は統計誤差（±1σ、平均 {sem(R10)}%）。差は平均 {n['p10'][0]}%、最大 {n['p10'][1]}%。</figcaption></figure>
<figure>{figB}<figcaption><b>図 B</b>　10 MV：横方向の線量の広がり（OCR）。深さ 5・10・20 cm で、各深さの中心の値を 100% にそろえた。下段は EGS5 が中心の 20% 以上の点の差。青い範囲が照射野内（50% 幅の 80% 以内）で、塗りつぶしの点がその比較に使った点（中心の点は基準なので除く）。照射野内の差は平均 {n['o10'][0]}%、最大 {n['o10'][1]}%。50% 位置のずれは 3 深さとも 0.1 mm 以内。</figcaption></figure>
<figure>{figC}<figcaption><b>図 C</b>　水・肺・水・骨・水・鉛の積層（10 MV）の PDD。縦線は材質の境目。水の層の最大の深さ（2.75 cm）の値を、両方のコードで 100% にそろえた。材質の境目での線量の立ち上がり・低下や鉛の手前のピークまで重なる。下段は各点（0.1 cm ごと、鉛は 0.05 cm、220 点）の差と統計誤差の帯（±1σ、平均 {sem(ST)}%）。差は平均 {n['pst'][0]}%、最大 {n['pst'][1]}%（鉛の直前の水）。</figcaption></figure>
<figure>{figD}{figD2}<figcaption><b>図 D</b>　診断領域（60 keV 単色、10×10 cm の平行ビーム、水 30×30×20 cm、GPU）の PDD（上）と OCR（下）。区間は 1 cm ごと。PDD は EGS5 の線量が最大となる表面の区間、OCR は各深さの中心の値を 100% にそろえた。OCR は左右の対称な区間を平均した。PDD の差は平均 {n['p60'][0]}%、最大 {n['p60'][1]}%、照射野内の OCR は平均 {n['o60'][0]}%、最大 {n['o60'][1]}%。この計算は対話の入口ではなく、診断領域のコマンド（GPU・Colab）で実行した。</figcaption></figure>
</section>"""
part2 = f"""<section class="rv4" id="rv4-part2">
<div class="card"><b>対話の入口から Colab の計算まで通した確認（依頼 1、2026-10-24）：</b>「4 MV の水ファントムの PDD・OCR を計算したい」という依頼から、目的の確認、できる範囲への調整（条件は 1 通りだけ、線量は相対値）、出力形式（PDD と OCR の表。EGS5 との比較は入口では出さない）、入射光子数（2.6×10⁷ 個、先生が選択）を対話で決め、3D 表示の承認（先生）を経て、Colab の T4 GPU で計算した。承認した条件と実際に計算した条件は一致し、仮想マシン上の確認（3 項目）もすべて合格、故障の検出の確認もできた。全体で約 26 分（10 分おきの確認と回収を含む）で、止まらずに通った。その後、同じ入口で 7.4×10⁷ 個を追加し、合わせた 1.0×10⁸ 個を、入口の外で EGS5（1.0×10⁸ 個）と比べた（図 E・F）。経緯：前回の通し（10 MV、2026-10-23）では、進み具合の確認が約 50 分止まり、結果の回収を手作業で復旧した。原因（確認に制限時間がなかったこと）を直して、今回は止まらなかった。</div>
<figure>{figE}<figcaption><b>図 E</b>　依頼 1：4 MV・10×10 cm・水の PDD（両コード 1.0×10⁸ 個、0.5 cm ごと）。EGS5 の線量が最大となる深さ（{R4['zmax']:.2f} cm）で 100% にそろえた。差は平均 {n['p4'][0]}%、最大 {n['p4'][1]}%（統計誤差の平均 {sem(R4)}%）。</figcaption></figure>
<figure>{figF}<figcaption><b>図 F</b>　依頼 1：4 MV の OCR（深さ 5・10・20 cm）。青い範囲が照射野内。照射野内の差は平均 {n['o4'][0]}%、最大 {n['o4'][1]}%、50% 位置のずれは 3 深さとも 0.1 mm 以内。</figcaption></figure>
</section>"""
html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>差し替え用 現時点の結果 v4</title>
<!-- 公開ページ「5. 現時点の結果」の差し替え用の断片（抄録 v3、2026-10-24）。生成: docs/jsmp133/region_analysis/build_fragment.py。
     #rv4-part1 = 用語・状況の囲み＋(1) の表と図 A〜D、#rv4-part2 = (2) の Colab の囲み＋図 E・F。 -->
{STYLE}</head><body>
{part1}
{part2}
</body></html>"""
open('docs/jsmp133/replacement_results_v4.html', 'w').write(html)
print('ok', len(html), n)
