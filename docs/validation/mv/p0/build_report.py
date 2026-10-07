"""Render RESULTS strictly from acquisition/check outputs; --check verifies reproducibility."""
from pathlib import Path
import json,sys
BASE=Path(__file__).resolve().parent
def read(name):return json.loads((BASE/name).read_text())
def report():
 ph=read('photon_check.json');es=read('estar_check.json');eg=read('egs5_check.json');sp=read('spectrum_check.json');perf=read('performance_check.json')
 total=ph['total_comparisons'];nist=[v for v in total if v['reference']=='bundled_NIST'];xr=[v for v in total if v['reference']=='xraylib']
 part={k:max(abs(v['relative_difference']) for v in ph['partial_comparisons'] if v['process']==k)*100 for k in ['photoelectric','compton','rayleigh']}
 synth={k:max(v['max_relative_difference'] for v in ph['synthesis'] if v['material']==k)*100 for k in ['water','air']}
 pair=ph['pair_thresholds'];c=eg['coupled'];k=eg['kerma'];low,high=perf['hours_range'];q=perf['reference_spectrum']
 s=f'''# 治療領域拡張 P0 — 成立性調査結果

対象計画: `docs/ai/plans/2026-10-07-mv-p0-feasibility.md`（approved）。
実施日: 2026-10-07。計算前に[事前登録](PREREGISTRATION.md)を固定し、
すべての成功runで計算前後のSHA256一致を確認した。

**10月13日の区分の判定案: 条件付き。** データ供給、線源の実ファイル、MV電子輸送の
基準系、非制限衝突阻止能と定性的ビルドアップは確保できた。
J1bの光電係数差の解釈、PEGS fit警告、未実装輸送の性能と期間の幅が残る。
J6の仮定付き上限は24時間予算を超える。これはP1着手の承認でも、10月16日の
水PDD/OCR合格記録でもない。最終の区分と次段階に進む判断はユーザーに委ねる。

## J1a〜J7 の値と判定

|項目|値・証拠|判定|
|---|---|---|
|J1a 全減弱係数|同梱NISTの1000–20000 keV全12点: 最大絶対相対差 {max(abs(v['relative_difference']) for v in nist)*100:.6f}%。xraylibの100–800 keV全8点: {max(abs(v['relative_difference']) for v in xr)*100:.6f}%。いずれも1%以内|合格|
|J1b 部分・合成|水の光電 {part['photoelectric']:.6f}%、Compton {part['compton']:.6f}%、Rayleigh {part['rayleigh']:.6f}%（最大絶対相対差）。質量比合成の最大差: 水 {synth['water']:.6f}%、空気 {synth['air']:.6f}%。対生成の閾値は下表|部分的確認・全体判定留保。光電差に対する許容差は計画にないので後付けしない|
|J2 利用条件|公式XCOM公開配布で取得可。配布ソースにnon-commercial使用許可、出典表示。SRD raw再配布は未確認、取得スクリプト・ハッシュで再現|今回の非商用研究利用について合格。商用利用・再配布へ一般化しない|
|J3 収支|電子輸送あり r={c['relative_residual']:.9g}、なし r={k['relative_residual']:.9g}。各20万履歴、未処理0、異常履歴0。基準 絶対残差≤1e−4|合格|
|J4 非制限衝突阻止能|ESTARとの最大絶対相対差 {max(abs(v['relative_difference']) for v in es['rows'])*100:.6f}%、0.1/1/10 MeVの全3点で2%以内|合格|
|J5 ビルドアップ|単色10 MeV、表層差 {eg['surface_difference_sigma_independent_approximation']:.6f} 合成SEM。SEMの和で評価しても {eg['surface_difference_over_sum_SEM']:.6f} 倍。電子輸送あり最大bin中心 {eg['dmax_bin_center_cm']:.1f} cm > 表面0.1 cm|定性的な試運転として合格。dmaxの位置・精度の検証はしていない|
|J6 性能の幅|Mohan公称10 MV、主評価点のSEM 0.5%に最大 {perf['maximum_required_histories']:.6g} historiesを要すると小標本から外挿。ChatCarlo未実装モデルの仮定付き範囲 {low:.3f}–{high:.3f} h、予算24 h|条件付き。仮定上限は予算超過、保証された上限ではない|
|J7 線源|Mohan 10 MV実ファイル、{sp['bins']} bins、0–10 MeV、0.5 MeV幅、mode=1、bin内一様。SHA256 `{sp['sha256']}`|合格|

## T1 データ経路と閾値

[選定表と利用条件](data_sources.md)、[全照合表](photon_comparison.csv)、
[機械可読照合](photon_check.json)。取得したNIST XCOM 3.1を変更せずscratchでビルドし、
公式の補間経路で元素・水・空気の係数を求めた。実ファイルを得た範囲外への外挿、
xraylibとの差をなくすための補正はしていない。

|過程|入力MeV（閾値未満）|係数 cm²/g|入力MeV（閾値直上）|係数 cm²/g|
|---|---:|---:|---:|---:|
|核場対生成|1.0219|{pair['nuclear_below']:.6g}|1.0221|{pair['nuclear_above']:.6g}|
|電子場対生成|2.0439|{pair['electron_below']:.6g}|2.0441|{pair['electron_above']:.6g}|

水は配布ATWTS.DATに基づくH/O合成と直接H2O、空気は取得したN/O/Ar/C質量比に基づく
直接mixtureと元素合成を比較した。合成は4有効桁の出力を用いるため厳密一致ではない。
J1aは共有基礎データによる経路整合性確認で、独立物理検証ではない。
J1bの「整合する」には部分係数の数値許容差が指定されておらず、光電差1.885%を
一律に合格とも不合格とも断定しない。次計画で物理モデル・精度の扱いを明示する必要がある。

## T2 EGS5 入力、PDD、収支

入力とログは[egs5/](egs5/)、PDDの2本は[pdd.csv](egs5/pdd.csv)と[pdd.png](egs5/pdd.png)。
各20万history、固定seed=20261007。水密度1.000 g/cm³、取得ESTAR組成、
EPSTFL=1（配布water_liquidの密度効果表）、IBOUND=1/INCOH=1/IRAYL=1。
PEGS AE=0.521 MeV全エネルギー、AP=0.001 MeV、UE=20.511 MeV、UP=20 MeV。
電子輸送ありECUT=0.521 MeV、なしECUT=21 MeV、共通PCUT=0.001 MeV。
電子を停止させる計算でも陽電子の停止後消滅光子はEGS5が輸送する。
したがって「全カーマ」はこの電子局所停止の比較設定を意味し、独立な解析kermaではない。

30×30×30 cm³水を深さ150層×横3×縦3の1350水領域に区切り、中央2×2 cm²の2 mm binへ
沈着を採点する。HOWFARで採点領域の境界までステップを制限し、長い電子ステップの
全沈着を1つの深さbinへ誤って振り分ける処理を避けた。
元の実績PDDユーザーコードからMAIN初期化を再利用し、幾何・採点・収支だけを新規入力に拡張。
EGS5本体は読取りのみで、scratchでビルド・実行した。

固定式 `E_in = E_dep + E_escape + E_unprocessed + ΔE_rest`、
`r = (E_in − E_dep − E_escape − E_unprocessed − ΔE_rest)/E_in`。
光子は全エネルギー、脱出荷電粒子は運動エネルギーを計上。
`ΔE_rest = 2 RM × (N_pair − N_annihilation)` を独立のイベント計数から求め、
`2 RM × N_escaped_positron`との一致も確認した。
生成・消滅に関与する媒質電子の静止質量を正味で差し引くので、停止陽電子の2RMを
沈着またはrest欄に二重加算しない。停止後の2消滅光子は光子エネルギーとして後段で採点。
カットオフ沈着はEGS5 EDEP、未処理はSHOWER後のstackから集計し、異常履歴を別に数える。

|収支項 (MeV)|電子輸送あり|電子輸送なし|
|---|---:|---:|
'''
 for field,label in [('incident_MeV','入射'),('deposited_MeV','沈着'),('escaped_kinetic_MeV','脱出（光子＋荷電粒子運動エネルギー）'),('unprocessed_MeV','未処理'),('net_rest_MeV','静止質量の正味変化'),('max_abs_residual_per_history_MeV','最大絶対収支残差/history')]:s+=f"|{label}|{c['ledger'][field]:.12g}|{k['ledger'][field]:.12g}|\n"
 s+=f'''
電子輸送ありの静止質量内訳: 生成 {c['rest_components_MeV']['pair_created']:.12g} MeV、
消滅 {c['rest_components_MeV']['annihilated']:.12g} MeV、脱出陽電子rest {c['rest_components_MeV']['escaped_positron_rest']:.12g} MeV。
同入力10kの反復c/dはPDD全150bin・収支・電子ステップ分布が完全一致。
同じseedでも電子停止の有無で履歴は分岐する。両runの共分散は未測定なので合成SEMは近似。
差/SEMの和も2を超えているため、この定性的表層差の結論は共分散の仮定で反転しない。

|単色10 MeVの位置|電子輸送あり/局所停止|比のSEM（独立近似）|
|---|---:|---:|
'''
 for v in eg['beyond_dmax_ratios']:s+=f"|{v['center_cm']:.1f} cm|{v['dose_to_local_kerma']:.6f}|{v['ratio_sem_independent_approximation']:.6f}|\n"
 s+='''
この比の大きなSEMを隠してdmax以深の一致と判定しない。PDD/OCRの正確さはP3以降の課題。

PEGS警告は全ログに保存。電子fit150区間、Rayleigh分布fit100区間について既定の相対精度
1%を達成できなかった警告が出た。MVでこの残存誤差を定量化できていない。
光子減弱の全データや電子輸送全体の検証が済んだとは扱わない。

## T3 阻止能

非制限はPEGSの`SPIONE(E0,E0)`、制限は`SPIONE(E0,AE)`をCALLで評価。
MeV/radiation length出力をRLC(cm)×ρ(g/cm³)で除した。
EPSTFL=1の密度効果は3点とも取得ESTARと一致し、組成・密度も同一。
制限のAE=0.521 MeV全エネルギーは明記するだけで、ESTARとの受入比較に使わない。

|運動エネルギー MeV|ESTAR非制限|PEGS非制限|相対差 %|PEGS制限 (MeV cm²/g)|
|---|---:|---:|---:|---:|
'''
 for v in es['rows']:s+=f"|{v['kinetic_MeV']:.1f}|{v['ESTAR_unrestricted_MeV_cm2_g']:.6f}|{v['PEGS_unrestricted_MeV_cm2_g']:.6f}|{v['relative_difference']*100:+.6f}|{v['PEGS_restricted_MeV_cm2_g']:.6f}|\n"
 s+=f'''
## T4 性能（実測と仮定を区別）

性能用線源は取得Mohan 10 MVの完全なhistogram。100kと200kを同固定seedのprefix拡張として実行。
結果を混ぜて40万historyなどとは数えない。主評価点は事前登録の表面、探索最大bin、5/10/20/29.9 cm。
5/10/20 cmはその深さを含むbinの中心5.1/10.1/20.1 cmで報告する。

|bin中心 cm|R (100k) %|R (200k) %|SEM(200k)/SEM(100k)|R=0.5%に必要なN（推定）|
|---|---:|---:|---:|---:|
'''
 for v in perf['points']:s+=f"|{v['depth_cm']:.1f}|{v['R_100k']*100:.4f}|{v['R_200k']*100:.4f}|{v['observed_SEM_200k_over_100k']:.4f}|{v['N_for_R_0_005']:.6g}|\n"
 s+=f'''
固定モデルのhistory統計に`R(N)=R(N0) sqrt(N0/N)`を適用。理想のSEM比は1/√2≈0.7071。
表面の観測比はこれより大きく、小標本の裾の不安定さが残る。最大binは全binから選んだ探索値なので
winner biasがあり、SEM 0.5%到達を保証しない。

公称10 MV一次光子あたりの電子・陽電子ステップ数: 平均 {q['charged_steps_mean']:.6f}、
median {q['charged_steps_quantiles']['0.5']}、90% {q['charged_steps_quantiles']['0.9']}、
99% {q['charged_steps_quantiles']['0.99']}、最大 {q['charged_steps_quantiles']['1.0']}。
全分布は[spectrum_charged_steps.csv](egs5/spectrum_charged_steps.csv)。
単色10 MeVでは平均 {c['charged_steps_mean']:.6f}、最大 {c['charged_steps_quantiles']['1.0']}。

既存ChatCarlo `kernel.py`の60 keV水・1 thread・JIT除外で光子正距離segment単価
{perf['photon_unit_seconds_range'][0]*1e6:.6f}–{perf['photon_unit_seconds_range'][1]*1e6:.6f} μsを実測。
輸送単価の分母は既存segment記録経路で数え、非記録経路と乱数・沈着が完全一致することを確認。
MVでの測定値ではない。[計測JSON](kernel_measurement.json)に全反復を残す。

EGS5公称10 MVの200k実測wall {q['wall_seconds']:.6f} s、実行プロセスpeak RSS {q['rss_bytes']/1e6:.6f} MB。
ビルドを含む子プロセス全体のRSSとは区別し、os.wait4のrun単独rusageを使用。
中央PDDの採点を無効にした同条件との差はCPU {perf['scoring_cpu_seconds_matched_difference']:.6f} s、
1履歴あたり {perf['scoring_unit_seconds_point']*1e6:.6f} μs。
100k/200kの単価変動も含めた採点の感度幅は0–{perf['scoring_unit_seconds_range'][1]*1e6:.6f} μs/history。
これは1対の差分で、信頼区間ではない。AUSGAB内の計時は計時自体の負荷が大きいので
実採点時間の値には使わない。EGS5採点・モーメント・steps histogramの主要配列は
{perf['scoring_array_bytes']} bytes、仮に30 cm立方を2 mmの3個のfloat64採点配列で保持すると
{perf['grid_2mm_30cm_cube_dose_and_moments_bytes']/1e6:.3f} MB（割付見積もり、未実装）。

未実装の電子単価を測定光子単価の1–20倍、二次粒子stackを輸送単価の5–50%増、
細かな採点を上記採点単価上端の1–10倍とした**工学的感度仮定**:

```
C_low = (Sγ + Se) cγ_min × 1.05
C_high = (Sγ + 20 Se) cγ_max × 1.5 + 10 cscore_high
T = max(N_required) × C
```

SγはEGS5の採点境界を含む光子ステップ、Seは荷電粒子ステップ。
未実装コストを実測値と称していない。得られた幅は {low:.3f}–{high:.3f} h。
EGS5自身の現在単価の外挿は {perf['EGS5_wall_extrapolation_hours']:.3f} h。
24時間内に入るための将来ChatCarlo平均単価の目安は {perf['budget_seconds_per_history']*1e6:.3f} μs/history。
この上端は確率的／数学的な上限ではなく、将来の実装・細かい境界条件で超え得る。

上限超過への具体策（次計画の候補、P0では実装しない）:

- 多重散乱・阻止能を前計算したNumbaスカラー輸送、step単位allocationを避ける固定容量stack。
- 同一媒体の採点境界を輸送の物理境界から分離し、既存解析DDAで採点する。
- historyを独立seed系列でCPUプロセスへ分割。単一threadのこの概算に未測定のspeedupを掛けない。
- 水・箱・今回の10×10 cm²に限定した試作で電子step単価とSEMの裾を実測し、次子計画の予算を再固定する。

## T5 線源、T6 モデル、期間

Mohanファイルのfluence加重平均は {sp['fluence_weighted_mean_MeV']:.9f} MeV（取得値からbin内一様で計算）。
NRC配布から実ファイルとparserを取得できたため、代替線源探索への切替は不要だった。
再配布条件の未確定部分と0–0.5 MeV binのカットオフ処理は[data_sources.md](data_sources.md)に明記。
第二の輸送コードは導入していない。

[モデル選定表](model_selection.md)に各モデルの候補・出典・最短/標準/難航時の幅を記録した。
全体の工学的目安は最短8–12作業週、標準16–24週、難航32–48週。
標準上限は2027年3月までに収まり切らない可能性があり、未検証の多重散乱・境界処理を
先行試作して幅を狭める経路が必要。単一の日数や成功確率で成立を保証しない。

## 未解決事項と想定外の挙動

1. J1b光電部分の1.885%差の説明と、「整合する」の定量的な受入定義は未確定。補正・外挿はしていない。
2. 電子fitとRayleigh分布fitのPEGS警告は未定量。PDD dmaxの精度、線量分布の独立検証、境界依存性は未確認。
3. 小標本SEMの裾と最大bin選択のバイアス、将来輸送モデルの分散・step数・stack負荷は未検証。
4. XCOM SRDとMohan由来rawの再配布条件、商用でのXCOM利用条件は未確定。rawは追跡しない。
5. 最初のPython urllibによるNIST取得はHTTP 403。curlに変更して公式取得が成功した。spectrum READMEとheader候補は404だったが、実ファイルとparser取得は成功。
6. 初回EGS5ビルドは150binへの拡張でF77の72列を超えた宣言が原因で失敗。継続行へ修正し、新runへ移した。
7. /usr/bin/time -lはsandboxのsysctl制限でexit 1、psもPermissionError。これらを正常物理runの証拠として使わず、直接起動とos.wait4へ変更。
8. stopping準備の最初のRM定数探索は失敗。配布DERCONの式と定数を取得して修正。DECK後のCALLはEPSTFL等が初期値に戻るため−9%/−13%の誤比較になった。DECK前へ移したstopping_v3だけをJ4に採用。データ補正や閾値変更はしていない。
9. 同seedによるcoupled/kermaの共分散は未測定。SEM和の上界によって表層差の定性的結論も併記した。
10. P0は成立性資料の作成まで。P1以降の実装、Fano試験、MV本番輸送、位相空間・測定との比較は未実施。

T1の公開データ経路は得られ、T2は原因調査1日以内に輸送と収支を成立させたため、
停止条件による作業打切りは発生しなかった。判定不能部分を推測で合格にしない。
期限前のこの結果を10月13日の判断材料として保存する。

## 再現コマンドとテスト

指定された全コマンドを実行した。回帰は **403 passed, 2 warnings**。
[全ログ](regression_test.log)の警告は既存のanode cutoff時のSpekPy zero emissionと、SpekPy不在を模擬するKramers fallback各1件。
`git diff --stat -- chatcarlo tests scripts examples`は空。
作業開始時から存在したCLAUDE.md/README/既存結果文書などの差分は変更せず残した。
コミットは実施していない。

```bash
.venv/bin/python -m pytest tests/ -q
git diff --stat -- chatcarlo tests scripts examples
.venv/bin/python docs/validation/mv/p0/check_photon_data.py
.venv/bin/python docs/validation/mv/p0/check_estar.py
.venv/bin/python docs/validation/mv/p0/analyze_egs5.py
# 追加のP0照合・文書再現
.venv/bin/python docs/validation/mv/p0/check_spectrum.py
.venv/bin/python docs/validation/mv/p0/check_performance.py
.venv/bin/python docs/validation/mv/p0/build_report.py --check
```

キャッシュをP0内に保つため回帰時にNUMBA_CACHE_DIRとPYTHONDONTWRITEBYTECODEを設定した。
照合スクリプトのexit 0は保存されたデータからの再解析成功を示す。
J1b留保やJ6予算超過をexit 0によって合格に読み替えない。
取得からEGS5入力生成・ビルドの再実行は[実行手順](egs5/README.md)を参照。
manifestに入力・スクリプト・データ・ログのSHA256、取得URL、配布EGS5ソースツリーのhashを保存。
'''
 return s
if __name__=='__main__':
 content=report();path=BASE/'RESULTS.md'
 if '--check' in sys.argv:
  assert path.read_text()==content,'RESULTS.md differs from saved primary/check data';print('RESULTS.md reproduced exactly')
 else:path.write_text(content);print(path)
