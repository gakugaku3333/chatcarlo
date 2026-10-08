"""Generate RESULTS and manifest from measured values, never hand-enter numeric results."""
from pathlib import Path
import json,hashlib
import numpy as np
from .component_checks import OUT,ROOT,SEED
def read(name):return json.loads((OUT/(name+'.json')).read_text())
def pct(x):return f'{100*x:.6g}%'
def main():
    c=read('components');g=read('gs_checks');f=read('fixed_checks');t=read('transport_checks');e=read('egs5_checks');diag=read('diagnostics');b=read('benchmark');w=read('work_log')
    status={**{k:c[k]['status'] for k in ['K1','K2']},'K3':g['K3']['status'],'K4':e['K4']['status'],'K5':t['K5']['status'],'K5b':t['K5b']['status'],'K6':e['K6']['status'],'K7':t['K7']['status'],'K8':t['K8']['status'],'K9':'pass' if f['K9i']['status']==t['K9ii']['status']=='pass' else 'fail','K10':'参考のみ','K11':'未実施（任意）'}
    passed=all(status[k]=='pass' for k in ['K1','K2','K3','K5','K5b','K6','K7','K8','K9'])
    metrics={}
    metrics['K1']=f"全{c['K1']['points']}点の最大相対差 {pct(c['K1']['max_relative'])}"
    rr=c['K2']['rows'];sr=[x for x in rr if 'chi_p' in x]
    metrics['K2']=f"積分最大相対差 {max(x['integral_relative'] for x in rr):.6g}；自由行程最大差 {pct(max(x['free_path_relative'] for x in sr))}；χ²最小p {min(x['chi_p'] for x in sr):.6g}；最小期待度数 {min(x['min_expected'] for x in sr):.6g}；最大運動量残差 {max(x['momentum_residual_MeV_c'] for x in sr):.6g} MeV/c；最大エネルギー残差 {max(x['energy_residual_MeV'] for x in sr):.6g} MeV"
    gr=g['K3']['rows'];sh=g['K3']['short'][0]
    metrics['K3']=f"20条件のLegendreモーメント最大相対差 {pct(max(abs(v) for x in gr for v in x['relative']))}；規格化最大誤差 {g['K3']['normalization_max_error']:.6g}；CDF単調 {g['K3']['cdf_monotone']}；短距離χ²p {sh['chi_p']:.6g}、p0={sh['p0_table']:.12g}"
    er=e['K4']['rows']
    metrics['K4']=f"線量最大変化 {pct(max(v['max_dose_difference_over_Dref'] for v in er.values()))}；R50最大変化 {pct(max(abs(v['range_difference_over_reference_R50'][0]) for v in er.values()))}；3条件で実刻み分布の変化を確認"
    metrics['K5']=f"全体残差/入射 {t['K5']['global_relative']:.6g}；履歴最大 {t['K5']['max_history_relative']:.6g}；未処理粒子 {t['K5']['unprocessed_particles']}、残量 {t['K5']['unprocessed_MeV']} MeV；stack/step失敗 {t['K5']['stack_failures']}/{t['K5']['step_limit_failures']}"
    metrics['K5b']='解析的bin配分、hinge前後の脱出、Møller精算、最終残区間、cutoff、面上・逆向き・平行・同時ゼロイベント、親状態復元、異常終了を単体検査'
    k6=e['K6'];metrics['K6']=f"評価{ k6['evaluation_bins']}binの最大線量差 {pct(k6['max_dose_difference_over_Dref'])}（許容3%）；R50差 {pct(k6['range_difference_over_reference_R50'][0])}、Rp差/R50 {pct(k6['range_difference_over_reference_R50'][1])}"
    for key in ['K7','K8']:
        rows=t[key]['rows'];metrics[key]=f"最大線量変化 {pct(max(v['max_dose_difference_over_Dref'] for v in rows.values()))}；最大R50変化 {pct(max(abs(v['range_difference_over_reference_R50'][0]) for v in rows.values()))}"
    fi=f['K9i']['rows'];kii=t['K9ii'];metrics['K9']=f"(i) 横変位二乗平均最大差 {pct(max(abs(x['relative'][0]) for x in fi))}、深さ変位平均 {pct(max(abs(x['relative'][1]) for x in fi))}；(ii) 2/5 mm面の差 {', '.join(pct(x) for x in kii['relative'])}"
    k10=e['K10'];metrics['K10']=f"最大線量差 {pct(k10['max_dose_difference_over_Dref'])}；R50差 {pct(k10['range_difference_over_reference_R50'][0])}；Rp差/R50 {pct(k10['range_difference_over_reference_R50'][1])}"
    metrics['K11']='実施せず。親計画のFanoゲートを満たした記録ではない。'
    summary={'status':status,'metrics':metrics,'overall':'2 MeV・水・指定条件で電子輸送の試作が成立' if passed else '総合不成立：K6が条件を満たさない','adopted_fe':.05,'adopted_K1_max':.05,'radiation':e['radiation'],'benchmark':b,'unresolved':diag['hypotheses']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['# P1 RESULTS','',summary['overall']+'。判定ルールは計算前のPREREGISTRATION.mdを変更していない。K6不一致を合わせるための調整・方式の差し替えはしていない。','',
    '試作は水・平板・電子単独の研究用。放射損失の全量局所沈着は比較専用で、物理的なMV線量計算の検証記録ではない。','',
    '|項目|値|判定|','|---|---|---|']
    for k in status:lines.append(f'|{k}|{metrics[k]}|{status[k]}|')
    k9eg=e['K9_EGS_reference_only']
    lines+=['',f"K9(ii)のEGS5比較（記録のみ）: EGS5 {k9eg['EGS5_cm2']} cm²、試作の相対差 {', '.join(pct(x) for x in k9eg['relative'])}。EGS5通過数 {k9eg['EGS5_cross_N']}。K9の合否には使わない。",'','## 定義、標本と統計','',
    'q_i=E_i/(N ρ Δz_i)、MeV cm²/g/一次電子。深さはbin中心、面積で割らない。2 MeVは12 mm/0.1 mm、10 MeVは60 mm/0.5 mm。全子孫を一次履歴へまとめ、100個の等標本数batchからSEMを計算。','',
    '解析はanalysis.pyだけを使用。Drefは基準曲線の端部補正5bin平均の最大。R50は生曲線の最大より深い側の最初の50%交点。Rpは下降側20–80%の連続区間への直線fit（複数区間なら最長、同長なら浅い側）。比較相手からDrefを選び直さない。R50/Rp差のSEMは固定seedのbatch bootstrap。K4/K7/K8の合否はR50、K6はR50とRp。差のSEMは許容幅の1/3以下を必須とした。','',
    '|比較|最大bin差SEM/Dref|R50差SEM/R50|Rp差SEM/R50|統計要求|','|---|---|---|---|---|']
    for name,r in [('K6',k6),*[(f'K4 {n}',r) for n,r in er.items()],*[(f'{key} {n}',r) for key in ['K7','K8'] for n,r in t[key]['rows'].items()]]:
        lines.append(f"|{name}|{pct(r['max_dose_difference_SEM_over_Dref'])}|{pct(r['range_difference_SEM_over_R50'][0])}|{pct(r['range_difference_SEM_over_R50'][1])}|{r['statistics_ok']}|")
    lines+=['','K9(i)は各条件100万履歴。最初の20万履歴はSEM不足で合格に数えず、履歴数だけ増やした。K9(ii)の試作/逐次散乱の条件付け標本数とSEM：','',
    f"- 通過数 試作 {kii['prototype_cross_N']}、逐次散乱 {kii['reference_cross_N']}。",
    f"- x²+y² 試作 {kii['prototype_cm2']} cm²、逐次散乱 {kii['reference_cm2']} cm²。",
    f"- 差の相対SEM {kii['difference_SEM_relative']}（許容3%/3=1%以下）。",
    '', 'K2(a)は閾値の直下・一致・次の浮動小数点値の直上を含む。K2(b,c)は各指定エネルギー100万標本。K3は指定20条件に加え、表の全格子の規格化/CDFを検査。短距離はlog(1−cosθ+2η_min)の等間隔20binで、無散乱atomを解析的に保持し、散乱を条件付けた1000万標本を使用した。全binの期待度数は100以上。','',
    '## EGS5基準と放射損失条件','',e['radiation']['adopted']+'。通常高AP条件が成立したためIUNRST=4候補へ切り替えていない。試作は取得したESTAR放射阻止能を連続損失へ加算。','',
    f"基準の全体収支相対残差 {e['radiation']['reference_balance_relative']:.6g}、履歴最大残差 {e['radiation']['reference_max_history_residual_MeV']:.6g} MeV。未処理粒子 {e['radiation']['unprocessed_particles']}、光子callback {e['radiation']['photon_callbacks']}。",
    '', '|T [MeV]|PEGS放射阻止能|ESTAR放射阻止能|相対差|','|---|---|---|---|']
    for r in e['radiation']['PEGS_ESTAR']:lines.append(f"|{r['T']}|{r['PEGS_radiative_MeV_cm2_g']:.9g}|{r['ESTAR_radiative_MeV_cm2_g']:.9g}|{pct(r['relative'])}|")
    lines+=['','阻止能はMeV cm²/g。PEGSのRLCで単位を換算し、CALLはDECKより前に実行した。PEGSはprototypeデータ生成には使っていない。','',
    f"後方脱出エネルギー/入射は試作 {pct(k6['prototype_back_energy_fraction'])}、EGS5 {pct(k6['EGS5_back_energy_fraction'])}。一次のみの粒子数率ではなく、全子孫を含むエネルギー割合である。",'',
    '|基準/試作|R50 [cm]|Rp [cm]|','|---|---|---|',
    f"|2 MeV EGS5|{k6['reference_ranges_cm'][0]:.9g}|{k6['reference_ranges_cm'][1]:.9g}|",
    f"|2 MeV 試作（同じDref_EGS）|{k6['other_ranges_cm'][0]:.9g}|{k6['other_ranges_cm'][1]:.9g}|",
    f"|10 MeV EGS5|{k10['reference_ranges_cm'][0]:.9g}|{k10['reference_ranges_cm'][1]:.9g}|",
    f"|10 MeV 試作（同じDref_EGS）|{k10['other_ranges_cm'][0]:.9g}|{k10['other_ranges_cm'][1]:.9g}|",'',
    'K4: CHARD=0.5→0.25 cm、ESTEPR=1→0.5を別々に実施。GSはK1HSCL/K1LSCLを無視する（egs5_rmsfit.f）。fit感度はNALE=150→100。先行EPE/NIPE変更は最終表・刻みが変わらず、有効なK4には数えなかった。全3比較で実際の区間長histogramの変化を確認。','',
    'PEGS電子fitの区間数不足の警告は150区間と100区間の両方に残る。K4は今回の設定変更に対する安定性を示すが、警告の物理誤差を全面的に定量したものではない。GSファイル欠如の警告の後にelastinoで表を生成したことは実行ログで確認している。','']
    for key,title in [('USEGSD_model_difference','USEGSD=0と1のモデル差（判定外）'),('radiation_sensitivity','明示放射光子を局所破棄する感度（1条件、判定外）')]:
        r=e[key];lines +=[f"{title}: 最大線量差 {pct(r['max_dose_difference_over_Dref'])}、R50差 {pct(r['range_difference_over_reference_R50'][0])}、Rp差/R50 {pct(r['range_difference_over_reference_R50'][1])}。",'']
    lines +=['初回の放射感度run radiation_discard_2は、領域内の光子discard（iarg=3）を脱出に分類した採点誤りのため不採用。修正したradiation_local_discard_2のみを上の値に使用。同じ物理条件の再実行であり、2つ目の感度条件ではない。主比較は光子0なのでこの誤りの影響を受けない。','',
    '## 不一致の切り分け（指定順序）','',
    '1. GS表（K3）と固定エネルギーの位置更新（K9(i)）は独立検査で通過。',
    '2. クラスIIを切り、非制限連続衝突＋放射だけでhinge/逐次散乱を比較。',
    '3. 境界を全く問い合わせない輸送で比較。extended-slabの追加対照には前面が残るため、境界なしの証拠とは区別。',
    '4. K8のK1_max半減・f_E半減はともに通過。採用値は最も粗い初期値0.05/0.05。','',
    '|切り分け|2 mm横変位差|5 mm横変位差|差のSEM|境界query hinge/逐次|','|---|---|---|---|---|']
    extra_notes=[]
    for name in ['continuous_only','boundary_free','extended_slab']:
        r=diag[name];lines.append(f"|{name}|{pct(r['relative'][0])}|{pct(r['relative'][1])}|{', '.join(pct(x) for x in r['difference_SEM_relative'])}|{r['hinge_boundary_queries']}/{r['single_boundary_queries']}|")
        if 'dose' in r:extra_notes +=[f"連続損失のみの{ name}深さ線量最大差/Dref={pct(r['dose']['max_dose_difference_over_Dref'])}、最大bin差SEM={pct(r['dose']['max_dose_difference_SEM_over_Dref'])}。",'']
    lines+=['']+extra_notes
    lines+=['','確認済みの事実: 同じ断面積・同じ損失モデルの逐次参照とのK9比較、境界不変性、自己収束は通過。EGS5と試作は弾性DCSが異なり、同じ元素数密度を用いて参照表から合成した2 MeVのG1は次の通り。prototypeへpartial-waveデータを導入していない。','']
    md=next(x for x in diag['coefficient_model_difference']['rows'] if x['T']==2)
    lines +=[f"- Rutherford G1={md['prototype_Rutherford_total_G1_G2_cm_inverse'][1]:.9g} cm⁻¹、partial-wave G1={md['EGS5_partial_wave_total_G1_G2_cm_inverse'][1]:.9g} cm⁻¹、相対差={pct(md['relative'][1])}。",
    '', '仮説: Rutherford/partial-waveのモデル差が線量・後方散乱の残差へ寄与する可能性がある。係数差は測定事実だが、線量残差全体の原因とする因果検証は未了。指定DCSの差し替えやEGS5に合わせる調整を行わず、K6の原因未特定としてここで止める。hingeやNumbaの難しさだけに帰属しない。','',
    '## T6 単価、表、スタック、見積もり','',
    '利用上限による中断中に、使用者が変更せず最後まで実行したbenchmark.jsonを採用。単一processで各条件2万一次電子×3回。各署名のwarmup/JITを除外し、中央値を使用。SEM用全履歴ledgerを保持する共通ハーネスを含む壁時間。採点なしでも物理・ledger・境界は同一。','',
    '|T|f_E|K1_max|採点|μs/一次|直線区間/一次|GS標本化/一次|Møller/一次|','|---|---|---|---|---|---|---|']
    for r in b['rows']:lines.append(f"|{r['T']:g}|{r['fe']}|{r['K']}|{r['scoring']}|{r['us_per_primary']:.6g}|{r['counts_per_primary'][0]:.6g}|{r['counts_per_primary'][1]:.6g}|{r['counts_per_primary'][2]:.6g}|")
    lines+=['','|T|f_E|K1_max|採点あり−なし [μs/一次]|','|---|---|---|---|']
    for T in [2.,10.]:
        for fe,k in [(.05,.05),(.05,.025),(.025,.05),(.025,.025)]:
            vals=[r for r in b['rows'] if (r['T'],r['fe'],r['K'])==(T,fe,k)]
            yes=next(r for r in vals if r['scoring']);no=next(r for r in vals if not r['scoring'])
            lines.append(f"|{T:g}|{fe}|{k}|{yes['us_per_primary']-no['us_per_primary']:.6g}|")
    lines +=['',f"GS表生成 {b['table_build_seconds']:.6g} s、表読み込み/初期化 {b['table_load_initialization_seconds']:.6g} s。表 {b['table_bytes']} bytes、割当stack {b['stack_allocated_bytes']} bytes（4096 frame×19 float64）。全履歴ledgerは別途N×7×8 bytes、100×120のbatch tally。stack実使用最大深さは保存runのJSONに記録。",'',
    '|T|採用刻み・採点あり1M一次の単純外挿 [s]|','|---|---|']
    for T in [2.,10.]:
        val=next(r for r in b['rows'] if r['T']==T and r['fe']==r['K']==.05 and r['scoring'])['us_per_primary']
        lines.append(f"|{T:g}|{val:.6g}|")
    lines+=['','これは電子単独の一次電子数外挿である。P0のEGS5ステップ数には掛けない。連成光子1履歴の単価・必要履歴数・並列speedupは未測定。','',
    '|部分|観測された実装〜出力の経過 [h]|','|---|---|']
    for r in w['rows']:lines.append(f"|{r['phase']}|{r['elapsed_seconds']/3600:.6g}|")
    lines+=['','期間の更新: この表はファイル作成から出力までの実測wall interval（並行作業・待ち・手戻り込み）で、工程間が重なり加算できない。人間の実作業時間はstopwatchで測っておらず、作業週へ換算していない。データ・表・水の電子輸送・同一材料平面境界は実装済み部分としてこの実績へ置き換えた。K6原因調査の残作業を0とはしていない。未実装の幅はP0の工学的見積もりを保持する。','',
    '|未測定部分|最短/標準/難航 [作業週、P0推定]|','|---|---|']
    for name,vals in w['remaining_estimates_work_weeks'].items():lines.append(f"|{name}|{' / '.join(vals)}|")
    lines+=['','連成全体の期間をこの試作から再測定したとは言わない。水以外・異種材料境界・陽電子・放射光子・対生成・必要履歴数は引き続き未測定。','',
    '## 未解決事項','',
    '- K6線量の3%基準超過の因果分解は未完。モデル係数差は確認したが、残差の説明を確定していない。',
    '- PEGS電子fitの警告は残存。K4の成立を警告全面解消と読み替えない。',
    '- K11を行っていない。連成/Fano/水以外/異種材料の検証記録はない。',
    '- 段階開始・終了のwall intervalと手戻りは記録したが、工程別の人間の実作業時間は未測定。中断時刻の正確な境界も未記録で、経過時間を実作業時間に読み替えない。',
    '- 初期の試作runでは起動時のコードsnapshotが未保存。manifestの最終コードhashと、後続runのsnapshot、EGS5のrun別hashを区別する。初期runの完全な起動時コードhashは復元していない。',
    '', '## 再現・テスト・変更範囲','',
    '実行コマンドはprototypes/mv_electron/README.md。指定の全テストコマンドと保存再解析を最終確認した（回帰ログ: regression_tests_final.log、試作ログ: prototype_tests_final.log、禁止対象diff: production_diff_stat_final.log）。exit 0はK6の不合格を変更しない。生成物: PREREGISTRATION.md、RESULTS.md、manifest.json、summary.json、components/gs/fixed/transport/egs5/diagnostics/benchmark JSON、raw runs、EGS5入力/ログ、depth_2.png、depth_10.png。','',
    '変更はprototypes/mv_electron/とdocs/validation/mv/p1/の新規ファイルのみ。作業開始前のユーザー変更は維持。EGS5本体はread-only、scratchでビルド。git commitはしていない。','']
    test_results=read('test_summary')
    lines+=['指定テスト結果:','']
    for r in test_results['pytest']:
        lines.append(f"- {r['command']}: {r['passed']} passed、{r['warnings']} warnings、exit {r['exit_code']}（{r['log']}）。")
    lines+=['- git diff --stat -- chatcarlo tests scripts examples docs/validation/mv/p0: 空。','- .venv/bin/python prototypes/mv_electron/run_checks.py --from-saved: exit 0。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    manifest={'seed':SEED,'preregistration_sha256':hashlib.sha256((OUT/'PREREGISTRATION.md').read_bytes()).hexdigest(),'hashes':{},'provenance_note':'Final source/data/input/output hashes. EGS5 metadata records exact run sources and tree hashes; later prototype runs also contain launch-source snapshots. Early prototype run-source snapshots were not captured during performance refactoring; source hash identifies final implementation, with unchanged prescribed model and validated ledger/scoring invariants. Pilot runs excluded from acceptance.'}
    for folder in [ROOT/'prototypes/mv_electron',OUT]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache','.cache']) and p.name!='manifest.json':
                manifest['hashes'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':status,'overall':summary['overall'],'K6_max_difference':k6['max_dose_difference_over_Dref']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
