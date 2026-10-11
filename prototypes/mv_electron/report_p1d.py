"""P1d Japanese report, fixed-scale plots and relative-path manifest."""
import json
import os
from pathlib import Path
import numpy as np
from .p1d_common import *
from .analysis import dref

def percent(x): return f'{x * 100:+.6f}%'

def report(s):
    lines = ['# P1d — 多重散乱の刻みを半分にした追試', '',
             '採用候補は `K1_max=0.025, f_E=0.05` の1条件。seed・履歴数・停止規則はPREREGISTRATION.mdと凍結したコードに従う。', '',
             '| 項目 | dcslib | mf |', '|---|---|---|']
    for k in ('R0', 'R1', 'R2', 'R3', 'R4'):
        vals = [s['R0'] if k == 'R0' else s['backends'][b][k] for b in BACKENDS]
        lines.append(f'| {k} | ' + ' | '.join(LABELS[v['status']] for v in vals) + ' |')
    lines += ['', 'R4は参考値を記録する項目で、合否の基準がないため状態は「未評価」。参考計算の完了・未完了は下記に区別する。', '',
              f"R0: 初期 {s['R0']['untouched']['initial_files']} ファイル、追加凍結 {s['R0']['untouched']['frozen_additional_files']} ファイルのSHA256一致。",
              '追加前後の10⁴履歴、seed 20261007、2 backend × K=0.05/0.025の4条件は、7つの数値配列が完全一致。',
              '開始前から存在するGit差分は initial_diff_stat.log と initial_git_status.log に保存。既存の変更・旧判定は保持した。', '']
    if (OUT / 'amendment1_hashes.json').exists():
        lines += ['追補1による再開: 最初の報告生成はdrefのimport漏れで終了コード1となり、その後、凍結対象に含めていたP2計画の変更でR0不合格として停止した。凍結1の不合格は取り消さず、停止時の成果物をstop1/に保全した。追補追記後の凍結1の既知の不一致はP1dとP2の計画2件。', '',
                  '上表のR0は再開後の凍結2の判定である。凍結2は試作の全Pythonファイル、旧成果物、GS表、EGS5基準データ、事前登録、保存済みR1の7標本とレシート、stop1/を明示して固定する。計画文書は含めない。対象の追加・削除も検出する。詳細はamendment1_hashes.jsonとamendment1_changes.json。', '',
                  '再開条件: 追補approved、凍結2保存、mf・0.05 cm・試作側のメモリ上再現が保存配列と完全一致。再現標本は保存せず、統計に追加しない。保存済み7標本は保持し、欠けていたdcslib・0.2 cm・試作側だけを追加した。', '',
                  'R1の0.05 cmの新旧差は、dcslibで合成SEMの約2.3倍、mfで約1.9倍。両backendは同じseedを使うため相関し、独立な2回の確認ではない。K=0.05の旧値（dcslib −1.956%、mf −2.020%）の合否の違いはseedによるゆらぎの幅の中の差だった可能性がある。原因は未確認で、旧判定は保持する。', '']
    for backend in BACKENDS:
        one = s['backends'][backend]
        lines += [f'## {backend}', '', 'R1: 2 MeV・損失なし、各100万履歴。参照seed 20261009、試作seed 20261010。許容差2%、差のSEM上限0.5%。', '',
                  '| 道のりcm | 横二乗平均の差 | SEM | 深さ平均の差 | SEM | 状態 | 旧値との横方向差/合成SEM |',
                  '|---|---|---|---|---|---|---|']
        for r in one['R1']['rows']:
            d, sem = r['relative'], r['difference_sem_relative']
            lines.append(f"| {r['path_cm']} | {percent(d[0])} | {percent(sem[0])} | {percent(d[1])} | {percent(sem[1])} | {LABELS[r['status']]} | {r['new_minus_historical_over_combined_SEM'][0]:+.6f} |")
        lines += ['', 'R2: K5、K5b、K7、K8、K9(ii)。K7の層厚は0.1 cm、ずらし0.037 cm。P1の計画値との相違はP1bから存在し、P1dで導入した変更ではない。', '']
        r2 = one['R2']
        if r2['status'] == 'not-evaluated' and 'pending_runs' in r2:
            lines.append('未完了run: ' + ', '.join(r2['pending_runs']))
        else:
            k5 = r2['K5']
            lines += [f"K5: {LABELS[k5['status']]}。全体残差 {k5['global_relative']:.9g}、最大履歴残差 {k5['max_history_relative']:.9g}（入射エネルギー比）。",
                      f"未処理粒子 {k5['unprocessed_particles']}、未処理MeV {k5['unprocessed_MeV']}、ステップ上限 {k5['step_limit_failures']}、スタックあふれ {k5['stack_failures']}。",
                      f"K5b: {LABELS[r2['K5b']['status']]}。実際の輸送用表で既存の全小テストを実行。", '',
                      '| 検査run | 線量最大差/Dref_proto | 最大SEM/Dref_proto | ΔR50/R50 | SEM/R50 | 状態 | 境界イベント/照会 |',
                      '|---|---|---|---|---|---|---|']
            for key in ('K7', 'K8'):
                for name, c in r2[key]['rows'].items():
                    lines.append(f"| {name} | {percent(c['max_dose_difference_over_Dref'])} | {percent(c['max_dose_difference_SEM_over_Dref'])} | {percent(c['range_difference_over_reference_R50'][0])} | {percent(c['range_difference_SEM_over_R50'][0])} | {LABELS[c['status']]} | {c['boundary_events']}/{c['boundary_queries']} |")
            c = r2['K9ii']
            lines += ['', f"K9(ii): {LABELS[c['status']]}。許容3%、SEM上限1%。", '', '| 深さcm | 横二乗平均差 | SEM | 試作/参照の通過数 |', '|---|---|---|---|']
            for i, depth in enumerate((.2, .5)):
                lines.append(f"| {depth} | {percent(c['relative'][i])} | {percent(c['difference_SEM_relative'][i])} | {c['prototype_cross_N'][i]}/{c['reference_cross_N'][i]} |")
        for key in ('R3', 'R4'):
            c = one[key]; lines += ['', f"{key}: {LABELS[c['status']]}。"]
            if 'max_dose_difference_over_Dref' in c:
                lines += [f"線量最大差 {percent(c['max_dose_difference_over_Dref'])} Dref_EGS（SEM {percent(c['max_dose_difference_SEM_over_Dref'])}、{c['evaluation_bins']} bins）。",
                          f"R50/Rp（参照cm）{c['reference_ranges_cm']}、試作cm {c['other_ranges_cm']}。",
                          'ΔR50/R50、ΔRp/R50: ' + ', '.join(percent(x) for x in c['range_difference_over_reference_R50']) + '。',
                          'そのSEM: ' + ', '.join(percent(x) for x in c['range_difference_SEM_over_R50']) + '。',
                          f"後方散乱エネルギー比: 試作 {c['prototype_back_energy_fraction']:.9g}、EGS5 {c['EGS5_back_energy_fraction']:.9g}。",
                          'R4の参考runは計算済み。' if key == 'R4' else 'R3: 線量・R50・Rpの許容3%、SEM上限1%。EGS5の保存済みK4合格を確認して判定。']
            else: lines.append(c['reason'])
        passed = s['R0']['status'] == 'pass' and all(one[k]['status'] == 'pass' for k in ('R1', 'R2', 'R3'))
        if passed:
            kind = '研究専用の対照（EGS5と同じ散乱データ）' if backend == 'dcslib' else '解析式のMcKinley–Feshbach補正（低Z限定、水のH/Oで検証）'
            lines += ['', f'2 MeV・水・指定条件（K1_max = 0.025）で電子輸送の試作が成立。散乱データ: {kind}。',
                      '刻みはP1cの結果を見たあとに変えた。選定の根拠はEGS5に依存しない検査である。']
        elif one['R1']['status'] == 'fail':
            lines += ['', '刻みを半分にしても横方向の広がりの検査を満たせなかった。R3の判定を停止した。さらに細かい刻みは別の計画にする。']
        else:
            lines += ['', '成立の条件は満たしていない。統計不足・未評価から物理的な未達は主張しない。']
        lines += ['', '| 時間計測 | 採点 | μs/一次電子 | 3回の秒数 |', '|---|---|---|---|']
        for r in one['timing']['rows']:
            lines.append(f"| {r['T']} MeV | {r['scoring']} | {r['us_per_primary']:.6f} | {r['wall_seconds']} |")
    lines += ['', '## 補助記録（合否に使わない）', '',
              '同じ固定EGS5評価mask、Drefを使用。r_P1 = 1 − d_new/d_P1、r_K = 1 − d_new/d_old。',
              '点推定と再標本化の比較元は同一。1000回、batch単位、seed 20261008、分子・分母に同じEGS5再標本を使用し、百分位法95%区間。', '',
              '| backend | K | 最大差/Dref_EGS | ΔR50 cm | ΔRp cm | 後方散乱比 |', '|---|---|---|---|---|---|']
    for backend in BACKENDS:
        aux = s['auxiliary'].get(backend)
        if not aux: continue
        for key, K in (('old_K005', .05), ('new_K0025', .025)):
            c = aux[key]
            lines.append(f"| {backend} | {K} | {percent(c['max_dose_difference_over_Dref'])} | {c['range_difference_cm'][0]:+.9g} | {c['range_difference_cm'][1]:+.9g} | {c['back_energy_fraction']:.9g} |")
    for backend in BACKENDS:
        aux = s['auxiliary'].get(backend)
        if not aux: continue
        lines += ['', f'{backend}:']
        for key in ('r_P1', 'r_K'):
            c = aux[key]
            lines.append(f"- {key}: {percent(c['point'])}、95%区間 [{percent(c['CI95'][0])}, {percent(c['CI95'][1])}]。")
        c = aux['saved_ms_half_consistency']
        lines.append(f"- 保存済みms_halfとの全bin最大差 {percent(c['max_abs_difference_over_Dref'])} Dref_EGS、最大|差/合成SEM| {c['max_abs_ratio']:.6f}、2SEM超のbin数 {c['bins_beyond_2SEM']}。bin配列はsummary.json。")
    if 'mf_minus_dcslib' in s['auxiliary']:
        c = s['auxiliary']['mf_minus_dcslib']
        lines += ['', f"mf−dcslib: 評価bin最大差 {percent(c['max_difference_over_Dref'])} Dref_EGS、深さ {c['depth_cm']} cm。",
                  f"EGS5のR50より深い全bin最大差 {percent(c['beyond_R50_all_bins_max_over_Dref'])}、評価bin最大差 {percent(c['beyond_R50_eligible_max_over_Dref'])}。各binの差はsummary.json。"]
    lines += ['', 'P1bの合格、P1cの未判定は変更しない。mfの旧K=0.05のEGS5最大差は計画前に既知。保存済みms_halfが過去にEGS5と比較されていないことは申告であり、ファイルから証明した事実ではない。', '',
              '## 確かめた事実と仮説', '',
              '確かめた事実: 上の値は固定seed・固定履歴数の保存結果から再計算した。輸送・損失・採点・解析・backend・GS表・旧成果物のハッシュ一致と、追加前後の数値配列の一致を確認した。EGS5は再実行していない。',
              '仮説: K9(i)のずれは有限の刻みによる位置更新の偏りである可能性がある。刻みの変化と観測差の変化は数値上の事実だが、その原因を特定する実験はこの計画では行っていない。', '',
              '時間計測はP1bと同じ単一プロセス・20000履歴×3回の中央値・JIT除外。採用K=0.025だけ測定。各繰り返しseedは20261009で、統計へ追加しない。表は凍結対象の既存表を読み、生成時間は既存表の記録値で再測定しない。', '',
              '## 未完了', '']
    pending = []
    for backend in BACKENDS:
        for name in CONFIGS:
            if not (OUT / backend / 'runs' / f'{name}.npz').exists(): pending.append(f'{backend}/{name}')
        if len(s['backends'][backend]['R1']['rows']) != 2: pending.append(f'{backend}/R1')
        rows = s['backends'][backend]['timing']['rows']
        if len(rows) != 4 or any(len(r['wall_seconds']) != 3 for r in rows): pending.append(f'{backend}/時間計測')
    tests = OUT / 'test_commands.json'
    if not tests.exists(): pending.append('受入テスト')
    else:
        receipt = json.loads(tests.read_text())
        pending += [r['command'] for r in receipt['commands'] if r['exit_code'] != 0]
    lines += ['- ' + name for name in pending] if pending else ['本計画の実行項目に未完了はない。']
    lines += ['', '## 実行コマンド', '', '```bash',
              '.venv/bin/python prototypes/mv_electron/run_tests_p1d.py',
              '.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --seal',
              '.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --reproduce',
              '.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py',
              '.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved',
              '.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched', '```', '',
              '受入テストの実コマンド・終了コード・出力は test_commands.json と対応するlog。再開後の進捗はamendment1_progress.json、実行記録はamendment1_commands.json。元のprogress.jsonとstop1/のrestart_allowed:falseは保持する。凍結1の前には初期ハッシュ取得・事前登録の転記と4条件の追加前回帰をインラインPythonで実行した。', '',
              '既存ファイルを編集していないため、.gitignoreにも追記していない。P1dのNPZはローカル成果物としてmanifestに記録し、commitしていない。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    plots(s)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json' and not p.name.endswith('.tmp') and not p.name.endswith('.pending.npz')]
    write('manifest.json', dict(plan='docs/ai/plans/2026-10-08-mv-p1d-half-step.md',
         seed=SEED, fixed_prototype_seed=FIXED_SEED, bootstrap_seed=BOOTSTRAP_SEED,
         run_matrix=CONFIGS, gs_tables=TABLES, files={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)},
         code_hashes=json.loads((OUT / ('amendment1_hashes.json' if (OUT / 'amendment1_hashes.json').exists() else 'code_hashes.json')).read_text()),
         EGS5_executed=False, existing_files_edited=(OUT / 'amendment1_hashes.json').exists(), commit=False))

def plots(s):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/p1d-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .check_egs5 import parse
    ref = parse('base_2'); D = dref(ref['batches'].mean(0))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for col, backend in enumerate(BACKENDS):
        if s['backends'][backend]['R1']['status'] != 'pass':
            axes[0, col].set(title=f'{backend}: R1 gate closed')
            continue
        axes[0, col].plot(ref['z'], ref['batches'].mean(0), 'k', label='EGS5 saved reference')
        for K, path in ((.05, OLD[backend] / 'runs/base_2.npz'), (.025, OUT / backend / 'runs/base_2.npz')):
            if not path.exists(): continue
            q = np.load(path)['batches'].mean(0)
            axes[0, col].plot(ref['z'], q, label=f'{backend} K={K}')
            axes[1, col].plot(ref['z'], (q - ref['batches'].mean(0)) / D * 100, label=f'K={K}')
        axes[0, col].set(title=f'{backend}: 2 MeV water', ylabel='MeV cm2/g / primary')
        axes[1, col].set(xlabel='Depth (cm)', ylabel='(prototype - EGS5) / Dref (%)')
        for y in (-3, 3): axes[1, col].axhline(y, color='gray', linestyle='--', alpha=.5)
        for ax in axes[:, col]: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / 'depth_2.png', dpi=160); plt.close(fig)
