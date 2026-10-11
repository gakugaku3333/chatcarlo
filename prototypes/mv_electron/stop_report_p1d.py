"""Saved-only stop report. No simulation, no changes to preregistered decisions."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prototypes.mv_electron.p1d_common import *
from prototypes.mv_electron.run_checks_p1d import r1

def main():
    initial = json.loads((OUT / 'initial_hashes.json').read_text())
    code = json.loads((OUT / 'code_hashes.json').read_text())
    changes = [dict(path=p, initial_sha256=h, final_sha256=sha(ROOT / p) if (ROOT / p).is_file() else None)
               for p, h in {**initial, **code}.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    assert changes, 'Use the preregistered normal replay when R0 is satisfied'
    # These checks are the frozen R1 analysis, with unchanged physical inputs.
    required = {p: h for p, h in initial.items() if p.startswith(('prototypes/mv_electron/', 'docs/validation/mv/p1/', 'docs/validation/mv/p1b/', 'docs/validation/mv/p1c/'))}
    assert all((ROOT / p).is_file() and sha(ROOT / p) == h for p, h in {**required, **code}.items())
    regression = regression_saved()
    summary = dict(R0=dict(status='fail', initial_files=len(initial), frozen_additional_files=len(code),
                          mismatches=changes, required_physics_and_saved_inputs_unchanged=True,
                          preregistered_code_unchanged=True, regression=regression), backends={})
    for backend in BACKENDS:
        summary['backends'][backend] = dict(R1=r1(backend))
        for key in ('R2', 'R3', 'R4'):
            summary['backends'][backend][key] = dict(status='not-evaluated', reason='R0不一致の停止規則。輸送run未実行。')
    write('summary.json', summary)
    write('untouched.json', summary['R0'])
    write('progress.json', dict(status='stopped-R0', no_simulation_running=True,
        completed='R0 before/after regressions, both K5b; all mf R1 samples; dcslib 0.05 reference/prototype and 0.2 reference',
        pending=['dcslib/R1/0.2 prototype']+[f'{b}/{n}' for b in BACKENDS for n in CONFIGS],
        restart_allowed=False, reason='Initial freeze mismatch must not be waived or re-baselined'))
    write('execution_interruption.json', dict(first_interruption='NameError: dref is not defined in report_p1d.plots',
        first_command='.venv/bin/python prototypes/mv_electron/run_p1d.py', exit_code=1,
        subsequent_stop='R0 freeze mismatch in the P2 plan file; startup reporting adapter was never frozen or used successfully; unused adapter removed',
        frozen_files_edited=False, completed_runs_rerun=False, seeds_reselected=False, statistics_added=False, EGS5_executed=False))
    receipts = json.loads((OUT / 'test_commands.json').read_text())
    receipts['commands'] = [r for r in receipts['commands'] if not r.get('expected_stop')]
    for option, log in (('--from-saved', 'from_saved_stopped.log'), ('--verify-untouched', 'verify_untouched_stopped.log')):
        command = ['.venv/bin/python', 'prototypes/mv_electron/run_checks_p1d.py', option]
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = (r.stdout + r.stderr).replace(str(ROOT), '.').replace(str(Path.home()), '~')
        (OUT / log).write_text(output)
        receipts['commands'].append(dict(command=' '.join(command), exit_code=r.returncode, log=log,
                                         expected_stop=True, reason='R0 freeze mismatch', output=output))
        assert r.returncode != 0 and 'R0 freeze mismatch' in output
    write('test_commands.json', receipts)
    lines = ['# P1d 停止報告', '',
        '**R0不合格のため停止。P1dは未完了であり、電子輸送の試作が成立したとは判定していない。**', '',
        '| 項目 | dcslib | mf | 値・理由 |', '|---|---|---|---|',
        '| R0 | 不合格 | 不合格 | 凍結したP2計画のSHA256が不一致。下記参照 |',
        '| R1 | ' + LABELS[summary['backends']['dcslib']['R1']['status']] + ' | ' + LABELS[summary['backends']['mf']['R1']['status']] + ' | 損失なしの2条件は完了。下表参照 |',
        '| R2 | 未評価 | 未評価 | K5bのみ両backendで合格。輸送run未実行 |',
        '| R3 | 未評価 | 未評価 | R0の停止規則。新しい2 MeV run未実行 |',
        '| R4 | 未評価 | 未評価 | 参考の10 MeV run未実行 |', '',
        '## R0で確かめた事実', '',
        f'初期 {len(initial)} ファイルと追加凍結 {len(code)} ファイルを比較し、次の不一致を検出した。', '',
        '| ファイル | 初期SHA256 | 最終SHA256 |', '|---|---|---|']
    for c in changes: lines.append(f"| {c['path']} | {c['initial_sha256']} | {c['final_sha256']} |")
    lines += ['', 'このファイルを編集する操作は本作業では行っていない。変更した主体・内容・意図は確認していない。',
        '初期スナップショットには、計画で必須の物理コード・旧成果物に加え、変更禁止領域の計画ファイル等も含めていた。結果を見たあとに不一致のファイルを凍結範囲から除外せず、事前固定したR0検証の停止規則を適用した。',
        '計画の必須範囲より広くP2計画までR0検証に含めたことは、実装上の問題である。ここでのR0不合格はこの追加対象の不一致によるもので、必須の物理コードや計算結果の改変を示すものではない。描画import漏れとともに、実装が未完了である理由として報告する。',
        '必須の物理コード、P1/P1b/P1cの全成果物（ignored NPZを含む）、使用GS表・EGS5基準、事前登録と初期P1dコードはすべて初期ハッシュと一致した。',
        'dcslib/mf × K=0.05/0.025、N=10000、seed=20261007の追加前後回帰は、7つの数値配列が4条件すべて完全一致した。',
        '既存テスト403件、試作テスト75件は合格。指定git diff --statは空で、初期状態と一致。', '',
        '## R1の保存結果', '',
        '2 MeV、損失なし、K=0.025。各標本N=1000000、逐次参照seed=20261009、試作seed=20261010。許容差2%、差のSEM上限0.5%。横二乗平均と深さ平均の2量で判定した。', '',
        '| backend | 道のりcm | 横二乗平均差 | SEM | 深さ平均差 | SEM | 状態 | 旧横方向差 | 新旧差/合成SEM |',
        '|---|---|---|---|---|---|---|---|---|']
    for backend in BACKENDS:
        for r in summary['backends'][backend]['R1']['rows']:
            d, se = r['relative'], r['difference_sem_relative']
            lines.append(f"| {backend} | {r['path_cm']} | {100*d[0]:+.6f}% | {100*se[0]:.6f}% | {100*d[1]:+.6f}% | {100*se[1]:.6f}% | {LABELS[r['status']]} | {100*r['historical_relative'][0]:+.6f}% | {r['new_minus_historical_over_combined_SEM'][0]:+.6f} |")
    lines += ['', '表の各標本は独立したNPZと完了レシートJSONに保存済み。seed・N・KとSHA256を検証した。旧値との整合は上表とsummary.jsonに記録するだけで、採用条件や判定を変えていない。',
        'dcslibの0.2 cmは逐次参照のNPZだけ保存済み。試作標本が未完了のため、この条件とdcslibのR1全体は未評価。mfは2条件とも完了して合格した。',
        'K5bは各backendの実際の輸送用表を用い、既存の全小テストを実行して合格した。R2全体は輸送結果がないため未評価。', '',
        '## 中断と未完了', '',
        'mfのR1完了後の報告生成で、凍結済みreport_p1d.pyの描画部分にdrefのimport漏れが判明し、最初の実行が終了コード1で中断した。実装上のミスであり、物理的未達ではない。並行していたdcslibは0.2 cm参照を保存したあと、次の試作標本の開始前検証でR0不一致を検出して停止した。',
        '既存解析関数を描画へ接続する追加起動補助を用意しかけたが、補助のハッシュ固定前の再検証でR0不一致を検出した。補助は未固定・未使用で削除した。凍結済みスクリプトは変更していない。',
        'R0不一致を検出してから、追加のシミュレーションは一切実行していない。R1の完了済み標本のやり直し、増統計、seed選択、EGS5の再実行もしていない。', '',
        '未完了項目:', '',
        '- dcslibのR1: 道のり0.2 cmの試作標本（N=1000000、seed=20261010、K=0.025）。参照標本は保存済み。',
        '- 両backendのbase_2、split_aligned、split_shifted、ms_half、energy_half、reference_2、base_10（計14run）。',
        '- R2のK5・K7・K8・K9(ii)、R3、R4。',
        '- EGS5に対する新旧刻みの補助比較、2種類の残差低減率とbootstrap区間、mf−dcslibの深さ線量差、保存ms_halfとの深さ線量の整合。',
        '- 性能計測、2 MeV深さ線量の重ね描きと差の図。新しい深さ線量がないため図は作成していない。',
        '- 通常の保存結果再生入口の描画import漏れへの対処。凍結済みファイルを編集するとR0が成立しないため、この計画の中では編集していない。',
        '- 受入の--from-savedと--verify-untouchedは実行したが、R0不一致で規定どおり終了コード1となり、完了条件は満たしていない。', '',
        '## 事実と仮説', '',
        '事実: 上記R1の値、統計精度、K5b合格、4条件の完全一致、物理コード・旧成果物のハッシュ一致、P2計画のハッシュ不一致、描画import漏れを確認した。',
        '仮説: 有限の刻みによる位置更新の偏りがK9(i)のずれの原因である可能性は未検証。新しい結果が許容幅内であることだけで原因を特定していない。P2計画変更の主体や意図も推測しない。',
        '刻みはP1cの結果を見たあとに変えた。選定の根拠はEGS5に依存しない検査である。P1bの合格・P1cの未判定は保持した。', '',
        '## 実行コマンド', '', '```bash',
        '.venv/bin/python prototypes/mv_electron/run_tests_p1d.py',
        '.venv/bin/python prototypes/mv_electron/run_p1d.py --freeze',
        '.venv/bin/python prototypes/mv_electron/run_p1d.py',
        '.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p1d.py -q',
        '.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p1d.py::test_stop_and_statistics_states prototypes/mv_electron/tests/test_p1d.py::test_reduction_same_comparator -q',
        '.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved',
        '.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched',
        '.venv/bin/python prototypes/mv_electron/stop_report_p1d.py', '```', '',
        'run_tests_p1d.pyは、計画にあるpytest tests/、git diff --stat、pytest prototypes/mv_electron/testsを実行した。テスト終了コード・出力はtest_commands.jsonと対応するlog。',
        '主入口が起動した内部コマンドは `run_p1d.py --worker dcslib --run fixed` と `run_p1d.py --worker mf --run fixed`、および実表K5bのpytest（backend別）。初期ハッシュ固定・PREREGISTRATION転記・追加前回帰はインラインPythonで実行した。',
        '描画補助付き--from-savedも一度試みたが、ハッシュ固定前のR0検証で停止し、計算は実行していない。', '',
        '成果物内のパスは相対表記。既存ファイルと.gitignoreは編集していない。commitはしていない。']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'RUN.md').write_text('# P1dの実行状態\n\nR0不一致のため停止。RESULTS.mdを参照。開始時のハッシュを更新して再開したり、不一致を除外して合格に変えたりしない。完了済みR1標本と回帰結果は保存済み。凍結した主入口には描画import漏れが残る。\n')
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json' and not p.name.endswith('.tmp')]
    write('manifest.json', dict(status='stopped-R0', plan='docs/ai/plans/2026-10-08-mv-p1d-half-step.md',
        seed=SEED, fixed_prototype_seed=FIXED_SEED, bootstrap_seed=BOOTSTRAP_SEED,
        run_matrix=CONFIGS, gs_tables=TABLES, files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        frozen_code_hashes=code, stop_report_script_sha256=sha(Path(__file__)),
        EGS5_executed=False, commit=False, initial_freeze_mismatches=changes))
    print(json.dumps({b:{k:LABELS[v['status']] for k,v in summary['backends'][b].items()} for b in BACKENDS},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
