# P1d 停止報告

**R0不合格のため停止。P1dは未完了であり、電子輸送の試作が成立したとは判定していない。**

| 項目 | dcslib | mf | 値・理由 |
|---|---|---|---|
| R0 | 不合格 | 不合格 | 凍結したP2計画のSHA256が不一致。下記参照 |
| R1 | 未評価 | 合格 | 損失なしの2条件は完了。下表参照 |
| R2 | 未評価 | 未評価 | K5bのみ両backendで合格。輸送run未実行 |
| R3 | 未評価 | 未評価 | R0の停止規則。新しい2 MeV run未実行 |
| R4 | 未評価 | 未評価 | 参考の10 MeV run未実行 |

## R0で確かめた事実

初期 4569 ファイルと追加凍結 14 ファイルを比較し、次の不一致を検出した。

| ファイル | 初期SHA256 | 最終SHA256 |
|---|---|---|
| docs/ai/plans/2026-10-08-mv-p2-photon-coupling.md | ce03fcc20bbf1b09a55e67cc7172038e1243d75660c31d4e8ed3e68f2896845d | 9c5442ae596e3280ff9232f913bf5490cec6524cecef6dee4c9a1d6f45a3db18 |

このファイルを編集する操作は本作業では行っていない。変更した主体・内容・意図は確認していない。
初期スナップショットには、計画で必須の物理コード・旧成果物に加え、変更禁止領域の計画ファイル等も含めていた。結果を見たあとに不一致のファイルを凍結範囲から除外せず、事前固定したR0検証の停止規則を適用した。
計画の必須範囲より広くP2計画までR0検証に含めたことは、実装上の問題である。ここでのR0不合格はこの追加対象の不一致によるもので、必須の物理コードや計算結果の改変を示すものではない。描画import漏れとともに、実装が未完了である理由として報告する。
必須の物理コード、P1/P1b/P1cの全成果物（ignored NPZを含む）、使用GS表・EGS5基準、事前登録と初期P1dコードはすべて初期ハッシュと一致した。
dcslib/mf × K=0.05/0.025、N=10000、seed=20261007の追加前後回帰は、7つの数値配列が4条件すべて完全一致した。
既存テスト403件、試作テスト75件は合格。指定git diff --statは空で、初期状態と一致。

## R1の保存結果

2 MeV、損失なし、K=0.025。各標本N=1000000、逐次参照seed=20261009、試作seed=20261010。許容差2%、差のSEM上限0.5%。横二乗平均と深さ平均の2量で判定した。

| backend | 道のりcm | 横二乗平均差 | SEM | 深さ平均差 | SEM | 状態 | 旧横方向差 | 新旧差/合成SEM |
|---|---|---|---|---|---|---|---|---|
| dcslib | 0.05 | -0.307841% | 0.291363% | -0.002218% | 0.004953% | 合格 | -1.257049% | +2.312505 |
| mf | 0.05 | -0.529827% | 0.287888% | +0.000832% | 0.004981% | 合格 | -1.297047% | +1.888016 |
| mf | 0.2 | -0.345136% | 0.167262% | +0.020969% | 0.011371% | 合格 | +0.035814% | -1.610793 |

表の各標本は独立したNPZと完了レシートJSONに保存済み。seed・N・KとSHA256を検証した。旧値との整合は上表とsummary.jsonに記録するだけで、採用条件や判定を変えていない。
dcslibの0.2 cmは逐次参照のNPZだけ保存済み。試作標本が未完了のため、この条件とdcslibのR1全体は未評価。mfは2条件とも完了して合格した。
K5bは各backendの実際の輸送用表を用い、既存の全小テストを実行して合格した。R2全体は輸送結果がないため未評価。

## 中断と未完了

mfのR1完了後の報告生成で、凍結済みreport_p1d.pyの描画部分にdrefのimport漏れが判明し、最初の実行が終了コード1で中断した。実装上のミスであり、物理的未達ではない。並行していたdcslibは0.2 cm参照を保存したあと、次の試作標本の開始前検証でR0不一致を検出して停止した。
既存解析関数を描画へ接続する追加起動補助を用意しかけたが、補助のハッシュ固定前の再検証でR0不一致を検出した。補助は未固定・未使用で削除した。凍結済みスクリプトは変更していない。
R0不一致を検出してから、追加のシミュレーションは一切実行していない。R1の完了済み標本のやり直し、増統計、seed選択、EGS5の再実行もしていない。

未完了項目:

- dcslibのR1: 道のり0.2 cmの試作標本（N=1000000、seed=20261010、K=0.025）。参照標本は保存済み。
- 両backendのbase_2、split_aligned、split_shifted、ms_half、energy_half、reference_2、base_10（計14run）。
- R2のK5・K7・K8・K9(ii)、R3、R4。
- EGS5に対する新旧刻みの補助比較、2種類の残差低減率とbootstrap区間、mf−dcslibの深さ線量差、保存ms_halfとの深さ線量の整合。
- 性能計測、2 MeV深さ線量の重ね描きと差の図。新しい深さ線量がないため図は作成していない。
- 通常の保存結果再生入口の描画import漏れへの対処。凍結済みファイルを編集するとR0が成立しないため、この計画の中では編集していない。
- 受入の--from-savedと--verify-untouchedは実行したが、R0不一致で規定どおり終了コード1となり、完了条件は満たしていない。

## 事実と仮説

事実: 上記R1の値、統計精度、K5b合格、4条件の完全一致、物理コード・旧成果物のハッシュ一致、P2計画のハッシュ不一致、描画import漏れを確認した。
仮説: 有限の刻みによる位置更新の偏りがK9(i)のずれの原因である可能性は未検証。新しい結果が許容幅内であることだけで原因を特定していない。P2計画変更の主体や意図も推測しない。
刻みはP1cの結果を見たあとに変えた。選定の根拠はEGS5に依存しない検査である。P1bの合格・P1cの未判定は保持した。

## 実行コマンド

```bash
.venv/bin/python prototypes/mv_electron/run_tests_p1d.py
.venv/bin/python prototypes/mv_electron/run_p1d.py --freeze
.venv/bin/python prototypes/mv_electron/run_p1d.py
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p1d.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p1d.py::test_stop_and_statistics_states prototypes/mv_electron/tests/test_p1d.py::test_reduction_same_comparator -q
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched
.venv/bin/python prototypes/mv_electron/stop_report_p1d.py
```

run_tests_p1d.pyは、計画にあるpytest tests/、git diff --stat、pytest prototypes/mv_electron/testsを実行した。テスト終了コード・出力はtest_commands.jsonと対応するlog。
主入口が起動した内部コマンドは `run_p1d.py --worker dcslib --run fixed` と `run_p1d.py --worker mf --run fixed`、および実表K5bのpytest（backend別）。初期ハッシュ固定・PREREGISTRATION転記・追加前回帰はインラインPythonで実行した。
描画補助付き--from-savedも一度試みたが、ハッシュ固定前のR0検証で停止し、計算は実行していない。

成果物内のパスは相対表記。既存ファイルと.gitignoreは編集していない。commitはしていない。
