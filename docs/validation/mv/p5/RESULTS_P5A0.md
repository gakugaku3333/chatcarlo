# P5a0 手元の検査結果（2026-10-14）

承認済み計画 `docs/ai/plans/2026-10-14-mv-p5a0-toolchain-trial.md` の手元部分。
輸送計算ではない。GPU実機と機械間比較は未実行で、P5aへ進める判定ではない。

## 成果物と再現条件

- 実装: `prototypes/mv_electron/gpu/p5a0.py`、共通ヘッダー、CPU/GPU入口、Colabノートブック。
- 機械的結果: `p5a0_local.json`（約5.6 MB。git では追跡しない。この場所に置いてある。再生成は計画のテストコマンド）。各項目の必須指定、状態、差の数値、配列のSHA-256、先頭1万点の値を含む。
- 環境: Darwin arm64、Python 3.11.13、NumPy 2.4.6、Apple clang 21.0.0。
- CPU指定: `-O2 -ffp-contract=off -std=c++11 -shared -fPIC`。fast-mathなし。
- 実行時HEAD: `e174faa81dd4794749330d85d8188053a89d892c`。実装は未コミットの新規ファイルであり、このHEADに含まれているという意味ではない。実際のソースと共通ヘッダーのSHA-256はJSONに別途記録している。
- Random123参照: commit `9545ff6413f258be2f04c1d319d99aaef7521150` の [philox.h](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/include/Random123/philox.h) と [kat_vectors](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/tests/kat_vectors)。実行時のダウンロードなし。
- 入力の束: `.p5a0_work/p5a0_inputs.npz`、48,230,592 bytes。SHA-256: `e69284253e074b614e6948822a1ea8c7a0446ed96d3e255e2f37837a56d4267a`。
- 表: energies 3,100、values 3,100×5、LOGET/ER 各167。評価点1,009,302、数学入力1,000,000×5。

## 手元の機械的な判定

| 検査 | 状態 | 数値 |
|---|---|---|
| T1 | pass | 既知解3例、12語すべて一致 |
| T2 | pass | 1,000,125組、整数4,000,500語・実数1,000,125個、不一致0 |
| T3 | pass | MT19937 2,000,000個、不一致0 |
| T4 | pass | 1,009,302点×4出力×disabled 2設定、不一致0。区間番号も不一致0。20 MeV直上の拒否を確認 |
| T5a | pass | sqrt・四則式 各1,000,000個、不一致0 |
| T5b | report | log/exp/sin/cos/pow/atan2/acos 各1,000,000個、不一致0、最大絶対差0、最大相対差0、非有限値0 |
| T6 | pass | 1,000履歴、メタデータ4,000個・ビン50,000個・合計50ビンの不一致0。正常1,000、最大深さ9、容量4であふれ945 |
| T7(a) | pass | 1ビン移動を検出: ビン不一致47,291、総量不一致0 |
| T7(b) | pass | 10回に1回の欠落を検出: ビン不一致19,105、履歴総量不一致945 |
| T7(c) | pass | 余分な乱数を検出: 引いた回数の不一致732 |
| T8 | pass | 一括と400+600履歴の分割、全出力の不一致0 |

T7の3変種はコンパイル・実行に成功し、各1,000履歴すべて状態0。
通常版の比較は、すべて最大絶対差0・最大相対差0・非有限値0。
おもちゃは世代8までの木（最大511ステップ）。各付与は1〜1,024の整数を1,024で割った値。
1,000万履歴でも全付与の整数分子は2^53未満なので、集計の順序で値が変わらない。
23成分はすべて重み付き和に入り、付与先と付与量に影響する。

## CuPyなしのColab入口

指定コマンドはCPU部分を実行し、X1用の出力を生成した。
`.p5a0_work/p5a0_colab_dryrun.json` に保存。
`physics_imported=false`、標準出力に「GPUなし」。G0/G1/G2a/G2b/G3a/G3b/G4/G5/G7/G6 はすべて `not_run`。
必須未実行を合格にしない規則に従い、総合判定は `fail`、終了コード1。
これはColabの実測結果ではないため、`p5a0_colab.json` としては保存していない。

## 未実行

- Colab T4でのコンパイル、G0〜G7（G2/G3の枝番を含む）、時間測定。
- 実機結果とのX1a/X1b。手元のdryrunとの比較は機械間検証に数えない。
- コミット・push、ノートブックの固定コミット設定、ユーザーへのColab実行引き継ぎ。

ノートブックは公開後の実装コミットを先頭の `CODE_COMMIT` に設定して使う。
現時点のHEADには新規実装がないため、そのHEADを代入してもColabでは実行できない。
公開コミットでbundle/localを再実行し、同じ束をColabへ渡す必要がある。
共通ヘッダーとGPU入口は固定長配列のみで、可変長配列、複合リテラル、restrict、例外、動的確保、標準コンテナはない。
CUDA C++での実際のコンパイル成功は未確認。

作業用の束・共有ライブラリ・検査結果は `.p5a0_work/` に残す。このディレクトリはコミットしない。
既存ファイルの編集や削除処理は行っていない。

## 最終回帰テスト

指定の `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` を `PYTHONPATH=.` で実行。
終了コード0、**654 passed, 2 skipped in 904.18s (0:15:04)**。
基準の645 passed / 1 skippedに、新規テストの9 passed / GPU検査1 skippedが加わった件数。
新規テスト単独は `9 passed, 1 skipped in 17.57s`。
`git diff --check` も終了コード0。
