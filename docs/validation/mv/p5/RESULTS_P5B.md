# P5b — GPU入口・入力束・固定判定の実装

日付: 2026-10-14。実装の基準は
[approved計画](../../../ai/plans/2026-10-14-mv-p5b-gpu-wrapper.md)。

**P5bの固定した検査 L0〜L4 はすべて合格（2026-10-15）。P5cは未実行（別の承認で行う）。**

| 検査 | 結果 |
|---|---|
| L0 指紋 | P5aの5ファイルの sha256 が計画§0と同じ。束の25配列が P5a の Philox 本番の `config.json` と一致 |
| L1 P5aの再現 | 2万履歴・保存済みバッチ0・2（各10万）の3件すべてで、`brems[1]` 以外の全出力が `np.array_equal`。`brems[1]` の差 2.8e-13 / 4.5e-13 / 2.7e-12 MeV（許容幅 1.1e-10 / 2.4e-9 / 2.4e-9） |
| L2 行の対応 | n=20000・100000 の全履歴で一致 |
| L3 誤りの検査 | F3 乱数の回数の一致 24.96%で拒否、F2 ビン一致 24.9%で拒否、F1（対照）健全性・回数・ビンとも100%で拒否せず、M 粗いビン9,087セルが幅を超えて拒否 |
| L4 コンパイル（Colab T4） | 7カーネル（M故障版を含む）すべて `compile()` 成功、NVRTCのログは空（警告なし）。全文の sha256 `2ce5fe57…32d0` が手元と一致。CuPy 14.0.1、CUDA 13.0、計算能力 7.5 |

`bundle`・`local` は Claude が手元で実行（`~/.cache/chatcarlo/p5/p5b/local.json`）。束は gmail の Drive `viveMonte/p5/p5b/` へ複写し sha256 で照合（入力 `7b02611d…920a`、比較の相手 `02f01237…289f`）。ノートブックは計算コード `b96e492` に固定。`p5_kernel.h` の書き換えも、入口側のマクロの追加も要らなかった。

## 成果物

- [共通入口](../../../../prototypes/mv_electron/gpu/p5b_wrap.h): 履歴ごとの局所出力、
  表の全フィールド、入力ごとの乱数初期化、Philox、行の対応。
- [CPU入口](../../../../prototypes/mv_electron/gpu/p5b_cpu.cpp) と
  [GPU入口](../../../../prototypes/mv_electron/gpu/p5b_gpu.cu): CPUは履歴ごと専用。
  GPUは局所出力を倍精度・整数の原子的操作で集計する。M故障版も含む。
- [共通判定](../../../../prototypes/mv_electron/gpu/p5b_core.py): NumPy以外の外部依存なし。
  L1/L2、C1〜C5、束の照合、途中保存の構成・履歴範囲・形・型・内容ハッシュの照合。
  C2-dは本番では報告のみ、L3だけ99%を適用する。C4はバッチごとの対の差のSEMを使う。
- [手元CLI](../../../../prototypes/mv_electron/gpu/p5b.py): `bundle` / `local` / `compare`。
  `p5a`は手元用の関数内でだけ読み込む。
- [Colab CLI](../../../../prototypes/mv_electron/gpu/p5b_colab.py) と
  [ノートブック](../../../../prototypes/mv_electron/gpu/p5b_colab.ipynb): コンパイル確認と、
  別承認後のP5c実行を別セルにした。Colabで物理・Numba・P5aを読み込まない。
- [速いテスト](../../../../prototypes/mv_electron/tests/test_p5b.py): 人工配列の境界・両側ゼロ・
  SEMゼロ・再開拒否、数百履歴のCPU実行、故障を指定の判定に直接入力する検査。

追加したファイルは計画§1の9ファイルだけ。P5a/P5a0、凍結ファイル、製品コードは編集していない。
削除処理は追加していない。ビルドやテスト用npzは `/tmp/chatcarlo_p5b_*` に保持する。

## 指紋とビルド

P5aの5ファイルは開始時・終了時とも次のSHA-256（計画§0に一致）。

| ファイル | SHA-256 |
|---|---|
| p5_cpu.cpp | `11fd05ed68100db0b3bc02231c659f86b72b7403ae17fc88085d91791e9a7f81` |
| p5_kernel.h | `39d82e04f5c969c92c1ee35d026241119364134fecf1db801ff5d0a4a44bd549` |
| p5_rng.h | `6c3257588ee4be2d0881a05f5d0869b65b942f68fd596a9f0ea981bcaaa9760f` |
| p5a0_common.h | `8c209cfe016241bbbb5d586830dfe5a1ddff648f8f75d3b431dd2ae6d61e70c2` |
| p5a.py | `2609e2a6ce666b3d56214e11929dcd642df261765b6bf55cdc1364d059743934` |

組み立てたGPUソース全文のSHA-256:
`2ce5fe575ee6317b1be5cd3c0f319ed1cd9bc07b19fb104e1addcb11782832d0`。
NVRTC用に入口側で `INFINITY` をIEEEの正の無限大のビット列として定義した。
核の意味・ファイルは変更していない。

CPU指定は `clang++ -O2 -ffp-contract=off -std=c++11 -shared -fPIC
-DP5_RNG_PHILOX -DP5_TRANSPORT_FAULT=k -DP5_STACK_CAP=64`。
GPU指定は計画どおり `-DP5_GPU -DP5_RNG_PHILOX --fmad=false
-DP5_TRANSPORT_FAULT=0 -DP5_STACK_CAP=64`、故障版にのみ `-DP5B_MERGE_FAULT=1`。
1ブロック128スレッド。速さの測定は行っていない。

## 砂場内のテスト

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5b.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
```

新規テスト: **42 passed in 14.99s**、終了コード0。
試作全体: **712 passed, 2 skipped in 894.38s (0:14:54)**、終了コード0。
既存の670合格・2スキップに新規42合格が加わった件数。
`git diff --check`と新規9ファイルの空白検査は合格。既存の追跡ファイルに差分なし。

数百履歴で、P5a Philoxの本番出力とのL1の規則による一致、履歴の分割の完全一致、
通常の集計のC3合格を確認した。F3はC2-c、F2はC2-dのL3条件、Mの模擬はC3が拒否し、
F1は健全性・乱数回数・L3のビン条件で拒否されなかった。
正式な2万履歴、保存済みバッチ0/2各10万履歴の再現は未実施。

初回の新規テストは38 passed / 2 failed。人工配列の単位変換と総和の丸めに対して
ビット一致を期待したテストの不備を修正した。物理、核、固定判定の閾値は変えていない。
途中の全体テストは2回中断した。1回目はSEM=0のzの表示を計画指定の「—」に揃えるため
（134 passed, 1 skipped）、2回目は許容範囲の大きいnでも履歴ごとの出力の配列オフセットが
32ビットであふれないようにするため（88 passed, 1 skipped）。行の式は計画どおりintのまま。
両回とも終了コード2で、最終回帰の合格件数には含めない。

## 砂場外で残る完全なコマンドと引き渡し

リポジトリのルートからClaudeが実行する（既定の保存先は `~/.cache/chatcarlo/p5/p5b/`）。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p5b bundle
.venv/bin/python -m prototypes.mv_electron.gpu.p5b local
```

`bundle`は保存済みP5aのconfigと25配列の形・型・ハッシュを照合し、入力・CPU比較相手・
manifestを作り、gmail側のDrive同期フォルダ `マイドライブ/viveMonte/p5/p5b/` へ複写し、
バイト数とSHA-256を照合する。gmailの同期フォルダが1つに特定できない場合は停止し、
`--drive`で実際の同期先を指定する。`local`はL0〜L3を行い `local.json` に保存する。
不合格なら計画§5に従って停止し、Claudeの独立確認を待つ。

この実装を含むコミットを引き渡した後、ノートブックの `CODE_COMMIT` にその40桁の
固定コミット、`EXPECTED_INPUT_SHA256` と `EXPECTED_REFERENCE_SHA256` にbundleの
報告値を設定する。現時点ではそのコミットも束も未作成なので、この3値は空欄で、
未設定のままではノートブックは停止する。

Colabで使用者がP5bコンパイル確認セルを実行する。セルが呼ぶ完全なコマンド:

```bash
python -m prototypes.mv_electron.gpu.p5b_colab compile \
  --bundle /content/drive/MyDrive/viveMonte/p5/p5b \
  --results /content/drive/MyDrive/viveMonte/p5/p5b/colab_results
```

`p5b_compile.json`の全文SHA-256、全7コンパイルの成功、NVRTCログ全文・警告・版を確認する。
L4不合格なら進まない。P5cは別の承認後にのみ、ノートブックで `P5C_APPROVED=True` とし、
次を実行する（今回未実行）。

```bash
python -m prototypes.mv_electron.gpu.p5b_colab run \
  --bundle /content/drive/MyDrive/viveMonte/p5/p5b \
  --results /content/drive/MyDrive/viveMonte/p5/p5b/colab_results
```

手元で戻った結果を再判定する場合は、実際のgmail同期先へ結果が戻った後、次を実行する。
アカウント名を含むパスは `bundle` が返すDriveのパスを使う。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p5b compare \
  --work "$HOME/.cache/chatcarlo/p5/p5b" \
  --results "$HOME/.cache/chatcarlo/p5/p5b/colab_results"
```

このコマンドは、結果一式を先に `~/.cache/chatcarlo/p5/p5b/colab_results/` へ複写してある場合。
Driveから直接比較する場合は `--results` を実際の同期先の `colab_results/` にする。
`compare`はC0〜C5を共通判定で再確認し `compare.json` に保存する。
P5cの数値報告は本ファイルに混ぜず、計画どおり `RESULTS_P5C.md` に記録する。
