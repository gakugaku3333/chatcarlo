# P8f の実装と引き継ぎ

承認計画: [2026-10-21-mv-p8f-gpu-materials.md](../../../ai/plans/2026-10-21-mv-p8f-gpu-materials.md)。
実装は P8e と同じ深さだけの積層。物理の変更・速度改善は行わない。
T4 の検証と判定 A/B は未実行であり、P8f を採用済みの核として扱わない。

## 実装

- `gpu/p8f_materials.h`: ハッシュで照合した材料表を値を変えずに格納し、材料関数・殻選択・緩和を移植。
- `gpu/p8f_kernel.h`: 共通の `p8f_history`。全粒子の xyz、材料境界 (a)、蛍光、3 系統の深さ採点、収支、スタック上限を扱う。採点は `p8f_sink` に集約。
- `gpu/p8f_cpu.cpp`: 連続 MT と履歴番号ごとの Philox。履歴途中の `brems` は MT では直接加算。
- `gpu/p8f_gpu.cu`: 同じ履歴関数を完走後、次の履歴を atomic で補充。40 ブロック × 128 スレッド。集計と履歴別の出力を別に保存。
- `gpu/p8f_core.py`: 格納の検査、履歴別の比較、付与の実数に基づく丸めの許容、対応する外側バッチの差の SEM、固定 Dref の A/B 判定。
- `gpu/p8f_colab.py` / `.ipynb`: 束を照合し、関門 2 → 3 → 本番。外側バッチごとの確定保存・再開・バッチ 0 の再計算。

Numba と C++ の関数試験は、Numba の MT 配列・添字・続く 16 個の乱数に加え、C++ が報告した消費数で NumPy の MT を進めた状態も照合する。
状態固定の試験には累積確率の境目・最後の正の要素、緩和の非放射・外殻・CUT の両側・K→L、制動放射の受理・複数棄却・包絡超過、境界の省略・反転・終端、事象の優先順位を含む。
実際の履歴に状態を渡す試験では、蛍光光子の CUT 未満の吸収、境界・エネルギー・hard の競合を P8e と照合する。鉛中の横向き 10 MeV 電子では、最初の事象を制動放射に固定し、親のスタックへの退避と再開を検査する。

CPU 上の GPU ドライバーのエミュレーションは、実際の起動コードで 15,377 履歴の補充と末尾、初期化を落とした故障を検査する。
これは CUDA 数学関数・T4 の atomic・デバイスコンパイルの検査ではなく、関門 2 の代わりにはならない。

## Codex で実行する検査

リポジトリ直下で実行する。これらは既存の `~/.cache` の入力を読むが、そこへ出力しない。

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k p8f
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m prototypes.mv_electron.gpu.p8f local
```

`local` は関門 0a〜0e の試験と手渡しの試験を実行し、`docs/validation/mv/p8f/gate0.json` に現在の入力とコードの指紋を記録する。
0f の 10⁶ 履歴は実行しない。ローカル試験のスタック上限 4096 は仮の安全な試験容量で、本計算の容量の決定ではない。

## Claude が実行するコマンド（未実行）

`~/.cache` への書き込みを伴うため、Codex からは実行していない。
順に実行し、前の関門が通るまで進まない。

```bash
# 0f: Numba 10^6 履歴で maxdepth を実測し、容量を max(16,2*maxdepth) に決定。
# 正常の C++ の健全性のあと、容量 1 の故障が counts[6] で拒否されることを確認。
# C++ の時間から 10^7 履歴の参照の所要時間も報告する。
.venv/bin/python -m prototypes.mv_electron.gpu.p8f stack

# 1a: 保存済みバッチ 0・50・99 の 12 出力をバイト照合し、枝の通過回数を報告。
# 1b: 旧来の水の条件。1c: 新しい Numba 参照に一致したあと、4 つの別ビルドの故障を数値で検出。
.venv/bin/python -m prototypes.mv_electron.gpu.p8f gate1

# CPU Philox: 固定状態の試験、20,000 履歴、100,000 履歴 × 100 外側バッチ。
.venv/bin/python -m prototypes.mv_electron.gpu.p8f reference

# 束の作成。材料・蛍光・スペクトル・幾何と定数・参照・ソース・ノートブックを照合して格納。
.venv/bin/python -m prototypes.mv_electron.gpu.p8f export
```

`stack` が参照の見積もりを 3 時間超と報告した場合は、分割・2 プロセスの段取りを使用者に示す。
`reference` は `P8F_BATCH_RANGE=0:50` または `50:100` で外側バッチを分けられる。最大 2 プロセスとする。
共有する固定状態・20,000 履歴の参照を最初に 1 プロセスで確定してから、外側バッチの範囲を分ける。

```bash
P8F_BATCH_RANGE=0:1 .venv/bin/python -m prototypes.mv_electron.gpu.p8f reference
P8F_BATCH_RANGE=1:50 .venv/bin/python -m prototypes.mv_electron.gpu.p8f reference &
P8F_BATCH_RANGE=50:100 .venv/bin/python -m prototypes.mv_electron.gpu.p8f reference &
wait
```

保存済みの参照は添え書きとハッシュを照合し、存在しないバッチだけを計算する。
合否の基準や履歴数は変えない。

### Colab CLI（2026-10-21 に実際に使った手順）

Drive とブラウザの代わりに Colab CLI（`colab`、uv で導入、gmail でサインイン済み）を使った。スクリプトは
[`prototypes/mv_electron/gpu/p8f_cli/`](../../../../prototypes/mv_electron/gpu/p8f_cli/)。

```bash
prototypes/mv_electron/gpu/p8f_cli/start_vm.sh   # T4 を作り、束を送り（150 MB 超は 50 MB に分割）、照合して起動
prototypes/mv_electron/gpu/p8f_cli/watch.sh      # 20 分ごとに確認し（2026-10-21 のユーザー決定。P8f は 4 分で保たれた。20 分は未検証）、結果を ~/.cache/chatcarlo/p8f/cli/results へ回収
colab stop -s p8f                                # 終わったら必ず止める
cp -R ~/.cache/chatcarlo/p8f/cli/results/. ~/.cache/chatcarlo/p8f/results/
.venv/bin/python -m prototypes.mv_electron.gpu.p8f analyse
```

- 確認を止めると VM が返却されることがある（1 回目は約 40 分で消えた。原因はカーネルが使われていないことと推定、n=1）。`watch.sh` を必ず回す。途中で消えたら `start_vm.sh` をもう一度実行すれば、回収済みの結果を送り直して続きから回る。
- Colab の既存の `cupy-cuda13x` に `cupy-cuda12x` を重ねない（`assemble.py` は重なっていたら起動しない）。

### Drive と Colab（当初の手順。今回は使っていない）

`export` は `~/.cache/chatcarlo/p8f/export/` を作る。全ファイルを gmail の
`MyDrive/viveMonte/p8f/` にコピーする。全ファイル名と SHA256 は `p8f_manifest.json` にあり、コマンドでも一覧を表示する。
少なくとも次のファイル群が Drive に揃うまで待つ。

- `water_bundle.npz`、`lung_bundle.npz`、`bone_bundle.npz`、`lead_bundle.npz`
- `water_fluorescence.npz`、`lung_fluorescence.npz`、`bone_fluorescence.npz`、`lead_fluorescence.npz`、`bundles.json`
- `p8f_inputs.npz` / `.json`、`p8f_spectrum.npz` / `.json`、`p8f_water_baseline.npz`
- `p8f_ref_fixed.npz` / `.json`、`p8f_ref_gate2.npz` / `.json`
- `p8f_ref_batch_000.npz` / `.json` から `p8f_ref_batch_099.npz` / `.json` まで（200 ファイル）
- `p8f_source.zip`、`p8f_manifest.json`、`p8f_colab.ipynb`

ノートブックを Colab で開き、T4 を選び「すべてのセルを実行」。Drive の許可は gmail アカウントを使う。
関門 2 は先頭の最大 2 診断地点での累積乱数呼び出し数の一致率 99% 以上を使い、総消費数の完全一致は求めない。
bin・境界・蛍光の一致率と最初の相違は報告だけ。相違を物理の差と断定せず、保存した診断と固定状態の検査から分岐を調べる。
関門 3 の精度が不足したらそこで停止し、追加計算は使用者が決める。

結果は Drive の `results/` に外側バッチ単位で保存する。本番は種 820101、各 200,000 履歴 × 100。
途中終了したら同じ「すべてのセルを実行」で再開する。確定済みのファイルは添え書きを照合し、バッチ 0 を再計算して丸めの許容内で比較する。

### 結果回収後の判定（Claude、未実行）

Drive の `results/` を `~/.cache/chatcarlo/p8f/results/` にコピーしてから実行する。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p8f analyse
```

全関門・100 バッチの健全性・参照の封印・種・履歴の範囲・束のハッシュを確認したうえで、固定
`Dref = 0.060387587985606006` を A/B の両方に渡す。
`run_p8e.block_statistics` と `classify_blocks` を再利用し、`judge_batches` に保存 MT を渡さない。
判定はこの配置の 19 ブロックで採用基準に届いたかまでとし、同等性とは言わない。
