# 計画: GPU エンジン G2b — CLI の `run --engine gpu --dose-grid` を Colab の承認と照合に通す

- 状態: draft（骨子。実装手順はまだ書かない）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-22
- Codexレビュー: 0巡
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md)。前段: [G2a](2026-10-22-chatcarlo-gpu-g2-voxel-tally.md)（implemented）、[G4](2026-10-21-chatcarlo-gpu-g4-cli.md)（implemented）
- 規模: 重（線量の数値と、承認した条件と実際の計算の一致。誤ると静かに違う格子の線量が出る）

## 目的

G2a で GPU の核に線量格子のタリーが入ったが、使えるのは Python の API（`run_transport(engine='gpu', dose_grid=True)`）だけ。
G4 の CLI（`preview --engine gpu --gpu-device colab` → `approve RUN_DIR` → `run`）は検出器の画像しか扱わず、`--dose-grid` を拒む。

この計画が答える問い: **CLI の `run --engine gpu --gpu-device colab --dose-grid --resolution r` で、使用者が承認した格子（解像度・原点・形）と同じ格子で GPU が線量を数え、その結果が本番と同じ形の `.npz`（`--dose-out`）と同じ表示（最大値、R、警告）になるか。承認した格子を改ざんすれば止まるか。**

## 0. 前提事実（検証済み、2026-10-22 に実行・読解）

**F7 既存コードとの契約**
- `gpu/condition.py` の `resolve` は、`engine == 'gpu'` で `dose_grid` があると `ValueError('G1 does not support voxel dose_grid tally')` を出す（G4 のまま）。`resolve` は `dose_grid` のとき `grid_resolution_cm`（`--resolution`、既定 5）も条件に入れる。
- `gpu/cli_run.py` の `run`:
  - 手元（`--gpu-device cpu`）は `engine.run(..., dose_grid=False, ...)` を呼ぶ。格子を渡していない。
  - Colab は `expected_run`（束を作り直して条件を作る）→ `approval.require` → `colab_driver.launch` → `approval.verify_batches` → `aggregate` の順。
- 条件（`condition.colab_condition`）は、束のハッシュ（`identity(manifest, job, ...)['bundle_sha256']`）、`job`（`n_histories`・`batch_size`・`seed`・`source_seed`）、`detector`、`roi`、`track_uncertainty` からなる。**格子の情報は入っていない。**
- 受領票の「核に渡った引数の観測」（`core.Core._observe`、`expected_observation`）は、G1 の入口の引数の並び（`CPU_INPUTS`・`CUDA_INPUTS`）だけを見る。**G2a の `g2_core.Core` が足した引数（`g2_table`・`cfg`（格子の原点・形・解像度・箱）・出力の配列）は観測されない。** つまり今のままでは、VM で格子の解像度を変えられても照合で止まらない。
- VM の `colab_driver.worker` は `core.Core`（G1）しか使わない。格子の設定は束にも `job.json` にも無い（G2a の関門 2 は、検証専用に `g2_tables.npz` に格子を書き足した）。
- `bundle.create(..., g2_tally=True)` は係数の表（`g2_tables.npz`）を束に入れる（G2a）。
- 結果のファイルは `write_result` が `np.savez_compressed` で書き、受領票に SHA256 と識別情報を入れる。
- `__main__.cmd_run` の後半（`--dose-grid` の表示と `--dose-out` の保存）は `result.grid`（`VoxelGrid`）を読む。G2a の `engine.run` は `result.grid` に本番と同じ `VoxelGrid` を入れる。
- `colab_driver.launch` は、G1 の関門の記録（`docs/validation/gpu/g1/gate1.json`・`gate2.json`）が合格であることを確かめてから送る。

**F10 実物の大きさ（G2a の関門 2 の実データ）**
- 2 cm 格子 298,900 ボクセル・10⁵ 履歴の 1 バッチの格子（カーマ＋H*(10)）は、圧縮して 3.28 MB。
- これから見積もると、angio_room の 5 cm 格子（1,137,600 ボクセル）は 1 バッチ約 12 MB、10⁷ 履歴を 2×10⁵ ずつ（50 バッチ）なら約 600 MB を Colab から回収する。

**Colab の運用（`.claude/rules/chatcarlo.md`・`mv.md`）**: 10 分ごとに確認し、確認のたびに出来上がった結果を回収する。`colab` の呼び出しには制限時間を付ける。起動スクリプトとは別に、生存記録を見張る。G4 の `launch` は終了を完了の印のファイル（`done.json`）で判断する。

## 1. 決めること（骨子のレビューで確かめたい点）

| # | 決めること | 推奨 | 理由 |
|---|---|---|---|
| H1 | 格子をどこで決めるか | ホストで `Geometry(...).tally_bbox` と `VoxelGrid.from_bbox(..., resolution)` から原点・形・解像度を決め、**条件（condition）に入れる**。束の `g2_tables.npz` にも同じ値を書き、条件と束の両方で照合する | 承認した条件に格子が入らないと、承認の意味が無い。本番と同じ作り方（G2a の契約） |
| H2 | 照合の対象を広げる | G2 の入口の引数の観測を、G1 の観測と同じ方式で足す: `g2_table`・`cfg` の SHA256（位置つき）。出力の配列は観測しない（中身は結果の SHA256 で照合済み）。G1・G4 の観測の並びは変えない | 格子の解像度・原点・形の改ざんを止めるため。既存の受領票の形を壊さない |
| H3 | VM の worker | `job` に `dose_grid: true`（と格子）があるときだけ、`worker` が `g2_core.Core` を使う。無ければ今と同じ（G4 の結果を 1 ビットも変えない） | G4 の回帰を守る |
| H4 | バッチの結果の形 | バッチごとの格子（カーマ・H*(10)）を結果の `.npz` に入れ、ホストで `VoxelGrid` に足して `end_batch` する（本番と同じ統計）。大きさは 0 節の見積もり | 本番と同じバッチ統計（R・寄与バッチ数）を作るには、バッチごとの格子が要る |
| H5 | 合格の判定 | 下の 2 節 | |
| H6 | 段の分け方 | G2b は「CLI と承認・照合・保存の配線」だけ。速さの工夫、cylinder 等（G3）、格子の転送の圧縮の工夫はしない | 計画を 25 KB 以下に |

## 2. 事前登録する判定（骨子）

- **関門 A（手元、CPU の双子）**: `run --engine gpu --gpu-device cpu --dose-grid` が、`run_transport(engine='gpu', dose_grid=True)`（G2a で合格した経路）と、同じ種でバイト一致の `.npz`（`--dose-out`）を書く。表示（最大値、R、警告）も同じ。
- **関門 B（手元、照合の故障検査）**: 承認の後に次を変えると、計算の前か結果の照合で止まる（G4 と同じ形の故障注入）。
  - 条件の解像度、束の `g2_tables.npz`、VM に渡る `cfg`（格子の形・原点・解像度）、VM の `g2_table`、`job` の `dose_grid` の有無。
- **関門 C（Colab T4、通し）**: G2a の S2（`scene_lead_slab.yaml`、2 cm）を `preview → approve → run` で 20 バッチ × 2×10⁵ 回し、
  - (i) 受領票の観測（G1 と G2 の引数）がすべて照合を通る
  - (ii) 保存した `.npz` のカーマ・H*(10) が、照合済みのバッチの格子の和とバイト一致する
  - (iii) G2a の関門 1 の参照（`data/gpu_g2/ref_lead_batches.npz`、本番 NumPy、固定済み）と、G2a の関門 1 と同じ判定（外れ・広がり・領域の合計・採点漏れ）で一致する（独立な標本の統計の比較）
  - 種は詳細の段で決めて、実行の前に書く。
- **関門 D（Colab、故障）**: 関門 B のうち VM で起きるもの（`cfg` の改ざん、`g2_table` の改ざん）を Colab で 1 回ずつ注入し、止まることを確かめる。

## 3. 対象範囲（骨子）

- 変更してよい: `chatcarlo/gpu/condition.py`・`cli_run.py`・`approval.py`・`colab_driver.py`（`worker` の分岐、結果の識別）・`g2_core.py`（観測の追加）・`aggregate.py`、`chatcarlo/__main__.py`（`run` の GPU の分岐で格子を表示・保存へ渡す部分だけ）、新規 `tests/test_gpu_g2b.py`、新規 `docs/validation/gpu/g2b/`。
- 変更禁止: `transport.py` の `numpy` 経路、`tally.py`、`materials.py`、`dose_coefficients.py`、`kernel.py`、G1・G2a の核（C++）、G4 の検出器の画像の経路の出力（G4 の既存のテストと結果が変わらない）、`data/`、`prototypes/`。
- スコープ外: cylinder・sphere・`mas`（G3）、`--workers`、格子の転送の工夫、`preview` の 3D 表示に格子の枠を描くこと（承認の HTML には文字で格子の解像度・原点・形を出すだけ）。

## 4. 停止条件

- 関門 A・B が落ちたら Colab に進まない。
- 判定の基準は結果を見てから変えない。変える必要が出たら、不合格を記録し、追補を書いてユーザーの承認を取ってから、新しい種で回す（G2a の追補 2 と同じ扱い）。
- G4 の既存のテスト・回帰（検出器の画像）が変わったら止まる。

## 5. まだ書いていないこと（詳細の段で書く）

関門 C・D の種と履歴数、故障注入の方法、観測の位置の番号、結果の識別（`identity`）への格子の入れ方、承認の HTML の文言、テストの一覧、Colab の手順（G2a の `colab_g2.py` を使うか G4 の `launch` を使うか）。
