# 計画: GPU エンジン G2b — CLI の `run --engine gpu --dose-grid` を Colab の承認と照合に通す

- 状態: draft・保留（2026-10-22 ユーザー判断で後回し。詳細まで書いたが Codex 2 巡目のレビューは未実施。再開するときは 2 巡目のレビューから）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-22
- Codexレビュー: 1巡（骨子: 致命 0・実質 7・軽微 2・参考 2。原文 `data/logs/codex_g2b_review1.md`。すべて反映。格子の仕様を `job` 1 か所に置く Codex の代替案を採用）
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md)。前段: [G2a](2026-10-22-chatcarlo-gpu-g2-voxel-tally.md)（implemented）、[G4](2026-10-21-chatcarlo-gpu-g4-cli.md)（implemented）
- 規模: 重（線量の数値と、承認した条件と実際の計算の一致。誤ると静かに違う格子の線量が出る）

## 目的

G2a で GPU の核に線量格子のタリーが入ったが、使えるのは Python の API（`run_transport(engine='gpu', dose_grid=True)`）だけ。
G4 の CLI（`preview --engine gpu --gpu-device colab` → `approve RUN_DIR` → `run`）は検出器の画像しか扱わず、`--dose-grid` を拒む。

この計画が答える問い: **CLI の `run --engine gpu --gpu-device colab --dose-grid --resolution r` で、使用者が承認した格子（解像度・原点・形）と同じ格子で GPU が線量を数え、その結果が本番と同じ形の `.npz`（`--dose-out`）と同じ表示（最大値、R、警告）になるか。承認した格子を改ざんすれば止まるか。**

## 0. 前提事実（検証済み、2026-10-22 に実行・読解）

**F7 既存コードとの契約**
- `gpu/condition.py` の `resolve` は、`engine == 'gpu'` で `dose_grid` があると `ValueError('G1 does not support voxel dose_grid tally')` を出す（G4 のまま）。`dose_grid` のとき `grid_resolution_cm` を入れるコードはあるが、GPU ではその前に例外になるので、**今の GPU の経路では格子の条件は作られない**。
- `__main__` の `preview` には `--dose-grid`・`--resolution` が無い（承認用の preview で格子を指定できない）。
- `gpu/engine.py`（G2a の Python API）は、観測を G1 の `expected_observation` と完全一致で比べる。G2 の観測を足すと、ここも直さないと止まる。
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

- 容量の目安: 1,137,600 ボクセルの 2 量は、圧縮しないと 1 バッチ約 18.2 MB（50 バッチで約 910 MB）。圧縮率は採点の密度と履歴数で変わるので、600 MB は保証ではない。

**Colab の運用（`.claude/rules/chatcarlo.md`・`mv.md`）**: 10 分ごとに確認し、確認のたびに出来上がった結果を回収する。`colab` の呼び出しには制限時間を付ける。起動スクリプトとは別に、生存記録を見張る。G4 の `launch` は終了を完了の印のファイル（`done.json`）で判断するが、`colab` の呼び出し（`subprocess.run`、191 行）に制限時間が無い。

## 1. 決めること

| # | 決めること | 決定（Codex の代替案を採用） | 理由 |
|---|---|---|---|
| H1 | 格子の正典 | **`job.grid = {origin_cm, shape, voxel_size_cm, layout_version}` を唯一の正典にする**（格子が有効なときだけ持つ。無効なら省略し、今の `job` とバイト一致）。ホストで `Geometry(...).tally_bbox` と `VoxelGrid.from_bbox(..., resolution)` から作る（本番と同じ）。条件は今どおり `job` 全体を含むので、格子も承認の対象になる。承認の HTML はこの仕様から文字で表示する。`g2_tables.npz` は係数の表だけにし、格子を重ねて書かない。受領票の識別（`identity`）は `job` 全体を含むので、格子専用の識別は増やさない | 1 か所に置けば、「表示・条件・束・job・VM の cfg・集計の格子・保存」が食い違う余地が減る |
| H2 | 値で結ぶ不変条件 | 承認の表示 ＝ `job.grid` ＝ ホストで作る期待の `cfg` ＝ VM の実際の `cfg` ＝ 集計の `VoxelGrid` ＝ 保存した `.npz` の原点・形・解像度、を**値で**照合する。形の取り違え（ボクセルの総数が同じで形が違う）も検出する | ハッシュがそれぞれ正しいだけでは、三者が同じ格子とは限らない |
| H3 | 観測の場所 | G2 の入口では、型の変換と故障注入の後の**最終の引数の並び**を 1 度作り、それを観測と呼び出しの両方に使う。追加の位置は CPU の 19・20、CUDA の 25・26（`g2_table`・`cfg`）。内容の書き換えを見逃さないよう、G2 の引数はアドレスでのキャッシュを使わず、毎回内容をハッシュする。G1 の観測の並びは変えない | 継承元が G1 の引数を観測した後に G2 の引数を足す今の形では、最終の引数の差し替えを検出できない |
| H4 | 出力の対応 | 出力の内容は呼び出しの前にハッシュしないが、**ABI の各出力の位置と、回収して名前を付けるバッファの対応**を確かめる（例: カーマと H*(10) の出力ポインタの交換を故障注入で検出）。ホストでも必須のキー、dtype、長さ、有限・非負を確かめる | 結果の SHA256 は転送中の改変しか保証しない |
| H5 | VM の worker | `job` に `grid` があるときだけ `g2_core.Core` を使う。無ければ今と同じ（G4 の結果を 1 ビットも変えない） | G4 の回帰を守る |
| H6 | バッチの結果と集計 | バッチごとの格子（カーマ・H*(10)）を結果の `.npz` に入れ、ホストで `VoxelGrid` に足して `end_batch(n)` を呼ぶ（本番と同じ統計）。手元（CPU の双子）の CLI の経路も同じ集計の関数を通す | 本番と同じ R・寄与バッチ数を作るため。手元と Colab の集計の経路を分けない |
| H7 | Colab の起動 | 通常の `preview → approve → run` の経路で、G4 の `launch` を使う。`launch` の各 `colab` の呼び出しに制限時間を足す（変更してよい範囲に入れる）。独立の見張りと、終了時の `colab stop` も確かめる。G2a の検証専用の `colab_g2.py` は使わない | 目的は通常の経路の検証。運用の決まりを満たす |
| H8 | 段の分け方 | G2b は「CLI と承認・照合・集計・保存の配線」だけ。速さの工夫、cylinder 等（G3）、格子の転送の工夫はしない | 計画を 25 KB 以下に |

## 2. 事前登録する判定（骨子）

- **関門 A（手元、CPU の双子）**: `run --engine gpu --gpu-device cpu --dose-grid --resolution 2 --dose-out X.npz` の `.npz` を、同じ種の `run_transport(engine='gpu', dose_grid=True)` の結果から既存の保存の規約で作った期待の配列と、キー・dtype・形・各配列のバイト列で比べて一致。線量に関わる表示（最大値、R、寄与バッチ数、警告）も一致（時間などの表示は除く）。
- **関門 B（手元、照合と集計の故障検査）**:
  - 承認の後に次を変えると、計算の前か結果の照合で止まる: `job.grid` の各値、形の取り違え（総数が同じ）、`g2_tables.npz`、最終の引数の `cfg`・`g2_table`（差し替えと内容の書き換え）、`job` から `grid` を消す、出力ポインタの交換。
  - 集計: 照合済みの同じバッチから独立に、順序つきの和、ΣS_b²/n_b、寄与バッチ数、履歴数・バッチ数を作り直し、保存した R・SEM・Gy/history・pSv/history と一致。端数のバッチ、バッチ 1 個、寄与 0、統計無効も扱う。
- **関門 C（Colab T4、通し）**: G2a の S2（`scene_lead_slab.yaml`、2 cm）を `preview → approve → run` で、**回収済みなしから** 20 バッチ × 2×10⁵ 回す（`new_batches=20`・`reused_batches=0` を求める）。
  - (i) 受領票の観測（G1 と G2 の引数）がすべて照合を通る。
  - (ii) 保存した `.npz` のカーマ・H*(10) が照合済みのバッチの格子の和とバイト一致し、R・SEM・寄与バッチ数が関門 B と同じ独立の作り直しと一致する。
  - (iii) G2a の関門 1 の参照（`data/gpu_g2/ref_lead_batches.npz`、sha256 9e6d3ab5…、本番 NumPy、固定済み）と、G2a の関門 1 と同じ判定で一致する（独立な標本の統計の比較。判定の前に参照のハッシュ、シーン、格子、履歴数、種を確かめる）。
  - 種は詳細の段で決めて、実行の前に書く。
- **関門 C2（再開）**: 関門 C とは別に、途中で止めて再開したとき、再利用したバッチも G2 の観測を照合する。
- **関門 D（Colab、故障）**: 関門 B のうち VM で起きるもの（最終の引数の `cfg`・`g2_table` の差し替え）を Colab で 1 回ずつ注入し、止まることを確かめる。一般的な GPU のメモリの破損への保証は求めない。

## 3. 対象範囲（骨子）

- 変更してよい: `chatcarlo/gpu/condition.py`・`cli_run.py`・`approval.py`・`colab_driver.py`（`worker` の分岐、`launch` の制限時間）・`g2_core.py`（最終の引数の観測）・`engine.py`（G2 の観測の照合）・`aggregate.py`・`bundle.py`（必要なら）、`chatcarlo/__main__.py`（`preview` の `--dose-grid`・`--resolution`、`run` の GPU の分岐で格子を表示・保存へ渡す部分）、新規 `tests/test_gpu_g2b.py`、新規 `docs/validation/gpu/g2b/`。
- 変更禁止: `transport.py` の `numpy` 経路、`tally.py`、`materials.py`、`dose_coefficients.py`、`kernel.py`、G1・G2a の核（C++）、G4 の検出器の画像の経路の出力（G4 の既存のテストと結果が変わらない）、`data/`、`prototypes/`。
- スコープ外: cylinder・sphere・`mas`（G3）、`--workers`、格子の転送の工夫、`preview` の 3D 表示に格子の枠を描くこと。

## 4. 停止条件

- 関門 A・B が落ちたら Colab に進まない。
- 判定の基準は結果を見てから変えない。変える必要が出たら、不合格を記録し、追補を書いてユーザーの承認を取ってから、新しい種で回す（G2a の追補 2 と同じ扱い）。
- G4 の既存のテスト・回帰（検出器の画像）が変わったら止まる。

## 5. 実装の形（詳細）

**`job.grid` の定義（`layout_version: "g2-grid-v1"`）**
- `origin_cm`: `Geometry(scene.raw['geometry'], detector_plane=plane).tally_bbox_min`（3 個の float）。
- `shape`: `VoxelGrid.from_bbox(...)` の形（3 個の int）。`voxel_size_cm`: `--resolution` の値（float）。
- 平らに並べる順は C 順（`(ix*ny + iy)*nz + iz`）。G2a の `g2_tally.h` と `VoxelGrid` の配列と同じ。
- `grid` は `--dose-grid` のときだけ `job` に入る。入らない `job` は今とバイト一致（G4 の条件と結果の識別が変わらない）。
- 値の作り方は 1 つの関数（`condition.grid_spec(scene, plane, resolution)`）にまとめ、preview・run・手元の経路がすべてそれを呼ぶ。

**値で結ぶ照合（H2）**: 次をすべて `job.grid` と値で比べ、違えば止める。
- `prepare`: 承認の HTML の文字（解像度、原点、形、ボクセル数）を `job.grid` から作る。`preview_snapshot.json` にも `job.grid` を入れる。
- `cli_run.run`（Colab）: `expected_run` で作り直した `job.grid` が承認済みの `job.grid` と一致（今の `require` の条件の一致に含まれる）。
- ホストの期待の `cfg`: `g2_tables.config(arrays, grid_from(job.grid), [], -1)` で作る。
- VM の実際の `cfg`: worker が `job.grid` から同じ関数で作る。観測で照合する。
- 集計の `VoxelGrid`: `job.grid` から作り、`origin_cm`・`shape`・`voxel_size_cm` が `job.grid` と一致することを確かめる。
- 保存: `--dose-out` の `.npz` の `origin_cm`・`voxel_size_cm`・`shape` を `job.grid` と比べる。

**観測（H3・H4、`g2_core.py`）**
- `invoke` の中で、型の変換と故障注入の後の最終の引数の並び（G1 の分＋`g2_table`・`cfg`・出力 6 個）を 1 度作り、それで観測と呼び出しをする。
- 観測の鍵: CPU は `19:g2_table`・`20:cfg`、CUDA は `25:g2_table`・`26:cfg`。値は内容の SHA256（アドレスでキャッシュしない）。
- 出力の対応: 出力 6 個の位置と名前（`kerma`・`h10`・`grid_additions`・`box_sum`・`box_sum2`・`box_additions`）の表を 1 か所に置き、呼び出しと回収の両方がそれを使う。
- `core.expected_observation` は変えず、G2 の分を足す関数 `g2_core.expected_observation(a, cfg, ...)` を作る。`engine.py` と `approval.verify_batches` は、`job.grid` があるときこちらを使う。
- ホストの検査（`health` の G2 版）: 必須のキー、dtype（float64）、長さ（`prod(shape)`）、有限・非負。

**故障注入（関門 B・D）**: G4 と同じく、VM 側は `fault.json`、手元は `CHATCARLO_G4_FAULT` で選ぶ。足す値:
- `g2_cfg`: 最終の引数の `cfg` の解像度を 2 倍にした配列に差し替える。
- `g2_cfg_inplace`: 同じアドレスのまま `cfg` の形の 1 要素を書き換える。
- `g2_table`: `g2_table` の 1 要素を 1 ulp ずらす。
- `g2_swap`: カーマと H*(10) の出力ポインタを入れ替える。
- 期待: `g2_cfg`・`g2_cfg_inplace`・`g2_table` は `observation mismatch: batch_000000 …:cfg`（または `g2_table`）で止まる。`g2_swap` は出力の対応の検査（H*(10) ÷ カーマが係数の比と合わない、を手元で人工データで確かめるのではなく、ポインタと名前の表の照合）で止まる。
- 手元の承認の故障（関門 B）: 承認の後に `job.json` の `grid` を書き換える、`grid` を消す、形を総数の同じ別の形にする（例: (70,70,61)→(61,70,70)）、`g2_tables.npz` を書き換える → G4 と同じく `generation`・`condition` の照合で止まる。

**集計（H6）**: `aggregate(batches, names, detector, roi, track_uncertainty, grid=None)`。`grid` があれば、各バッチで `grid.kerma_keV += r['kerma'].reshape(shape)`、`grid.h10_track_pSv_cm3 += …`、`grid.end_batch(n)`。`engine.run`（G2a）の格子の集計もこの関数に寄せる（同じ結果になることをテストで確かめる）。`result.grid` に入れ、`__main__.cmd_run` の後半（表示と `--dose-out`）は今のまま使う。

**`launch` の制限時間（H7）**: `colab_driver.launch` の `cli()` の `subprocess.run` に `timeout=300` と 3 回までの再試行を足す（G2a の `colab_g2.py` と同じ）。確認の間隔（600 秒）と完了の印（`done.json`）は今のまま。

**CLI**: `preview` に `--dose-grid`・`--resolution` を足す（`run` と同じ意味）。`run --engine gpu --dose-grid` を `resolve` が受け付ける。`--workers` は今どおり 1 だけ。

## 6. 関門の詳細（事前登録）

- **関門 A**（手元）: S2・2 cm・2×10⁵ 履歴・バッチ 5×10⁴（4 バッチ）・seed 2401・source_seed は既定の導出。CLI の `.npz` と、`run_transport(engine='gpu', dose_grid=True, grid_resolution_cm=2, …)` から既存の保存の規約で作った配列を、キー・dtype・形・バイト列で比べる。
- **関門 B**（手元）: 5 節の故障すべてと、統計の独立の作り直し（端数のバッチ: 2.5×10⁵ をバッチ 10⁵、バッチ 1 個、寄与 0 のボクセル、`--no-uncertainty`）。
- **関門 C**（Colab T4）: S2・2 cm・4×10⁶ 履歴・バッチ 2×10⁵（20 バッチ）・**seed 2410・source_seed 2411**。新しい実行一式（回収済みなし）。判定は 2 節 (i)〜(iii)。(iii) の参照は `data/gpu_g2/ref_lead_batches.npz`（sha256 `9e6d3ab523f2101205b374a4188cc17135eb9b5b811029da959dc8ca8a44c84e`）。判定の関数は G2a の `run_gates.judge` をそのまま使う。
- **関門 C2**（Colab）: 関門 C の直後に同じ `run` をもう一度実行し、`new_batches=0`・`reused_batches=20` で、保存する `.npz` が関門 C とバイト一致し、再利用したバッチの観測も照合を通る。
- **関門 D**（Colab）: S2・2 cm・10⁵ 履歴・バッチ 10⁵（1 バッチ）・seed 2420・source_seed 2421。`fault.json` に `g2_cfg`、`g2_table` を入れて 1 回ずつ。どちらも計算の後の照合で止まり、`.npz` を書かない。

## 7. テスト（`tests/test_gpu_g2b.py`、手元、GPU なし）

- T1 `grid_spec` が本番の `VoxelGrid.from_bbox` と同じ原点・形を返す（S2・水ファントム・2 cm と 5 cm）。
- T2 `--dose-grid` の無い `job`・条件・HTML が今とバイト一致（G4 の既存のテストも通る）。
- T3 関門 A。
- T4 関門 B の故障 7 種（5 節）がそれぞれ止まり、止まる場所の文言が期待どおり。
- T5 統計の独立の作り直し（端数、1 バッチ、寄与 0、統計無効）。
- T6 観測の位置（CPU 19・20、CUDA 25・26）と出力の名前の表。CUDA の引数の並びは、CuPy なしで組み立てる関数をテストする。
- T7 `aggregate` の格子の集計と G2a の `engine.run` の結果が一致。
- T8 `launch` の `cli()` が制限時間で例外を出し、再試行する（`subprocess.run` を差し替えて）。
- T9 `preview --dose-grid --resolution 2` の承認の HTML に格子の文字が出る。
- 既存のテストすべて。

## 8. テストコマンド

```bash
.venv/bin/python -m pytest tests/test_gpu_g2b.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m chatcarlo run docs/validation/gpu/g2/scene_lead_slab.yaml --engine gpu --gpu-device cpu --seed 2401 -n 2e5 --batch-size 50000 --dose-grid --resolution 2 --dose-out /tmp/g2b_a.npz
```

関門 C・C2・D（Colab）は Claude が実行する。Codex は手元の受入（関門 A・B、テスト）までで止まる。

## 9. 受入条件（Codex の実装）

- 7 節のテストと既存のテストがすべて通る。
- 3 節の変更禁止のファイルに差分が無い（`git diff --stat`）。
- `docs/validation/gpu/g2b/RESULTS.md` に関門 A・B の結果を書く（Colab の関門は書かない）。

## 10. 枠組みについての Codex の意見（記録だけ）

「この標本の精度で不一致を検出しなかった」は、今回の研究用の配線の検証には適している。線量の精度を保証する段に進むときは、検出したい最小の差と同等性の幅を別に決める価値がある。
