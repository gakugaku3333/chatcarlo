# 計画: G5-A1 — 共通の土台・共通の束（目録）・組の識別子を足し、診断の核を載せ替える（診断の駆動は今のまま）

- 状態: implemented（2026-10-30。A1a・A1b とも全条件合格。結果は docs/validation/gpu/g5/A1.md）
- 作成者: claude
- 実装担当: codex（A1a: コード・単体テスト・駆動の作成と、リポジトリ内で閉じる検査）。Claude（`~/.cache` を使う A1a-7・9・10・12、旧 core の固定、A1b: Colab T4・EGS5・Drive）
- 日付: 2026-10-29
- Codexレビュー: 1巡（致命 0・実質 9・軽微 2・参考 3。原文 `data/logs/codex_g5a1_review1.md`。実質はすべて反映）
- 親計画: [2026-10-29-chatcarlo-gpu-g5-unified-energy.md](2026-10-29-chatcarlo-gpu-g5-unified-energy.md)（approved）の段 A1。A0 の結果: [RESULTS](../../validation/gpu/g5/RESULTS.md)（許容の比 **1.0571**）
- 規模: 重（診断の核と束と承認の同一性に触れる。誤ると静かに違う線量が出る、または承認済みの条件が取り違えられる）

## 目的

親計画の A1 を実装できる形にする。成果は 4 つ。

1. **共通の土台**: Philox・`P5_FN`・型の定義を `g5_base.h` に分け、診断のヘッダーがそれを使う。治療の組（A2）が同じ土台を使えるようにする。
2. **組の識別子**: 物理の組（D9）をコンパイル時の識別子にし、実際にコンパイルされた核から読み出して束の宣言と照合する。
3. **共通の束の目録（catalog）**: 各欄の単位・エネルギーの帯・出典・生成元・ハッシュを持つ目録。**診断の欄を実際に載せ、治療の P8 の 33 欄が同じ形式で往復できること（A2 で形式を作り直さないこと）を A1 で示す**。
4. **関門**: 載せ替えた診断の核が、記録済みの判定を同じ基準で再現し、同じセッションの旧い核より遅くならない（比 ≤ 1.0571）。

方針（ユーザー、2026-10-29）: バイト一致は合否に使わない。統計と速さで決める。**ただし A1 の変更は「既存の行を動かさず、行を足す」形にする**（§1 A1-D1）。そうすれば既存のテスト（MT の鎖 0 を含む）が無改造で通る。**既存テストの全通過は、算術を変えていない部分の回帰の見張りであって、A1 の採否を決める新しいバイト一致の条件ではない**（採否は A1a-9 以降の統計・速さ・各検査で決める。新しく変える部分と A2 以降にはバイト一致を使わない）。

## 0. 前提事実（2026-10-29 に確認）

### 0.1 起点と旧い核の固定

`git rev-parse HEAD` → `19c281e7adbcc702d1df172033166684788137d3`。A1 は `chatcarlo/gpu/` をその場で編集するので、「同じセッションで測り直す旧い核」はこのコミットから `git archive` で取り出して固定する（A1-D4）。A0 の速さの基準・設定は `a0_configs.json`（SHA256 `7392eda4a2b96e880d1071e5370f029328afb8a35f46434d2e545d0eb05340ab`）。

### 0.2 既存の GPU テストが依存している配置

次のうち、ABI・関数名・欄の順・INFINITY の定義位置は A1 で**変えない**（変える必要が出たら止めてユーザーに報告する。§7）。**束の形式名・必須メンバー・ヘッダー構成は A1-D1・D3 で変える**（承認の対象）。既存のテストはこれらを固定していない（`test_gpu_g4.py:288` は `core.SOURCES` から期待集合を作り、束は現行の `bundle.create()` で生成して読む。旧い形式名を固定する assert はない）。ただし `cuda_source()` に `#include "` が残らないこと（`test_gpu_g2.py:212`）は守る。

| 依存 | 場所 |
|---|---|
| `g1_cpu.cpp` を `-DP5_RNG_MT` だけで直接コンパイル（`-D` で組を渡さない） | `tests/test_gpu_g1_functions.py:34` |
| ABI の関数名 `g1_physics_probe`・`g1_geometry_probe` | `tests/test_gpu_g1_faults_tally.py:27,37`、`test_gpu_g1_geometry.py:17`、`chatcarlo/gpu/validation.py:88,142,161` |
| `cuda_source()` の中身（INFINITY の定義位置、`#include "` が展開後に残らないこと）、`g2_tally.h` の本文 | `tests/test_gpu_g1_regression.py:15`、`tests/test_gpu_g2.py:212-220` |
| `core.SOURCES` を束の「重要ファイル」とする | `tests/test_gpu_g4.py:288`、`validation.py:247` |
| 欄の順 `FIELDS`、位置表 `CPU_INPUTS`・`CUDA_INPUTS` | `core.py:13,174-175` |
| 束の形式名 `chatcarlo-g1-v1` と必須メンバーの集合（**A1 で変更。上のとおり承認の対象**） | `bundle.py:17,21` |

`g1_physics_probe` と `fault` の MT 版テスト（`test_gpu_g1_functions.py:126` など）は、物理の移植が `kernel.py` と同じ乱数の鎖で一致することをバイトで比べる。A1 は算術を変えないので、これは**回帰の見張りとして必須のまま**残す。A1 の採否の新しい条件にはしない。

### 0.3 上書きの危険（既存の駆動をそのまま使えない）

| 駆動 | 書く場所 | 扱い |
|---|---|---|
| `docs/validation/gpu/g1/run_gates.py` の `gate1()` / `gate2_merge()` | `HERE/'gate1.json'`・`gate2.json`（記録済みの判定を上書きする） | **使わない**。判定関数 `chatcarlo.gpu.validation.independent_judge` と `collect` の考え方だけ使う |
| `docs/validation/gpu/g2/run_gates.py` の `gate1()` | `ROOT/'data/gpu_g2'`（参照を無条件に作り直す）と `gate0/1.json` | **使わない**。純関数 `judge(reference, twin, regions)` だけ import する |
| `make_reference.py` | 既定 `data/gpu_g2` | 使わない（A0 が `~/.cache/chatcarlo/g5/a0/g2_ref/` に作り直した） |
| `run_egs5_60kev.py` の `prepare(out)`・`compare(out)` | 引数 `out` | `out` を `~/.cache/chatcarlo/g5/a1/g2gate3/` にして import して使う。`compare` は `docs/` の旧記録を読むだけで、出力（BSF・比較・図・出典・RESULTS）は `out` に書く |

### 0.4 レジスターの崖

T4（sm_75）、128 スレッド/ブロックでは、レジスターが、今回の 98〜129 の範囲では **128 以下なら 4 ブロック/SM、129 以上なら 3 ブロック/SM**（一般には 96 なら 5 など他の値もある。割り当ては 1 ワープ 256 単位。`a0_speed.resident_blocks` と同じ式で計算）。A0 の実測: 診断 g2b **120**（余裕 8）、g4 98。足す関数がレジスターを増やすと速さの関門に落ちるので、A1b の最初に確かめる（§3 A1b-1）。

### 0.5 展開と同一性

- `core.cuda_source()` は `#include "…"` を**そのまま再帰展開**する（重複の除去はヘッダーの `#ifndef` 頼み）。新しいヘッダーは必ず同じ形の守りを持つ。
- CPU の `.so` キャッシュは `SOURCES` の本文＋rng＋fault のハッシュ（`core.py:57`）。**新しいヘッダーを `SOURCES` に入れないと、変更しても古い `.so` が使われる**。
- 束の SHA（`bundle.py:55`）は `SOURCES` と `RUNTIME` の全ファイルを含み、承認の条件ハッシュ（`condition.colab_condition`）に入る。**よって A1 の変更の後は、`~/.cache/chatcarlo/gpu_runs/` にある承認済みの G4 の実行フォルダは再開できなくなる**（ファイルは読めるが、同じ条件ハッシュにならない）。

### 0.6 旧い核と新しい核を 1 つの VM に同居させる

`chatcarlo/gpu/*.py` の import は相対のみ（絶対 import は `colab_driver.py:178` の起動用の文字列だけ）。旧い版を `git archive 19c281e chatcarlo/gpu` で取り出し、最上位のパッケージ名を `chatcarlo_g5_old` に変えて置けば、新旧が互いを import しない。

### 0.7 G1 関門 1 の手元の所要

`run_gates.py gate1`（3 シーン × 10 回 × 100 万履歴、NumPy 4 ワーカー＋CPU の双子）は **約 8 分**（`docs/validation/gpu/g1/RESULTS.md:67`）。手元で繰り返せる。

### 0.8 故障の検出力（試し計算、CPU の双子・Philox、乱数 seed は参照 11・正常 22・故障 33）

物理の probe（`g1_physics_probe`）の出力 1 列（Rayleigh は散乱角の余弦、Compton は余弦）を、`kernel.py` の独立な標本（`kernel._sample_rayleigh_cos_theta_scalar`・`_sample_compton_bound_scalar`、MT）と**二標本 KS 検定**で比べた。

| N | 検査 | 正常 vs 参照 の p | 故障 vs 参照 の p |
|---|---|---|---|
| 50,000 | Rayleigh 水 30 keV（故障 1） | 0.674 | 9.6×10⁻¹⁹⁶ |
| 50,000 | Rayleigh 鉛 30 keV（故障 1） | 0.778 | 1.9×10⁻¹⁷¹ |
| 50,000 | Compton 水 30 keV（故障 2） | 0.798 | 9.4×10⁻³⁰ |
| 50,000 | Compton 鉛 30 keV（故障 2） | 0.925 | 6.1×10⁻²⁵⁹ |

100 keV は水の Compton で故障 vs 参照の p が 2.6×10⁻⁴ まで落ちる（検出力が弱い）ので採らない。**判定の閾値は p < 10⁻⁶ で故障、正常は p ≥ 10⁻⁶**。故障 3（強制 4 keV の蛍光線）と故障 4（重なりの優先）は既存テストのとおり**決定論的**（乱数に依らない固定値）なので、そのまま移す。

### 0.9 P8 の束の形

`prototypes/mv_electron/gpu/p8f_core.py:11` の `FIELDS`（33 欄）を `pack()` で 1 本の float64 の塊と記述子（offset・size・dim0・dim1）に詰める。スカラー欄 6 つは `shape==()`、`gs_y` は 3 次元、`lines` は `(nz,4,nline,4)` の 4 次元。旧い 4 整数の記述子は欄ごとの固定構造を使った圧縮で、一般の shape の記録ではない。`prototypes.mv_electron.gpu.p8g2_colab` は CuPy なしで Mac で import できることを確かめた（`load` がある）。実物は `~/.cache/chatcarlo/p8g2/export/`（`*_bundle.npz`・`*_fluorescence.npz`・`bundles.json`、820 MB、A0 で作り直したもの）にある。

### 0.10 その他

- 作り直した G2 の参照のハッシュ（固定する）: `ref_water_batches.npz` `a515113841e1c6e55dbae32423e435b92f53d9d33465dbd693adbcc8957d43d0`、`ref_lead_batches.npz` `cb12f5c8fad79fc9fb36c48763ed09fce3ae6c082870c4f2ba4aac84df5673aa`、`reference.json` `740a73d9a9cd54fae14c957b5de54040bede61fe934ce51ceb6b2f0cb2731d13`（`~/.cache/chatcarlo/g5/a0/g2_ref/`）。
- ディスクの空きは **7.2 GiB（97% 使用）**。A1 は P7 の pilot（2.9 GB）を使わない。
- `numpy` 経路・`kernel.py` は変更禁止（バイトで守る）。

## 1. 決めたこと・決めること

| # | 決めること | 決定・推奨 | 理由 |
|---|---|---|---|
| **A1-D1** | 変更の形 | **加算のみ**。`g1_*.h`・`g1_cpu.cpp`・`g1_gpu.cu` は今のファイル名・ABI・既存の行を保つ。新しいヘッダーを足し、`g1_common.h` の先頭の Philox の部分（5〜46 行）を `#include "g5_base.h"` に置き換える（この 1 か所だけ既存の行が動く。移した本文は一字も変えない）。`FIELDS`・`CPU_INPUTS`・`CUDA_INPUTS` は不変。**既存のテストは 1 本も変えない** | 0.2 のとおり多くのテストが配置に依存する。加算なら MT の鎖 0 が残り、展開した CUDA ソースの差分が「足した行だけ」であることを機械的に確かめられる（コード生成が変わっていないことの手がかりになる。速さの保証ではなく、速さは A1b で測る） |
| **A1-D2** | 組の識別子の持たせ方 | `chatcarlo/gpu/physics_sets.py`（登録表）に 2 つの名前を置く: `diagnostic-photon-kerma-v1`（実装済み）、`therapy-electron-v1`（登録のみ、`implemented: false`、A1 では使えない）。各組に**記述子**（成分・線量の量・エネルギー帯）を持たせ、その正準な JSON の SHA256 の先頭 8 バイトを **tag** とする。コンパイル時に `-DG5_SET_ID=<番号> -DG5_SET_TAG=<tag>ULL` を渡す。核の ABI は変えず、**引数は足さない**。実際にコンパイルされた核から **(ID, tag)** を読み出す小さな関数（CPU は `extern "C" g5_physics_set`、CUDA は `g5_physics_set_kernel`）を足し、`Core` が構築時に読んで登録表・束の `settings.json` の宣言と照合する（不一致は `ValueError`）。**`Core` はコンパイルのオプションを 1 つの tuple（`CUDA_OPTIONS`＋組の `-D`＋`fault != 0` のときの `-DP5_FAULT`）で持ち、`g1_histories_kernel`・`g2_histories_kernel`（`g2_core.py:26-27` も今の `core.CUDA_OPTIONS` ではなくこの tuple を使う）・probe・識別子の読み出しを、同じソースと同じ tuple でコンパイルする（CUDA は 1 個の `RawModule` から履歴核・probe・読み出しを取るのが単純。CPU は履歴関数と読み出しを同じ `self.lib` から取る）。検査は、実際に起動する核と同じコンパイルに対して行う（別の核を読んで一致と見なさない）。CPU の `.so` のキャッシュの鍵（`core.py:57`）にも組の `-D` を入れる。** 直接コンパイルするテスト用に `physics_set.h` が `G5_SET_ID`＝1・`G5_SET_TAG`＝0 を既定で定義する。**`physics_set.h` は `g1_common.h` から `#include` して展開の連鎖に入れる**（`cuda_source()` が展開するのは `g1_history.h` 以下だけで、`.cu` 本体は展開されない。`.cu` に `#include` を書くと展開後に残り `test_gpu_g2.py:212` が落ちる）。CUDA の読み出しカーネルは `.cu` の末尾にマクロだけで書き（`#include` なし）、CPU の読み出しは `g1_cpu.cpp` の末尾に足す。CPU 末尾の関数は CUDA の probe 切り出しの範囲外なので自動では CUDA に入らない | 引数を足すと観測位置（`CPU_INPUTS`/`CUDA_INPUTS`）がずれ、レジスターが増えるおそれがある。コンパイル済みの核そのものから読むので、「宣言どおりの核が動いた」ことを実測で確かめられる |
| **A1-D3** | 目録（catalog）の形 | 束に `catalog.json` を足し、形式名を `chatcarlo-g5-v1` にする（旧い `chatcarlo-g1-v1` の束は `verify` が拒否する）。各欄に次を持たせる: `name, used_by(組の名前の一覧), blob(tables/g2_tables などの塊), offset, size, dtype, shape, unit, energy_unit, energy_band_keV, grid_kind, grid_nodes_sha256, provenance, generator{path, sha256, options}, data_sha256`。`verify`/`load` は**全欄のスライスのハッシュを再計算**して目録と照合する。**核に渡す塊（`tables.npz` の `data`/`offsets`）は今の `core.pack` の出力とバイト単位で同じ**（目録は説明を足すだけ）。`catalog.py` は numpy と標準ライブラリだけ（VM で読めるように）。`catalog.py`・`physics_sets.py` は `bundle.RUNTIME` に入れる | 核に渡る値が変わらないので「データの取り違え」の検査（親計画 A1 の (3)）が簡単で強い。治療の P8 の欄を同じ形式で書けるかは A1-D6 で実測する |
| **A1-D4** | 旧い核の固定 | `git archive 19c281e chatcarlo docs/validation/gpu/g2` を `~/.cache/chatcarlo/g5/a1/old_src/` に展開する。旧い束は旧い版の `bundle.create` で別に作る（`..kernel`・`..geometry`・`..spectrum` の import と、`docs/validation/gpu/g2` の `run_gates.py`・`run_egs5_60kev.py` のコピーが要るため、`chatcarlo` の全体と g2 の docs を展開して `PYTHONPATH` を切り替えた別プロセスで作る）。VM に送るのは旧い `gpu/` だけで、最上位の名前を `chatcarlo_g5_old` に変える。VM には新旧を並べて送り、1 プロセスで両方を import する。展開の SHA256 を `a1_run/` に記録する。旧い側は計時に必要な `Core`・`g2_core.Core`・`bundle.load` を直接使うだけで、旧い worker は使わない | 同じセッションで旧い核を測り直す（親計画 §2 の速さの測り方） |
| **A1-D5** | 段の分け方 | **A1a（手元・Codex）→ 停止して確認 → A1b（T4・Claude）**。A1a の全条件が通るまで T4 の時間を使わない | 移植の誤りを安いところで先に出す |
| **A1-D6** | 治療の欄が目録に入ることの示し方 | A1a の検査として、P8g-2 の束（0.9）を読み、33 欄 × 4 材料を目録の形式で書き出し、**塊から再構成して `getattr(material, field)` とバイト単位で一致**することを確かめる（`docs/validation/gpu/g5/a1_catalog_p8.py`。`prototypes` を import するのは `chatcarlo/` の外に置くこのファイルだけ。`chatcarlo/` は `prototypes/` を import しない）。**欄の定義表**（A0 表 2 から作る。欄ごとに、材料の識別、参照する格子の欄、有効長とパディング、ln 変換の有無、MeV↔keV と密度の因子、適用できない属性は `null`）を `chatcarlo/gpu/catalog.py` の入力データとして固定する。エネルギーの帯は、エネルギー格子を持つ欄だけ、その格子の両端から**データ自身で**計算して記録する（`gs_K` のような無次元の欄、`gs_u` のような確率軸の欄、定数欄は `null`。成分ごとに単位が違う `lines`・`brems_V` は成分の単位を列挙する）。`catalog` の `shape` は元の配列の全次元を保存し、旧い 4 整数の記述子から推定しない | 親計画が A1 に束の形式を置いた理由は A2 で作り直さないため。実物で示さないと A2 で破れる |
| **A1-D7** | エンジン層のエネルギー上限 | `Core.run` が、光子のエネルギーが**表の上端**を超えたら `ValueError` で拒否する。比較は対数で行い、`log(e) <= max(有効な log_e) + 1e-12`（有効な点は `n_grid` までで、パディングは含めない）とする（`exp` で戻すと丸めで 150.0 keV が落ちるおそれがある）。診断では上端は 150 keV。目録の帯（`energy_band_keV`）と表から計算した帯が一致することも検査する。下端は検査しない（散乱で下がる光子は核の内部の話） | 今は表の外側で `g1_lerp` が端の値で黙って丸める。D10 の判定は A3 で `scene.py` に入るが、エンジン自身も取り違えで範囲外を黙って出さないようにする（多重の防御） |

### ユーザーに決めてもらう項目（承認のとき）

1. **既存テストを変えない**（A1-D1）。変える必要が出たら、その場で止まってユーザーに諮る（推奨: この方針）。
2. **A1a／A1b に分け、間で停止する**（A1-D5。推奨: 分ける）。
3. **エンジン層の上限の拒否を A1 で入れる**（A1-D7。推奨: 入れる）。
4. **承認済みの G4 の実行フォルダが再開できなくなる**（0.5）。記録（`docs/validation/` の受領書）は残るので影響は再開だけ。推奨: 許容。
5. ディスクの空きが 3 GiB を切ったら A1 の計算を止める（見張りを入れる）。P7 の pilot（2.9 GB）の整理は A2 の前にユーザーが判断する。

## 2. 作業

### 作業 0: 駆動の 2 つの直しを引き継ぐ（A0 の持ち越し）

`a0_speed.py`・`a0_launch.py` は A0 の記録なので**変更しない**。A1 の駆動を新しいファイルにする。

1. ncu の時間切れは**プロセスの群れごと**止める（`start_new_session=True` で起動し、`os.killpg`。VM 側の `sh` ラッパーの内側の ncu とその子まで）。ncu は報告だけ。A1 では新旧の g2b・g4 の核だけ取る。
2. 起動役は `a0_launch.call`（exec を 3 回まで試し直し、`--timeout limit-30`）と `ASSEMBLY_SECONDS=1800` を **import して再利用**する（写さない）。

### 作業 1: 共通の土台と組の識別子（A1a）

- `chatcarlo/gpu/g5_base.h`（新規）: `g1_common.h` の 5〜46 行をそのまま移す（INFINITY・`P5_FN`・型・`static_assert`・`P5_CAP`・`P5_FAULT`・`philox`・`uniform`・`draw`）。守りは `G5_BASE_H`。
- `chatcarlo/gpu/g1_common.h`: 5〜46 行を `#include "g5_base.h"` に置き換え、残り（`RngState` の選択、MT）は不変。
- `chatcarlo/gpu/physics_set.h`（新規）、`g1_cpu.cpp`・`g1_gpu.cu` に識別子を読む関数を**末尾に足す**（既存の行は動かさない）。
- `chatcarlo/gpu/physics_sets.py`（新規）: 登録表と tag の計算。
- `core.py`: `SOURCES` に `g5_base.h`・`physics_set.h` を足す（先頭側）。`Core(…, physics_set=…)`（既定は診断の名前）。CPU の `-D` と CUDA の `-D` に組を渡す。**`fault` を CUDA のビルドにも渡す**（`fault != 0` のときだけ `-DP5_FAULT=n` を足す。`fault == 0` のオプションは今のまま）。構築時に tag を読み出して照合する。`run` が上限を検査する（A1-D7）。
- 束の `settings.json` に `physics_set`（名前・番号・tag）を足し、`engine.run` と `colab_driver` の worker が `Core` の宣言と照合する。

### 作業 2: 目録（A1a）

`catalog.py`（新規）と `bundle.py`（形式名・必須メンバー・`create`/`verify`/`load`）。診断の 20 欄と G2 の補助の欄を載せる。`docs/validation/gpu/g5/a1_catalog_p8.py` が治療の 33 欄の往復を示す。

### 作業 3: 関門の駆動（新規、すべて `docs/validation/gpu/g5/`、出力は `~/.cache/chatcarlo/g5/a1/` と `a1_run/`）

| ファイル | 内容 |
|---|---|
| `a1_gate1.py` | G1 関門 1 の 51 量。NumPy の参照（`transport.py`）と新しい核（CPU の双子、A1b では T4）を `validation.independent_judge` で判定。コードの SHA256 と履歴数・seed を照合してから解析 |
| `a1_g2vox.py` | G2 のボクセルの照合。参照は固定したハッシュの NPZ。判定は `g2/run_gates.py` の `judge` を import |
| `a1_g2gate3.py` | `run_egs5_60kev` の `prepare`/`compare` を `out` 差し替えで再実行。**報告のみ**。`compare` の前に、EGS5 の `egs5job.out` 3 つの SHA256（A0 表 3-3）、シナリオの seed（3101/3102/3103）と履歴数（500万/50万/50万）、組の識別子を照合する |
| `a1_speed_configs.json` | 新旧 6 タスクの順（A1a-13）。測定の前に固定 |
| `a1_health.py` | 履歴ごとのエネルギー収支の共通関数（A1a-15） |
| `a1_faults.py` | 故障 1〜4 を非バイトで検出（0.8）。CPU と T4 |
| `a1_speed.py` | 新旧を 1 つのセッションで測る（`a0_configs.json` の g2b・g4 を使う）。`a1_launch.py` が起動役 |

### 作業 4: 記録

`docs/validation/gpu/g5/A1.md`（表）と RESULTS に段の結論とデータの扱いを追記。`STATUS.md` に 1 行。

## 3. 受入条件

### A1a（手元、Codex が実装、Claude が再実行）

- [ ] **A1a-1** `git diff --stat` が §4 の「変更してよい」だけ。変更禁止（`transport.py` の numpy 経路、`kernel.py`、`prototypes/mv_electron/`、`docs/validation/` の既存の記録、`data/`）に差分がない。
- [ ] **A1a-2** 既存のテストが**無改造で**全部通る（本体 745＋新規、MV は 1808 passed・17 skipped）。MV のテストが書き換える 2 つの追跡記録（`p8g1/faults_gate0.json`・`p8g3/fault_observations.json`）は `git checkout` で戻す。
- [ ] **A1a-3** 展開した CUDA ソース（`core.cuda_source()`）を、`19c281e` のもの（固定した SHA256 つきの `tests/fixtures/g5_a1_old_cuda_source.txt`）と行単位で比べ、**旧い行がすべて同じ順序で新しい版に含まれる（挿入のみ）**。**識別子の読み出しカーネルまで入った完成した CUDA ソースで検査する**。挿入された行は、ガード・`g5_base.h` の外側の守り・`physics_set.h` の本文・識別子の読み出しカーネルだけ（許可リストをテストに書く）。展開後に `#include "` が残らないこと（`test_gpu_g2.py:212`）も同じテストで確かめる。
- [ ] **A1a-4** `g1_cpu.cpp`・`g1_gpu.cu` の `#include` の閉包が `SOURCES`（この 2 つを除く）と等しい。
- [ ] **A1a-5** 識別子: CPU の核から読んだ tag が登録表と一致。読み出しは (ID, tag) の両方を照合する。次の故障を入れると `Core` の構築が `ValueError`: ID だけ違う、tag だけ違う、G2 の核だけ既定値 0、同じ `build_dir` で組の `-D` だけ変える（古い `.so` を再利用しない）、束の宣言との不一致。`therapy-electron-v1` は構築を拒否する（`implemented: false`）。
- [ ] **A1a-6** 目録（診断）: シーン S1・S2・S3・G2 の水 60 keV で、新しい束の `data`/`offsets`/幾何の配列が、`19c281e` の `core.pack(bake_scene_materials)` から作った配列と**バイト一致**する。目録のスライスのハッシュが再計算と一致する。次の破損を、**外側の manifest を更新したうえで** catalog の検査まで到達させて欄名つきで拒否する: 1 バイト破損、欄の重複・欠落、範囲外の offset、shape と size の不一致、誤った格子の参照、同じ shape の 2 欄の取り違え。外側のファイル SHA の検査の拒否とは別に試験する。旧い形式の束は拒否する。G2 の `prepare()` が足す採点用メタデータは許す欄として明示する。
- [ ] **A1a-7** 目録（治療）: P8g-2 の束（0.9）の 33 欄 × 4 材料が、目録＋塊から再構成して**バイト一致**する（A1-D6）。往復試験は少なくともスカラー欄・3 次元の `gs_y`・4 次元の `lines` を明示して含める。束がなければ、この条件は「未実行」と明記して A1b に進まない。
- [ ] **A1a-8** エネルギー上限（A1-D7）: ちょうど 150.0 keV の単色の光子と、`kvp: 150` の SpekPy スペクトルの最上段のビンは通り、150.0001 keV を含む入力は `ValueError`。表から計算した帯と目録の帯が一致する。
- [ ] **A1a-9** G1 関門 1（CPU の双子）: 3 シーン、各 NumPy 10 回（seed 5000..5009）と新しい核 10 回（seed 6000..6009、線源 7000..7009）、各 100 万履歴・バッチ 20 万。**比較できる 51 量（両側とも厳密に 0 の量を除く、`independent_judge` の規則）がすべて \|z\| ≤ 3**。3 を超える量が 1 個で 4 未満なら、一度だけ seed を +100 して全部やり直す。4 以上、または 3 を超える量が 2 個以上、または非有限なら不合格。健全性の不合格（A1a-15）は再試行の対象にしない。
- [ ] **A1a-10** G2 のボクセル（CPU の双子）: 参照は 0.10 のハッシュの NPZ（実行前に照合）。水（seed 221100+k）・鉛（221200+k）の各 20 バッチ × 20 万履歴、格子 2 cm。`g2/run_gates.py` の `judge` で kerma・H*(10) とも `pass`。**再試行できるのは、`judge` の検査のうち `outliers` か `spread` が、ちょうど 1 つの（シーン, 量）で落ちた場合だけ**。そのとき seed 231100+k／231200+k で一度だけ全部やり直す。`nonempty`・`missing` の失敗、非有限、領域総量の不一致、健全性の不合格、2 つ以上の失敗は再試行せず不合格。A1b-3 にも同じ規則を適用する。
- [ ] **A1a-11** 故障（CPU）: 0.8 の条件で、故障 1・2 は p < 10⁻⁶（4 検査すべて）、正常は p ≥ 10⁻⁶（4 検査すべて）。故障 3・4 は既存テストと同じ固定値で検出。**どの故障の検出も MT の鎖のバイト比較に頼らない**（Philox のビルドで行う）。
- [ ] **A1a-12** 開発中の手がかり（**合否に使わない**、報告のみ）: 新旧の CPU の双子（Philox）を同じ入力・seed で 2,048 履歴走らせ、出力がバイト一致するか。一致しなければ A1b に進む前に理由を書く。
- [ ] **A1a-13（T4 の駆動の手元の検査。A0 の最初の実行を失った型の事故を A1b で繰り返さないため）** `a1_launch.py`・`a1_speed.py`・T4 側の各モードを、偽の `colab` 実行ファイルで通しで動かす単体テスト: (i) `exec` の時間切れで 3 回まで試し直す、`new` は試し直さない、すべての呼び出しに上限がある、(ii) ncu の時間切れでプロセスの群れごと止まり、子が残らない、(iii) 旧い `chatcarlo_g5_old` と新しい `chatcarlo` が互いを import しない（偽の VM のディレクトリで確認）、(iv) 測定の順番が `a1_speed_configs.json` と一致する、(v) 転送の破損・パス逸脱の拒否、(vi) 設定ファイルのハッシュが全測定の行に付く。**`a1_speed_configs.json`**（A1b の前に固定し SHA256 を記録）: タスクは g2b-50万・g2b-10万・g4-10万 × {旧, 新} の 6 つ。温めの順と 3 巡の順番を**ファイルに直接列挙する**（回転では対の中の新旧の順が変わらないため）。定義は「構成の並びを 0/2/4 回転し、2 巡目だけ各対の中の新旧を入れ替える」とし、1・3 巡目は旧が先、2 巡目は新が先になる。この定義を受入条件・偽 VM の試験・本番の駆動で同じにする。
- [ ] **A1a-15（統計の関門に入れる出力のエネルギー収支）** 統計の関門（A1a-9・10、A1b-2・3）の**すべてのバッチ**で、`record=True` の再実行から履歴ごとの収支（付与＋持ち出し＝入射、A0 の `diagnostic_health()` と同じ考え方）を確かめる共通の関数を `docs/validation/gpu/g5/a1_health.py` に置いて使う。有限・非負・件数が正常でも付与を欠落させた記録（人工）を拒否する試験を付ける。落ちたバッチを含む出力は判定に使わない。
- [ ] **A1a-14** 健全性の検査（`core.health`、履歴数の保存、`g2_core` の出力検査）が新しい経路でも働き、既存の故障入れテスト（`CHATCARLO_G4_FAULT`）が全部通る。判定に使う出力は健全性を通ったものだけ。

### A1b（Colab T4、Claude が実行。A1a の全条件が通ってから）

- [ ] **A1b-1（最初）** 新旧の g2b・g4 の核のレジスター・局所メモリー・ブロック/SM を、手計算と occupancy API で記録。**新しい核が 128 レジスターを超える、またはブロック/SM が旧い核より少ない場合は、ここで停止して報告**（以降の長い実行をしない）。
- [ ] **A1b-2** G1 関門 1（T4）: A1a-9 と同じ seed・履歴数・規則で、T4 の核を NumPy の参照と比べる。比較は `chatcarlo run --engine gpu` の集計と同じ経路（`aggregate`）。
- [ ] **A1b-3** G2 のボクセル（T4）: A1a-10 と同じ。
- [ ] **A1b-4** 故障（T4）: A1a-11 の故障 1〜4 を T4 の核（probe カーネル、CUDA のビルドに `-DP5_FAULT=n`）でも検出する。独立な標本（`kernel.py`）はホストで作る。
- [ ] **A1b-5** G2 関門 3 の EGS5 60 keV を新しい核でやり直し、旧 `compare.py` で分類して**並べて報告**する（今の GPU は 37 合格・6 不合格・4 統計不足、NumPy は 42・1・4）。全項目合格や件数の完全な再現は条件にしない。EGS5 の Rayleigh の警告をそのまま添える。
- [ ] **A1b-6 速さ** 同じ T4 セッションで新旧を交互に測る（`a0_configs.json` の g2b＝50 万履歴・g4＝10 万履歴、A0 と同じ seed・開始位置・温め・3 巡）。各構成で **新の `kernel_s` の中央値 ÷ 旧の中央値 ≤ 1.0571**。計時の外で健全性を検査し、1 回でも落ちたら基準にしない。ncu は報告のみ。
- [ ] **A1b-9（NVRTC のオプションは T4 でしか試せない）** わざと違う tag／違う組の `-D` で T4 の核を構築し、`Core` の構築が `ValueError` になること。1 個の `RawModule` から取った `g1_histories_kernel`・`g2_histories_kernel`・読み出しが、同じ (ID, tag) の核であること（G2 の核に既定の tag 0 が入っていないこと）。A1a-5 の ID だけ違う・tag だけ違う故障も T4 の構築で検出する。
- [ ] **A1b-7** 束の転送: VM 側の `data`/`offsets` のハッシュがホストと一致（目録の全欄の再計算を含む）。
- [ ] **A1b-8** 記録: `A1.md`・RESULTS・STATUS、データの扱い（§6）の実行。

## 4. 対象範囲

- 変更してよい:
  - `chatcarlo/gpu/`: 新規 `g5_base.h`・`physics_set.h`・`physics_sets.py`・`catalog.py`。変更は `g1_common.h`（5〜46 行の置き換え）、`g1_cpu.cpp`・`g1_gpu.cu`（末尾に足す）、`core.py`、`bundle.py`、`engine.py`、`colab_driver.py`（worker の宣言照合）、`g2_core.py`（`Core` の引数の通し）だけ
  - 新規 `tests/test_gpu_g5_a1_*.py`、`tests/fixtures/g5_a1_old_cuda_source.txt`
  - 新規 `docs/validation/gpu/g5/`: `A1.md`、`a1_*.py`、`a1_*.json`、`a1_run/`、RESULTS への追記
  - `docs/validation/mv/STATUS.md`（1 行）、親計画（子計画の行）
- 変更禁止:
  - `chatcarlo/transport.py` の numpy 経路、`kernel.py`
  - `prototypes/mv_electron/` のすべて（読むだけ。`a1_catalog_p8.py` が import するのは読み出しのため）
  - 既存の `tests/*`（0.2 のとおり、**1 本も変えない**）
  - `docs/validation/gpu/g1〜g4/`・`water60_100M/`・`docs/validation/mv/` の既存の記録と、`g1/run_gates.py`・`g2/run_gates.py` などの既存の駆動
  - `docs/validation/gpu/g5/` の A0 のファイル（`a0_*`）
  - `data/`、`docs/jsmp133/`、Mohan のスペクトルの値

## 5. テストコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest tests/test_gpu_g5_a1_*.py -q
git diff --stat -- chatcarlo prototypes/mv_electron tests
git checkout -- docs/validation/mv/p8g1/faults_gate0.json docs/validation/mv/p8g3/fault_observations.json
```

最後の `git checkout` は MV のテストが書き換える 2 つの追跡記録を戻すため。T4 の関門は A0 と同じ決まり（`colab` の呼び出しにすべて時間の上限、`colab exec` を 10 分ごと、heartbeat の別監視、最後に `colab stop`）。

## 6. 段階の終わりのデータの扱い

| 出力（場所） | 分類 | 理由 | 復元が要る作業 |
|---|---|---|---|
| `A1.md`、判定・速さ・故障・目録の要約 JSON、駆動のコード（`a1_run/`） | リポジトリ | 小さい。A2・A3 の基準になる | — |
| G1 関門 1 の NumPy 参照 30 本と新しい核の 60 本（`~/.cache/chatcarlo/g5/a1/gate1/`） | 表＋Drive: 51 量の表・seed・コード SHA256 はリポジトリ。生 JSON は Drive | NumPy 参照の作り直しは約 8 分。A2・A3 でも使う | 再集計 |
| G2 のボクセルの twin バッチ（NPZ） | 手元。判定の要約はリポジトリ | 作り直せる | — |
| 旧い核の固定（`old_src/`）と旧い束 | 手元（A3 まで比較に使う） | `19c281e` から作り直せる | — |
| G2 関門 3 の GPU の JSON 3 つ・`comparison.json` | リポジトリ（小さい） | 報告の元 | — |
| ncu の `.ncu-rep` | 手元（50 MB 超は Drive） | CSV で足りる | — |
| Colab の束・転送の断片 | 消す | 作り直せる | — |

## 7. 停止条件

- `numpy` 経路・`kernel.py` の出力が 1 ビットでも変わったら止まる。
- **既存のテストが 1 本でも落ちる、または変える必要が出たら、止まって報告する**（直さない）。
- A1a-3（挿入のみ）が破れたら、その行の理由を書いてユーザーに諮る。
- A1a のいずれかが落ちたら A1b に進まない。偶然の不合格の扱い（A1a-9・10 の一度だけのやり直し）の後も落ちたら、原因を調べるまで進まない。判定の基準は結果を見てから変えない。
- A1b-1 でレジスター＞128、またはブロック/SM の減少なら止まる。
- 速さの比が 1.0571 を超えたら、その変更を採らずに報告する（比、ブロック数、ncu）。
- 健全性の検査が落ちた出力は判定に使わず止まる。
- ディスクの空きが 3 GiB を切ったら止まる。
- 参照・固定したハッシュ（0.1、0.10）が照合で合わなければ、その判定の入力は使わず止まる。

## 8. 書かなかったこと

- 治療の組の核の実装・駆動の選択（A2）。入口・`scene.py` の `physics:` 欄・承認と受領書への組の識別子の入れ込み・D10 の判定（A3）。
- 診断の駆動の変更・レジスターの削減などの速さの工夫（A5）。
- MT19937（CPU 専用の検証）の削除。残す（既存テストが使う。GPU のビルドには入らない）。
- 150 keV を超える検証（A4）。
- 目録を使った束の自動生成の一般化（治療の束を目録から作るのは A2）。
- G4 の Colab 入口を通した経過時間の測定（A3）。
