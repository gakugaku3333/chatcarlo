# 計画: P3u — 試作のバッチを複数のプロセスで実行する（結果はプロセス数によらず同じ）

- 状態: implemented（2026-10-12。全項目合格。結果は `docs/validation/mv/p3u/RESULTS.md`。承認の経緯: 2026-10-12。隔離された試作で、Codexレビューの致命が0のため、これまでの取り決めどおりClaudeが承認。プロセス数の上限は既定の2のまま）
- 作成者: claude
- 実装担当: codex
- 日付: 2026-10-12
- Codexレビュー: 1巡（原文: `reviews/2026-10-12-mv-p3u-review1-codex.md`。致命0、実質10、軽微3。すべて反映、再レビューなし）
- 規模: 中（物理も乱数の使い方も変えない。合否はバイト単位の一致で機械的に決まる。本体`chatcarlo/`には触れない）

## 目的

次の段階（P3b: 5×5 cm²の照射野でのPDDと側方分布）は、集計する領域が照射野の一部になるので、平板より多くの粒子が要る。1粒子あたりの時間はP3tで当面の打ち止めなので、待ち時間を縮めるためにバッチを複数のプロセスへ配る。あわせて、長い実行を途中で止めても続きから再開できる形と、バッチ間のばらつきからSEMを出す形を、P3b以降で共通に使える部品にする。

**この段階の約束**: 同じ `seed` と同じ `N` なら、プロセス数が1でも2でも、途中で止めて再開しても、全バッチと集計結果の配列がバイト単位で同じになる。

## 0. 前提事実（検証済み。2026-10-12に実行）

- `run_p3r.py` の `run_saved` は、すでに実行を100バッチに分けている。各バッチの乱数の種は `[int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(100)]`、バッチごとに `batch_NNN.npz` を保存し、保存済みのバッチは飛ばして再開する。集計は `np.stack([... for b in range(100)])` でバッチ番号の順。**逐次実行で、プロセスへは配っていない**（`ProcessPoolExecutor` は import されているだけで使われていない）。
- `run_p3r.SOURCES`（ハッシュ固定の対象）は次の33個。これらは書き換えない:
  `run_p3a.py, photon_transport_p3.py, p3_egs.f, p3_egs_common.inc, run_p2c.py, photon_transport_p2c.py, flight_annihilation_p2c.py, run_coupled_p2.py, p2_common.py, photon_data_p2.py, photon_sampling_p2.py, physics.py, transport.py, sampling.py, backend_sampling.py, elsepa.py, dcs.py, mf.py, single_table.py, gs_backend.py, reference_p2.py, p1e_common.py, p1d_common.py, workflow_p2.py, compare_photon_p2.py, run_p3s.py, photon_transport_p3s.py, run_p3r.py, bremsstrahlung_p3r.py, photon_transport_p3r.py, p3r_egs.f, p3r_egs_common.inc, tests/test_p3r.py`
- `run_p3a.py` の `save` は一時ファイル（`*.pending.npz`）に書いてから `Path.replace` で置き換える（途中で止まっても、壊れた `batch_NNN.npz` は残らない）。`job_lock` は `fcntl.flock` の排他ロック。`write`（JSON）も同じ方式。
- Numbaの `cache=True` は試作のどこにも使われていない。macOSのプロセス起動は spawn なので、ワーカーは毎回 import・表の読み込み・コンパイルをやり直す。
- ワーカー1個の起動の実測（新しいプロセス、Mohanスペクトル、`elsepa`、離散事象を輸送）: import 0.5 s、`inputs('elsepa')` 4.9 s、コンパイル＋制動放射の表＋100履歴 9.6 s、合計 約15 s。最大メモリ 486 MB。起動後の2,000履歴は0.47 s（235 µs/履歴）。
- 1プロセスの速さ（P3t、`docs/validation/mv/p3t/timing.json`）: 228.2 µs/履歴（Mohan、離散事象を輸送、50,000履歴）。
- このMacは物理8コア（高性能4、高効率4）。運用ルールは「同時に計算するプロセスは2個まで」（ユーザーの既定の指示）。
- `photon_transport_p3t.run_coupled` は、板の厚さが0.2 cmの倍数でないと拒否する。集計の区切り（0.2 cm、0.05 cm）は核の中に固定されている。P3bの区切り（0.25 cm、0.1 cm）と照射野は、この段階では作らない。

## 対象範囲

- 変更してよい（すべて新規）:
  - `prototypes/mv_electron/batch_runner_p3u.py`（バッチを配る仕組み）
  - `prototypes/mv_electron/batch_kernels_p3u.py`（ワーカーが実行する関数。P3tの平板と、試験用の偽の核）
  - `prototypes/mv_electron/run_p3u.py`（1プロセスと2プロセスの時間測定）
  - `prototypes/mv_electron/tests/test_p3u.py`
  - `docs/validation/mv/p3u/`（結果）
- 変更禁止: `chatcarlo/`、`tests/`、`scripts/`、`examples/`、既存の `prototypes/mv_electron/` のファイルすべて（上の33個と `photon_transport_p3t.py`、`run_p3t.py` などを含む。import して使うのはよい）、過去の段階の結果ファイル、計画ファイル。
- スコープ外: P3bの照射野・横方向の集計・区切りの変更、EGS5側の並列実行、Numbaのスレッド並列（`prange`）、コンパイル結果のキャッシュ（`cache=True`）、プロセス数の上限を2より上げること、表の引き方の高速化。

## 実装方針

Codexレビュー1巡目（原文: `reviews/2026-10-12-mv-p3u-review1-codex.md`、致命0・実質10・軽微3）を反映した版。採った単純化: **保存は親だけが行う／同時に投入するのは `workers` 件まで／バッチの検証は1つの関数に集約／離散事象は mode 1・2 に限定**。

**約束の範囲**: 同じ `config.json`（核、`params`、`N`、`seed`、`nbatch`、ソース・入力・実行環境の指紋）のもとで、**プロセス数と中断の位置だけ**を変えたときに、全バッチと集計結果の配列がバイト単位で同じになる。`nbatch` や `params` を変えた場合、別のOS・CPUの場合は約束の外。バッチの中身はP3rの `run_saved` のバッチと同じだが、最終結果の形式（Gy換算後）の互換は約束しない。

### 1. `batch_kernels_p3u.py`

- `KERNELS`: 名前 → `Kernel(run, keys, healthy, fingerprint, check_params)`。関数はすべてモジュールの最上位に置く（spawn のワーカーから import するため）。
  - `run(n, seed, params) -> dict[str, np.ndarray]`、`keys`: 戻り値のキーの組、`healthy(batch) -> bool`（バッチ単体用）、`fingerprint(params) -> dict`（そのプロセスが実際に読み込んだ入力の指紋）、`check_params(params, n)`（不正なら `ValueError`）。
- `'p3t_slab'`:
  - `params` は `E`、`backend`、`brems_mode`、`spectrum`（`'mohan'` か `None`）だけ。知らないキーは `ValueError`。**`brems_mode` は 1 か 2 だけ**（0 はP3sへ委譲されて戻り値が13個になり、`brems` と `first_record` が欠けるので受け付けない）。`n % 100 != 0` は `ValueError`（核の内側の100分割と合わせる）。
  - `photon_transport_p3t.run_coupled(float(E), n, seed, .05, .025, 30., *run_p3s.inputs(backend), annihilate_in_flight=True, spectrum_edges=…, spectrum_cum=…, brems_mode=…, score_kerma=False, score_pair=False, **run_p3s.rule_kwargs('own'))` を呼ぶ。戻り値が15個であることを確かめ、`run_p3r.KEYS` の名前を付け、**`run_p3r.run_saved` と同じ縮約**（`KEYS[:5]` は軸0で和、残りはそのまま）をかける。
  - `inputs` はすでに `lru_cache` を持つので、独自のキャッシュは重ねない。スペクトルは `run_p3a.load_spectrum()` で実行時に読む（値を追跡対象のファイルに書かない）。
  - `healthy` は `run_p3r.healthy` と同じ条件。
  - `fingerprint`: `run_p3a.table_hashes(backend, inputs(backend))`、スペクトルを使うときは `run_p3a.spec_config(edges, cum, SPECTRUM_SHA)`、`bremsstrahlung_p3r.tables()` の2配列の `run_p3a.array_hash`、Python・NumPy・Numba・llvmlite・SciPy の版、`platform.machine()`。
- `'fake'`（試験用、外部のデータに一切依存しない）: `np.random.default_rng(seed)` から決まった形の配列（`value`、`violations`、`counts`）を返す。`params` で次を起こせる: 指定した種のバッチで NaN を入れる／有限だが不健全にする（`violations` を非0に）／例外を投げる／`os._exit(1)` で子を異常終了させる／`sleep` する。`fingerprint` は `params['fingerprint']`（試験が差し替えられる値）と版を返す。

### 2. `batch_runner_p3u.py`

- `MAX_WORKERS = 2`（運用ルール）。
- 引数の検査（合わなければ `ValueError`）: `workers` は整数で `1..MAX_WORKERS`、`N` と `nbatch` は正の整数で `N % nbatch == 0`、`stop_after` は `None` か `1..nbatch`、`kernel` は `KERNELS` にある名前、`check_params(params, N//nbatch)` が通る。
- `batch_seeds(seed, nbatch)`: 前提事実の式と同じ。
- `validate_batch(batch, kernel, n, seed) -> None`（**検証はこの関数1つ**。保存の前、保存済みの読み込み、集計の前のすべてで使う）: キーの集合が `keys + ('N','seed','seconds')` と一致、数値がすべて有限、`N == n`、`seed` が一致、`seconds` が正で有限、`healthy` が真。合わなければ `RuntimeError`。加えて、全バッチで各キーの `dtype` と `shape` が番号0のバッチと同じであることを集計の前に確かめる。
- `run_batches(work, kernel, params, N, seed, workers, nbatch=100, stop_after=None) -> Path | None`
  1. 親が `work` に `run_p3a.job_lock` をかける。ワーカーはロックもファイルも触らない。
  2. スレッド数: `NUMBA_NUM_THREADS`、`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`VECLIB_MAXIMUM_THREADS` が未設定なら `'1'` を設定する（子に引き継がれる）。`'1'` 以外が設定済みなら `ValueError`。
  3. `config.json`: `kernel`、`params`、`N`、`seed`、`nbatch`、`source_hashes`（`run_p3r.SOURCES` のうち `.py` のものに、`photon_transport_p3t.py`、`batch_runner_p3u.py`、`batch_kernels_p3u.py` を足した一覧のsha256）、`fingerprint`（親のプロセスで計算）。**`workers` は入れない。** 既存の `config.json` と1か所でも違えば `RuntimeError`（実行環境の版が変わった再開もここで拒否される）。
  4. `terminal.json` があれば `RuntimeError`（やり直さない）。
  5. `result.npz` がすでにあれば、下の8と同じ検証（保存済みのバッチから番号順に組み直したものと `same` で一致、`N`・`seed`・`nbatch`・`batch_seeds` が一致）をして、そのパスを返す。
  6. 保存済みの `batch_bbb.npz` を `validate_batch` で確かめる。合わなければ `RuntimeError`（上書きしない）。
  7. 残りのバッチを `ProcessPoolExecutor(max_workers=workers, mp_context=spawn)` で実行する。**同時に投入するのは `workers` 件まで**で、1件が正常に終わるたびに番号の小さい順に次を投入する。ワーカーは `(番号, 配列の辞書, 計算時間, そのプロセスの fingerprint, 最大メモリ)` を返す。親は1件ごとに次の順で処理する:
     - ワーカーの `fingerprint` が `config` と違う、またはソースのハッシュが変わっている → **中断**（`RuntimeError`、そのバッチは保存しない、再開できる）。
     - 数値が有限でない、または `healthy` が偽 → **異常**（`terminal.json` に番号と理由を書き、`RuntimeError`。やり直さない）。
     - それ以外 → `validate_batch` を通して `run_p3a.save` で保存する。
     - 核の例外、子の異常終了（`BrokenProcessPool`）、保存の失敗 → **中断**（`RuntimeError`、`terminal.json` は書かない、再開できる）。
     - 中断でも異常でも、それ以降は投入せず、**実行中の仕事が終わるのを待ってから**（`shutdown(wait=True)`）ロックを外して例外を出す。異常・中断のあとに届いた結果は保存しない。
  8. `stop_after=k`: 番号 `0..k-1` だけを実行して `None` を返す。
  9. 全バッチがそろったら、入力の指紋とソースのハッシュをもう一度確かめ、**番号の順**に読み込んで全バッチを `validate_batch` にかけ、各キーを `np.stack` して `result.npz` に保存する（形は `(nbatch, …)`）。完了した順には決して足さない。加えて `N`、`seed`、`nbatch`、`batch_seeds`、`batch_seconds`、`wall_seconds`、`workers`、`fresh` を入れる。Gyへの換算やブロックの解析は呼ぶ側の仕事。
     - `wall_seconds`: `run_batches` に入った直後（ロックの前）から `result.npz` を保存し終えるまで。親の入力の準備、子の import とコンパイル、保存、集計、プールの終了待ちをすべて含む。
     - `fresh`: この1回の呼び出しで `nbatch` 個すべてを計算したとき真。再開や `stop_after` を経た結果は偽で、`wall_seconds` は時間の比較に使わない。
  - **ファイルを消す処理を入れない。**
- `same(a, b, skip=TIMING_KEYS)`: キーの集合が同じで、各配列の `dtype`・`shape`・`tobytes()` が同じかを返す。`TIMING_KEYS = ('seconds','batch_seconds','wall_seconds','workers','fresh','max_rss')`。npzファイルそのもののバイト列は比べない（zipの時刻が入るため）。
- **親を強制終了した場合**: ワーカーはファイルを書かないので保存の競合は起きないが、旧ワーカーが計算を続けることがある。再開の前に、旧ワーカーが残っていないことを人が確かめる（運用。仕組みは作らない）。

### 3. `run_p3u.py`（時間測定。Claudeが実行する）

`python -m prototypes.mv_electron.run_p3u` で、Mohan・`elsepa`・離散事象を輸送（mode 2）・`N=2,000,000`・`seed=20291630`・`nbatch=100` を、`workers=1`、`workers=2` の順に、別々の作業フォルダ（`CHATCARLO_P3U_CACHE`、既定は `~/.cache/chatcarlo/p3u`、その下の `timing_w1`・`timing_w2`）で実行する。

- **どちらかの作業フォルダがすでにあれば、何もせずに止まる**（T2は2回の新規の実行だけで判定する。測り直しをしない）。
- 2つとも `fresh` であることを確かめ、**先に2つの結果の一致（T1）を確かめる。一致しなければ時間を保存も表示もせずに止まる。**
- 一致したら `docs/validation/mv/p3u/timing.json` に、各条件の `wall_seconds`、バッチの計算時間の合計÷`N`（µs/履歴）、子プロセスの最大メモリの最大値、比、`config.json` の中身（ソース・入力・環境の指紋）、判定を書く。
- U1〜U10の確認は人（Claude）が行う: 試験コマンドの出力を見て、失敗0かつ `test_p3u.py` に skip が0であることを確かめてから起動し、その出力を結果の文書に貼る。スクリプトは関門を自動では確かめない。

## 受入条件（測定の前に固定）

試験の仕組み（配る・保存・再開・ロック・異常）は `'fake'` 核だけで確かめる（外部のデータなし）。U2・U3は物理の核を使う結合試験で、P2の外部キャッシュ（`photon.npz` など）と生成済みのGS表が要る。**必要なデータが無くて skip になった場合は「未実施」であり、時間測定に進まない。**

| # | 内容 | 合格の条件 |
|---|---|---|
| U1 | 種の作り方 | `batch_seeds(s, 100)` が前提事実の式と、3つの `s` で完全に一致 |
| U2 | プロセス数によらない | `'fake'`（`nbatch=8`）と `'p3t_slab'`（10 MeV単色、`mf`、mode 2、`nbatch=8`、`N=1600`）のそれぞれで、`workers=1` と `workers=2` の全バッチと `result.npz` が `same` で一致 |
| U3 | 既存の逐次実行と同じ中身 | `'p3t_slab'` の各バッチが、`photon_transport_p3r.run_coupled` を同じ種で直接呼んで `run_saved` と同じ縮約をかけたものと、`run_p3r.KEYS` の全キーでバイト単位に一致（mode 1 と mode 2） |
| U4 | 再開 | `'fake'` で、`workers=2`・`stop_after=3` のあと `workers=1` で続けた結果が、止めずに実行した結果と `same` で一致し、`fresh` は偽 |
| U5 | 取り違えの拒否 | 同じ作業フォルダで `N`・`seed`・`params`・`nbatch`・指紋（`'fake'` の `fingerprint`）のどれかを変えると `RuntimeError`。保存済みバッチについて、種の違い・キーの欠落・形の違い・`dtype` の違い・NaN のそれぞれで `RuntimeError` になり、そのファイルは書き換えられていない。`result.npz` の `N` や中身がバッチと食い違う場合も `RuntimeError` |
| U6 | 二重起動の拒否 | ロックを持ったまま同じ作業フォルダで起動すると `RuntimeError` |
| U7 | 引数の境界 | `workers` が0・3、`N` が0・負・`nbatch` で割り切れない、`nbatch=0`、`stop_after=0`、知らない核、`'p3t_slab'` の mode 0・知らない `params`・`n%100!=0`、スレッド数の環境変数が `'2'` — すべて `ValueError` |
| U8 | 異常と中断 | `'fake'` で (a) NaN、(b) 有限だが不健全 → `terminal.json` が書かれて `RuntimeError`、再起動も `RuntimeError`、それより前の正常なバッチのファイルは残る。(c) 核の例外、(d) 子の異常終了、(e) ワーカーの指紋の不一致 → `RuntimeError`、`terminal.json` は無く、原因を除いて再起動すると完走し、止めずに実行した結果と `same` で一致 |
| U9 | 消す処理がない | 新規の3つの `.py` を `ast` で調べ、`unlink`・`rmtree`・`remove`・`rmdir`・`removedirs` の呼び出しと、`subprocess`・`os.system` の使用が無い |
| U10 | 既存の試験 | `pytest prototypes/mv_electron/tests -q --ignore=prototypes/mv_electron/tests/test_p3u.py` が 190 passed, 1 skipped のまま（失敗0、skip が増えない） |
| T1 | 本番の大きさでの一致 | `run_p3u.py` の2条件（各200万履歴、どちらも `fresh`）が `same` で一致 |
| T2 | 2プロセスの効果 | `wall_seconds`（1プロセス）÷ `wall_seconds`（2プロセス）≥ 1.7。このMacでの1回の測定で判定する |

報告のみ（判定しない）: 各条件のバッチの計算時間の合計÷`N`（各ワーカーの最初のバッチにはコンパイルと表の生成が入るので、純粋な輸送の速さではない）、子プロセスごとの最大メモリの最大値。

**停止条件**: U1〜U10のどれかが通らない、または未実施なら、時間測定（T1・T2）に進まずに報告する。T1が不一致なら時間を報告しない。T2が1.7未満なら「不合格」と記録し、条件を変えて測り直さない（原因の調査は別に計画する）。

## テストコマンド

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p3u.py -q -rs
.venv/bin/python -m pytest prototypes/mv_electron/tests -q --ignore=prototypes/mv_electron/tests/test_p3u.py
# 以下はClaudeが実行（リポジトリの外に書くため、Codexのサンドボックスでは動かない）
.venv/bin/python -m prototypes.mv_electron.run_p3u
```

試験は作業フォルダを引数で受け取り、pytestの `tmp_path` を使う（`~/.cache` に書かない）。

## 書かなかったこと（スコープ外）

- **プロセス数を2より増やすこと。** 運用ルールの変更はユーザーの判断。この段階は2までで測る。高性能コアは4個なので、4までは2倍近くずつ伸びる見込みだが、5個目以降は高効率コアに乗るので伸びは鈍る（未測定）。
- **EGS5側の並列実行。** 基準側はP3bの計画で扱う（種を変えた複数の実行を合算する形になる）。
- **`cache=True`。** 起動の15 sのうちコンパイル分を省けるが、キャッシュの場所と無効化の条件を決める必要がある。1バッチが数十秒以上なら効果は小さい。
- **既存の `run_saved` の置き換え。** P3rの記録はそのまま残す。P3b以降の新しい実行がこの仕組みを使う。

## 運用メモ

- Codexのサンドボックスは `~/.cache` に書けない。`run_p3u.py` はClaudeが実行する。サンドボックスでプロセスの起動や外部キャッシュの読み込みができずに試験が動かない場合は、その旨を報告する（Claudeが実行して確かめる）。
- 2プロセスの実行中、親プロセスは待つだけで計算しない（同時に計算するのは2個）。
- ハッシュで固定されるのは上の `source_hashes` の `.py` だけで、計画ファイルは含まれない。それとは別に、これまでの運用どおり、時間測定の間はリポジトリのファイルを書き換えない。
