# 計画: P5a0 — 小さい試作品で道具立てを確かめる（clang++ と Colab の RawKernel）

- 状態: approved（隔離された試作・Codexレビューで致命0のため、Claudeが承認。2026-10-14）
- 作成者: claude / 実装担当: codex / 日付: 2026-10-14 / Codexレビュー: 1巡（致命0・実質9・軽微3、すべて反映。原本 [reviews/2026-10-14-mv-p5a0-review2-codex.md](reviews/2026-10-14-mv-p5a0-review2-codex.md)）
- 骨子: [2026-10-14-mv-p5-gpu-rewrite.md](2026-10-14-mv-p5-gpu-rewrite.md)（決定と用語はそちら。この計画はその段階P5a0）
- 規模: 中（輸送の計算はしない。数十〜数百行の試作品と検査）

## 目的

核の全体（約1,500行）をC++に写す前に、次を小さい試作品で確かめる。

1. 1つのC++のソースを、手元の `clang++` と Colab の CuPy `RawKernel` の両方でコンパイルして同じ入力で動かせるか。
2. 新しい乱数（履歴ごと）の整数列と実数への変換が、CPUとGPUで完全に一致するか。
3. 実際の表の読み出しと補間（制動放射の `quantities`）が、Numba・clang++・GPUでどこまで一致するか。
4. 粒子の状態・固定長のスタック・作業領域・引数の受け渡し・原子的な加算が、GPUで意図どおりに動くか。
5. ColabのT4の環境（CuPyとCUDAの版、`--fmad=false` が通るか、数学関数の差、倍精度の速さ）。

## 0. 前提事実（検証済み）

骨子のF-a〜F-j。加えて:

- `quantities(T, energies, values, disabled)`（`bremsstrahlung_p3r.py` 76〜89行）は、`np.interp`（ESTARの放射阻止能、対数）、`np.searchsorted`、`log`、`exp` を使う。表は `tables()` が返す `energies`（3100個）と `values`（3100×5）、および `physics.py` のグローバルの `LOGET`、`ER`。ELSEPAの表とMohanのスペクトルは使わない。
- **ESTARの `estar.npz` はリポジトリで追跡されていない**（`docs/validation/mv/p1/.gitignore` の `data/*.npz`。追跡は `metadata.json` だけ。初稿は「リポジトリにある」と誤記、Codexの指摘で訂正）。`physics.py` は読み込みの時にこのファイルを読むので、**公開リポジトリを clone しただけのColabでは既存の物理のモジュールを import できない**。→ Colabでは既存のモジュールを import しない。表と評価点は手元で1つの入力の束にまとめ、ユーザーがColabへアップロードする（束はリポジトリに入れない）。
- Colab側で未確認のこと（CuPyが入っているか、`g++` があるか、版）は、ノートブックの最初のセルで記録する。入っていなければその旨を出して止まる。

## 設計

### ファイル（すべて新規）

| ファイル | 内容 |
|---|---|
| `prototypes/mv_electron/gpu/__init__.py` | 空 |
| `prototypes/mv_electron/gpu/p5a0_common.h` | 共通部分（C++の小さい部分集合）。関数の修飾はマクロ `P5_FN`（GPUでは `__device__ inline`、CPUでは `static inline`）。標準ライブラリは `<cmath>`・`<cstdint>` だけで、CPUのときだけインクルードする。GPUのときは `typedef unsigned int uint32_t; typedef unsigned long long uint64_t;` と、幅を確かめる `static_assert`（4バイト・8バイト）を置く |
| `prototypes/mv_electron/gpu/p5a0_cpu.cpp` | CPUの入口（`extern "C"`）。`p5a0_common.h` を読む。検証専用のMT19937もここ |
| `prototypes/mv_electron/gpu/p5a0_gpu.cu` | GPUの入口（`extern "C" __global__`）。共通ヘッダーは `#include` せず、所定の印の行（`//@@COMMON@@`）にPythonが `p5a0_common.h` の中身を展開してからRawKernelに渡す（RawKernelは文字列を受け取るので、ヘッダーの探索に頼らない）。CPUは通常の `#include`。両方で使ったヘッダーのSHA-256を結果に記録する |
| `prototypes/mv_electron/gpu/p5a0.py` | 入力の準備、`clang++`/`g++` でのビルド（出力先は引数の作業ディレクトリ。リポジトリの中に生成物を書かない）、ctypes の呼び出し、Numbaとの比較、GPUの実行（CuPyがあるときだけ）、検査、結果のJSONの書き出し。入口は4つ: `bundle --out <npz>`（手元のみ。既存のモジュールから表を読み、評価点を作って入力の束を書く）、`local --bundle <npz> --work <dir> --out <json>`、`colab --bundle <npz> --work <dir> --out <json>`（**既存の物理のモジュールを import しない**。NumPy・CuPy・ctypes だけ）、`compare <local.json> <colab.json>`。`import` は入口ごとに遅らせ、`colab` と `compare` が `physics.py` などを読み込まないようにする |
| `prototypes/mv_electron/gpu/p5a0_colab.ipynb` | Colab用。セル: (1) 環境の記録（`nvidia-smi`、CuPy・CUDA・g++・Pythonの版） (2) 公開リポジトリを作業用の新しいディレクトリへ clone し、**固定のコミット**を checkout して `git rev-parse HEAD` が先頭の変数と一致することを確かめる（すでに clone があれば fetch と checkout だけ）。その場所へ移動する (3) 入力の束のアップロード（`google.colab.files.upload()`）とSHA-256の表示 (4) `python -m prototypes.mv_electron.gpu.p5a0 colab ...` の実行（Colabに初めからあるPythonを使う。追加のパッケージは入れない） (5) 結果のJSONを表示し、ダウンロードさせる。ノートブックを取ってくるコミットと、計算のコードを固定するコミット（先頭の変数）は別でよい |
| `prototypes/mv_electron/tests/test_p5a0.py` | 手元で走る検査（GPUを使うものは `cupy` が無ければ skip） |
| `docs/validation/mv/p5/` | 結果（`p5a0_local.json`、`p5a0_colab.json`、`RESULTS_P5A0.md`） |

### 共通部分の中身

0. **入力の束**（`bundle`。リポジトリに入れない）: `energies`、`values`（C連続、float64）、`LOGET`、`ER`、`CUT`、評価点 `T_eval`、数学関数用の入力、束を作ったコードのコミットと各配列のSHA-256。評価点は**四則だけ**で作る（`log`/`exp` を使うと機械で末尾が変わる）: `T = CUT + (20−CUT)·u·u`（u はPhiloxの実数、seed 20291900、履歴番号0、100万点）、`energies` の節点すべて、各節点の `nextafter` の直前と直後（ただし 20 MeV を超えるものと 0 以下は除く）、`T ≤ CUT` の値（`CUT`、`CUT/2`、`nextafter(CUT, 0)`）。手元もColabも**同じ束のバイト列**を読む。結果のJSONに束のSHA-256を入れ、`compare` は束が違えば「比較不能」とする。
1. **乱数（新）**: Philox4x32-10。定義はRandom123（DEShawResearch/random123 の `include/Random123/philox.h`。参照したコミットのハッシュをコードのコメントとJSONに記録）: 4語のカウンター、2語の鍵、10ラウンド、乗数 `0xD2511F53`・`0xCD9E8D57`、鍵の増分 `0x9E3779B9`・`0xBB67AE85`、ラウンドごとに鍵を更新。既知の答えは同じ配布物の `tests/kat_vectors` の `philox4x32 10` の3例（零、全ビット1、非対称）をテストの中に持つ（実行時にダウンロードしない）。零カウンター・零鍵の期待値は `6627e8d5 e169c58d bc57ac4c 9b00dbd8`（実装者は配布物で確かめる。食い違えば止まって報告）。
   - 呼び出しの規約: 鍵 = `(seed の下位32ビット, 上位32ビット)`、カウンター = `(履歴番号の下位32, 上位32, 回数の下位32, 上位32)`。履歴番号と回数は `uint64_t`。**回数は0から始め、引いてから1増やす**（最初の乱数は回数0）。1回引くごとに1ブロックを計算し、出力の語0を `a`、語1を `b` として `((a>>5)*67108864.0+(b>>6))/9007199254740992.0`（MT19937と同じ変換）。語2・3は使わない。
   - 状態は「履歴番号と回数」だけ。履歴番号は**全体を通した番号**で、GPUのラッパーは `開始番号 + スレッド番号` で与える。
2. **表の補間**: `quantities` を演算の順を保って写す（`np.interp` と `np.searchsorted` はNumbaの実装の演算の順を確かめて同じ順に書く。2次元の `values` は行優先の1次元のポインタと寸法で受け取る）。戻り値は4つの出力ポインタ。`raise` に当たる入力（T > 20 MeV）は呼ぶ側で除く。
3. **数学関数の表**: 1つずつ評価する関数。入力 u は束にある100万個の実数（Philox、(0,1)。0は含まない）。`log(u)`、`exp(-20u)`、`sqrt(u)`、`sin(2πu)`、`cos(2πu)`、`pow(u+0.1, 1.3)`、`atan2(u−0.5, u2)`、`acos(2u−1)`（u2 は別の100万個）。四則だけの式は `a*b+c*u-b/(1.0+a)`（a,b,c は束にある別の3組）。どれも有限の値になる。
4. **小さい履歴（おもちゃ）**: 数学関数を使わず、四則・乱数・比較だけで動く。固定長23個の倍精度の状態を持つ「粒子」を、深さ16の固定長のスタックに積み下ろしする。手順の細部は実装者に任せるが、次を満たすこと。
   - 乱数で枝分かれする（子を0〜2個積む）。子の状態は親の23成分すべてから四則で作り、**23成分すべてが出力（付与先のビンか付与量）に影響する**。
   - 付与量は**小さい整数÷2のべき乗**（たとえば k/1024、k は1〜1024の整数）にし、1つのビンの総和が倍精度で正確に表せる範囲に収める。こうすると加算の順によらず和が正確に等しくなる。ビンは50個。
   - 履歴ごとの出力: 引いた乱数の回数、ステップ数、スタックの最大の深さ、状態の符号（0=正常、1=スタックのあふれ、2=ステップの上限1,000、3=有限でない値）、ビンごとの量。
   - 最初の1,000履歴（seed 20291901、履歴番号0〜999）のなかに、最大の深さが8以上の履歴がある。全履歴が状態0で終わる。
   - スタックの容量はコンパイル時のマクロで変えられる（容量4のビルドで、あふれが状態1として数えられることを確かめる）。
   - 出力の経路は2つ: (A) 履歴ごとの専用のビン配列へ書く、(B) 共有のビン配列へ足す（GPUでは原子的な加算、CPUでは履歴番号の順に加算）。

### 検査（合否は測る前にここで固定）

記号: N = CPU版（手元は clang++、Colabは g++）、G = GPU版、P = Python/Numba。「完全一致」は `tobytes()` の一致。各検査は `pass / fail / report / not_run` と「必須」の印でJSONに記録し、最終の判定はその記録から作る（必須が1つでも `fail` か `not_run` なら不合格）。有限でない値が出たら、報告の項目でも実行の異常として `fail` にする。数値の差の報告は「不一致の数、最大の絶対差、参照値が0でない点での最大の相対差、有限でない値の数」。

**手元**

| # | 内容 | 必須 |
|---|---|---|
| T1 | Philoxの既知の答え3例とNが一致 | 必須 |
| T2 | 純粋なPython（NumPyの整数演算）で書いた独立のPhiloxとNが、整数4語と実数で完全一致。入力は、(seed 20291902, 履歴 0〜999, 回数 0〜999) の100万組に加えて、seed・履歴番号・回数のそれぞれについて上位32ビットが0でない値と 2^32−1、2^32、2^32+1 を含む組 | 必須 |
| T3 | CのMT19937がNumbaの `np.random.seed(12345)`＋`np.random.random()` と200万個で完全一致 | 必須 |
| T4 | `quantities`: P（凍結ファイルの関数をそのまま呼ぶ）とNが、束の評価点すべてで4つの戻り値とも完全一致。`disabled=True` でも同じ。Nは選んだ区間の番号 `i` も返す（早く戻る場合は `i=-1`）。20 MeV を超える入力（`nextafter(20, inf)`）は、Nの入口（Python側）が `ValueError` で拒否することを別に確かめる | 必須。一致しなければ止まる（骨子2-2の3分類。丸めの差と特定できても、実装の調整か計画の追補とレビューを経るまで合格にしない） |
| T5a | 四則だけの式と `sqrt`: PとNが完全一致 | 必須 |
| T5b | そのほかの数学関数: PとNの差 | 報告 |
| T6 | 小さい履歴: 純粋なPythonで書いた独立の実装とNが、最初の1,000履歴で経路(A)の全出力が完全一致。最大の深さが8以上の履歴がある。経路(B)は、経路(A)を履歴番号の順に足した和とビンごとに完全一致。容量4のビルドであふれが数えられる | 必須 |
| T7 | 誤りの検査: まず正常版のT6の合格を確かめる。次に、コンパイル時のマクロで (a) 付与先を1ビンずらす、(b) 10回に1回の付与を落とす、(c) 乱数を1回多く引く、を入れた3つのビルドが**正常にコンパイル・実行され**、T6の比較がそれぞれ (a) ビンごとの量の不一致、(b) ビンごとの量と総量の不一致、(c) 引いた回数の不一致、で拒否する（コンパイルの失敗や例外は「見つけた」に数えない） | 必須 |
| T8 | 履歴番号の付け方: 履歴 0〜999 を、一度に実行した場合と、(0〜399)＋(400〜999) に分けて開始番号を与えた場合で、経路(A)の出力が完全一致 | 必須 |

**Colab**（N = Colabのg++ `-O2 -ffp-contract=off`。GはNVRTCの既定の最適化に `--fmad=false`。実際に渡した指定を記録）

| # | 内容 | 必須 |
|---|---|---|
| G0 | 環境の記録。共通ヘッダーを展開したソースで `RawKernel(..., options=('--fmad=false',)).compile()` が通る。通らなければ、エラーの文面を記録して止まる（原因を「融合を切れない」と決めつけない。指定を外すと通るかどうかは診断として記録する） | 必須 |
| G1 | Philox: GとNが、T2と同じ入力で整数と実数とも完全一致 | 必須 |
| G2a | 表の読み出し: Gが、選んだ区間の番号 `i` と、読んだ表の生の値（`energies[i]`、`energies[i+1]`、`values[i,0..4]`、`values[i+1,0..4]`）を返し、Nと完全一致。評価点は束のすべて、`disabled=True` を含む | 必須 |
| G2b | `quantities` の4つの戻り値: GとNの差 | 報告 |
| G3a | 四則だけの式と `sqrt`: `--fmad=false` でGとNが完全一致 | 必須 |
| G3b | そのほかの数学関数（`--fmad=false`）、および既定の設定での全関数と四則の式: GとNの差 | 報告 |
| G4 | 小さい履歴: GとNが、最初の100万履歴（seed 20291901）で経路(A)の全出力が完全一致（履歴ごとの専用の配列はチャンクに分けて確保）。Gの経路(B)（原子的な加算）は、経路(A)を足した和とビンごとに完全一致。全履歴が状態0 | 必須 |
| G5 | 誤りの検査をGPU側に対して行う: T7と同じ3つのマクロをGPUのソースだけに入れ、G4の比較（最初の1,000履歴）が同じ項目で拒否する | 必須 |
| G7 | 履歴番号の付け方: 履歴 0〜9,999 を、1回で起動した場合、開始番号を変えて2回に分けた場合、GPUのブロックの大きさを変えた場合（64と256）で、経路(A)の出力が完全一致 | 必須 |
| G6 | 時間: 小さい履歴1,000万個を**経路(B)**で、100万個ずつのチャンクで実行。Nの1スレッドとG。Gは1回の空実行のあと3回測り、開始前と終了後に同期する。コンパイル、転送、集計配列の初期化は別に計時。**これは小さい履歴の速さで、輸送の速さの見積もりではない** | 報告 |

**機械をまたぐ比較**（`compare`。両方のJSONの束のSHA-256とコードのコミットが同じでなければ「比較不能」）

| # | 内容 | 必須 |
|---|---|---|
| X1a | 手元のNとColabのN: Philox（T2の入力）と小さい履歴（最初の1,000履歴、経路(A)）の出力のSHA-256が一致 | 必須 |
| X1b | 手元のNとColabのN: `quantities` と数学関数の差。JSONには配列そのものではなく、評価点ごとの値のSHA-256と、先頭1万点の値を入れ、差の大きさは先頭1万点で報告する | 報告 |

- 乱数の標本による統計の検定は使わないので、標本の偶然による不合格は起こらない。
- CPUのビルドは `-O2 -ffp-contract=off`。`-ffast-math` に当たる指定は使わない。

### 止まる条件

- 手元の必須（T1〜T4、T5a、T6〜T8）のどれかが `fail`/`not_run` → Colabへ進まない。原因を報告する。
- Colabと機械間の必須（G0、G1、G2a、G3a、G4、G5、G7、X1a）のどれかが `fail`/`not_run` → P5aへ進まない。原因を報告する。
- 報告の項目（数学関数と補間の差、時間）はP5aの詳細計画（鎖2の基準）に使う。

## 対象範囲

- 変更してよい: 上の表の新しいファイルだけ。
- 変更禁止: それ以外のすべて。特に `chatcarlo/`、`tests/`、`scripts/`、`examples/`、`prototypes/mv_electron/` の既存のファイル（ハッシュで凍結されている。読むだけ）。
- スコープ外: 輸送の核の移植、ELSEPA・Mohanのデータ、速さの最適化、単精度。
- 規則: 生成物（共有ライブラリ、入力の束）はリポジトリの外の作業ディレクトリに書く（例外: Codexが検査を自分で走らせるときだけ、リポジトリの直下の `.p5a0_work/` を使ってよい。Codexはこのディレクトリを `.gitignore` に足さず、終わったら中身を残したまま報告する。コミットには入れない）。コードやノートブックに絶対パス・ユーザー名を書かない。ファイルを消す処理（`rm`、`unlink`、`shutil.rmtree`）を書かない。同時に動かす計算のプロセスは2つまで。
- Codexのサンドボックスはリポジトリの外に書けない。リポジトリの外を使う正式の実行はClaudeが行う。

## 受入条件

- [ ] `bundle` が入力の束を書き、`local` が T1〜T8 を実行して、必須の項目がすべて `pass`、報告の項目の数値がJSONに入る。
- [ ] `colab` の入口が、CuPyの無い手元では「GPUなし」と出して G の項目を `not_run` にし、Nの部分（X1用の出力）だけを実行できる。このとき `sys.modules` に `prototypes.mv_electron.physics` が無い。
- [ ] `pytest prototypes/mv_electron/tests/test_p5a0.py -q` が通る（GPUの項目はskip）。既存の試作のテスト（`pytest prototypes/mv_electron/tests -q`）の結果が変わらない（645 passed, 1 skipped に新しい分が加わるだけ）。
- [ ] `p5a0_gpu.cu` と `p5a0_common.h` に、CUDA C++で通らないことが分かっている書き方（可変長配列、複合リテラル、`restrict`、例外、動的な確保、標準ライブラリのコンテナ）が無い。
- [ ] ノートブックは上から順に実行するだけで結果のJSONを出す。手で値を書き換える箇所は、先頭のコミットのハッシュだけ。
- [ ] 実装の開始時にあった差分（`STATUS.md` の変更、追跡されていない計画とレビューのファイル）のほかに、新しいファイル以外の差分が増えていない。
- [ ] （Colabの実行の後・Claudeが確認）Colabと機械間の必須の項目がすべて `pass` で、結果が `docs/validation/mv/p5/` に記録されている。

## テストコマンド

```bash
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a0 bundle --out <作業ディレクトリ>/p5a0_inputs.npz
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a0 local --bundle <作業ディレクトリ>/p5a0_inputs.npz --work <作業ディレクトリ> --out <作業ディレクトリ>/p5a0_local.json
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a0 colab --bundle <作業ディレクトリ>/p5a0_inputs.npz --work <作業ディレクトリ> --out <作業ディレクトリ>/p5a0_colab_dryrun.json
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5a0.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
git status --short
```

## 進め方（Colabの部分）

1. 手元の検査が通ったら、Claudeが結果を確認し、ユーザーの了承を得てコミットとプッシュをする（公開リポジトリ。絶対パス・個人情報が無いことを確かめる）。
2. Claudeがノートブックの先頭にコミットのハッシュを書き込み、もう一度プッシュする。
3. Claudeが入力の束（`p5a0_inputs.npz`）を、ユーザーが開ける場所（リポジトリの中の ignore 済みの場所）に置いて場所を伝える。ユーザーがColabでノートブックを開き（ランタイムはT4）、「すべて実行」し、途中で束をアップロードし、出てきた `p5a0_colab.json` をダウンロードして渡す。
4. Claudeが `compare` を実行し、`RESULTS_P5A0.md` を書く。
