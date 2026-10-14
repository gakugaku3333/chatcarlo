# 計画: P5a — 輸送の核をC++に写し、いまの核と完全一致させる（CPU版）

- 状態: implemented（2026-10-14、固定した検査はすべて合格。結果 [RESULTS_P5A.md](../../validation/mv/p5/RESULTS_P5A.md)。承認: 隔離された試作・Codexレビューで致命0のため、Claudeが承認。2026-10-14）
- 作成者: claude / 実装担当: codex（`-m gpt-6.1-sol`） / 日付: 2026-10-14 / Codexレビュー: 1巡（致命0・実質8・軽微3、すべて反映。簡素化案3つも採用。原本 [reviews/2026-10-14-mv-p5a-review2-codex.md](reviews/2026-10-14-mv-p5a-review2-codex.md)）
- 骨子: [2026-10-14-mv-p5-gpu-rewrite.md](2026-10-14-mv-p5-gpu-rewrite.md)（用語・鎖0/1/2・翻訳の規則・配列の受け渡しの規則はそちら）
- 前段: [P5a0](2026-10-14-mv-p5a0-toolchain-trial.md)（implemented。結果 [RESULTS_P5A0.md](../../validation/mv/p5/RESULTS_P5A0.md)）
- 規模: 重

## 目的

Mohan 10 MV・30 cmの水の板・深さの線量について、いまの核（`photon_transport_p4.py` の `run_coupled`、Numba）と**同じ計算をするC++の核**（CPU版、GPUと共通のソース）を作り、次を示す。

1. 鎖0: 旧乱数（MT19937）で動かすと、いまの核と**出力が完全に一致**する（同じ機械＝手元のM3で）。
2. 鎖1: 新しい乱数（Philox）に替えても、深い5ブロックの線量が統計の範囲で一致する。

GPUで動かすのはP5c。**GPU用の入口はこの段階では作らない**（P5bで作る）。この段階では共通の核がGPU向けの修飾・型・禁止構文の規則を守っていることだけを確かめる。

## 0. 前提事実（検証済み、2026-10-14）

- **対象の設定**（P3rの本番 `p3r_proto` と同じ）: `run_coupled(10., N, seed, .05, .025, 30., *run_p3s.inputs('elsepa'), annihilate_in_flight=True, spectrum_edges=edges, spectrum_cum=cum, brems_mode=2, score_kerma=False, score_pair=False, **run_p3s.rule_kwargs('own'))`。照射野・領域・層・Fanoの線源・横方向の集計は使わない。`run_p3r.py` 95〜125行。
- **P3rの本番は保存済み**: `~/.cache/chatcarlo/p3r/prototype/p3r_proto/batch_000.npz`〜`batch_099.npz`。各バッチは**独立の1回の呼び出し**（`N=100000`、種は `np.random.SeedSequence(20291621).spawn(100)` から作った値。たとえばバッチ0は 4067114529）。保存されているのは `coarse`(150)、`fine`(600)、`counts`(16) などの合計。→ **Cの核のMT19937の経路で、保存済みのバッチを丸ごと再現できるか**を検査に使える。
- P4の核は、層などを指定しなければP3b・P3t・P3rと出力がバイト単位で同じ（P4a、P3b-1、P3t で確認済み）。したがって比べる相手は `photon_transport_p4.run_coupled` でよい。
- 使う入力: ELSEPAの表（`~/.cache/chatcarlo/elsepa/`、リポジトリの外）、GSの表、NIST XCOMの光子の表、ESTAR（追跡されていない `estar.npz`）、Mohanのスペクトル（追跡されていない）。**どれもリポジトリに入れない**。この段階は手元だけで動かす。
- 1履歴あたりの平均ステップは660、使ったスタックの深さは最大8（P3b-3の本番、骨子F-d/F-e）。
- P5a0の結果: Cで写した `quantities` は Numba と100万点で完全一致。MT19937・数学関数も手元では完全一致。**別の機械やGPUでは数学関数の末尾が違う**（鎖0は手元のM3だけで行う）。

## 設計

### ファイル（すべて新規。既存のファイルは編集しない）

| ファイル | 内容 |
|---|---|
| `prototypes/mv_electron/gpu/p5_kernel.h` | 共通部分（P5a0で検証済みの `p5a0_common.h` の Philox・実数への変換・`interp`・`quantities` は、**そのファイルを読み取り専用で `#include` して再利用**し、写し直さない。そのほかはP5a0の書き方を引き継ぐ。`P5_FN`、GPUでの型の定義、`--fmad=false`／`-ffp-contract=off` を前提）。粒子の状態、表の構造体（ポインタと寸法）、補間、光子の相互作用、電子の輸送、制動放射、消滅、集計、1履歴の関数 `history(...)` |
| `prototypes/mv_electron/gpu/p5_rng.h` | 乱数の差し替え: `P5_RNG_MT`（CPUのみ。全体で1本の状態、`history_rng` のときは履歴の先頭で `seed+h` を入れ直す）と `P5_RNG_PHILOX`（P5a0と同じ定義・規約）。核からは同じ関数名 `rng_uniform(state)` で呼ぶ |
| `prototypes/mv_electron/gpu/p5_cpu.cpp` | CPUの入口（`extern "C"`）。履歴のループ、バッチの集計 |
| `prototypes/mv_electron/gpu/p5a.py` | 表の束の作成、ビルド、ctypes の呼び出し、検査、結果の書き出し。入口: `gate`、`chain0`、`sem-check`、`chain1 --part k`、`analyse`。**入口は固定の設定専用**で、変えられるのは履歴数・種・開始の履歴番号・乱数の方式・診断の有無だけ（ほかの設定を個別に拒否する作りにしない） |
| `docs/validation/mv/p5/p5a_summary.json` | 小さい要約（各検査の状態と数値のみ。スペクトルの値・配列を含まない） |
| `prototypes/mv_electron/tests/test_p5a.py` | 速い検査（数秒〜数分） |
| `docs/validation/mv/p5/RESULTS_P5A.md` | 結果 |

### 写す範囲

対象の設定（固定）の下で、**乱数と粒子の状態によって到達しうる枝のすべて**を写す（試しに走らせて通った枝だけではない。低いエネルギーの二次粒子、棄却のループ、事象の同着、失敗の処理を含む）。実装者は最初に、`_run_coupled` から到達しうる関数と枝の一覧（写すもの／固定の設定では到達しないので写さないもの、と理由）を作り、`RESULTS_P5A.md` の付録に載せる。固定の設定は「0. 前提事実」の呼び出しそのもの（`source_state`・荷電粒子の線源なし、`disable_brems=False`、`annihilation_scale=1`、P4の内部の `mode=0`・`hard_on=True` など）で、これ以外は扱わない。

出力は、P4の核の出力のうち次の10個だけ（「全出力」と書くときはこの10個）。

| P4の核の名前 | 形・型 | P3rの保存の名前 | 保存の形 |
|---|---|---|---|
| `batches` | 100×150, float64（MeV、0.2 cm） | `coarse` | `np.sum(..., axis=0)`（150） |
| `fine_batches` | 100×600, float64（0.05 cm） | `fine` | 同（600） |
| `totals` | 6, float64 | `totals` | そのまま |
| `maxima` | 3, float64 | `maxima` | そのまま |
| `violations` | 3, int64 | `violations` | そのまま |
| `counts` | 16, int64 | `counts` | そのまま |
| `maxdepth` | int | `maxdepth` | そのまま |
| `nspec` | スペクトルのビン数×10, int64 | `nspec` | そのまま |
| `step_batches` | 100, int64 | `steps` | そのまま |
| `brems` | 10, float64 | `brems` | そのまま |

Cの核も100群のまま返し、合計はPython側で同じ `np.sum(..., axis=0)` で作る（C側で別の順に足すと、輸送が同じでも完全一致が崩れる）。

### 写し方の規則（骨子2-2の再掲と追加）

- 乱数は1文に1回だけ引いて一時変数に入れる。**乱数を引く順番と回数は、いまの核と同じにする**（鎖0の前提）。これは乱数を消費する補助関数（`initial` の電子の初期化と制動放射の残りの距離、`gs_mu_cached`、`compton`、`pair`、`sample_W` など）にも当てはまる: 呼ぶ順、条件の分かれ、棄却の回数、スタックへ積む・取り出す順、状態を更新する前後を保ち、枝の外で先に引かない。カットオフより低い二次粒子でも、元のコードが初期化で引く乱数は省かない。
- 演算の順を保つ。`np.interp`、`np.searchsorted`、`np.sum`（短い配列は前から順に足す）、`**`（整数べきは Numba の実装を確かめて同じ形に）、`max`/`min` の引数の順、`np.log`/`np.exp` の型。
- スタックは固定長（容量は64。あふれは `counts[6]` と失敗の扱いを、いまの核と同じにする）。いまの核の `maxstack=4096` と結果が変わらないことは、あふれが0件であることで確かめる。
- 表はPython側で作った配列を、C連続・float64でそのまま渡す。Cの側で表を作り直さない。

### 段階と検査（合否は測る前にここで固定）

**A. 関門（全体を写す前）**

| # | 内容 | 判定 |
|---|---|---|
| A1 | 代表の関数を写し、Numbaの関数と比べる。決定的な関数: `collision`、`hard`、`coefficients`、`table_rates`、`score_segment`、`local`。入力は、各表の節点、節点の `nextafter` の前後、閾値の前後（10 keV、20 keV（Møller）、1 MeV（ステップの規則の境）、光子の対生成の閾値）、ビンの境界、対数で一様な10万点（P5a0と同じく四則だけで作る）。乱数を使う関数 `gs_mu_cached` と `sample_W`: Numba側は**njitの小さいラッパーの中で** `np.random.seed(s)` をしてから関数を呼び、Cの核はMTの経路で同じ種から呼ぶ。種 20291940〜20291949 の10通り×各1,000回、戻り値と呼んだあとの乱数列の位置（次の1個の値）を比べる。定義域の外の入力は「値の一致」ではなく「両方が拒否する（Pythonの入口で `ValueError`）」を確かめる | すべての出力が完全一致（`tobytes()`）。1つでも違えば止まり、骨子2-2の3分類で扱う |

**B. 鎖0（完全一致）**

| # | 内容 | 判定 |
|---|---|---|
| B1 | `history_rng=True`、種 20291950、2万履歴: Cの核（MT）とP4の核の10個の出力 | 完全一致。加えてC2の健全性の条件 |
| B2 | 入れ直しなし（いまの核の通常の使い方）、種 20291951、2万履歴 | 完全一致。C2の健全性の条件 |
| B5 | Cの核の診断の記録を有効にしても、B1の10個の出力が変わらない | 完全一致 |
| B6 | 失敗の扱い: スタックの容量を小さく（例: 2）したビルドとステップの上限を小さくした実行で、いまの核と同じ失敗の数え方（`counts[5]`、`counts[6]`、`totals[3]`）になる（P4の核も同じ `maxstack`・`maxsteps` で呼んで比べる） | 完全一致 |
| B3 | **保存済みのP3rの本番のバッチ0と2**（種 4067114529 と 3112960579、各10万履歴）を、Cの核（MT、入れ直しなし）で再現し、上の表の保存されている名前の量をすべて比べる（`seconds` は除く。`N`・`seed` はメタデータとして照合）。保存時の `config.json` の入力のハッシュ（表、スペクトル）が、いまの入力と同じであることを先に確かめる | 完全一致。C2の健全性の条件 |
| B4 | わざと入れた誤り（コンパイル時のマクロ）: (a) 制動放射の光子のエネルギーに `(1+1e-12)` を掛ける、(b) 電子の付与を隣のビンへ、(c) 光子の相互作用の選択で乱数を1回多く引く。それぞれがコンパイル・実行でき、B1が不一致になり、**最初に違う履歴の番号、出力の名前・添字・両側の値**が報告される | 3つとも検出 |

- 一致しないときの手順: (1) `history_rng=True` では履歴 h は種 `seed+h` だけで決まるので、両方の核を `n=1, seed=seed+h` で呼ぶと1履歴ずつ比べられる。履歴0から順に（必要なら二分探索で）最初に違う履歴を探す。(2) 実装者（Codex）が報告するのは、最初に違う履歴、出力の名前・添字・両側の値、Cの核の診断の記録（1履歴の各ステップの粒子の種類・エネルギー・位置・引いた乱数の回数）まで。(3) **最初に違うステップと原因の特定はClaudeが行う**。Python側で `NUMBA_DISABLE_JIT=1` を使う場合は、その履歴の出力がNumbaの核と一致することを先に確かめてから参照に使う（丸めが同じとは限らないため）。

**C. 鎖1（新しい乱数）**

| # | 内容 | 判定 |
|---|---|---|
| C1 | Cの核（Philox、種 20291960、履歴番号 0〜9,999,999）を1,000万履歴。バッチは100（履歴番号の順に10万ずつ）。相手は保存済みのP3rの本番（1,000万履歴、MT） | 深い5ブロック（5〜10, …, 25〜30 cm、0.2 cm のビンをまとめる）の平均の差が `Dref` の1%以内、かつ差のSEM（両側のバッチ分散から）が0.33%以下。`Dref` は `docs/validation/mv/p3a/pilot_lock.json` の固定値（EGS5のパイロットの粗い深さ線量の5ビン移動平均の最大値、9.798535080485913×10⁻¹² Gy cm²）。再推定しない。ブロックの平均は、各バッチで25ビンの平均を先に作り、その100個の標本分散から各側のSEMを出す。差のSEM = √(s²_Philox/100 + s²_MT/100)（バッチ番号で対にしない） |
| C2 | 健全性: 数値の出力がすべて有限（事象の距離に内部で使う `inf` は出力ではないので対象外）、`violations[0:3]` がすべて0、`counts[5]` と `counts[6]` が0、`brems[2]` と `brems[3]` が0、`totals[3]`（未解決のエネルギー）が0。B1〜B3にも同じ条件を当てる | 合否 |
| C3 | 相互作用の数（Compton、光電、対生成、制動放射、Møller、飛行中の消滅）の1履歴あたりの差とその標準誤差、深い5ブロックの差、ビルドアップ（0〜2 cm）の差 | 報告 |
| C4 | 時間（1プロセスの µs/履歴）: Cの核とP4の核（同じ機械、同じ設定、5万履歴） | 報告 |

- 差のSEMの見積もり: 保存済みのP3rの100バッチから再計算したMT側のSEMは 0.064〜0.089% of `Dref`（Codexのレビューで計算）。同程度なら差のSEMは 0.090〜0.126% で、0.33%に届く。C1の実行の前にClaudeが同じ計算をして記録する（0.23%を超えていたら止まって報告）。
- 統計の検定は使わない（合否は上の2条件）。種を変えた取り直しはしない。
- C1は2プロセスまで。バッチ b（0〜99）の履歴番号は `[100000·b, 100000·(b+1))`、part 0 がバッチ 0〜49、part 1 が 50〜99。共通の種 20291960 と**全体を通した履歴番号**を核へ渡す（呼び出しごとに0から数え直さない）。各バッチの結果はリポジトリの外（`~/.cache/chatcarlo/p5/p5a/`）に保存し、途中で止まっても再開できる。再利用の条件は、ソースとビルドの指定、入力の配列、乱数の方式、種、履歴の範囲がすべて一致すること。欠け・重複・構成の違うバッチが混ざれば止まる。
- B7（Codexが実行）: 小さい輸送の例（Philox、2,000履歴を4バッチ）で、一度に実行した場合と、part を分けて途中で止めて再開した場合で、バッチごとの出力が完全一致。
- P3uのランナー（CLAUDE.mdの方針「新しい本番はP3uのランナーを通す」）は使わない。**例外とする理由**: P3uは各バッチに独立の種を渡す作りで、今回の「共通の種と全体の履歴番号」とは渡し方が違い、既存ファイルの編集も禁止しているため。保存・再開の検査はB7と上の再利用の条件で代える。

### 止まる条件

- A1が不一致 → 全体を写さずに止まる。
- B1〜B4のどれかが不合格 → C（鎖1）に進まない。
- C1の前のSEMの確認で 0.23% を超える → 止まる。
- C1・C2が不合格 → P5bに進まない。

## 対象範囲

- 変更してよい: 上の表の新しいファイルだけ（`p5a0_common.h` は読むだけ）。
- 変更禁止: 既存のすべてのファイル（特に凍結ファイル、`photon_transport_p4.py`、P5a0のファイル）。`chatcarlo/`、`tests/`、`scripts/`、`examples/`。
- リポジトリに入れない: 表の束、Mohanのスペクトルの値、結果の `.npz`・大きいJSON。`docs/validation/mv/p5/` に置く結果は、`RESULTS_P5A.md` と小さい要約のJSON（数値のみ、スペクトルの値を含まない）だけ。
- 規則: 生成物はリポジトリの外。Codexが検査を走らせるときだけ、リポジトリの直下の `.p5a0_work/`（ignore済み）を使ってよい。絶対パス・ユーザー名をコードに書かない。ファイルを消す処理を書かない。計算のプロセスは同時に2つまで。
- Codexのサンドボックスはリポジトリの外（`~/.cache`）を読めても書けない。**Codexがすべての道具（B3・C1〜C4・解析・再開を含む）を実装し、A1・B1・B2・B4〜B7と速い検査を実行する。ClaudeがB3・C1〜C4を実行し、検証する**。B3を飛ばした状態（`--skip-saved`）は、P5a全体の合格・P5bへ進める状態にしない。

## 受入条件

- [ ] A1、B1、B2、B4〜B7が合格（Codexが実行し、Claudeが再実行して確かめる）。
- [ ] B3が合格（Claude）。
- [ ] C1、C2が合格、C3・C4が記録されている（Claude）。
- [ ] `pytest prototypes/mv_electron/tests/test_p5a.py -q` が通る。既存の試作のテストの結果が変わらない（654 passed, 2 skipped に新しい分が加わるだけ）。
- [ ] 到達する関数と枝の一覧が `RESULTS_P5A.md` にある。
- [ ] `p5_kernel.h` に、CUDA C++で通らない書き方（可変長配列、複合リテラル、`restrict`、例外、動的な確保、標準ライブラリのコンテナ、再帰）が無い。
- [ ] 実装の開始時にあった差分のほかに、新しいファイル以外の差分が増えていない。

## テストコマンド

```bash
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a gate --work .p5a0_work/p5a
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a chain0 --work .p5a0_work/p5a --skip-saved
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5a.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
git status --short
```

（`--skip-saved` はB3を飛ばす指定。）

Claudeが実行するコマンド（Codexが同じ入口として実装する）:

```bash
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a chain0 --work ~/.cache/chatcarlo/p5/p5a
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a sem-check --work ~/.cache/chatcarlo/p5/p5a
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a chain1 --part 0 --work ~/.cache/chatcarlo/p5/p5a
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a chain1 --part 1 --work ~/.cache/chatcarlo/p5/p5a
PYTHONPATH=. .venv/bin/python -m prototypes.mv_electron.gpu.p5a analyse --work ~/.cache/chatcarlo/p5/p5a --summary docs/validation/mv/p5/p5a_summary.json
```
