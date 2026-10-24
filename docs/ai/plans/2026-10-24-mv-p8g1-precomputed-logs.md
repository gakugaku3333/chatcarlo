# 計画: P8g-1 — 積層の核の表の定数 log の事前計算（A1）

- 状態: implemented（2026-10-24。関門 0〜2 合格、T4 で ×1.25〜1.31。結果 `docs/validation/mv/p8g1/RESULTS.md`。T4 の 1 回目は合成格子の試験の局所配列への atomic で停止し、Claude が `p8g_colab.py` を 1 行修正）
- 作成者: claude
- 実装担当: codex（`~/.cache` に書く関門と Colab は Claude）
- 日付: 2026-10-24
- Codexレビュー: 1巡（詳細。致命 0・実質 7・軽微 2・参考 2、原文 `reviews/2026-10-24-mv-p8g1-precomputed-logs.review2-codex.md`。骨子は 骨子は上位計画 `2026-10-24-mv-p8g-fast-stack-kernel.md` で 1 巡、原文 `reviews/2026-10-24-mv-p8g-fast-stack-kernel.review1-codex.md`）
- 上位計画: [2026-10-24-mv-p8g-fast-stack-kernel.md](2026-10-24-mv-p8g-fast-stack-kernel.md)（§8 の順番 A1 → B → A2 の 1 段目）

## 1. 目的（この計画が答える問い）

P8f の積層の核（T4 で約 333 µs/履歴）から、**表の値だけに依存する log（表の定数 log）を、輸送の前に端末ごとに 1 回だけ計算して配列から読む形**にしたとき、
(a) 結果が P8f と**履歴ごとにバイト一致**したまま、(b) T4 で何倍速くなるか。

- ユーザーの決定（2026-10-24）: 進める順は A1（既存の格子のまま前計算）→ B（状態機械と補充）→ A2（必要な量だけ再格子化）。検証は「ビット一致に縛られない 4 段の検証」に替える。
- A1 は格子も演算も変えず、同じ端末で同じ式・同じコンパイル指定で計算した値を置き換えるだけなので、**ビット一致が成り立つと見込む**（P5i が P5g と一致したまま ×1.25 になった前例）。ただし保証ではない（同じ綴りでも、コンパイラが定数を畳み込むかどうかで評価の経路が変わりうる。Codex が clang 21 で `log(.01)` が定数に畳み込まれることを確認）。**関門で一致を確かめる仮説**として扱う。費用のかからないところでは最も強い検査（バイト一致）を使い、4 段の検証のうち表の精度・部品の照合・本計算の検定は**不要**（P8f の判定 A・B の合格をバイト一致で引き継ぐ）。4 段の検証は、ビット一致が成り立たない A2 で使う。
- 期待する速さ（未測定の見込み）: 行の地図で `p8f_log_interp` が時間の標本の 17.9%（うち二分探索のループ内の `log(x[j])` が 11.4%）、`quantities` 7.9%、`p5_table_rates` 5.8%、`p5_gs_mu_cached` の一部。全部を除いても 1.2〜1.4 倍程度。大きな改善は B に期待する。

## 2. 前提事実（検証済み、2026-10-24）

- 材料の束 `~/.cache/chatcarlo/p8e/materials/`: `bundles.json` の `files` の 8 ファイルすべて SHA256 一致（Claude が確認）。
- P8f の手元の関門 0 は今も通る: `.venv/bin/python -m prototypes.mv_electron.gpu.p8f local` → `32 passed in 58.75s`。P8f の追跡ファイルに差分なし（`git status --short docs/validation/mv/p8f` が空）。
- P8f の export（関門 2 の参照を含む 220 ファイル）は `~/.cache/chatcarlo/p8f_prof/p8f/export/` に取り出し済み。`p8f_manifest.json` の全ハッシュ一致、`core.fingerprint()` が記録の `identity.sources` と一致（2026-10-24 の測定で確認）。
- 核の組み立て: `p8f_core.cuda_source()` が `p8f_gpu.cu` から `#include` を展開する。P8f は凍結した P5 の共有ヘッダ `p5_kernel.h`・`p5a0_common.h`・`p5_rng.h` と `p8f_materials.h`・`p8f_kernel.h` を使う。これらと P5k F は指紋で凍結されているので**編集しない**。
- 表の定数 log がある関数（行の地図の上位。`linemap_summary.txt`）:
  - `p8f_log_interp`（`p8f_materials.h`）: `log(x[0])`・`log(x[n-1])`・探索のループ内の `log(x[j])`・`log(y[j])`・`log(y[j+1])`・`log(x[j+1])-log(x[j])`。呼び出し元は `p8f_collision`（`stop_T`・`stop_collision`）、`p8f_coefficients`（`photo_E`・`photo`）。
  - `p8f_coefficients` の吸収端の近くの枝: `log(m.photo_E[i+1]/m.photo_E[i])`、`log(m.photo[left])`、`log(m.photo[right]/m.photo[left])`。
  - `quantities`（`p5a0_common.h`）: `log(e[i])`・`log(e[i+1])`、`log(v[5*i+k])`・`log(v[5*(i+1)+k])`（k = 0, 1, 2）。
  - `p5_table_rates`（`p5_kernel.h`）: `log(e1/e0)`、`log(t->rates[...])` 4 個。
  - `p5_gs_mu_cached`: `log(t->ts[i+1]/t->ts[i])`。
  - `p5_pair_interp`: `log(es[...])`、`p5_pair_value` の `log(vs[j*stride]/((q*q)*q))`。
  - `p5_shape`（`p8f_history → p8f_sample_k → p5_shape`、元素ごと）: `log(1194.*pow(z,-2./3.))`・`log(184.15*pow(z,-1./3.))`（`z!=1` の枝だけ）・`log(z)`。`z` だけで決まる。`log(d+.952)` は粒子に依存するので対象外。同じ式の `pow`・`exp((21.12-xl)/4.184)` は今回は置き換えない（log だけ）。
  - 純定数（コンパイル時に決まる）: `p8f_sample_k` の `log(.01)`、`p8f_hard → p5_primitive0/1` の上端 `.5` での `log(.5/(1-.5))`・`log(.5)`。**列挙するが置き換えない**（P5i も `log(.01)` は残した）。`p5_sample_k` は P8f の履歴から呼ばれない。
  - 変数を含む log（`log(E)`・`log(T)`・`log(T/e0)`・`log(T/ts[i])`・`log1p(...)` など）と `exp` は**対象外**（そのまま）。
- P5i の前例: `p5i_logs.h`。同じ端末で初期化（`p5i_initialize`）、無効な添え字は NaN、検査ビルドで読み出しの数と無効な読み出しを数える（`P5I_CHECK`）。
- P8f の比較の道具: `p8f_core.compare_histories`、`check_local_bins`、`check_aggregation`、`run_health`（`p8f_colab.py`）。

## 3. 設計

### D1. 対象の log の決め方（列挙を実装者に任せきりにしない）
- 規則: `p8f_history` から到達する関数の中で、**引数が表の配列・定数だけで決まる** `log`（乱数・粒子のエネルギー・K・u などの変数を含まない）を対象にする。§2 の一覧は最低限。
- 実装者は、到達する関数すべてについて `log(` の出現を一覧にし（ファイル・行・対象か否か・理由）、`docs/validation/mv/p8g1/log_sites.md` に残す。
- `log_sites.md` は字面だけでなく呼び出し条件（定数の引数で呼ばれる箇所、到達しない関数）も区別する。
- 対象外の log を配列に置き換えない。演算の順（`(log(a)-log(b))/(log(c)-log(d))` の各項など）は元の式のまま、項を配列の読み出しに替えるだけにする。

### D2. 配列と初期化
- 新しいヘッダ `p8g_logs.h`: 材料ごと・表ごとに、元の表と同じ添え字順の `double` 配列（重複の除去・並べ替えをしない）。無効な添え字（例: 対生成のしきい値以下）は NaN。
- **値は端末ごとに、輸送と同じ `log` で計算する**（CPU は clang の `log`、GPU は NVRTC の `log`。ホストの NumPy で計算して送らない）。GPU は輸送の前に初期化の核を 1 回呼ぶ。CPU は ctypes の関数で同じことをする。
- 比（`log(e1/e0)`、`log(ts[i+1]/ts[i])`、`log(vs/q³)`）は、元の式と同じ割り算をしてから `log` を取る（`log(e1)-log(e0)` に変えない）。
- 初期化と輸送は同じ翻訳単位（CPU は同じ `.cpp`、GPU は同じ NVRTC のソースと指定）でコンパイルする。不一致の調査用に、比の配列は商・`q`・`q³`・log の引数も補助記録に出せるようにする（合否には使わない）。
- 読み出しは添え字の範囲 `0<=j<length` を**読む前に**確かめる（検査ビルド）。
- 検査ビルド（`-DP8G_CHECK=1`）: 配列の種類ごとに読み出しの数と、無効な読み出し（範囲外・NaN の位置）の数を数える。本番のビルドでは数えない。
- 探索の向きは元のまま: `p8f_log_interp` は `l<log(x[j])` の右側の探索で `lo-1`（同じ値の節点の最後）を選ぶ。`p5_pair_interp` は `log(es[j])<x` で最初の同値節点を選ぶ。吸収端の `i` は元のエネルギーの格子で `p8f_upper` により決める。吸収端の `log(photo[right]/photo[left])` は通常の補間の `log(photo[right])-log(photo[left])` と配列を共有しない。

### D3. 新しいファイル（P8f・P5 のファイルは編集しない）
- `prototypes/mv_electron/gpu/p8g_logs.h`、`p8g_materials.h`（`p8f_materials.h` の写しに配列の読み出しを入れたもの）、`p8g_kernel.h`（`p8f_kernel.h` の写しで、`p8g_*` の関数と表の定数 log を使う版の P5 の関数を呼ぶ）。
- 表の定数 log を含む P5 の共有関数（`quantities`、`p5_table_rates`、`p5_gs_mu_cached`、`p5_pair_interp` など）は、`p8g_` を付けた写しを `p8g_kernel.h` または別のヘッダに置く。凍結したヘッダは `#include` するだけ。
- `p8g_cpu.cpp`、`p8g_gpu.cu`、`p8g_core.py`（`p8f_core` を使い回し、束の詰め方・指紋・ソースの組み立てだけ新設）、`p8g.py`（手元の関門と Colab への書き出し）、`p8g_colab.py`（T4 の関門と測定）。
- 起動の大きさは引数にする（ブロック数・スレッド数、動的な取り出しの初期値 = 総スロット数）。既定は P8f と同じ 40 × 128。P8f の GPU の駆動は 40 × 128・`next=5120` に固定なので、40 × 256 の P8f は新規側（`p8g_colab.py`）の起動のラッパーから呼ぶ（P8f のファイルは変えない）。

## 4. 関門（事前登録。順に通す）

種と履歴の範囲は P8f と同じ（関門 2 の種 `core.SEED2` = 2026102102 で 0〜19,999）。

**関門 0（手元、Codex が実装・実行できる範囲）**
- 0a 列挙: `log_sites.md` があり、§2 の一覧をすべて含む。対象にした各箇所の置き換えが元の式と同じ項の順であることを、写しと元の差分で示す。
- 0b 初期化の値: CPU の初期化した配列の各要素が、同じ CPU で元の式を計算した値とバイトで一致する（無効な添え字は NaN のまま）。
- 0c 検査ビルド: 2,000 履歴（`start=7000`）で、無効な読み出しが 0。各配列の読み出しの数を報告する（0 の配列は理由を `log_sites.md` に書く）。さらに同じ 2,000 履歴で、通常版・検査版・P8f の履歴ごとの全出力がバイト一致する。
- 0e 部品の照合（P5i の正式な関門 0 と同じ型）: 変更した各関数を、凍結した元の関数と同じ入力で直接呼び、出力がバイト一致する。入力は、元の節点とその `nextafter` の両隣、重複した節点、異なる入力が同じ log に丸まる合成例、吸収端の両側、対生成のしきい値の両側、`quantities` の `i==0`、`p5_shape` の `z==1` と `z!=1`、表の両端の外。4 材料すべて。
- 0d 故障の注入（それぞれ別のビルド、正常なビルドが同じ検査を通ったあと）:
  - f1 固定した 1 要素（配列の種類・材料・添え字を計画の実装時に `log_sites.md` に固定。0e の入力で読まれる要素）を 1 ulp ずらす → 0b と 0e で拒否。
  - f2 固定した 1 材料・1 配列の先頭位置を隣の材料の配列にずらす → 0e で拒否（関門 1 でも拒否されることを Claude が確かめる）。
  - f3 固定した配列・材料で、読み出しの添え字を範囲外または NaN の位置に 1 つずらす（通常の読み出し関数を必ず通る形。しきい値の判定を外すだけの故障は、後の探索で NaN を読まないことがあるので使わない）→ 0c の無効な読み出しで拒否。
  - 正常版の通過・故障の要素への到達（その要素の読み出しの数 > 0）・意図した検査での拒否を記録する。ビルドの失敗やハッシュの不一致は検出に数えない。故障版の作成（Codex）と、関門 1・2 での拒否の確認（Claude）を分けて記録する。

**関門 1（手元、Claude。`~/.cache` を使う）**
- 1a CPU の P8g と CPU の P8f を、同じ種・履歴番号で 20,000 履歴（0〜19,999）と 2,000 履歴（`start=7000`）回し、**履歴ごとの全出力がバイト一致**（`brems[1]` を含め、履歴ごとの出力に加算の順の違いはないので全項目バイト）。
- 1b 保存済みの P8f の CPU 参照と、CPU の P8g の同じ範囲の結果がバイト一致（P8f の読み出しの関数で封印を照合してから比べる）。`p8f_ref_gate2.npz` は履歴ごとの出力（`histories` の API で比べる）、`p8f_ref_batch_000/050/099.npz` は各 10 万履歴の集計（`run` の API で、元の CPU の集計の経路のまま。履歴の記録を後から足し合わせない — `brems[1]` は輸送の途中で足すため順が変わる）。`seconds` は比べない。

**関門 2（T4、Claude）**
- 2a GPU の P8g と CPU の P8g を 20,000 履歴、P8f の `compare_histories` の規則で比べて合格。
- 2b **GPU の P8g と GPU の P8f を、同じ VM・同じ種で 20,000 履歴、履歴ごとの全出力がバイト一致**（同じ端末の `log` なので一致するはず。一致しなければ停止して原因を調べる）。
- 2c 集計: 起動 40 × 128 と 40 × 256 の両方で、100 群の集計を履歴ごとの出力から作り直した値と照合する。規則は出力ごとに固定する: 加算する項目は履歴の和、`maxima`・`p8_max`・`maxdepth` は最大値、診断の記録など集計しない項目は対象外。整数は完全一致。浮動小数点の加算（符号つきの `brems` を含む）は、絶対値の和と加算の回数から計算した丸めの上限以内（`check_aggregation`・`check_resume` を材料に新しい照合を `p8g_core` に作る）。
- 2d 補充の経路: 総スロット数の直前・一致・直後の履歴数（5,119・5,120・5,121、40 × 256 では 10,239・10,240・10,241）と、`start≠0`、2 回続けて呼ぶ場合で、**同じ VM の GPU の P8f（または 2b で確かめた P8g の同じ履歴番号の出力）と履歴ごとの全出力がバイト一致**（`compare_histories` は線量などを報告だけにするので、この用途には使わない）。
- 2e 検査ビルドを T4 でも 2,000 履歴回し、無効な読み出し 0、通常版・検査版・GPU の P8f の履歴ごとの全出力がバイト一致。
- 2f 0e の部品の照合を T4 の GPU の関数でも行う（GPU の元の関数と GPU の新しい関数のバイト一致）。代表的な故障（f1・f3）を T4 でも 1 回ずつ回し、拒否されること。

**測定（T4、報告。合否にしない）**
- 同じ VM で交互に 3 回: P8f 40 × 128、P8f 40 × 256、P8g 40 × 128、P8g 40 × 256。履歴数は 20,000 と 200,000（本番と同じ大きさのバッチ）。核の時間の µs/履歴、初期化の時間は別に記録。
- 各構成の `num_regs`・ローカルメモリ・共有メモリと、そこから計算した 1 SM あたりの常駐ブロック数（mv.md）。
- 通常のビルドで測る（行の情報付きのビルドは命令の総数が通常と違った理由が未確認のため、速さには使わない）。

## 5. 判定

- 関門 0〜2 にすべて合格した場合だけ、「P8g-1 は、検査した履歴（関門 1・2 の範囲）で P8f と履歴ごとにバイト一致し、採用の方針に従って P8f の判定 A・B の合格を引き継ぐ」と書く（全履歴の同一性の証明ではない。P8f の合格範囲を他の積層・照射野・境界近くのビンへ広げない）。新しい本計算・EGS5 との検定はしない。
- 速さは測定の中央値の比で報告する（比べるのは同じ VM の交互の値だけ）。採否（B の土台にするか）はユーザーが決める。

## 6. 対象範囲

- 変更してよい: 新規 `prototypes/mv_electron/gpu/p8g*`（`p8g.py` を含む）、新規 `prototypes/mv_electron/tests/test_p8g*.py`、新規 `docs/validation/mv/p8g1/`。
- 変更禁止: `p8f_*`・`p5*`・`p6*`・`p7*` の既存ファイル、`docs/validation/mv/p8f/` と保存済みの結果、材料の束、`p8f_prof/` の測定記録。
- Codex は `~/.cache` に書けない。関門 1・2 と書き出しは Claude が回す（Codex はコードと関門 0 まで）。

## 7. テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k p8g
.venv/bin/python -m prototypes.mv_electron.gpu.p8g local      # 関門 0
.venv/bin/python -m prototypes.mv_electron.gpu.p8f local      # P8f が無傷であること（32 passed）
# 以下は Claude
.venv/bin/python -m prototypes.mv_electron.gpu.p8g gate1      # 関門 1
.venv/bin/python -m prototypes.mv_electron.gpu.p8g export     # T4 用の束
# T4: Colab CLI で p8g_colab の関門 2 と測定（p8f_cli と同じ手順、10 分おきの確認）
.venv/bin/python -m pytest prototypes/mv_electron/tests -q    # 全体（最後に 1 回）
```

## 8. 書かなかったこと（スコープ外）

- B（状態機械・補充）、A2（再格子化）、起動の大きさの最適化の採否、集計の整理（D）。A1 の測定のあとで別の計画にする。
- 変数を含む log・exp の書き換え、二分探索そのものの置き換え、表の配置の変更、物理の近似の変更。
- 新しい本計算と EGS5・P8e との検定（バイト一致で P8f の合格を引き継ぐため）。
- 入口（`chatcarlo run`）への接続。

## 9. 停止条件

- 1a・1b・2b のどれかでバイト一致しない → 速さの測定に進まず、どの配列・どの履歴で最初に分かれたかを調べて報告する（許容幅を後から足さない）。
- 0c・2e で無効な読み出しがある → 停止。
- 故障の注入で拒否されない故障がある → 検査を足す計画の修正をユーザーに諮る。
