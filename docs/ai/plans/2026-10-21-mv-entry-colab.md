# 計画: 治療領域（10 MV 水ファントム）を対話の入口につなぎ、Colab の GPU で 1 回通す（JSMP133 依頼 A・B）

- 状態: implemented（2026-10-21、結果 docs/validation/mv/entry/RESULTS.md。承認: ユーザー承認 2026-10-21。1: 計画と chatcarlo/__main__.py の振り分け、2: 本番前に使用者が 3D 表示を確認して承認する）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-21
- Codexレビュー: 2巡（骨子: 致命 0・実質 5・軽微 1・参考 1。詳細: 致命 0・実質 7・軽微 2・参考 1。すべて反映。2 巡目の実質はどれも同じ設計の精密化なので、手順どおり Claude が反映し 3 巡目は行わない。原文 `docs/ai/plans/reviews/2026-10-21-mv-entry-colab-review{1,2}-codex.md`）
- 打ち切り: 2026-10-26 までに Claude の独立検証が終わらなければ、その時点で止まった所を報告する（抄録の文を弱める判断に使う。抄録登録 2026-11-01）

## 目的

抄録（`docs/jsmp133/abstract_ja_draft.md`）の Methods「使用者が 3D 表示で確認・承認した条件が Colab の GPU で計算される」
「承認条件と実計算条件の整合性を検証した」と、Results の依頼 1 を、1 回の実績で裏づける。答える問いは次の 3 つ。

1. `python -m chatcarlo validate / preview / approve / run` に scene.yaml を渡すだけで、10 MV・10×10 cm²・SSD 100 cm・水 30 cm の
   PDD/OCR を Colab T4 で人の手を介さずに計算し、結果を手元に戻せるか。
2. 使用者が 3D 表示で確認して承認した条件と、GPU の核の起動に実際に渡った引数から作り直した条件が一致するか。
   わざと条件を変えると不一致を検出するか。
3. その結果は、保存済みの EGS5（P3b-3 の `main_egs`、4,600 万履歴）と、P3b-3 と同じ 11 量で許容の内側か。

新しい物理の検証段階ではない。**検証済みの核（P7）を再利用し、今回の接続構成（配置 `'p6'`・新しい起動役）を確認したうえで**、
入口から呼べることと条件の取り違えがないことを示す段階。

## 0. 前提事実（検証済み、2026-10-21）

- P6 の GPU 核は半幅 5・SSD 100 を定数で埋め込む（`p6_kernel.h` 14・19 行）。P7 の核は引数で受け取る
  （`p7_kernel.h` 243 行 `p7_initialize(...,bool field,double field_half,double ssd,...)`）。P6 からの差は `p7.py` の `PERMITTED`
  だけ（`check_kernel_diff` pass。Codex レビューで再確認済み）。**P7 案を採る理由**: 半幅と SSD を核の実引数として観測・照合できる（問い 2）。
- P7 の関門 1a（`~/.cache/chatcarlo/p7/work/gate1.json`、`checks.1a.status = pass`）: CPU・MT・配置 `'p6'`・半幅 5 で、保存済み P3b-3 本番
  バッチ 0/2/99（各 46 万履歴）を 21 出力すべてバイト再現。1,836 bin より後ろ（容量 3,420）はゼロ。`h.reduced(actual,'p6')` で読める。
  **GPU での配置 `'p6'` の実績はない**（P7 の GPU は配置 `'p7'` の A/B/C、P6 は P6 核）。だから §1-3 の VM 上の接続確認を置く。
- `p7_colab.GPU.__init__` は `h.geometry('p7')` と `h.condition(name)`（A/B/C のみ）を固定で使う。P7 の `c2` の領域の報告は配置 `'p7'` 固定。
  `p7_core.c3` は BINS 全体で比べるので配置に依らない。P7 のファイルは編集せず、入口側にアダプターを書く。
- 比較相手の EGS5: `run_p3b3.read_egs('main_egs',46000000,20291763)` が読むのは `docs/validation/mv/p3b/egs5/main_egs/result.npz`
  （git で無視）と `summary.json`・`config.json`（追跡）。`read_egs` は設定の一致を見ないので、P6 の `analyse`（`p6.py` 409〜444 行）と同じく
  比較入力の SHA256 と `config.json == p3.egs_config(46000000,20291763,edges,cum)` を別に確かめる。`Dref = 8.90663759934512e-14`
  （`docs/validation/mv/p3b/pilot_lock_p3b3.json`、`N_main` 46,000,000）。
- 解析関数: 11 量は `analysis_p3b3.compare`、半影（20–80% 幅）は `analysis_p3b3.reports` の `profiles[*].width20_80`、
  ビルドアップ（横に無限の板の付与の比 − 1）は `analysis_p7.slab_ratios`（P6 RESULTS の「板全体の付与の比」と同じ方法）。
- スペクトル `docs/validation/mv/p0/data/mohan10.spectrum`（git で無視、SHA256 06fe2ba4…）。値を追跡ファイルに書かない。
- 表の入力（`p5a.load_inputs(full=True)`）は 25 配列・151 MB。Colab の `upload` は 150 MB 超で失敗した（P8f）ので分割を前提にする。
- P6 本番の実績: 4,600 万履歴、T4 のカーネル時間 1,080 s（23.5 µs/履歴）、最大 SEM 0.27%。
- `codex-cli 0.160.1`、`colab`（`~/.local/bin/colab`）。手順と落とし穴は `docs/validation/mv/p8f/HANDOFF.md`「Colab CLI」と `gpu/p8f_cli/`。
- `chatcarlo/preview.py` の `_TEMPLATE`・`_SHARED_CANVAS_JS` はモジュール変数、`scene_to_json` は `Scene.raw` だけを読む。preview.py は編集しない。
- 共有チェックアウト（他 2 セッション）。`git diff --stat chatcarlo/` は 2026-10-21 時点で空。実装の直前にもう一度確かめる。

## 1. 設計

### 1-1. 入口（`chatcarlo/` の変更は `__main__.py` の分岐と `approve` だけ）

- `main()` は診断側の parser で解析する**前に**生の argv を見る。サブコマンドが `validate`・`preview`・`approve`・`run`・`trace`・`animate` で、
  最初の位置引数の YAML の最上位に `domain: therapy_mv` があれば、argv をそのまま `prototypes.mv_electron.entry.cli.main(argv)` に渡す（遅延 import、
  MV 専用の parser が引数を解析するので、診断側の引数は明示した既定値でも未知の引数としてエラーになる）。`trace`・`animate` は MV 側で
  「治療領域では未対応」終了コード 1。`plot` は `--scene` の YAML が MV なら同じ文言で終了コード 1（`--scene` が無ければ従来どおり）。
  `approve` は診断側の parser にも足し、`domain` の無いシーンには「診断領域では不要」で終了コード 1。YAML が読めないときは診断側に任せる
  （従来のエラーのまま）。`domain` の無いシーンの経路は変えない。
- シーンの形（`examples/therapy/water_10mv_10x10.yaml`）:
  ```yaml
  domain: therapy_mv
  source: {spectrum: mohan10, field_cm: [10, 10], ssd_cm: 100}   # 照射野は SSD 100 cm の面での大きさ
  phantom: {material: water, thickness_cm: 30}                    # 横は無限の板
  scoring: pdd_ocr                                                # P3b-3 の配置（中心軸と 5/10/20 cm の横方向）
  run: {histories: 46000000, seed: 20261022, engine: colab_t4}    # engine: colab_t4 | cpu_local
  ```
- 許可リストは 1 通り（上の source/phantom/scoring）。違えば、項目のパスと「この入口で計算できる条件」を並べたエラー
  （例 `source.field_cm: 10×10 のみ`）。`histories` は 100 の倍数かつ 100 バッチに等分、`colab_t4` は 2.4×10⁸ 以下、`cpu_local` は 2×10⁵ 以下。
  `seed` は 0 以上 2⁶⁴ 未満の整数。既定値は持たない（書いていなければエラー）。
- `run` の引数: `--out DIR`（既定 `~/.cache/chatcarlo/mv_entry/<承認ハッシュ先頭 12 桁>/`）。ほかの `run` の引数を付けたらエラー。

### 1-2. 条件の 2 つの記録と照合

**承認条件**（人が確認する内容）と**資材の目録**（核・表・オプション）を分ける。

- 承認条件 `condition.json`: `field_half_cm`=5.0、`ssd_cm`=100.0、スペクトル名とファイル SHA256、`edges`/`cum` 配列の SHA256、
  幾何（配置 `'p6'` の `geometry` 構造体のバイト列）の SHA256、板の厚さ 30、採点 `pdd_ocr`、`histories`、`seed`、バッチ 100 × n、`maxsteps`、
  `field`=1、`regions`=1、engine。キーを並べた JSON（`ensure_ascii=False, sort_keys=True, separators=(',',':')`）の SHA256 が**条件ハッシュ**。
- 資材の目録 `materials.json`: P7 の `identity` と同じ材料（核のソース各 SHA256、表の `describe`、コンパイルオプション）と、
  NVRTC に渡す連結ソース文字列の SHA256、表のスカラー値と寸法（`p5b_core.dimensions`/`SCALARS` の値）。SHA256 が**資材ハッシュ**。
- `preview`: 承認条件と資材目録を作り、`preview.html`（3D 表示）と、同じ内容の `preview_snapshot.json` を書く。HTML の画面の文字として
  条件ハッシュ・資材ハッシュ・条件の一覧（照射野 10×10 cm² @ SSD 100 cm、厚さ 30 cm、採点の範囲、履歴数、種）を出す。
  3D の図: 線源点 (0,0,130)、SSD 100 cm の面 z=30（板の表面）での照射野 ±5 cm、板 z=0〜30（横は ±30 cm で切って描き、
  「横に無限の板を ±30 cm で切って表示」と書く）、中心軸の 2×2 cm の柱と 5/10/20 cm の横方向の帯（±9 cm）。
  深さと表示の z の変換は `z = 30 − depth` の 1 か所に固定し、帯のラベル（深さ）と頂点を対応させる。
- `approve SCENE --preview DIR`: `preview_snapshot.json` を読み、(i) 今のシーンから作り直した条件・資材がスナップショットと一致し、
  (ii) `preview.html` にその 2 つのハッシュが文字として入っていることを確かめてから、承認記録 `approval.json`
  （両ハッシュ、スナップショットの SHA256、HTML の SHA256、時刻）を書く。**`approve` を呼ぶのは、使用者が画面を見て承認と答えた後だけ**
  （AI の手順として RESULTS に書く。今回の 1 回はユーザーが承認する）。
- `run`: `approval.json` が無い、またはシーンから作り直した両ハッシュが承認記録と違えば、束を作らずに止まる。
- **実計算条件**（Colab の起動役、手元の CPU も同じ枠組み）: 核の起動呼び出しに**実際に渡す引数の列そのもの**を、呼ぶ直前に 1 か所で観測する。
  位置ごとに型と値（スカラー）、または `cp.asnumpy` で読み戻した内容の SHA256（配列）を記録する。対応する承認条件の項目は、
  入口側の固定の位置表（P7 の `measure` の引数の順番）で引数列から取り出して作り直す（`field_half`・`ssd`・`seed`・`start`・`n`・`maxsteps`・
  `field`・`regions`・幾何・`edges`・`cum`）。資材は表の配列と寸法・スカラー、コンパイルしたソース文字列とオプションから作り直す。
  手元に戻ってから、作り直した条件ハッシュ・資材ハッシュを承認記録と比べる。違えば解析に進まず、違った項目を並べて止まる。
- **作り直しの規則**（承認条件の各項目を 3 種に分ける。一意に実装するための固定表）:
  - 実引数から観測: `field_half_cm`・`ssd_cm`・`seed`・各バッチの `start`/`n`・`maxsteps`・`field`・`regions`・幾何の SHA256・`edges`・`cum` の SHA256。
    幾何は板の厚さ 30 と採点の配置（`pdd_ocr`）を含むので、`thickness_cm` と `scoring` は幾何の SHA256 で観測したことにする。
  - 観測値から導出: `histories`（100 バッチの検証済み範囲の和）、バッチ数と 1 バッチの履歴数。
  - 照合済みの束から持ち越す（独立には観測しない、と記録に明記する）: スペクトル名、スペクトルのファイル SHA256、engine の名前。
  - GPU の引数の位置表（0 始まり、`p7_colab.GPU.measure` の並び。Codex が 2 巡目でコードから確認）: 0–32 基本表のポインター・寸法・スカラー
    （25・26 が `edges`・`cum`）、33・34 デバイス生成の `logs`・診断カウンター、35 幾何、36–39 seed・start・n・maxsteps、40–43 field・field_half・ssd・regions、
    44・45 next・history。実装はこの表を定数で持ち、起動の直前に位置ごとの型と値を検査する（ずれたら止まる）。
  - CPU: 表は `p5j_configure` が共有設定に格納し、`run` の引数には無い。CPU は `configure` の引数列と `run` の引数列の両方を観測し、それぞれの位置表を固定する。
  - 資材ハッシュは核に渡る入力だけ（基本表の配列・寸法・スカラー、コンパイルしたソース文字列、オプション）。`describe(data)` のうち核の実引数でない
    `single`・`nist_E`・`nist_muen` は束の目録の SHA256 で照合するだけにする。`logs` は各デバイスで生成する値なので照合の対象にせず、`describe` を報告だけ残す。
- **わざと変える検査**（期待値は承認記録のまま、最終の引数だけを変える）:
  (a) 承認後に束の `condition.json` を書き換える → 束の照合（目録の SHA256）で止まる。照合を外して進めても、引数から作り直したハッシュで止まる。
  (b) 検査用の旗（環境変数 `MV_ENTRY_FAULT=field_half`）で、観測の直前に引数列の `field_half` だけを 5.5 にする → 照合で止まる。
  CPU（ctypes の呼び出しの境界）で pytest にする。CUDA の引数の結合は CPU の検査では確かめられないので、**VM でも (b) を 1 回**行う
  （2,048 履歴、本番の後。結果は手元の照合で不一致になることを記録する）。

### 1-3. GPU 実行（核は合格構成のまま）と VM 上の接続確認

- 新しいアダプター（`entry/gpu_adapter.py`）: `p7_colab.GPU` の初期化を写し、幾何を `h.geometry('p6')`、半幅・SSD を承認条件から取る。
  核のソース・オプション・`blocks=40`・ブロック 256 は P7 と同じ。高速化はしない。
- VM での順番（どれかが落ちたら本番に進まない）:
  1. 束の全ファイルの SHA256 と、分割したファイルの組み立て後の SHA256 を目録と照合。`cupy-cuda12x` が入っていたら起動しない。
  2. **2a 相当**: 手元の C++ CPU（Philox、配置 `'p6'`、同じ半幅・SSD）で作った 2,048 履歴の履歴ごとの出力と GPU を比べる。
     合否は P7 の `c2` と同じ健全性・線源（≤ 1e-13）・乱数の呼び出し数。領域の差は配置 `'p6'` で区切って報告だけ。
  3. **2c 相当**: 同じ 2,048 履歴の履歴ごとの出力の和と、本番の経路の集計（`count_additions=True`）を `p7_core.c3` で比べる。
  4. **2d 相当**: GPU の `p7_score_kernel` は 9 列の付与記録を読む（`p7.scorer_vectors` の 16 列は単体の採点用で、直接は渡せない）。
     入口側で `p7.scorer_fixture` と同じ変換を行い、配置 `'p6'` 用の端点（±9 cm の端、粗 0.25・細 0.125 の bin 境界、各深さの帯の端、
     点の付与と線分の付与）を足す。期待値は `layout='p6'` の CPU の `p7_sinks` から手元で作り、GPU の `regions`/`sums` とバイト一致。
     容量の未使用部分（1,836〜3,420）がゼロであることも確かめる。
  5. 本番: 100 バッチ × 460,000 履歴、バッチ b は `start = 460000*b`。バッチごとに結果と、その起動の引数の観測記録を保存する。
- 各バッチの保存物の添え書き: 条件ハッシュ、資材ハッシュ、バッチ番号、種、`start`/`end`、出力の `describe`、引数の観測記録の SHA256、VM のセッション ID。
  解析の前に、100 バッチの範囲が 0〜4,600 万を重なり・欠けなく覆い、全バッチの観測記録から作り直したハッシュが承認記録と一致することを確かめる
  （VM が替わって再開した場合も、各 VM の記録が対象）。既にあるバッチは計算し直さない。

### 1-4. Colab CLI の自動化（`entry/colab_cli.py`、P8f のファイルは編集しない）

- `colab new -s mventry --gpu T4` → 束の送付（150 MB 超は 50 MB に分割し VM で組み立て）→ 照合して `nohup` で起動 → **10 分ごと（ユーザー決定 2026-10-21、20 分から変更）**に
  `colab exec` で状態と終わったファイルの一覧を取り、未回収を `colab download` → 起動役の終了を見たら回収を終えて `colab stop`。
- 止まって理由を出す場合: アップロードの失敗（2 回試して駄目）、組み立て後の照合の不一致、`cupy-cuda12x` の重なり、VM の関門 2〜4 の不合格、
  セッションの消失（回収済みの結果は残す。もう一度 `run` すると回収済みのバッチを送り直して続きから回る）。所要時間を段ごとに記録する
  （束づくり、送付、VM の照合・コンパイル、関門、本番、回収、照合、解析）。
- 認証情報・アカウント名をファイルやログに書かない。`colab` の実行は Claude が行う（Codex のテストは `colab` を偽物の実行ファイルに置き換える）。

### 1-5. 比較と記録

- 解析（`entry/compare.py`）: 100 バッチを P3b-3 の形にまとめる関数を 1 つ置く（`p7_core.reduced(r,'p6')` → 外側 100 バッチを積む → `N`=総履歴数・
  `nbatch`=100 を付ける）。GPU と EGS5 の両方を `analysis_p3b3.dose()` に通して線量にし（期待形状 `axis_coarse`/`slab_coarse` (100,120)、`axis_fine` (100,300)、
  `profile` (100,2,3,72)、`profile_fine` (100,2,3,144)）、その出力を `compare`・`reports`・`slab_ratios`・`symmetry` に渡す。`reduced` の値は MeV なので、
  換算を飛ばすと同じ履歴数の比では相殺されて見えない。既知の MeV 入力から既知の線量差が出る試験を置く。EGS5 の比較入力（`result.npz`・`summary.json`・
  `config.json`・`pilot_lock_p3b3.json`）の SHA256 と、`config.json == p3.egs_config(46000000,20291763,edges,cum)`（`edges,cum` は承認した
  スペクトルから）を確かめてから、`analysis_p3b3.compare(gpu,egs,Dref)` と `reports`、`analysis_p7.slab_ratios` を呼ぶ。
- 出力（抄録の書き方、ユーザー決定 2026-10-21）: (i) 深部 8 比較（深さ 5 ブロック + 横方向 3 点の中央平均）の最大 |差|・RMS（8 個の差の
  二乗平均の平方根）・SEM の範囲（どれも `Dref` の %）、(ii) 50% 位置 3 点の最大 |差| と SEM の範囲、(iii) ビルドアップ: 板の 0–2 cm の比 − 1 と SEM、
  中心軸の直方体 0–2 cm の差と SEM（`reports.intervals[0]`）、(iv) 半影: 5/10/20 cm の 20–80% 幅の GPU − EGS5。平均 ± 標準偏差は出さない。
  抄録のたたき台の半影の行は「試作（CPU）− EGS5」なので、今回の行は「GPU − EGS5」で意味が変わる、と記録に書く。
  (v) 対称性（常設ルール「本番データで確認」）: GPU の本番の換算済み横方向分布に `analysis_p3b3.symmetry()` を当てて記録する。11 量の判定とは分け、
  偏りが検出されたら記録したうえで、抄録に数値を渡す前に使用者に報告する（判定は書き換えない）。
- 記録: `docs/validation/mv/entry/RESULTS.md` と `run_record.json`（所要時間の内訳、履歴数、両ハッシュ、関門の結果、比較の数値、わざと変える検査の結果）。
  束・結果の配列・スペクトルは `--out` の下（`~/.cache`）に置き、コミットしない。`docs/validation/mv/STATUS.md` に 1 行足す。

## 2. 事前登録の判定

- 条件の一致: 全 100 バッチで、引数から作り直した条件ハッシュ・資材ハッシュ = 承認記録（完全一致）。わざと変える検査 (a)(b) がどちらも不一致で止まる（VM の (b) を含む）。
- VM の接続確認: 2a 相当・2c 相当・2d 相当がすべて合格。
- EGS5 との比較（P3b-3 と同じ 11 量・同じ基準）: 深部 5 ブロックと横方向 3 点の差 ≤ `Dref` の 1%、50% 位置の差 ≤ 0.1 cm。
  精度の条件は既存コード（`analysis_p3b3.classify` 系）と同じ式（SEM ≤ 1/3% と 0.1/3 cm）。届かなければ「統計不足」と記録し、履歴数の追加は使用者が決める。ビルドアップ・半影は報告だけ。
  言えるのはブロック・中央平均・50% 位置の範囲までで、同等性ではない。この 1 条件・今回の環境の 1 回の実績である。

## 3. 対象範囲

- 変更してよい: `chatcarlo/__main__.py`（分岐と `approve` だけ）、新規 `prototypes/mv_electron/entry/`（`__init__.py`・`scene.py`・`condition.py`・
  `preview.py`・`bundle.py`・`gpu_adapter.py`・`vm_driver.py`・`colab_cli.py`・`compare.py`・`cli.py` を目安）、新規 `prototypes/mv_electron/tests/test_entry_*.py`、
  新規 `tests/test_mv_dispatch.py`、新規 `examples/therapy/water_10mv_10x10.yaml`、新規 `docs/validation/mv/entry/`（RESULTS は Claude が書く）、`docs/validation/mv/STATUS.md`（Claude が 1 行足すだけ）。
- 変更禁止: `chatcarlo/` のほかのファイル、`prototypes/mv_electron/` の既存ファイル（P5〜P8f の核・Colab・CLI スクリプトを含む）、`docs/jsmp133/`、
  凍結対象（`.claude/rules/mv.md` の「Data, freezing and deletion」）、`~/.cache/chatcarlo` の既存データ、`data/`。`rm` と `git add -A` は使わない。
- スコープ外: 積層（P8e/P8f）の入口、P7 の他の条件、高速化、`trace/animate/plot` の治療領域対応、Drive 経由の手順、スキル文書（依頼 C の担当）。

## 4. 受入条件

- [ ] `tests/` は全件合格（既存の件数 + 新規）。`domain` の無い `examples/*.yaml` で `validate`・`preview` の出力が変更前と同じ（新規テストで、
  変更前のコミットの `__main__.py` を使った出力と比べる、または同等の固定）。
- [ ] 許可リスト: 照射野 10×12、SSD 90、`mohan4`、厚さ 20、`seed` 無し、`histories` が 100 で割れない、のそれぞれで、項目のパスを含むエラーで止まる。
  型と境界（パラメーター化）: `histories` が 0・負・上限の直上・小数・bool、`seed` が負・2⁶⁴・bool・小数、未対応の engine、必須キーの欠け、未知のキー。
- [ ] 入口: MV のシーンで `run --workers 1`（明示した既定値）がエラー、MV の `trace`/`animate`/`plot --scene` が未対応で終了コード 1、
  診断シーンへの `approve` が終了コード 1、診断側 `run` の既存オプション（`--dose-grid` など）が従来どおり受け付けられる。
- [ ] 条件ハッシュは同じシーンで常に同じ（2 回の実行と、キーの順番を入れ替えた YAML で同じ）。項目を 1 つ変えると変わる。
- [ ] `approve`: スナップショットとシーンが違う、HTML にハッシュが無い、のどちらでも承認記録を書かない。`run` は承認記録無し・不一致で束を作らない。
- [ ] 3D 表示の JSON: 照射野の角が z=30 で (±5, ±5)、線源 (0,0,130)、板の上面 z=30・下面 z=0、深さ 5/10/20 cm の帯の z 範囲がそれぞれ 24.5–25.5・19.5–20.5・9.5–10.5 cm（ラベルと対応）。
- [ ] 手元の CPU の通し（`engine: cpu_local`、2,000 履歴 = 100 × 20）: 束づくり → 起動役（CPU）→ 回収 → 照合が通り、承認ハッシュと一致。
  わざと変える検査 (a)(b) が不一致で止まり、止まった項目名（`condition.json` の SHA256、`field_half_cm`）が出る。
- [ ] 配置 `'p6'` の手元の確認: 「P7 の CPU・配置 `'p6'`・半幅 5 は P3b-3 と同じ計算」は関門 1a で済んでいるので再計算しない。
  入口の CPU 経路（承認条件から引数を組み立てる経路）が、P7 の `p7.CPU(work,data,'philox',layout='p6').histories(2048,seed,5.,100.)` と
  全出力でバイト一致することを確かめる（取り出し・組み立ての誤りの検出）。`start` が 0 でない場合（`start=2048`）も同じ。本番の経路
  （入口の CPU の `run`、1 バッチ目以降の `start`）が `p7.CPU(...).run(n,seed,5.,100.,start=n)` と全出力でバイト一致。
- [ ] 誤結合の検出: 引数列で幾何・`edges`・`cum`・表のスカラー・寸法を入れ替えた（または別の値にした）場合に、位置表の検査か作り直したハッシュで止まる。
- [ ] 来歴の改ざんの検出: 回収した出力だけを変えた、観測記録だけを変えた、別の種・別の条件で作ったバッチを混ぜた、のどれでも解析前に止まる。
- [ ] Colab CLI の偽物（`colab` を置き換える実行ファイル）で: 分割・組み立て・照合、アップロード失敗で止まる、セッション消失で止まり回収済みを残す、
  再実行で回収済みのバッチを送り直して計算し直さない、`cupy-cuda12x` で起動しない。
- [ ] 解析: 比較入力の SHA256 を 1 バイト変えた複製、`config.json` の不一致、バッチの欠け・範囲の重なり、で止まる（一時フォルダの偽データで）。
- [ ] Colab の通し（Claude）: §2 の判定をすべて記録。所要時間・履歴数・両ハッシュを `run_record.json` に残す。

## 5. テストコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k entry
.venv/bin/python -m chatcarlo validate examples/therapy/water_10mv_10x10.yaml
.venv/bin/python -m chatcarlo validate examples/chest_room.yaml
```

Codex のテストは一時フォルダで完結させる（`~/.cache` に書けない）。2d 相当の入力・2a 相当の参照・束の実物づくり、`~/.cache` の EGS5 比較入力を
読む検査、Colab の通しは Claude が行う。

## 6. 停止条件

- 手元の検査（許可リスト、ハッシュ、わざと変える検査、CPU の小さな通し、配置 `'p6'` の確認）が通らなければ Colab に進まない。
- VM の 2a/2c/2d 相当が落ちたら本番に進まない。実計算ハッシュが承認記録と違えば解析しない。EGS5 の比較入力の照合が落ちたら判定しない。
- 判定の基準・履歴数・種は結果を見てから変えない。統計不足なら追加は使用者が決める。

## 書かなかったこと

- P6 核を直接呼ぶ案（Codex の代替案 1）: 半幅・SSD が定数で、問い 2 の「実引数の照合」ができないので採らない。
- 関門 3（1,000 万履歴の CPU 参照との統計比較）: 核は合格済みで、今回の変更は接続だけなので、2a/2c/2d 相当で確かめる（Codex 指摘 3 の提案どおり）。
- 悪意ある起動役への対策（署名など）: 取り違えの検出が目的なので扱わない。
