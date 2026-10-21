# 計画 G4: `chatcarlo run --engine gpu`（CLI の入口・検出器の宣言・画像の保存・承認と照合・Colab の自動実行）

- 状態: approved（ユーザー承認 2026-10-21。3D 表示の承認は、ユーザーの指示で Claude が代わりに `approve` を実行する（事後承認）。骨子 + 詳細。U1〜U4 はユーザー決定 2026-10-21: U3 は (a)、U1・U2・U4 は推奨どおり）
- 作成者: claude
- 実装担当: codex（Colab の実行は Claude）
- 日付: 2026-10-21
- Codexレビュー: 2巡（骨子: 致命 0・実質 6・軽微 4・参考 2、`data/logs/codex_g4_review1.md`。詳細: 致命 0・実質 4・軽微 5、`data/logs/codex_g4_review2.md`。すべて反映）
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md) の段 G4。前段 G1（implemented）: [2026-10-21-chatcarlo-gpu-g1-diagnostic-core.md](2026-10-21-chatcarlo-gpu-g1-diagnostic-core.md)

## 目的

ユーザーの指示（2026-10-21）: 「G4（CLI への接続）を行って」。いまの GPU エンジン G1 は `run_transport(engine="gpu")` と
依頼 D 専用のランナーからしか呼べない。これを、対話の入口（scene.yaml → `chatcarlo` の CLI）から使えるようにする。

この計画が答える問い: **使用者が 3D 表示で承認した診断領域の条件が、人の手を介さずに Colab の T4 で G1 の核に渡り、
結果（検出器の画像と材料ごとの付与）が手元に戻ることを、CLI の 1 本の流れで示せるか。** 親計画の G4 の合格条件
「入口から人の手を介さずに 1 回通る。わざと条件を変えると止まる」を、G1 の対象範囲（box・`kvp`/`spectrum`・`rect`/`cone`/`parallel`・
平面検出器）で満たす。

**順番について**: 親計画は G2（ボクセルのタリー）・G3（cylinder・sphere・`mas` など）の後に G4 を置いていたが、ユーザーの指示で G4 を先に行う。
G4 は G1 の対象範囲だけを入口につなぎ、G2・G3 が入ったら同じ入口がそのまま広がる形にする（`gpu_engine_compatible` が断る理由を CLI がそのまま出す）。

## 0. 前提事実（検証済み、2026-10-21、コマンドの出力から）

- **CLI**（`chatcarlo/__main__.py`、531 行）:
  - `run` の `--engine` は `choices=["numpy", "kernel"]`（498 行）。`cmd_run`（131〜340 行）は `run_transport(..., engine=engine, ...)` を呼び、
    `detector` を渡していない（CLI からは検出器を使えない）。
  - `_mv_dispatch`（384〜423 行）は値を取るオプションを集合 `value_options` で列挙して、シーンのパスを探す。
    **新しく値を取るオプションを足すときは、この集合にも足さないと、値をシーンのパスと取り違える。**
  - `approve` は診断領域では `cmd_approve` が `[エラー] 診断領域では不要` で 1 を返す（426〜428 行）。
    `tests/test_mv_dispatch.py` 19 行がこの挙動（`['approve', DX]` → 1、`['approve', DX, '--preview', ...]` → 1）を検査している。
  - 依頼 A+B の未コミットの変更は、もう残っていない（`git status --short` は `?? data/` とこの計画ファイルだけ）。G1 の「`__main__.py` は G4 まで触らない」の縛りは、ここで解ける。
- **シーンに検出器の宣言がない**: `chatcarlo/scene.py` に `detector` の語は無い。`examples/` の `detector` は box の名前（材料の物体）で、平面検出器ではない。
  scene.py は `physics` の未知のキーを拒む（290 行）が、**最上位の未知のキーは拒まない**（Codex が確認）。今 `detector:` を書いても黙って無視される。
  `DetectorPlane`（`detector.py`）の引数: `center_cm`・`normal`・`u_axis`（単位ベクトル、直交）・`size_cm`（2 要素）・`shape`（2 要素）。
  依頼 D は `DET = dict(center_cm=[0,0,0], normal=[0,0,1], u_axis=[1,0,0], size_cm=(15,18), shape=(100,120))` をコードに直書きしている
  （`chatcarlo/gpu/run_detector_image.py` 9 行）。
- **G1 の API**（`chatcarlo/gpu/engine.py`）: `run(scene, n_histories, seed, batch_size, dose_grid, n_workers, track_uncertainty, detector, detector_roi, gpu_device, source_seed)`。
  `seed` は必須、`gpu_device` は `cpu`（手元の CPU の双子）か `cuda`（その計算機の CUDA）。Mac には CUDA が無いので、手元の `cuda` は使えない。
- **Colab の起動役**（`chatcarlo/gpu/colab_driver.py`）: `launch(bundle_path, job, out, gate1=, gate2=, session=, colab=, poll_seconds=)`。
  関門 1・2 の JSON が `pass` でなければ起動しない。束を部品に分けて送り、VM で SHA256 を照合して組み立て、`worker` を nohup で起動し、
  `poll_seconds` ごとに結果と受領票を回収する。回収済みの結果は再開時に送り直して計算し直さない。
  受領票の識別情報 `identity(manifest, job, batch, start, count)` は、**束の目録全体のハッシュ**（表・幾何・検出器・線源の辞書・スペクトル・
  `settings.json` の `scene_sha256`・核のソース）と**ジョブ全体**（`n_histories`・`batch_size`・`seed`・`source_seed`・`device`）を含む。
  手元の `read_result` は、受領票の識別情報が期待値と違えば `result identity mismatch` で止まる。
  ホストでの集計は `run_detector_image.collect_image`（依頼 D 用。`.npz`/`.json` の形は依頼 D と同じ）。
- **治療領域の承認の層**（`prototypes/mv_electron/entry/preview.py`・`condition.py`）: `preview` が条件の辞書と資材の辞書を正準 JSON にして
  SHA256 をとり、3D 表示の HTML の**見える文字**にハッシュを書く。`approve` は HTML とスナップショットを照合して `approval.json` を書く。
  `run` は束を作る前に `require_approval` で照合する。さらに核に渡す引数を観測して条件を作り直し、承認と照合する（観測の層）。
  **注意（Codex が確認）**: `approve` は HTML の中に見えるハッシュの文字があることを確かめるだけで、描画データ全体をスナップショットと照合しない。
  見えるハッシュを残して描画データだけ書き換えた HTML も承認できる。写すときはこの穴を塞ぐ（§2）。
  この層は `prototypes/` の中にあり、`chatcarlo/` から import しない（親計画 D4・CLAUDE.md の「治療領域は `chatcarlo/` の外」）。
- **核の呼び出し境界**（`chatcarlo/gpu/core.py` 71〜94 行 `Core.run`）: `seed`・`start`・`n`・線源の 3 配列・`fluorescence` と、`self.a`（束の表と幾何）から
  ネイティブの関数（CPU は ctypes、GPU は RawKernel）の引数を組み立てる。`worker`（`colab_driver.py` 87〜104 行）は受領票の識別情報をジョブから作り、
  `Core.run` への引数は別の式で渡す。**受領票の照合だけでは、呼び出し側が `seed` や `start` を取り違えても通る**（Codex の指摘 1）。
- **依頼 D の保存済み 10⁶ 履歴**（`~/.cache/chatcarlo/jsmp133_request_d/gpu/N001000000_A.npz/.json`）: seed 2040、source_seed 3712336832425968660、batch_size 100000、
  検出 435,186、検出数 0 の画素 477 個。`collect_image` は時間を `null` で保存する。束の `RUNTIME` には `colab_driver.py`・`core.py` が入るので、G4 で
  これらを変えると束の目録のハッシュは依頼 D と変わる（核の C++ は変えないので、計算の値は変わらない想定。§3 で確かめる）。
- **`collect_image` の制限**（`run_detector_image.py` 32〜49 行）: 検出器あり・CUDA・不確かさ有効が前提で、ROI を受けず、材料ごとの付与・SEM・転帰を集計しない。
- 関門の記録: `docs/validation/gpu/g1/gate1.json`・`gate2.json` はどちらも `pass`。
- `colab` コマンドは手元にある（`colab --help` が応答する。gmail 側の有料アカウント）。

## 1. ユーザーが決めること（推奨つき）

| # | 決めること | 推奨 | 理由 |
|---|---|---|---|
| U1 | 検出器の宣言の場所 | **scene.yaml の最上位に `detector:`**（`DetectorPlane` と同じ 5 項目と任意の `roi`。未知のキー・非有限値・整数でない画素数・ROI の範囲外を拒む）。`numpy` の経路も同じ宣言を使い、`kernel` は理由つきで断る | 対話の入口は宣言的な scene.yaml。CLI の引数にすると承認の対象から漏れやすい。`numpy` で読めれば GPU の結果をいつでも参照実装と比べられる（Codex も賛成） |
| U2 | 承認を求める範囲 | **Colab で実行するとき（`--engine gpu --gpu-device colab`）だけ**。手元の CPU の双子と `numpy` は今までどおり承認なし | 抄録の主張は「承認した条件が Colab の GPU で計算される」。手元の試し計算まで承認を課すと対話の往復が増える（Codex も賛成） |
| U3 | 照合の深さ（**ユーザー決定: (a)**） | **(a) 核の呼び出し境界で実際の引数を観測する小さな層を足す**: `Core.run` が、ネイティブの関数に渡した `seed`・`start`・`n`・`fluorescence`・線源 3 配列の SHA256・表と幾何の配列の SHA256 を返し、`worker` がそれを受領票に入れる。ホストは承認済みの束とジョブから同じものを作り直して照合する。`core.py` の Python 側だけを最小に変える（C++・CUDA は変えない） | (b) 足さない場合、保証は「固定した実装を信頼し、承認済みの束とジョブがそのまま VM に届いたことの確認」に狭まり、親計画 G4 の「承認した条件と実際の引数の一致」とは別の決定になる（Codex の指摘 1）。(a) は数十行で、治療の入口（P7 の ABI に合わせた観測）より小さい |
| U4 | 画像の保存 | **検出器があれば実行ディレクトリに必ず `.npz`（依頼 D と同じ配列名）・`.json`（完全な条件ハッシュ・検出器の位置と向きと画素の軸・単位・統計の設定・照合の結果・VM の時間）・確認用の PNG 1 枚を保存**。`--detector-out PATH` は写しの置き場所 | `plot` は線量の `.npz` だけを扱う。画像を見られないと対話の確認ができない |

## 2. 流れ（骨子）— preview が実行一式を固定し、run はそれを実行する（Codex の代替案を採用）

1. `chatcarlo preview scene.yaml --engine gpu --gpu-device colab -n N --seed S [--batch-size B] [--source-seed T]`
   - シーンを検証し、`gpu_engine_compatible` を通し、Colab で使えない組み合わせ（`--workers` ≥ 2、`--dose-grid` など）を**送る前に拒む**（黙って無視しない）。
   - 実効の条件を一度だけ決める: 履歴数の整数化、`source_seed` の省略時の導出（G1 と同じ式）、`batch_size`、統計の設定、ROI。ジョブの `device` は `cuda`（`colab` は実行場所の名前）。
   - **実行一式**を一度だけ作る: 束、ジョブ、表示のスナップショット（検証済みのシーンと、束から復元した検出器・幾何が一致することを確かめて作る）、HTML、
     それらの SHA256 を書いた生成記録。出力先は `~/.cache/chatcarlo/gpu_runs/<条件ハッシュの先頭 12 文字>/`。**照合には常に完全なハッシュを使う**（12 文字はフォルダ名だけ）。
   - 3D 表示に**検出器の面**と、**条件ハッシュ・履歴数・種・検出器の大きさと画素数**を見える文字で出す。
   - 同じ出力先に承認済みの一式があれば上書きしない（止まって知らせる）。承認前の一式は作り直してよい。
2. 使用者が 3D 表示を見て承認 → `chatcarlo approve <出力先>`: 生成記録と、束・ジョブ・スナップショット・HTML の SHA256 を照合してから `approval.json` を書く。
   **HTML とスナップショットは生成記録のハッシュと照合する**ので、見えるハッシュを残して描画データだけ書き換えた HTML は承認できない（Codex の指摘 2）。
3. `chatcarlo run scene.yaml --engine gpu --gpu-device colab -n N --seed S ...`
   - シーンと引数から条件を作り直すのは**出力先を探すためと、承認済みの条件との一致を確かめるため**だけ。作り直しは一時フォルダで行い、**実行には承認済みの束とジョブをそのまま使う**（Codex の指摘 3）。
   - 送る前に、承認記録・生成記録・束・ジョブ・HTML を照合する。どれか違えば止まる。
   - `colab_driver.launch` で実行・回収（関門 1・2 の `pass` が前提）。VM の `worker` は計算の時間を測って結果と一緒に返す。
   - 回収した全バッチを、受領票の識別情報と（U3 (a) なら）観測した実際の引数で照合する。
   - **共通の集計関数**（新規）で、照合済みのバッチから `TransportResult` 相当（材料ごとの付与・転帰・`ScalarMoments`・検出器があれば `DetectorTally`）を作る。
     端末の表示と U4 の保存は、この 1 つの結果から作る。ROI・`--no-uncertainty`・検出器なしもここで扱う（Codex の指摘 5）。`collect_image` はこの関数を呼ぶ形に置き換え、依頼 D の `.npz` の形は保つ。
   - 端末への出力は今の `cmd_run` と同じ形に、エンジン・装置・完全な条件ハッシュ・VM の計算時間・新しく計算したバッチ数と再利用したバッチ数を足す。
4. `chatcarlo run scene.yaml --engine gpu --seed S`（既定 `--gpu-device cpu`）は、承認なしで手元の CPU の双子で計算し、同じ共通の集計関数と同じ保存を使う。

## 3. 判定（事前に決める合格の条件）

- **通し（新規の計算）**: 依頼 D の肖像のシーンの写し（`docs/validation/gpu/g4/` に置き、`detector:` を足す。元の `docs/jsmp133/` は変えない）を、
  **回収済みの結果がない状態から** `preview` → ユーザーの承認 → `run --gpu-device colab` で、人の手を介さずに 1 回通す。新しい GPU のバッチが計算された記録を残す
  （条件ハッシュ、履歴数、VM の計算時間、新規のバッチ数）。再開（回収済みの再利用）の確認は別に行い、通しの 1 回には数えない。
- **入口・転送・集計の回帰確認**: 依頼 D の保存済みの 10⁶ 履歴 A（seed 2040、source_seed 3712336832425968660、batch_size 100000）と同じ条件で走らせ、画像を比べる。
  - `photon_count` は完全一致。
  - 浮動小数の配列（`category_fluence`・`energy_sum_keV`・`energy_sum2_keV2`）は、**加算の回数と寄与の大きさから計算した絶対差の上限**の内側（G1 関門 2c と同じ作り。装置の中の atomic 加算と、ホストのバッチの足し合わせの両方を数える）。両方 0 の画素は一致として扱い、片方だけ 0 でない画素も同じ絶対差の判定にする。試しの実行は加算の回数の取り方を確かめるためだけに使い、**試しの差から許容の幅を決めない**。
  - 比べる相手の前提（線源の配列・表・核のソース・実効のジョブ・実行環境）を記録し、前提が違えば「不合格」ではなく「回帰比較の前提が成り立たない」として区別して報告する。
  - これは入口・転送・集計の回帰確認で、物理の検証ではない（物理は G1 の関門で済み）。
- **集計の正しさ**: 同じ照合済みバッチを、共通の集計関数とは別の素朴な足し合わせ（テストの中に書く）で集計し、材料ごとの付与（keV→MeV の換算と材料の並びを含む）・転帰・SEM・履歴数・バッチ数・`n_batches_hit`・R の有限な部分と未定義の部分が一致すること。PNG は、左右・上下が非対称な小さな既知の配列で、向き・軸・単位をテストする。
- **わざと変える検査（すべて止まること）**: (a) 承認なしで `run`、(b) 承認後にシーンの値を 1 つ変える、(c) 承認後に `-n` か `--seed` を変える、
  (d) 束のファイルを 1 つ書き換える、(e) 見えるハッシュを残して HTML の描画データ（検出器の位置）だけ書き換えて `approve`、
  (f) VM 上のジョブの値を書き換える（受領票の識別情報の不一致）、(g)（U3 (a) のとき）ジョブは変えずに `Core.run` に渡す `seed` か `start` だけを変える（故障注入の環境変数。既定では無効）。
  (a)〜(e)・(g) は手元のテスト（CPU の双子）、(f) と (g) の 1 つは Colab で 1 回。
- **既存の経路を変えない**: `detector:` のないシーンで、`numpy`・`kernel` の結果がバイト一致（既存のテストがすべて通る）。G1 の関門のテストもすべて通る。

## 4. 対象範囲（骨子）

- 変更してよい: `chatcarlo/__main__.py`（`run`・`preview`・`approve` の診断の分岐、`_mv_dispatch` の `value_options` に新しいオプションを足す）、`chatcarlo/scene.py`（`detector:` の検証）、
  `chatcarlo/preview.py`（検出器の面の描画、承認用の文字の欄）、
  `chatcarlo/gpu/` の新規ファイル（承認・条件ハッシュ・CLI の駆動・共通の集計・保存）、`engine.py`（集計を `aggregate` に置き換える）、`validation.py`（回帰確認の判定を足す）、`run_detector_image.py`・`colab_driver.py`（`worker` の時間の計測、観測の受領票への記録）、
  `core.py` の Python 側（U3 (a) のときだけ、観測の戻り値と故障注入。ネイティブ関数への引数の値と順は変えない）、新規テスト、
  `tests/test_mv_dispatch.py` 19 行（診断の `approve` の挙動が変わるため）、`docs/validation/gpu/g4/`（通しの記録・シーンの写し・比較の結果）、
  `CLAUDE.md` の Commands 節と `.claude/rules/chatcarlo.md` の GPU 段落（doc-impl 同期）。
- 変更禁止: G1 の核の C++・CUDA（`chatcarlo/gpu/*.h`・`*.cpp`・`*.cu`）、`source_host.py`、`bundle.py` の束の形式、`transport.py`（`run_transport` は既に `detector`・`detector_roi` を受けるので、CLI がシーンから作って渡す）、
  `physics.py`・`geometry.py`・`detector.py`・`source.py`・`spectrum.py`・`materials.py`・`tally*.py`、`prototypes/mv_electron/`（読んで写すだけ）、
  `data/`、`docs/jsmp133/`、関門の記録（`docs/validation/gpu/g1/`）。
- スコープ外: G2（ボクセルのタリー）・G3（cylinder・sphere・`mas` など）、治療領域、`trace`・`animate` の GPU 化、`plot` の画像対応（PNG は `run` が直接書く）、
  スマートフォンからの実行、EGS5 との照合や大規模な物理検証。

## 5. 停止条件

- 関門 1・2 の記録が `pass` でない、または束の形式・核の C++ を変えないと G4 ができないと分かった → 実装せず報告。
- §3 の回帰確認で、事前に固定した比較の基準（`photon_count` の完全一致と、絶対差の上限）を満たさない → Colab の通しを記録に使わず、止まって報告。
- Colab のセッションが消えた → `launch` の既存の再開（回収済みは送り直し）で 1 回だけ再実行し、2 回目も消えたら報告。

## 6. 詳細: 宣言・ファイル・関数

### 6-1. scene.yaml の `detector:`（U1）

```yaml
detector:
  center_cm: [0, 0, 0]
  normal: [0, 0, 1]
  u_axis: [1, 0, 0]
  size_cm: [15, 18]
  shape: [100, 120]
  roi: [[40, 60], [50, 70]]   # 任意。画素の番号の半開区間 [(u0,u1),(v0,v1)]、DetectorTally の roi と同じ形
```

- `scene.py` の `validate_scene` に検証を足す: 未知のキー、`center_cm`/`normal`/`u_axis` が有限の 3 要素、`normal`・`u_axis` が単位で直交
  （`DetectorPlane.__post_init__` と同じ許容 `np.isclose`）、`size_cm` が正の有限 2 要素、`shape` が正の**整数** 2 要素（`100.0` や `True` は拒む）、
  `roi` が `0 ≤ u0 < u1 ≤ shape[0]`・`0 ≤ v0 < v1 ≤ shape[1]` の整数。誤りは既存と同じ `SceneError("detector.<key>", "...")` で返す。
- 新しい関数 `chatcarlo.scene.detector_from_scene(scene) -> (DetectorPlane | None, roi | None)`。`cmd_run` と G4 の駆動が使う。
- `numpy` の `run` は `detector:` があれば `run_transport(..., detector=..., detector_roi=...)` に渡し、U4 と同じ保存をする。`kernel` は既存の拒否（`run_transport` の「kernel engine は detector を…」）をそのまま出す。
- 最上位の未知のキーを拒む変更はしない（既存のシーンを壊すおそれがあるので G4 の範囲外）。

### 6-2. CLI（`__main__.py`）

- `run --engine` の choices に `gpu` を足す。新しいオプション: `--gpu-device {cpu,colab}`（既定 `cpu`、`--engine gpu` 以外で指定したら誤り）、`--source-seed`、`--detector-out`、
  `--run-dir`（Colab の実行一式の場所。既定は §2 の `~/.cache/chatcarlo/gpu_runs/<12 文字>/`）。値を取る新しいオプションは `_mv_dispatch` の `value_options` にも足す。
- `preview` に `--engine`・`--gpu-device`・`-n`・`--seed`・`--batch-size`・`--source-seed`・`--no-uncertainty`・`--run-dir` を足す（`run` と同じ実効の条件を作れるように。`--no-uncertainty` は条件の `track_uncertainty` に入る）。`--engine gpu --gpu-device colab` のときだけ §2-1 の実行一式を作る。それ以外の `preview` は今と同じ出力（ただし `detector:` があれば検出器の面を描く）。
- `main` は今どおり argparse より先に `_mv_dispatch` を呼ぶ（治療のシーンは従来の入口へ、実行一式のフォルダは診断の `cmd_approve` へ）。
- `approve`: 診断のシーンでは `chatcarlo approve <実行一式のフォルダ>` を受ける（引数がフォルダで、中に生成記録があるとき）。シーンのファイルを渡したときと、生成記録のないフォルダは、今と同じく 1 を返す
  （`tests/test_mv_dispatch.py` 19 行の 2 例は、そのまま 1 を返すことを保つ。治療の `approve SCENE --preview DIR` の分岐は変えない）。
- `--engine gpu` で `--seed` がなければ `run` の前に誤り（G1 は種が必須）。`--workers` ≥ 2・`--dose-grid`・`--kernel-*` との併用は誤り。

### 6-3. 新しいファイル

- `chatcarlo/gpu/condition.py`: 実効の条件の解決（§2-1）と正準 JSON・SHA256。**手元の実行（`numpy`・`gpu` の `cpu`）の保存用**には、検証済みのシーン（`scene_sha256`）・実効のオプション・実際の engine と device から作る条件記録を使い、束は作らない（`numpy` は G1 の束を作れないシーンも扱うため）。保存先は `--detector-out` の場所（省略時は `./chatcarlo_out/<条件ハッシュ 12 文字>/`）。Colab の承認用の条件は次のとおり。条件 = `{format: "chatcarlo-g4-v1", bundle_sha256（目録全体）, job, detector（宣言の値）, roi, track_uncertainty}`。
  `job` は `{mode: "batches", device: "cuda", n_histories, batch_size, seed, source_seed}`（`colab_driver.identity` が使う形のまま）。
- `chatcarlo/gpu/approval.py`: `prepare(scene, options, run_dir)`（束・ジョブ・スナップショット・HTML・`generation.json` を作る。承認済みなら止まる）、
  `approve(run_dir)`、`require(run_dir, expected_condition)`（送る前の照合）。`generation.json` は条件の完全なハッシュと、束の目録・`job.json`・`preview_snapshot.json`・`preview.html` の SHA256 を持つ。
  `approval.json` は `generation.json` の SHA256 と時刻を持つ。
- `chatcarlo/gpu/aggregate.py`: 照合済みのバッチ結果の列（`Core.run` の戻り値の辞書）から、`TransportResult` 相当と `DetectorTally` を作る共通の集計（§2-3）。
  `engine.run` の集計部分もこれに置き換える（同じ値になることをテストで確かめる。足し算の順は今の `engine.run` と同じにする）。
- `chatcarlo/gpu/cli_run.py`: `cmd_run` から呼ぶ駆動。手元（`cpu`）は `engine.run` を呼ぶ。`colab` は `approval.require` → `colab_driver.launch` → 受領票と観測の照合 → `aggregate` → 保存。
- `chatcarlo/gpu/outputs.py`: U4 の保存（`.npz` は `run_detector_image.save` と同じ配列名、`.json`、PNG）。PNG は matplotlib で `photon_count` を、u を横軸・v を縦軸、
  画素 (0,0) を左下にして描き、軸の名前と単位（画素の番号、cm の目盛り）を付ける。`run_detector_image.save` はこの関数を呼ぶ形にする。統計を無効にしたとき（`--no-uncertainty`）は、`.npz` のキーをそろえたまま `total_rel_err` を NaN の配列、`n_batches_hit` を −1 の配列（未集計の印）にし、`.json` に `track_uncertainty: false` を書く。

### 6-4. 観測の層（U3 (a)）

- `Core.run` に引数 `observe=False` を足す。**型変換と故障注入を済ませた最終の呼び出し引数の列を 1 回だけ組み立て、ネイティブ関数の呼び出しと観測の両方がその同じ列を参照する**（Codex の指摘 2）。
  観測 = 引数の位置ごとの記録: スカラーはその値、配列はその位置に渡した配列の SHA256（CPU は ctypes のポインタの元の NumPy 配列、GPU は RawKernel に渡す CuPy 配列を `cp.asnumpy` して求める。静的な表と幾何の配列は `Core` の作成時に 1 回だけ求めてキャッシュし、**引数の位置と配列の名前の対応を固定の表で持つ**）。
  ホストは同じ位置の表から期待値を作るので、`centers` と `halves` の取り違えのような位置の誤りも検出する。GPU のメモリの破損一般は対象にしない。
- 戻り値: `observe=True` のとき `Core.run` は `(result, observed)` の組を返す（`result` は今と同じ数値だけの辞書）。`health` と `.npz` には `result` だけを渡す（辞書を入れると `allow_pickle=False` で読めないため。Codex の指摘 3）。
- 故障注入: 環境変数 `CHATCARLO_G4_FAULT` が `seed` か `start` のとき、ネイティブ関数に渡す値だけを 1 ずらす（観測はずらした値を記録する）。既定は無効で、無効のときの引数と結果は今とバイト一致。
- `worker`（`colab_driver.py`）は `observe=True` で呼び、受領票に `observed` と `elapsed_s` を入れる（識別情報 `identity` の形は変えない。どちらも `identity` の外の新しいキー）。
  受領票に `observed` が無い、または項目が足りないバッチは、集計の前に拒む。**再利用したバッチ**（回収済みを送り直したもの）も、受領票の `observed` を同じく照合する。
- 故障を VM に設定する手順: 手元の環境変数は VM に伝わらないので、ジョブとは別の小さなファイル `fault.json`（照合の対象外）を送り、`worker` が読んで環境変数と同じ扱いにする。本番の通しでは送らない。
- ホストの照合（`approval.verify_batches`）: 承認済みの束から `bundle.load` → 表と幾何の配列の SHA256、承認済みのジョブから `source_host.sample_source_photons` を同じ順に回して各バッチの線源 3 配列の SHA256 を作り直し、
  受領票の `observed` と全項目を比べる。1 つでも違えば `observation mismatch: batch_000003 seed` の形で止まる（集計しない）。
- 手元の `cpu` の経路も `observe=True` で同じ照合を通す（テストで故障注入を検出するため。`seed`・`start` の両方を CPU で検査する）。

### 6-5. 時間

- 時間はバッチの受領票の `elapsed_s`（`identity` の外）に入れる（Codex の指摘 4。既存の転送と再利用でそのまま運ばれる）。照合の指紋には入れない。
  ホストが合計と内訳（新規のバッチの秒と、再利用したバッチの秒）を作り、端末と `.json` に出す。「新規だけ」「全部再利用」「途中から再開」で、時間とバッチ数の扱いをテストする。

## 7. 受入条件

- [ ] `detector:` の検証: 正しい例が通り、§6-1 の誤り（未知のキー、非単位、非直交、非整数の画素数、`True`、ROI の範囲外、非有限）がそれぞれ `detector.<key>` の誤りになる。`detector:` のない既存の `examples/` の全シーンの検証結果（誤りと警告）が変わらない。
- [ ] `numpy` で `detector:` のあるシーンを `run` すると、`run_transport(..., detector=DetectorPlane(...))` を直接呼んだ結果と画像の全配列がバイト一致し、§6-3 の 3 ファイルが保存される。
- [ ] `--engine gpu`（`cpu`）で依頼 D の肖像のシーンの写しを 10⁴ 履歴で `run` すると、`run_transport(engine="gpu", gpu_device="cpu", ...)` を直接呼んだ結果と付与・画像がバイト一致する。
- [ ] `aggregate` に置き換えた後の `engine.run` が、置き換える前の `engine.run` と付与・SEM・R・転帰・画像でバイト一致する（G1 の S1・S2 の小さな実行、テストに固定値を入れる）。
- [ ] 集計の正しさ（§3）: 別の素朴な足し合わせとの一致、PNG の向きのテスト。
- [ ] わざと変える検査（§3 の (a)〜(e)・(g)）が手元のテストでそれぞれ止まり、止まった理由の文が検査ごとに違う（`approval.json` なし／条件の不一致／束の SHA256／HTML の SHA256／観測の不一致）。
- [ ] `observe=False`（既定）と故障注入なしのとき、`Core.run` の戻り値が今とバイト一致する（G1 の既存テストがすべて通る）。
- [ ] 統計無効: `--no-uncertainty` で preview → approve → run（手元の CPU の経路で承認の照合を通す試験の形）が通り、承認後に統計の設定だけを変えると条件の不一致で止まる。
- [ ] 集計の追加ケース: 検出器なし、ROI あり（共分散と SEM）、統計無効、1 バッチ、端数のバッチ、検出 0 のバッチ。NaN の位置も確かめる。独立の足し合わせとの比較は、足し算の順をそろえればバイト一致、順が違えば上の丸めの式の内側、と分けて判定する。
- [ ] `detector:` の追加の誤り: `detector: null`、辞書でない、必須のキーの欠落、要素数の違い、ROI に bool。どれも例外で落ちず `SceneError` になる。
- [ ] 描画: 移動・傾けた検出器も初期の画角に入る。面の頂点は `corners_cm()` の順（周回の順ではない）をそのまま使わず、周回の順に並べ替える。
- [ ] 既存の経路のバイト一致: 「既存テストが通る」とは別に、`numpy` は既存の固定値、`kernel` は小さな固定の実行（box・単色・parallel、10⁴ 履歴、seed 固定）で、変更の前後の付与がバイト一致することを確かめる。
- [ ] `approve` の挙動: `tests/test_mv_dispatch.py` の既存の検査がすべて通る。生成記録のあるフォルダでだけ承認が成り立ち、生成記録があっても中身の照合に失敗すれば承認しない。承認済みのフォルダに `preview` をもう一度書こうとすると止まる。治療の側の新しいテスト: `approve SCENE --preview DIR` が引数を変えずに MV 側へ渡る、`--preview DIR` がシーンより前でも渡る、新しい値つきのオプションがシーンより前にあっても MV と認識され、MV 側が対応しないオプションとして拒む。
- [ ] Colab（Claude が実行、ユーザーの承認が要る）: §3 の「通し」「回帰確認」と、(f)・(g) を 1 回ずつ。結果を `docs/validation/gpu/g4/RESULTS.md` に書く。
- [ ] 回帰確認の判定の式: 画素ごとに `|a−b| ≤ (k_a + k_b + 2B) · 2⁻⁵² · (|a| + |b|)`。`B = ceil(N / batch_size)`（ホストのバッチの足し合わせの回数）。
  `k` はその画素の装置の中の加算の回数の上界で、**`photon_count[u,v]` を使う**（1 件の検出ごとに、そのカテゴリの `category_fluence`・`energy_sum_keV`・`energy_sum2_keV2` にそれぞれ 1 回ずつ加算する。`g1_history.h` 83 行付近。依頼 D の保存済みの `.npz` には `category_count` が無いので、カテゴリ別の配列にも `photon_count` を上界として使う）。両辺 0 は一致。`photon_count` は完全一致。
  この式は、今回の 10⁶ 履歴の回帰確認のための保守的な基準で、任意の加算回数や輸送の差を保証する一般式ではない。判定のコードは `chatcarlo/gpu/validation.py` に足す。
  判定の前に、比べる相手の `.json` の seed・source_seed・batch_size・履歴数と、束の目録のうち**核のソース・表・幾何・スペクトル・線源の辞書**のハッシュが一致することを照合する。G4 で変わる runtime の Python（`colab_driver.py`・`core.py` など）と `settings.json` のハッシュは、核と入力の不一致と区別して報告する。
- [ ] doc-impl 同期: `CLAUDE.md` の Commands 節に `run --engine gpu` の例を 2 行（手元・Colab）、`.claude/rules/chatcarlo.md` の GPU 段落の「API only」を書き換える。

## 8. テストコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -q -k "gpu or mv_dispatch or scene or detector"
.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k entry
.venv/bin/python -m chatcarlo validate docs/validation/gpu/g4/scene_portrait_detector.yaml
.venv/bin/python -m chatcarlo run docs/validation/gpu/g4/scene_portrait_detector.yaml --engine gpu -n 1e4 --seed 2040 --detector-out /tmp/g4_cpu.npz
```

## 9. 書かなかったこと（スコープ外）

- 最上位の未知のキーを拒むこと（既存のシーンに影響する）。
- `numpy` の `run` の承認（U2）。`plot` で検出器の `.npz` を描くこと。
- 観測の層を治療の入口と共通にすること（`prototypes/` を `chatcarlo/` から import しない）。
- 束の形式の変更、核の C++・CUDA の変更、速さの工夫。
