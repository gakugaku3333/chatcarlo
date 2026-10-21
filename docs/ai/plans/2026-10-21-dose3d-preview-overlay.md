# 計画: 3D 表示（preview の HTML）に線量グリッドを色で重ねる

- 状態: codex-reviewed
- 作成者: claude
- 実装担当: codex
- 日付: 2026-10-21
- Codexレビュー: 1巡（致命 0・実質 5・軽微 5・参考 3。すべて反映。記録は data/logs/codex_dose3d_review1.md）
- 規模: 中。新機能で、線量の計算は増えず、既存の .npz を読んで見せるだけ。ただし「どのボクセルを見せ、どれを隠すか」を誤ると、信用できない値を信用できるように見せてしまう。そのため隠す規則は受入条件で固定する。

## 目的

JSMP133 の抄録の依頼 3 では、「IVR の空間線量分布をカラーマップで 3D 表示したい」という、今のシステムにない要素を含む依頼を、対話で調整する例として使う。
今の chatcarlo の 3D 表示（`preview`/`trace` の自己完結 HTML）は、形状と軌跡しか描けない。線量は `plot` の 2D 断面図（PNG）でしか見られない。
そこで、`run --dose-grid --dose-out` の .npz を、`preview` の 3D 表示に半透明の色付きボクセルとして重ねられるようにする。
その際、CLAUDE.md の「R の指針」と「Known sharp edges」（線源のすぐ近くの空気、空気中の吸収線量）の規則を、表示の段階でも守らせる。

## 0. 前提事実（検証済み、2026-10-21 に実行）

**F7 既存コードとの契約**
- `.npz` のキー（`chatcarlo/__main__.py` の `cmd_run`）:
  - 常に入る: `dose_per_history_Gy`, `h10_per_history_pSv`, `kerma_keV`, `h10_track_pSv_cm3`, `origin_cm`, `voxel_size_cm`, `shape`
  - `source.mas` があるとき: `dose_Gy`, `h10_pSv`
  - `--no-uncertainty` を付けないとき: `rel_err_dose`, `rel_err_h10`, `sem_*`, `n_batches`, `n_batches_hit`
  - 校正の方式と設定した mAs は入らない。
- **.npz には、シーンの情報（線源の位置・背景材料・シーンの指紋）が入っていない。** そのため、3D 表示には scene.yaml が必須。
- グリッドは `VoxelGrid.from_bbox(geometry.tally_bbox_min, geometry.tally_bbox_max, resolution)` で作られる（`transport.py` 465/518 行）。同じシーンと `voxel_size_cm` から、`origin_cm` と `shape` を再計算して突き合わせられる。angio_room（5 cm）で実測すると、npz の `shape=(120,120,79)`, `origin=[-300,-300,-65]` は再計算の値と一致した。
- 量の選び方は `plotting.select_quantity` に従う（校正済みのキー `dose_Gy`/`h10_pSv` を優先）。
- 統計の閾値は `diagnostics._UNRELIABLE_R_THRESHOLD = 0.10` と `_UNRELIABLE_HIT_FRACTION_THRESHOLD = 0.25`。判定は `R > 0.10` または `hit/M < 0.25`（`unreliable_max_warning`）。
- 線源のすぐ近くの空気の判定: 材料が background で、かつ線源からの距離が `geometry.nearest_object_distance_cm(src)` より小さい（`near_source_air_warning`）。この距離は、曲面までではなく**各物体の境界の箱までの距離**。
- `preview.py` の `_SHARED_CANVAS_JS` は `animate.py` も埋め込んでいる。`render_html`/`write_html` は `trace` も使う。
- 同じチェックアウトで、別のセッション（依頼 A+B）が `chatcarlo/__main__.py` に未コミットの変更を入れている（2026-10-21 時点、HEAD=85630be）。
  - 中身は `_mv_dispatch` で、`preview` 等のシーンが `domain: therapy_mv` なら試作の CLI に回す。
  - 値を取るオプションは `value_options` の固定の集合で読み飛ばしている。
  - 振り分け先の import は関数の中で `from prototypes.mv_electron.entry.cli import main as mv_main` と書かれている。

**実測（examples/angio_room.yaml、5 cm 格子、H*(10)、mAs 校正済み）**

| 履歴数（batch） | 時間 | 全ボクセル | 非ゼロ | 統計不足 | 線源近くの空気 | 統計条件を満たす |
|---|---|---|---|---|---|---|
| 2e5（1e4、M=20、1 worker） | 18 s | 1,137,600 | 560,784 | 557,711 | 968 | 2,564 |
| 2e6（5e4、M=40、4 workers） | 41 s | 1,137,600 | 825,320 | 778,884 | 968 | 45,468 |

- 統計不足は、R>0.10 または寄与バッチが 25% 未満のもの。
- 2e6 では、寝台から約 1 m の室内の点で R=0.106（寄与 35/40、点 (100,0,100)）と R=0.136（点 (-150,0,100)）だった。全体の R の中央値は 0.32。
  - **統計不足のボクセルを既定で隠すと、履歴数が少ないうちはほぼ主線束しか見えない。** これは正しい挙動なので、HTML に「隠した数」と「履歴数を増やす目安」を出す。
- H*(10) の最大値は 4.2e10 pSv で、線源から 42.6 cm の、主線束の中の空気のボクセルにあった。線源近くの空気の判定（30.4 cm より近い）には当たらない。
  - 室内の散乱は約 5e6 pSv で、最大値の約 1e-4 にあたる。**色の範囲は 6 桁とる**。また、主線束に色の範囲を引っ張られないよう、「色の上限」を下げられるようにする。
- 2e6 の .npz の、統計条件を満たすボクセルの桁ごとの数（最大値に対する比の log10 の切り捨て）:
  - 0 桁: 1、-1 桁: 63、-2 桁: 238、-3 桁: 5,122、-4 桁: 39,765、-5 桁: 279（計 45,468）
  - 統計不足のほう: -4 桁 37,878、-5 桁 467,186、-6 桁 251,955（計 757,019、最大値の 1e-6 以上のもの）
- 描画の速さ（ブラウザペイン、Canvas 2D、半透明の四角、奥行きでの並べ替え込み、1 フレームあたり）: 10,000 個で 8.7 ms、30,000 個で 21.6 ms、100,000 個で 79.8 ms。
- .npz は 1 ファイル 95.6 MB（120×120×79）で、JSON に全部は埋め込めない。
- 独立検証に使う .npz（scratchpad に置いてあり、リポジトリには入れない）は、`run examples/angio_room.yaml -n 2e6 --seed 2 --batch-size 50000 --workers 4 --dose-grid --resolution 5` の出力。sha256 は `52421ff937a6f9f9ae15a677756dff425506117e2e58827f80a0c0ad67fe8ec0`。

**回帰の基準（変更前の `render_html(load_scene(p), title="golden")` の UTF-8 の sha256。HEAD=85630be、preview.py は未変更）**

| scene | sha256 |
|---|---|
| examples/abdomen_table.yaml | f0fcabbd61f842a4a928cbf7cb380fd674823dbfaf4d26dfaf4df346344d8c18 |
| examples/angio_room.yaml | 3f14bb5e71f5ac9ec7a0d17e5781db8daa7e572d3df8044fe3f89800915f0e76 |
| examples/chest_room.yaml | f0d0630737277be9a4c69e1449007cc27f5eea018fe694131f7cdfd82062cfab |
| examples/ct_room.yaml | 7c07a89707d189afa339de09e0d1546a9e37ceb8146707cd8bd39a63099fd332 |
| examples/portable_ward.yaml | 61a4bb4bef46b7035899a00006b77d2936e8b8da2ce425e662da872a039e0b0e |
| examples/water_phantom_pdd_ocr.yaml | e9c8d05bfbe0adb67ffee627cd0a7c9f2be9c8498d80cf1e3bf154e5a36d74e2 |

## 決めたこと（設計判断）

1. **コマンドの形**: `chatcarlo preview <scene.yaml> --dose <dose.npz> [--quantity h10|dose] [-o out.html] [--title ...]`。
   - `--dose` が無いときは、今の `preview` と同じ動作で、同じバイト列を出す。
   - `-o` を省いたときの出力は `<npz の拡張子を除いた名前>_3d.html`。
   - 新しいサブコマンドは作らない。形状と線量を同じ画面で確かめられることを優先する。
2. **既定の量は `h10`**。IVR の空間線量分布は空気中の場所の評価なので、H*(10) が意味を持つ。空気中の吸収線量は媒質に固有の量で、`background_medium_warning` の対象になる。
   - `--quantity dose` も選べる。その場合は HTML に「空気のボクセルの吸収線量は、人や検出器がいる位置の評価には使えない」という注意を常に出す。
3. **表示は、ボクセルごとに半透明の四角を置く方式（スプライト）にする。**
   - 各ボクセルの中心に、画面に平行な正方形を描く。辺はボクセルの大きさを透視投影した長さ。奥から手前の順に描き、色はカラーマップ、不透明度はスライダーで決める。
   - グリッドの外形は破線の箱で描く。
   - 6 面の箱は、Canvas 2D で数万個を毎フレーム並べ替えるには重いので使わない。
   - 3 断面を 3D 空間に置く案も採らない。2D の断面は既に `plot` にあり、今回の目的は空間への広がりを見せることだから。
   - 半透明の四角は重なるので、画面の色は凡例と一対一には対応しない。凡例は「`plot` と同じ LUT（inferno）」とだけ書き、数値はクリックで読むものとする。
4. **カラーマップは対数の inferno**。生成時に `matplotlib.colormaps["inferno"]` から 256 色を取り、JSON に埋め込む。
5. **HTML 内の操作**（範囲の値はすべて、埋め込んだ丸めの後の値から求める）:
   - (a)「表示下限 L」のスライダー: log10 で 0.1 桁刻み。範囲は [埋め込んだ全ボクセル（灰色も含む）の最小値, 最大値]。L 未満は描かない（色付き・灰色とも）。**色の下限も L とする。**
   - (b)「色の上限 H」のスライダー: 同じ範囲。H 以上は最も明るい色で描く。常に `log10 H ≥ log10 L + 0.1` を保つ。片方を動かして条件が崩れたら、もう片方を押し動かす。
   - (c)「不透明度」のスライダー。
   - (d)「統計条件を満たさないボクセルも表示（灰色）」のチェック。既定はオフ。
   - (e) クリックすると、描画中の中で最も手前のボクセルの値・単位・R・寄与バッチ数・統計不足の理由・材料・中心座標を表示する。
     - 非表示のボクセルは選ばない。
     - マウスが押されてから 4 px 以上動いた操作（回転・平行移動のドラッグ）はクリックとして扱わない。
   - 初期値は、L = max(色付きの最小値, 色付きの最大値×1e-6)、H = 色付きの最大値。
   - **色付きが 0 個のときは、(a)(b) を無効にし、色は使わない**（灰色の表示は (d) で使える）。色付きがすべて同じ値のときは、L = その値/10、H = その値とする。
   - 既存の面塗り・ビーム・ラベル・視点のボタンは、そのまま残す。
6. **単位の表示**: 凡例の見出しに、量と単位を必ず出す。根拠は .npz とシーンに分けて書く。
   - .npz に校正済みのキーがあるとき: 「H*(10) [pSv]（.npz は校正済み。シーンの設定: mAs=100）」。シーンに `ctdi_vol_mGy` があれば「CTDIvol=… mGy」と書く。
   - 校正済みのキーが無いときは、シーンの設定にかかわらず「[pSv/history]（相対値・校正なし）」と書く。
   - クリックで値を表示するとき、校正済みの pSv と Gy にだけ、読みやすい単位への換算を添える（例: 4.21e10 pSv（= 42.1 mSv））。
7. **ボクセルの分類**（Python 側で、丸める前の値で全ボクセルに 1 回だけ当てる。上から順に判定し、各ボクセルはどれか 1 つに入る）:
   - `zero`: 値が 0 以下。統計がある場合は、値が正でも `n_batches_hit == 0` ならここに入る。埋め込まない。
   - `near_source_air`: 材料が background で、線源からの距離が `nearest_object_distance_cm` より小さい。埋め込まず、数を凡例に出す。
     - `near_source_air_warning` と同じ規則を、ボクセルごとに当てる。材料はボクセルの中心で判定する（`dose_map_Gy` と同じ近似）。
     - `source.rotation` があるシーンでは、最初の線源の位置だけで判定している旨を、凡例に 1 行書く。
   - `unreliable`: `R > 0.10`、`hit/M < 0.25`、`R` が nan のいずれか。閾値は diagnostics の定数を import して使い、複製しない。理由をビットで持つ（1 = R が大きい、2 = 寄与バッチが少ない、4 = R が計算できない）。灰色で別に埋め込む。
   - `main`: 残り。統計があれば「統計条件を満たす」、無ければ「未確認」として色で描く。
   - 画面の文言は「信頼できる」ではなく「統計条件（R<0.10 かつ寄与バッチ≥25%）を満たす」とする。R は必要条件であって十分条件ではない（CLAUDE.md）。
   - **統計が無い場合**（`--no-uncertainty` で `rel_err_<q>`/`n_batches`/`n_batches_hit` がすべて無い場合、または `n_batches < 2` の場合）:
     - `unreliable` は空になり、`zero` と `near_source_air` 以外はすべて `main`（未確認）になる。
     - HTML の上部に「統計の情報が無いため、どのボクセルの信頼性も確かめていない」という赤い帯を常に出す。
     - 3 つのキーの一部だけがある場合は、壊れた .npz としてエラーにする。
8. **色の範囲は、隠したボクセルを除いてから決める。** 色の範囲の初期値は `main` の値だけから求める。線源近くの空気の 1 ボクセルが最大値を決め、部屋の残りが黒く沈むのを防ぐため。
9. **埋め込む数の上限と間引き**:
   - `main` は最大 30,000 個、`unreliable` は最大 20,000 個。どちらも、`main` の最大値の 1e-6 未満は捨てる（`main` が空なら `unreliable` の最大値を基準にする）。
   - 上限を超えたら、**桁ごとの水位方式で間引く**。最大値に対する比の log10 の切り捨てで、桁ごとの組に分ける。上限を組の数で等分し、数が割り当てに満たない組はすべて残し、余りを残りの組に配り直す。これを繰り返す。
     - 各組の中では、固定の seed（`np.random.default_rng(20261021)`）で値に依存せず抜き出す。組の最大値のボクセルは必ず残す。
   - 値の上位だけを残す方式は採らない。統計条件を満たす低線量の広がり（2e6 の例では -4 桁に 39,765 個）が丸ごと消えるため（Codex の 1 巡目の指摘）。
   - 2e6 の例で `main` に当てはめると、0〜-3 桁と -5 桁の計 5,703 個はすべて残り、-4 桁は 39,765 個から 24,297 個に間引かれる。
   - 凡例には桁ごとに「残した数/元の数」を出す。間引きがあれば「表示は間引き済み。空間的な広がりは保つが、隣り合うボクセルが欠ける」と書く。
   - `counts` は、`zero`、`near_source_air`、`below_range`（1e-6 未満で捨てた数、`main`/`unreliable` 別）、`thinned`（間引いた数、別）、`embedded`（別）に分ける。全部を足すとボクセルの総数になること。
   - HTML の大きさは、独立検証の .npz で 6,000,000 バイト以下（`os.path.getsize`）。
10. **シーンとの照合**: `VoxelGrid.from_bbox(Geometry(...).tally_bbox_min, tally_bbox_max, voxel_size_cm)` を計算して比べる。
    - `origin_cm` は `np.allclose(..., rtol=0, atol=1e-9)` で、`shape` は完全一致で比べる。合わなければ終了コード 1。`[エラー]` に、期待値と実際の値の両方を日本語で出す。
    - 入力の検査も行い、問題があれば同じエラーにする:
      - 必要なキーが欠けている。
      - 選んだ量・`rel_err_<q>`・`n_batches_hit` の配列の形が `shape` と違う。
      - `voxel_size_cm` が正の有限値でない。
      - 量に NaN か Inf が含まれる。
      - .npz が読めない。
    - これらはすべて `DoseOverlayError` に変換する。
    - 外形が同じ別のシーンは見分けられない。この限界は凡例に 1 行書く。
11. **ターミナルへの出力と終了コード**:
    - 書き出しの行に加え、分類ごとの数・色の範囲・間引きの有無を `[情報]` で出す。
    - 量が全部 0 以下: `plot` と同じく終了コード 1 で、HTML は書かない。
    - `main` が 0 個で `unreliable` がある: HTML を書いて終了コード 0。`[警告]`「統計条件を満たすボクセルが無い。-n を増やすか --batch-size を下げる」。
    - `main` も `unreliable` も 0 個で、正の値がすべて `near_source_air`: HTML を書いて終了コード 0。`[警告]`「正の値はすべて線源のすぐ近くの空気（点線源の 1/r² による非物理的な値）にある。評価したい位置に細かいグリッドを敷く」。この場合、履歴数を増やす案内は出さない。
    - 統計が無い場合と `--quantity dose` の場合の注意は `[警告]` で出す。
12. **外部ライブラリを使わない**: 出力の HTML は、今の preview と同じく vanilla JS と canvas だけで作る。`<script src`・`http://`・`https://` を含めない。Python 側の依存は既存のもの（numpy、matplotlib）だけ。
13. **検証のための窓口**: HTML の JS は `window.__dose3d` に次を出す。独立検証をスクリプトで行うため。
    - `pick(clientX, clientY)`: クリックと同じ選択の結果を返す。描画中でなければ null。
    - `state()`: L, H, 不透明度, 灰色表示, 描画数, 直前の描画時間 ms。
    - `setState({...})`: 値を変えて再描画する。

## 対象範囲

- 変更してよい:
  - `chatcarlo/dose3d.py`（新規）。.npz とシーンの照合、分類、間引き、埋め込みデータの構築、LUT、HTML/JS の断片。
  - `chatcarlo/preview.py`。`scene_to_json`/`render_html`/`write_html` に、省略できる `dose_overlay=None` を足す。テンプレートに、空文字に置き換えられる差し込み口を足す。`dose_overlay=None` のときの出力は、今とバイト単位で同じにする。
  - `chatcarlo/__main__.py` の次の 2 か所だけ:
    - (i) `preview` のパーサーに `--dose`・`--quantity` を足し、`cmd_preview` から渡す。
    - (ii) `_mv_dispatch` の `value_options` に `'--dose'`・`'--quantity'` を足す。
  - `tests/test_dose3d.py`（新規）
  - `CLAUDE.md` の Commands 節に 2 行（コマンド例と注釈）
- 変更禁止:
  - `_SHARED_CANVAS_JS` の中身、`animate.py`、`plotting.py`、`diagnostics.py`、`tally.py`、`transport.py`、`kernel.py`
  - `cmd_run`（.npz の形式は変えない）
  - `prototypes/`、`data/`、`.claude/skills/`、既存のテスト
- **`__main__.py` に触る直前の停止条件**: `git status --short chatcarlo/__main__.py` と `git diff chatcarlo/__main__.py` を確かめる。
  - 依頼 A+B の未コミットの変更（`_mv_dispatch`/`cmd_approve`）が残っていれば、`__main__.py` には触らない。他の部分だけを実装して停止し、その旨を報告する。
  - 理由: コミットに A+B の差分が混ざるのを防ぐため。A+B のコミットの後に、この 2 か所だけを足す。
  - `__main__.py` に依存するテスト（T2・T10・T11）は、そのとき失敗してよい。報告に明記すること。

## 受入条件

テストは `tests/test_dose3d.py` に置く。合成の小さなシーンを tmp_path に YAML で書き（水の箱 1 個と点線源）、それに合う合成の .npz を作って使う。.npz の `origin_cm`/`shape` は `VoxelGrid.from_bbox` で求める。

- [ ] **T1 照合と入力の検査**: 次のそれぞれで、`build_dose_overlay` が `DoseOverlayError` を出し、メッセージに期待値と実際の値（形の不一致では両方の形）が含まれる。
  - `shape` を 1 つずらす。
  - `origin_cm` を 1e-6 cm ずらす（許容差より大きい）。
  - 必要なキーが欠けている。
  - 量の配列の形が違う。
  - 統計のキーの一部だけがある。
  - `voxel_size_cm` が 0 か NaN。
  - 量に Inf が含まれる。
  - `origin_cm` のずれが 1e-10 cm なら通る。
- [ ] **T2 量がすべて 0**: CLI が終了コード 1 になり、HTML は書き出されない。
- [ ] **T3 線源近くの空気**: 線源から最寄りの物体より近い空気のボクセルに、他のボクセルの 1e6 倍の値を置く。
  - そのボクセルは埋め込まれず、`near_source_air` が 1 になる。
  - `vmax`（H の初期値）は、残りのボクセルの最大値を 4 桁に丸めた値と一致する。
  - 別のケースとして、正の値が線源近くの空気だけにある場合は、HTML を書いて終了コード 0 になり、出力に「線源のすぐ近く」の文言があって「-n を増やす」の文言が無い。
- [ ] **T4 統計の規則**:
  - R=0.10 ちょうどで hit/M=0.25 ちょうど（例: M=4, hit=1）のボクセルは `main`。
  - R=0.1001 のボクセルは `unreliable`（理由 1）、hit/M<0.25（例: M=8, hit=1）は `unreliable`（理由 2）、R=nan は `unreliable`（理由 4）。
  - 値が正で `n_batches_hit=0` のボクセルは `zero`。
  - diagnostics の 2 つの定数を monkeypatch で変えると判定が変わる（定数を参照していることの確認）。
  - `main` が 0 個で `unreliable` だけのとき、`vmin`/`vmax` は null。CLI は終了コード 0 で、「-n を増やす」の文言を出す。
- [ ] **T5 統計が無い場合**:
  - 統計の 3 キーが無い .npz では、`zero` と `near_source_air` 以外の全ボクセルが `main` になり、`stats_available` は False、HTML に赤い帯の文言が入る。
  - `n_batches=1` の .npz でも同じになる。
- [ ] **T6 間引き**:
  - (a) 1 桁だけの組で 30,001 個のとき、埋め込みは 30,000 個、`thinned` は 1。組の最大値は残る。
  - (b) 桁が 3 つで個数が 10 / 100 / 50,000 のとき、10 と 100 はすべて残り、残りの 29,890 個は 3 つ目の組から取られる。
  - (c) `unreliable` が 20,001 個のとき、埋め込みは 20,000 個になる。
  - (d) 最大値のちょうど 1e-6 倍のボクセルは残り、それより小さいボクセルは `below_range` に入る。
  - (e) 同じ入力では、2 回実行しても埋め込みの添字が完全に一致する。
  - (f) `counts` を全部足すと、ボクセルの総数になる。
- [ ] **T7 単位**:
  - `h10_pSv` があれば、校正済みの値と単位の文字列（「校正済み」と「mAs=」）になる。
  - `h10_pSv` が無ければ `h10_per_history_pSv` と「/history」になる。シーンに mas があっても「/history」のまま。
  - シーンに `ctdi_vol_mGy` があり、かつ校正済みのキーがあれば「CTDIvol」になる。
  - `--quantity dose` のときは、空気の注意の文言が HTML に入る。
- [ ] **T8 HTML**:
  - `--dose` 付きの出力に `<script src`・`http://`・`https://` が含まれない。
  - HTML の `<script>` の中身を取り出した JS が `node --check` を通る（node が無い環境では skip）。
  - 埋め込みの JSON に、`materials`（材料名の表）、`lut`（256 個の "#rrggbb"）、`counts` がある。
- [ ] **T9 回帰（`--dose` 無し）**: 0 節の表の 6 個のシーンで、`render_html(load_scene(p), title="golden")` の UTF-8 の sha256 が表の値と一致する。値はテストに固定で書く。
  - 今後シーンの例を変えたら、この値も更新が必要になる。その旨をテストのコメントに書く。
- [ ] **T10 CLI の通し**:
  - `examples/chest_room.yaml` を `-n 2e4 --batch-size 1000 --dose-grid --resolution 10 --seed 1` で run し、`preview examples/chest_room.yaml --dose <npz> -o <html>` が終了コード 0 で HTML を書く。
  - `--dose <npz> --quantity h10` をシーンより前に置いても同じになる。
  - 別のシーン（`examples/water_phantom_pdd_ocr.yaml`）を渡すと、終了コード 1 になる。
- [ ] **T11 MV への振り分け**: tmp_path に `domain: therapy_mv` の YAML を置く。`sys.modules` に `prototypes.mv_electron.entry.cli` の stub を入れて、`main` が受け取った argv を記録させる。
  - `chatcarlo.__main__._mv_dispatch(["preview", "--dose", "x.npz", "--quantity", "h10", <mv.yaml>])` で、stub が呼ばれ、argv がそのまま渡る。
  - 試作の側の実装は変えない。
- [ ] 既存のテストがすべて通る（`tests/` 全体。とくに `test_plotting.py`、`test_mv_dispatch.py`、preview/trace/animate のテスト）。

**Claude の独立検証**（Codex の受入条件ではない。実装の後に Claude が行う。行うまでは、操作・性能・容量は未検証と報告する）:
- 0 節の 2e6 の .npz（sha256 で照合する）で、`--quantity h10` の HTML を作る。大きさは 6,000,000 バイト以下であること。
- ブラウザペイン（Claude 内蔵のブラウザ、既定の窓の大きさ）で開き、`window.__dose3d` を使って確かめる。
  - 等角の視点で、灰色の表示をオンにし、L を最小にする（描画数は最大の 50,000 個まで）。2 回描いて慣らした後、10 回描いた描画時間の中央値が 60 ms 以下であること。
  - 画面の要素を確かめる: 主線束、寝台のまわりの散乱、灰色の切り替え、H のスライダー、L ≥ H にしようとしたときの押し動かし。
  - `pick` を確かめる:
    - 重なる 2 ボクセルで手前を返す。
    - 非表示のボクセルは返さない。
    - ズームとリサイズの後も、投影した座標で同じボクセルを返す。
    - ドラッグの後にクリックの表示が出ない。
    - コンソールにエラーが無い。
- `pick` で 3 点の値を読み、.npz の同じボクセルの値と比べる（相対差 5e-4 以下。4 桁に丸めているため）。

## テストコマンド

```bash
.venv/bin/python -m pytest tests/test_dose3d.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m chatcarlo run examples/chest_room.yaml -n 2e4 --batch-size 1000 --seed 1 --dose-grid --resolution 10 --dose-out /tmp/dose3d_chest.npz
.venv/bin/python -m chatcarlo preview examples/chest_room.yaml --dose /tmp/dose3d_chest.npz -o /tmp/dose3d_chest.html
```

（`/tmp` の出力ファイルは消さずに残してよい。削除は利用者が行う。）

## 実装方針

- **`chatcarlo/dose3d.py` の公開関数**
  - `build_dose_overlay(npz_path, scene, quantity="h10") -> dict`: 照合・分類・間引き・埋め込みデータの構築を行う。HTML は作らない。テストの主な対象。
  - 失敗はすべて `DoseOverlayError(ValueError)` で返す。
- **戻り値の dict**（JSON にそのまま入れる）:
  - `quantity`, `unit_label`, `calibration_note`
  - `voxel_size_cm`, `origin_cm`, `shape`
  - `stats_available`, `n_batches`
  - `counts`（設計判断 9）、`decades`（桁ごとの残した数と元の数）
  - `main`, `unreliable`, `vmin`, `vmax`（null あり）
  - `materials`, `lut`, `notes`（凡例と帯に出す日本語の文のリスト）
- **`main`/`unreliable` の形**: 列ごとの配列にして JSON を小さくする。
  - `ix`/`iy`/`iz` は整数、`v` は有効数字 4 桁、`R` は小数 4 桁か null、`hit` は整数か null、`why` は理由のビット、`mat` は `materials` の添字。
  - 分類と間引きは丸める前の値で行う。HTML の中の比較とスライダーの範囲は、すべて埋め込んだ値で行う。
- **`preview.py`**:
  - `scene_to_json(..., dose_overlay=None)` は、None でなければ `"dose"` キーを足す。None ならキー自体を足さない。
  - テンプレートに `__DOSE_HEADER__`（スライダー類）・`__DOSE_LEGEND__`・`__DOSE_JS__` の差し込み口を置く。置く位置は、空文字に置き換えた後のバイト列が今と同じになるところ（行の途中で、前後に改行を増やさない位置）。T9 で確かめる。
  - 描画は `draw()` の中で、面塗りの後、軌跡とラベルの前に呼ぶ `drawDose()` で行う。JS の断片は `dose3d.py` に置く。
  - 初期化の順: テンプレートの最後の `resize()` より前に、線量の状態の初期化を済ませる（`resize` がすぐに `draw` を呼ぶため）。
- **描画**:
  - 表示下限以上（灰色の表示がオンなら `unreliable` も）を選び、投影し、奥行きで並べ替え、`fillRect` で描く。
  - 色は log10(v) を [log10 L, log10 H] で 0〜255 に写して LUT を引く。灰色は `#9ca3af`。
  - クリックの選択には、直前の描画で使った画面上の四角の一覧を使い、手前から当たりを調べる。
- **凡例**:
  - 縦のカラーバー（CSS の linear-gradient）、1 桁ごとの目盛り、単位、分類ごとの数、桁ごとの間引き、照合の限界の 1 行、`notes`。
  - 統計不足で隠したボクセルがあれば、「灰色のボクセル（N 個）は R>0.10 か寄与バッチ<25%。履歴数を増やすと減る」と書く。
- **`cmd_preview`**: `--dose` があれば `build_dose_overlay` を呼ぶ。`DoseOverlayError` なら `[エラー]` を出して 1 を返す。ターミナルへの出力は設計判断 11 のとおり。

## 書かなかったこと（スコープ外）

- 等値面、WebGL、ボクセル間の補間、3 断面を 3D 空間に置く表示。2D の断面は `plot` にあり、必要になったら別の計画にする。
- 値の上位だけを残す埋め込み。高い値の側に偏り、空間の広がりを削るので採らない（設計判断 9）。
- 小さい固定の大きさの点の群で描く案（Codex の参考の指摘）。重なりは減るが、ボクセルが占める範囲が伝わらないので採らない。
- `animate`・`trace` への線量の重ね合わせ。`trace` は同じ `write_html` を使うが、今回は `--dose` を足さない。
- 治療領域（`domain: therapy_mv`）のシーン。`_mv_dispatch` が試作の CLI に回すので、試作の側は対象外。
- .npz にシーンの指紋や校正の方式を入れる変更。`cmd_run` を変えることになるので、外形での照合で止める。
- vive-check・vive-interview などのスキルの手順への組み込み。別のセッションがスキルを編集中なので、後で別に行う。
- 統計の閾値や、線源近くの空気の判定規則そのものの変更。diagnostics の規則をそのまま使う。
- 主線束のボクセルを別に扱う機能。色の上限のスライダーで代える。
