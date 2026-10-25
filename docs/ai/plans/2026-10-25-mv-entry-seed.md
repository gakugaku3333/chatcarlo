# 計画: 治療領域の入口を P9c（前立腺シード 58 本の 7 門照射）に対応させる

- 状態: implemented（2026-10-26。承認 2026-10-25 Claude。隔離された試作で致命 0、実質 9 は反映済み。実 Colab 完走・記録は docs/validation/mv/entry_seed/RESULTS.md）
- 作成者: claude
- 実装担当: codex（入口の追加・偽クライアントでのテスト）。Colab の実行・`~/.cache` への書き込みは Claude
- 日付: 2026-10-25
- Codexレビュー: 1巡（致命 0、実質 9、軽微 2、参考 3。原文: [reviews/2026-10-25-mv-entry-seed-review1-codex.md](reviews/2026-10-25-mv-entry-seed-review1-codex.md)。対応は末尾の表）
- 規模: 中〜重（物理の演算・P9c の核は変えない。入口の許可リスト・承認・来歴の接続だけを足す。ただし「承認した条件 = 実際に計算した条件」の保証を新しい条件に広げるので、誤りに気づきにくい種類）

## 目的

JSMP133 の依頼 3（前立腺内の I-125 シードが外照射の線量分布に与える影響）を、**対話の入口（vive-interview → scene.yaml → `chatcarlo validate / preview / approve / run`）から Colab の GPU まで通した実績**を作る。
今の入口（`prototypes/mv_electron/entry/`）は「水 30 cm・10×10 cm²・1 門・`pdd_ocr`」だけを許可リストに持つ。P9c は個別の駆動（`gpu/p9c_cli.py`、`gpu/p9c_colab.py`）で回したため、入口からは通らない。
光子数は少なくてよい（通せることの実績が目的。シードの近くの線量を統計的に示す計算ではない）。

## 0. 前提事実（検証済み、2026-10-25）

1. 入口の許可リストは `entry/scene.py` の `validate`（`scoring` は `'pdd_ocr'` だけ、`phantom.material` は `water`、`thickness_cm` は 30）。`chatcarlo/__main__.py:487` が `domain == 'therapy_mv'` を `entry/cli.py` に回す。
2. 入口の承認は `entry/preview.py`: 条件ハッシュ・資材ハッシュを `condition.digest` で作り、`preview.html` の**見える文字**にハッシュを出し、`approval.json` に `condition_hash`・`materials_hash`・`snapshot_sha256`・`html_sha256` を書く。`run` は束を作る前に `require_approval` で照合する。
3. 入口の 3D 表示は `chatcarlo.preview` の `_TEMPLATE`・`_SHARED_CANVAS_JS` を使い、`scene_to_json` が受ける図形は `box`・`cylinder`（と球の描画）で、**楕円柱・楕円体は図形としては受けない**。描画器の最終入力は頂点・辺・面（§3 参照）。この食い違い（球がある）は Codex 1 巡目で指摘済みで、§0-3 を直した。停止条件にしない。
4. P9c の駆動: `gpu/p9c_cli.py run --session S --export DIR --output DIR --job '--mode production --histories-per-field N --t0 DIR --beams prostate --seed S' --t0 DIR`。Colab 側 `gpu/p9c_colab.py` は production のとき `--histories-per-field`（100 の倍数）と `--t0` が必須。関門 V1c・V2b は production で毎回計算する。
5. 既存の資産（`~/.cache/chatcarlo/`、git の外）: 束 `p9c2/export/`（`manifest.json`・`input.json`・`material_0..4.npz`・`cpu_reference.npz`・`source.zip`）、T0 `p9c/results_t0/job.json`（`status: partial-accepted`、`sources` を持つ）。`p9c_colab.require_t0` は T0 の `mode == 't0'` と `sources` が束の `input.json` の `sources` と一致することを見る。
6. 速さ（RESULTS.md）: 前立腺形 36〜38 µs/履歴。各側の光子数 = 7 × `histories_per_field`。空きディスク 11 GB（2026-10-25、95%）。
7. P9c の図と数値の関数は `prototypes/mv_electron/p9c_figures/`（`combined.py` が環境変数 `P9C_RUNS`・`P9C_OUT` でチェックポイントを足し合わせ、`make_radial.py`・`make_top3.py` が表と図を作る）。

**実装前に Codex が自分で確かめること**: 上の 3・4・5・7 を実ファイルで読み、計画と食い違えば食い違いを報告して止まる。

## 1. 入口の新しい条件（scene.yaml）

```yaml
domain: therapy_mv
scoring: seed_ratio
source: {spectrum: mohan10, sad_cm: 100, gantry_deg: [0, 45, 90, 135, 225, 270, 315], field: prostate_projection}
body: {material: water, ellipse_cm: [36, 24], length_cm: 30}
target: {prostate_cm: [4.5, 3.0, 3.5], rectum_diameter_cm: 3}
seeds: {preset: p9a_58, material: ag_ti_mixture}
run: {histories_per_field: 200000, seed: 941000, engine: colab_t4}
```

- 許可リストは**この値だけ**（`sad_cm` は 100。線源は回転軸（原点）から 100 cm で、体表までは 0° で 88 cm・90° で 82 cm なので SSD ではない。画面にも SAD と書く。`gantry_deg` はこの 7 角、`field` は `prostate_projection` のみ、`engine` は `colab_t4` のみ、`histories_per_field` は 100 の倍数で 100 以上 7,142,900 以下、`seed` は整数で 0 以上 2⁶¹/8 未満）。許可リスト外・未知のキー・型違いは、理由つきで止める（黙って近似しない）。既存の `pdd_ocr` の検査の文面と挙動は変えない。
- 振り分け: `entry/scene.py` の `validate` が `scoring` を見て、`seed_ratio` なら新しい検査（新ファイル `entry/seed_scene.py`）を呼ぶ。`pdd_ocr` の経路は**1 文字も変えない**（既存テストが無傷であること）。
- 速さの目安（前立腺形 36〜38 µs/履歴）は RESULTS の別 VM の測定で、所要時間の約束には使わない。`seed` の上限は P9c の実装（2⁶¹ 未満）より厳しい入口の制限（安全側）。
- 実行は `run.engine == colab_t4` のみ（手元 CPU の本計算は P9c にない）。

## 2. 条件ハッシュ（承認 = 実計算の保証）

新ファイル `entry/seed_condition.py`。条件 `cond` に入れるもの（`condition.digest` と同じ正規化で SHA256）:
- scene の全項目（上の値）、
- 束 `export/manifest.json` の全ファイルのハッシュ、`input.json` の `sources`（核のソースのハッシュ）、現在の `p9c_core.fingerprint()`（コードが束と同じであることの照合。違えば止める）、
- T0 の 3 ファイル（`job.json`・`gates.json`・`rectangle_summary.json`。`p9c_cli` が実際に送るもの）の SHA256 と `job.json` の `sources`・`status`・partial-accepted の理由（無ければ止める）、実行直前にも再照合する、
- Colab に渡す job 引数の完全な文字列（`--mode production --histories-per-field N --t0 /content/p9c_t0 --beams prostate --seed S`）。

束の場所は固定の既定 `~/.cache/chatcarlo/p9c2/export`、T0 は `~/.cache/chatcarlo/p9c/results_t0`。環境変数 `CHATCARLO_P9C_EXPORT`・`CHATCARLO_P9C_T0` で差し替え可（テスト用）。束や T0 が無い・ハッシュが合わない・`p9c_colab.require_t0` 相当の条件を満たさない場合は、理由つきで止める（束を作り直す手順は案内だけで、入口では作らない）。

## 3. 3D 表示と承認

新ファイル `entry/seed_preview.py`（`entry/preview.py` の `write`/`approve`/`require_approval` と同じ形で、`condition_hash`・`snapshot_sha256`・`html_sha256` を照合）。

- 表示する図形（**計算と同じ数値から作る**。軸順は計算の座標（x=左右、y=頭尾、z=前→後）で、scene の `prostate_cm` は [左右, 前後, 頭尾] を計算の全径 [4.5, 3.5(頭尾 y), 3.0(前後 z)] に対応づける変換を 1 か所に書いてテストする。直腸は中心 z=3 cm、頭尾長 3.5 cm の評価領域。シード 58 本は規則を再実装せず `p9c_core.seeds()` を直接呼ぶ。前立腺・直腸は水中の**評価領域**であると画面に書く）: 体の水の楕円柱、前立腺の楕円体、直腸の円柱、シード 58 本（銀・チタンの混合物、半径 0.05・半長 0.25 cm の頭尾方向の円柱）、7 門の線源の位置（`S(θ) = (100 sin θ, 0, −100 cos θ)`）と中心線。
- 描画器の最終入力は頂点・辺・面で、`chatcarlo/preview.py` には球もある。楕円柱・楕円体は、**描画 JSON に辺・面を直接作る（体の外形は楕円の多角形 64 辺以上、前立腺は球のワイヤーを軸別に拡大）方法を先に試す**。`chatcarlo/` を変えずに実現できなければ、軸に沿った薄い箱の積み重ね（24 枚以上）で近似し、近似であることを画面に書く。どちらでも計算の入力は近似しない。
- 画面の見える文字に、条件ハッシュ・束のハッシュ・T0 のハッシュ・光子数（各門・各側の合計）・種・シードの本数・材料（密度 2.47、銀 59.6%・チタン 40.4%）・調整で外した項目（強度変調・MLC・半影・人体形状・シード内部構造・EGS5 との照合は入口の外）を出す。
- `approve` は `preview_snapshot.json` を作り直した値と照合し、見える文字にハッシュがなければ止める（既存と同じ）。

## 4. 実行（`run`）

新ファイル `entry/seed_run.py`。承認の照合（`require_approval`）を**束・出力フォルダを作る前**に行う。通ったら:
1. 出力フォルダ `~/.cache/chatcarlo/mv_entry/seed_<条件ハッシュ先頭 12 桁>/`（`--out` で変更可）を作る。
2. `gpu/p9c_cli.py` の `run` を、条件ハッシュから作った job 引数で呼ぶ（`--session` は `p9c` + 条件ハッシュ先頭 8 桁（別条件の VM 名が衝突しない）、`--export` と `--t0` は条件の束と T0）。出力フォルダには条件ハッシュのファイルを置き、既存の出力が別条件なら止める。実行ロックを置き、二重実行は拒否する（同条件の部分再開は許す）。承認用の置き場（HTML など）とGPU結果のフォルダは分ける。`p9c_cli.py` 自体は**変更しない**。見張り（10 分ごとの確認・停滞の判定・再開）はそのまま使う。
3. 完了後、回収した出力が条件と一致するか検査する: `job.json` の `status == complete`・`mode == production`・`beams == ['prostate']`・`sources`。チェックポイント `prostate.npz` の identity（`n`・`seed`・`angles`・`grids`・`beam`・`sources`・`bundles`）の**全項目を条件から作り直した値と照合**し、`completed == list(range(100))`（欠番・重複を拒否）、summary の `checkpoint_sha256` が実ファイルと一致することを確かめる。関門は `p9c_colab.validate_gates(..., required=True)` と束の検査を再利用し、V2b の `insufficient-statistics` + `fallback: V2a` は正当な代替として受け付ける（V2a・V6・V6b は束の `input.local`、GPU の関門は `job.gates`、V3 の受け入れ（`V3_CUDA_acceptance`）は summary から読む。必須キーと許容状態は列挙して書く）。これらはメタデータの照合であり、既存入口のような実際の GPU 引数の独立観測ではない。`run_record.json` にそう書く。
4. 結果の数値と図は、**今回の出力フォルダのチェックポイント 1 つだけ**から作る（合算処理は使わない）。既存の `p9c_figures/`（`combined.py` は `~/.cache/chatcarlo/p9c2/` を前置し、`make_top3.py` は過去の `prod_prostate/prostate_summary.json` を固定で読むため、そのままでは使えない）は**変更せず**、同じ計算式を入口の側（`entry/seed_analysis.py`、明示的な入力パスを取る関数）に写す。写した式が既存の図スクリプトと同じ数値を出すことを、既存の `prod_prostate`（あれば）で一度突き合わせる（無ければ飛ばし、そう記録する）。出すもの: 前立腺全体・直腸の平均の比、シードの軸からの距離区間（0.6/0.9/1.4/2 mm の丸めた区間の中心と「3 mm 以上」を別の集計として）の比と SEM、各値に実際の区間・セル数を持たせる。分母 0・有効セルなし・全ゼロでも完了し、未定義の値は `null` と理由を保存する。距離別の SEM は「セルを独立とみなした値」で小さめの可能性、少ない光子数では 3% の増加は分解できない、近傍の EGS5 照合なし、P9b は不合格のまま、を `run_record.json` の `notes` と図に残す。資料用の図は 1 点: 横断面のあり・なし・比（`make_top3` 相当）。
5. `run_record.json`: 条件ハッシュ、承認の時刻、光子数、所要時間（束の検査・Colab・解析）、関門の結果、持ち越し項目、結果の数値。

コマンドは既存の `chatcarlo run scene.yaml` の形（`entry/cli.py` の振り分けを `scoring` で分ける）。`validate` の成功メッセージは新しい条件の文面にする。

## 5. 受入条件（測定可能な形）

- [ ] `pytest prototypes/mv_electron/tests -q` が、既存の `test_entry_*.py` を含めて全部通る（既存テストの変更なし）。
- [ ] 新しい `prototypes/mv_electron/tests/test_entry_seed.py`（偽の Colab クライアント・偽の束で CPU だけで動く）が次を含む: (a) 許可リスト内の scene は `validate` を通り、許可リスト外（角度の欠け、`field` が `rectangle_7x7`、`histories_per_field` が 99・100 の倍数でない、`engine: cpu_local`、未知のキー、型違い）は理由つきで止まる。(b) 承認なしの `run` は束も出力フォルダも作らずに止まる。(c) 承認後に scene の種・光子数・T0 のどれかを変えると、同じ `--out` を指定したときは `approval condition_hash: mismatch` で止まる（既定の出力先では出力先が変わるため「承認なし」で止まる。この 2 通りを分けて試す）。承認後の `preview.html`・`preview_snapshot.json` の改変、見えるハッシュの削除も止まる。(d) 束のファイルを 1 バイト変える、T0 の 3 ファイルのどれかを変える・欠く、partial-accepted の理由を欠く、`sources` を変えると止まる。(e) `preview.html` の見える文字に条件ハッシュ・束のハッシュ・T0 のハッシュ・58 本が出る。(f) プレビューの図形が計算の入力と一致する: 軸順、体・前立腺の寸法、直腸の中心 z=3 と頭尾長、シード 58 本の中心・半径 0.05・半長 0.25・頭尾方向、7 門の線源位置（原点から 100 cm）と、中心線が体表に当たるまでの距離（0° で 88 cm・90° で 82 cm）。(g) 偽クライアントで完走し、`run_record.json` が書かれる。(h) `job.json` の `mode`・`beams`・バッチ数・`status`、チェックポイントの identity（種・光子数・格子・角度・資材）、欠番・重複したバッチ、古い summary、必須の関門の欠落・fail が違えば止まり、V2b の正当な代替は通る。(i) 解析は旧キャッシュなし・任意の `--out` で図と JSON まで完成する。分母 0・全ゼロ・有効セルなしでも完了し `null` と理由が入る。(j) 二重実行は拒否され、別条件の既存出力は拒否され、同条件の部分再開は通る。(k) 型・境界: 光子数の上限超過・99・浮動小数・bool、種の両境界、null、mapping でない root、角度の重複・順序、各階層の未知キー。
- [ ] 変更範囲は未追跡を含むファイル一覧（`git status --short`）と `git diff` で確かめ、`pdd_ocr` の経路のファイルに差分が無い（`scene.py`・`cli.py` は振り分けの数行のみ）。変更が `prototypes/mv_electron/entry/`（新ファイル `seed_scene.py`・`seed_condition.py`・`seed_preview.py`・`seed_run.py` と、`scene.py`・`cli.py` の振り分け数行）・`prototypes/mv_electron/tests/test_entry_seed.py`・`examples/therapy/`（scene.yaml の例）・`docs/validation/mv/entry_seed/`（空の RESULTS の雛形は作らない。Claude が書く）に限られる。
- [ ] `chatcarlo validate examples/therapy/prostate_seed_7field.yaml` が OK を返す。`preview` が `preview.html` を作り、ブラウザで開ける（Claude が画面で確認する）。

## 5b. 計画全体の完了条件（Codex の実装完了とは別）

Codex の実装完了 = §5 の充足。計画全体の完了 = さらに、Claude による**使用者の 3D 承認、実 Colab の完走、回収物の照合、数値と図の提示、`docs/validation/mv/entry_seed/RESULTS.md` への記録**。前者だけで計画を `implemented` にしない。

## 6. テストコマンド

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m chatcarlo validate examples/therapy/prostate_seed_7field.yaml
```

## 7. 実装方針（理由）

- **P9c の駆動と核は変えず、入口は薄い皮にする**: 物理と見張りは検証済み。入口が足すのは「許可リスト・承認・来歴の照合」だけで、ここが「承認した条件 = 実際に計算した条件」を守る。
- **束と T0 は既存を使い、入口では作らない**: 束の作成は CPU で数十分かかり（`p9c_io.export`）、今回の目的（入口の通し）と無関係。ハッシュで同一性を担保する。
- **`pdd_ocr` の経路を触らない**: 既に合格した入口を壊さない。振り分けは `scoring` の値だけ。
- **持ち越しは隠さず列挙する**: Colab の出力に光子数・種が記録されていない項目は「観測していない」と書く（既存の入口の流儀、`condition.CARRIED`）。

## 8. 書かなかったこと（スコープ外）

- 矩形 7×7 の照射野、他の角度・本数・材料・体の大きさ、手元 CPU での本計算、EGS5 との比較、シードの近くの線量の検証。
- 束・T0 の作成を入口から行うこと。`p9c_cli.py`・`p9c_colab.py`・`p9c_*` の核の変更。
- 診断領域（`chatcarlo/`）の変更。`domain: therapy_mv` の振り分け（`__main__.py`）の変更。
- Colab の実行（Claude が行う。Codex は実行しない）。

## 9. 停止条件

- §0 の前提が実ファイルと食い違う、束・T0 のハッシュが現状のコードの `fingerprint()` と合わない、既存の `test_entry_*.py` が通らなくなる、のどれかなら、実装を止めて報告する。

## Codex 1 巡目の反映

| # | 分類 | 指摘 | 対応 |
|---|---|---|---|
| 1 | 実質 | SSD ではなく SAD 100 cm | `sad_cm` に変更、画面にも SAD、7 門の距離のテスト（§1・§3・§5(f)） |
| 2 | 実質 | 既存の図スクリプトは固定パスで使えない | 変更せず式を入口側へ写す。旧キャッシュなしで完成するテスト（§4-4・§5(i)） |
| 3 | 実質 | 光子数・種はチェックポイントに残る | identity 全項目・`completed`・`checkpoint_sha256` を照合（§4-3・§5(h)） |
| 4 | 実質 | 関門の合格の定義 | `validate_gates` を再利用、V2b の代替を許容、必須キーを列挙（§4-3） |
| 5 | 実質 | T0 の 3 ファイル | 3 つともハッシュに含め、直前に再照合（§2・§5(d)） |
| 6 | 実質 | 軸順・直腸・シードの取り違え | 変換を 1 か所に、`p9c_core.seeds()` を直接呼ぶ、テスト（§3・§5(f)） |
| 7 | 実質 | VM 名・出力の衝突 | 条件ハッシュ由来の接頭辞・ロック・別条件の出力を拒否（§4-2・§5(j)） |
| 8 | 実質 | 低光子数の未定義値 | `null` と理由、区間とセル数を持たせる（§4-4・§5(i)） |
| 9 | 実質 | 実 Colab が受入にない | §5b を追加 |
| 10〜11 | 軽微 | 承認後変更テストの分け方・型境界・変更範囲 | §5 に反映 |
| 12 | 参考 | 箱の積み重ねは不要に複雑 | 直接メッシュを先に試す（§3） |
| 13〜14 | 参考 | 統計の近似の明記・速度とseed上限 | notes と §7 に反映 |
