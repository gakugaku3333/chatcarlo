# 計画: G5 A3 — シーンの検査・承認・受領・入口を新エンジンへつなぐ（骨子）

- 状態: approved（2026-10-31 ユーザー承認。A3 の速さも有効 1 回）
- 作成者: claude
- 実装担当: codex（Colab・`~/.cache` を使う検査・承認の表示の送付は Claude）
- 日付: 2026-10-31
- Codexレビュー: 2巡（骨子: 致命 0・実質 5・軽微 1・参考 1。詳細: 致命 0・実質 8・軽微 3・参考 2。原文 `data/logs/codex_g5a3_review{1,2}.md`。すべて反映。§2e が他の節に優先する）
- 親計画: [2026-10-29-chatcarlo-gpu-g5-unified-energy.md](2026-10-29-chatcarlo-gpu-g5-unified-energy.md) の段 A3。D5・D8〜D13 は親計画で決定済みで、ここでは蒸し返さない
- 規模: 重（入口の取り違えは、静かに違う線量を出す）

## 目的（この計画が答える問い）

`domain: therapy_mv` の水の照射野（10 MV・4 MV）を、凍結した試作の核ではなく `chatcarlo/gpu/` の新エンジンで計算し、
承認した条件が核まで届くことと、EGS5 の判定が同じ基準で再現されることを示せるか。
同時に、検証していない組とエネルギーを `validate` が理由つきで止められるか。

`scoring: seed_ratio` は A2b まで試作の経路のまま残す（親 D8）。

## 0. 前提事実（2026-10-31 に実物で確認）

### 入口の今の形

- 分岐は `chatcarlo/__main__.py` の `_mv_dispatch`。`raw.get('domain') == 'therapy_mv'` なら `prototypes.mv_electron.entry.cli.main`。
- `entry/cli.py:31-33` が `seed_ratio` を先に `seed_run` へ流す。つなぎ替え後も `seed_ratio` はこの経路を通す必要がある。
- `entry/scene.py`: キーは `domain, source{spectrum,field_cm,ssd_cm}, phantom{material,thickness_cm}, scoring, run{histories,seed,engine}` だけ。値は `mohan10|mohan4`、10×10、SSD 100、水 30 cm、`pdd_ocr`。`physics` キーはなく、未知のキーとして拒否される。histories は 100 の倍数、上限 colab_t4 2.4×10⁸、cpu_local 2×10⁵。
- `entry/condition.py:OPTIONS` と `observe_gpu` は P7 の核の引数 80 個の位置に固定されている（`field_half_cm` は位置 41）。**新しい核には流用できず**、観測は作り直しになる。
- `entry/vm_driver.py:gates` の 2a・2c・2d は P7 の出力（`regions`・`core.HISTORY_ORDER`）が前提。2a は CPU と GPU の線源・乱数回数の照合を含む。
- `entry/bundle.py:build` の `source.zip` は `prototypes/` だけを詰める。新エンジンでは `chatcarlo/gpu/` の実行に要るファイルを詰める必要がある。
- `entry/compare.py:analyse` の EGS5 入力はリポジトリの `docs/validation/mv/p3b/{egs5/main_egs/{result.npz,summary.json,config.json},pilot_lock_p3b3.json}`。実物あり。ただし `result.npz` は gitignore。
- わざと変える検査は入口の RESULTS の 5 つ（承認なし、承認後の種の変更、束の `condition.json` の書き換え、手元 CPU で核の引数の `field_half` を変更、Colab GPU で同じ変更）。

### 配置の違い（今回新しく確認。親計画に書かれていない）

```text
p6 sizes (120,300,120,72×6,144×6) bins 1836  extent 9 cm
p7 sizes (120,300,120,160×6,320×6) bins 3420 extent 20 cm
P7 条件 C: mohan4, field_half 5.0, ssd 100
```

- 入口の 10 MV は `p6`（EGS5 は `docs/validation/mv/p3b`）。A2 の新しい核は `p7` しか通していない（`a2_common.field_geometry` は `geometry('p7')`）。
- 親計画は 4 MV の相手を P7 の条件 C とする。条件 C の EGS5 は `p7` の配置で、今の入口の 4 MV は `p6`。**今の入口のままでは 11 量を同じ定義で比べられない。**

### 新エンジン側

- `therapy.CPU` の `field_geometry` は `FieldGeometry` の構造体を受ける。採点の容量は 3,420。`p6` の 1,836 bin が通るかは**未検査**。
- `a2_speed.NewCore` と `a2_gates.T4` は `docs/validation/gpu/g5/` の中にあり、`chatcarlo/gpu/` に GPU の起動役はない。`therapy.py` は CPU の双子とソースの展開だけ。
- `therapy.CPU.histories` は 1 回 20 万履歴まで。GPU の本番は A2 で 1 バッチ 138 万履歴まで通した（条件 B）。
- `a2_common.water_arrays` は `prototypes` から幾何と配列を作る。`therapy.py` の冒頭は「No prototype imports, tables or spectrum values in runtime」。
- 表の束は `therapy_bundle`（4 材料、`chatcarlo-g5-v1`）。スペクトルを含むのでリポジトリの中には作れない（`create` が拒否する）。
- `physics_sets.require_quantity` が治療の組の H*(10) を拒否する。組の識別子は `therapy.CPU` が native から観測する。
- 診断の承認は `chatcarlo/gpu/approval.py`（`generation.json`）。条件は `condition.colab_condition`（形式 `chatcarlo-g4-v1`）で、**組の名前は条件に直接入っていない**。組は `bundle` の settings 経由で、`bundle_sha256` を通して間接に効く。
- 診断の `physics:` 欄は既にあり、許すキーは `fluorescence` だけ（`scene.py:287-300`）。

### 150 keV の検査

- `scene.py:83`（`spectrum`）、`:114`（`kvp` 20〜150）。`materials.py:96` の `_XS_GRID_MAX_KEV=150.0` は xraylib の表の範囲で、D10 で 150 keV 超を止める限り変える必要がない。

### 手元の資源

- 空き 4.5 GiB（停止閾値 3 GiB。余裕 1.5 GiB）。入口の 10 MV の生データは 2 回の実績で約 430 MB（`.tar.zst`）。`~/.cache/chatcarlo/mv_entry/9734038057e4/` は圧縮済みで手元にない（アーカイブは `~/.cache/chatcarlo/archive/`）。
- Mohan 4 MV のスペクトルは `~/.cache/chatcarlo/p7/spectra` 側。

## 1. 決めること

| # | 決めること | 案 | 理由 |
|---|---|---|---|
| **A3-D1** | 治療の入口をどこに置くか | `chatcarlo/gpu/therapy_entry.py`（必要なら数ファイル）に新規に書く。`prototypes/mv_electron/entry/` は `pdd_ocr` について呼ばれなくなるが、消さない（D12）。`_mv_dispatch` が `scoring` を見て `seed_ratio` だけ試作へ渡す | 親 D12 |
| **A3-D2** | 配置の扱い | **両方のスペクトルで `p7` を使う**（A2 で検証済み、`p6` の経路は作らない）。10 MV の解析では、横方向の粗格子 `[44:116]`・細格子 `[88:232]` を切り出し、先頭 3 領域と合わせて P3b-3 の定義へ写す。4 MV は P7 C の定義のまま。切り出しは保存済みの入口のバッチと P3b-3 の定義で検査する。表示と出力は実際の ±20 cm を示す | `p6` は bin 幅・軸・体積が `p7` と同じで範囲だけが違う（Codex 1 巡で実配列を確認）。核の経路が増えず、4 MV も同じ定義で比べられる |
| **A3-D3** | 表と入力の出どころ | 実行時は `therapy_bundle` の形式の封印済みの束だけを読む。作るのは `docs/validation/gpu/g5/` の書き出しスクリプトで、試作を import してよいのはそこだけ。束は `~/.cache/chatcarlo/g5/` に置き、SHA256 を `chatcarlo/gpu/` に固定する | runtime に試作の import を入れない。ELSEPA の表の配布は未決なので、配布の形は決めない |
| **A3-D4** | 承認の層 | シーン形式・条件の内容・3D 表示は領域別のまま。**生成記録の封印・承認の照合・受領書の条件／組／観測の照合を共通の 1 モジュールにする**。G4 の公開関数は薄い包みとして残す | 親の「1 つの承認・受領の層」を満たしつつ、シーン形式の統合は避ける |
| **A3-D5** | 診断の条件の版を上げる影響 | 既存の `gpu_runs` の承認は無効になる（条件が変わるので承認し直し）。互換の読み替えは作らない | 既存の承認の再利用より、組が必ず条件に入ることを優先する |
| **A3-D6** | VM 上の関門 | 2a 相当: 識別子・健全性・CPU と GPU の線源と乱数回数の照合。**今の GPU は線源を出力しない**（`a2_gates.py:63` はゼロ埋め）ので、検査ビルドだけで線源の生成直後の状態を出力する。本番ビルドでは無効にする。2c 相当: 履歴ごとと合計の一致。2d 相当: 固定の付与の採点。線量のバイト一致は合否にしない | 親の方針。核のソースが変わるので、H6 と A2 の T4 検査を最終のコードでやり直す |
| **A3-D7** | `physics:` 欄の形 | `physics: {set: diagnostic-photon-kerma-v1 | therapy-electron-v1}` を両方の領域で任意のキーとして足す。省くと線源の最大エネルギーで選ぶ。`fluorescence` は診断の組でだけ許す | D9 |
| **A3-D8** | D10 の判定 | 最大エネルギーが 150 keV 以下 → 診断、許可リストの MV スペクトル → 治療、それ以外 → 止める（文言に「検証していない区間」と許される範囲を書く）。`kvp` 151 以上・`spectrum` の 150 keV 超も同じ文言にする。診断の組で MV、治療の組で低エネルギーは止める | 親 D10 |
| **A3-D9** | 量の名前 | 最終成果物の NPZ と受領書に `physics_set`、配列ごとの量・単位・履歴の正規化を書く。診断はカーマ／H*(10)、治療は吸収線量。生バッチの付与エネルギー（MeV）は別の量として記述する。読み戻して検査する。治療の組の H*(10) の指定は `validate` で止める | 親 D11。生の MeV と換算済みの線量を混同しない |
| **A3-D10** | 入口の速さの測り方 | **親の条件のまま測る。** 検証スクリプトが T4 を 1 台持ち、旧入口には注入できる `client`（`colab_cli.run` の引数）でその VM を渡す。凍結ファイルは変えない。旧・新を同じ `histories`（460 万）で順に、それぞれ専用のコンパイルキャッシュで初回、次に同じキャッシュで全履歴を再計算する（バッチの再利用なし）。承認から結果まで（送付・待機・回収・照合・解析を含む）の全体と内訳を保存し、全体の比を上限 1.0571 で判定する。確認の間隔は短くして比に入れる。NVRTC のキャッシュが効かないと分かったらキャッシュ済みの値はその旨を書いて報告だけにする | 同じセッションの比にできるので、親の 2 つの条件は両立する。測定は有効 1 回（A2 のユーザー決定に揃える。A3 への適用も 2026-10-31 にユーザーが承認） |

## 2. 関門（統計の seed は計算の前に固定）

| 関門 | 内容 | 合格 |
|---|---|---|
| H1 手元 | シーンの検査、組の拒否、`p7`→`p6` の切り出し、条件の往復、偽の VM で全モード | テスト |
| H2 わざと変える | 入口の 5 つ＋組だけを変えたシーン＋束の組の書き換え | すべて検出して止まる |
| H3 T4 10 MV | 4,600 万、100 バッチ、種 20261101。再試行の種 20261111 | P3b-3 の 11 量が入口の RESULTS と同じ基準（線量 1%・SEM 1/3%、位置 0.1 cm・SEM 0.1/3 cm）で全合格 |
| H4 T4 4 MV | 2,600 万、100 バッチ、種 20261102。再試行の種 20261112 | P7 の条件 C の EGS5 に対して P7 と同じ基準の 11 量で全合格 |
| H5 診断 | G4 の入口の例を通す（Colab）。G1 の 51 量・G2 の照合を最終のコードでやり直す | 既存の基準 |
| H6 速さ | 最終のコードで診断 `g2b`・`g4`、治療 `p7A/B/C`・積層を同じセッションで旧と比べる。入口は A3-D10 | 各比が上限 1.0571 以下 |

偶然の不合格は親計画の扱い（1 項目だけ落ちたら上の種で 1 回だけやり直す）。H3・H4 の統計は保存済みの実績（同じ履歴数で SEM 0.15〜0.26%）から届く。

**ユーザーの関門**: H3・H4 の `run` の前に 3D 表示の承認が要る（入口の決まり）。

## 2b. 受入条件

- [ ] `__main__` の分岐: `pdd_ocr` は新エンジン、`seed_ratio` は試作へ。`examples/therapy/` の 2 例の `validate` が通る
- [ ] 実行時のコード（`chatcarlo/gpu/`）が `prototypes` を import しない（テストで検査）。書き出しスクリプトだけが例外
- [ ] D10: 151 kV、`spectrum` 151 keV・1 MeV、許可リスト外の MV スペクトル、診断の組＋MV、治療の組＋低エネルギー、未知の組の 6 シーンを `validate` が理由つきで止める。150 kV・150 keV は通る
- [ ] H2 の 7 つの変更をすべて検出する（検査ビルドと T4 の両方を含む）
- [ ] 受領書の組が、native で観測した識別子と一致する。NPZ と受領書を読み戻して量・単位・組を確認する
- [ ] 切り出しの検査: 保存済みの入口の 10 MV の判定値を、`p7` の EGS5 ではなく定義の写像の単体の固定値で再現する。入力に 10 MV の EGS5 の配列をそのまま使う自己比較で差 0
- [ ] H3〜H6 の合格。判定スクリプトは解析の前に EGS5 の入力のハッシュ・N・seed・組を照合する
- [ ] 偽の VM で全モードを端から端まで通す（PLAN_TEMPLATE の運用メモ）
- [ ] `CLAUDE.md`・`mv.md`・`chatcarlo.md` の方針の文と vive-interview の対応表（4 MV の範囲 ±20 cm を含む）を実装に合わせる
- [ ] `git diff` で変更禁止の範囲が空

## 2e. 詳細レビューの反映（他の節に優先）

1. **切り出しの検査は 2 本。** (a) 写像: バッチ・領域・bin を識別できる人工の `p7` 配列（切り出す範囲の直前・直後にも別の値）を、独立に決めた期待値と全要素で比べる。(b) 回帰: 保存済みの入口の表を `p7` の中央へ埋め込み、実際の切り出し関数を通して元の表と判定値を再現する。§2b の「自己比較で差 0」の行はこれに置き換える。
2. **A3-D10 の測定。** 検証スクリプトが旧 CLI の `main` 全体を呼び、メモリー上の包みで下位の `colab_cli.run` に client を渡す（凍結ファイルは変えない）。VM の作成と停止はスクリプトが持つ。計時は `main` の呼び出しの前後。確認の間隔は旧新とも 60 秒。測定ごとに手元と VM の結果の保存先を新しくし、コンパイルキャッシュだけを引き継ぐ。
3. **キャッシュ。** 置き場を固定し、2 回目のコンパイル時間とキャッシュのファイルで利用を確認する。確認できなければ「未判定」とし、A3 を完了にしない（報告だけに落とすなら親の条件の変更なのでユーザーの承認を取る）。
4. **2c 相当。** 同じ 2,048 履歴を 1 回と、256 履歴ずつ 8 回に分けた和で、`field`・`ledger`・深さの採点・`field_additions` を比べる。数え上げは完全一致、浮動小数は |差| ≤ 加算回数 × 2⁻⁵³ × 絶対値和。核の追加の出力は要らない。上限が厳しすぎないかは計画の決まりどおり試しに回して実物を見てから固定する。
5. **対象範囲に追加:** `tests/test_gpu_g4.py`・`tests/test_gpu_g2b.py` の期待値の更新。4 MV の入力は新規テストの中に作る（examples は増やさない）。
6. **H2 に追加:** 承認と束を保ったまま native の `id/tag` だけを変える、別条件の受領書の混入、バッチの欠落・重複・範囲ずれ、関門記録のない再開。偽の VM は正常完走に加えて、中断→回収→再開と、失敗時の停止・回収を通す。
7. **H*(10):** 量の選択の機能は足さない。治療のシーンに `quantity` など未知のキーを書けば `validate` が止め、`physics_sets.require_quantity` の API の拒否は別のテストで確かめる。`run --dose-out` のような H*(10) の CLI は治療の入口に存在しないことを `--help` の引数で確かめる。
8. **判定コードのレビュー:** 切り出し・単位換算・入力照合・速さの比を含む判定スクリプトは、合否を出す前に Codex がレビューする。
9. **試作の import:** 製品の実行経路には直接・間接の依存を入れない（`sys.modules` で実行時に検査する）。検証スクリプトは比較相手として使ってよい。
10. **D10 のテスト:** §2b の一覧（「6 シーン」は 7 ケース）に、60 keV と 151 keV の混合を順序を変えて止めること、省略と正しい明示の `set` の正常系、診断の `fluorescence` の true/false、治療の `fluorescence` の拒否を足す。

## 2c. テストコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q
```

## 2d. 実装の順

1. `T4` の起動役を `chatcarlo/gpu/` へ移し、検査ビルドの線源の出力を足す。2. 共通の承認・受領。3. `scene.py` の D10 と `physics:`。4. 治療の入口と切り出し。5. `__main__`。6. 文書。Codex の範囲は手元（H1・H2 の手元分）まで。T4 と判定は Claude。

## 3. 対象範囲

- 変更してよい: `chatcarlo/gpu/`、`chatcarlo/scene.py`、`chatcarlo/__main__.py`、新規 `tests/test_gpu_g5_a3_*.py`、既存 `tests/test_mv_dispatch.py`（分岐先の変更に合わせる範囲）、`tests/test_dose3d.py` の `test_t11_mv_dispatch`（2026-11-01 追加。分岐先の変更に合わせ入力を `seed_ratio` に）、`docs/validation/gpu/g5/`、`.claude/skills/vive-interview/SKILL.md`、`CLAUDE.md`、`.claude/rules/mv.md`、`.claude/rules/chatcarlo.md`、`docs/validation/mv/STATUS.md`（1 行）
- 変更禁止: `transport.py`・`kernel.py`、`prototypes/mv_electron/` の既存ファイルすべて、`materials.py`、MV と G1〜G4 の既存の RESULTS、`data/`、`docs/jsmp133/`、Mohan のスペクトルの値（リポジトリに写さない）
- `prototypes/mv_electron/tests/` の入口のテストは試作の入口を直接呼ぶので、そのまま通るはず（通らなければ止まる）

## 4. 停止条件

- 親計画 §7 のすべて。
- 空きが 3 GiB を割りそうなら止める。H3〜H5 の前に本番の生データの Drive への複製が要るかを見積もる。
- 線源の観測の追加で本番ビルドの速さが落ちたら、その変更を採らずに報告する。

## 5. 段階の終わりのデータの扱い

| 出力 | 分類 | 理由 |
|---|---|---|
| 判定・速さ・観測の JSON、外側 100 バッチの表 | リポジトリ | 小さい。RESULTS を再現できる |
| H3・H4 の生データ（各 約 430 MB の見込み） | Drive、手元から消す候補（消すのはユーザー） | 空きが少ない |
| 治療の束（スペクトルを含む） | 手元 | リポジトリに置けない |
| Colab の断片 | 消す | 作り直せる |

## 6. 書かなかったこと

- `seed_ratio` の移植（A2b）、5×5・20×20 など入口の条件を広げること、積層を入口に通すこと、150 keV〜1 MeV の許可（A4）、ELSEPA の表の配布方法、シーン形式と 3D 表示の統合、速さの追加の工夫（A5）。
