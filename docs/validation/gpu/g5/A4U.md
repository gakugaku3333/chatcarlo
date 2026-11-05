# G5 A4u — S1（構造だけ）

2026-11-03。対象は計画 §2b の S1 のみ。S2 は未実装・未判定。

## 実装

- GPU の登録名は `chatcarlo-photon-electron-v1` の 1 つ。
  `local-deposition`（ID 1）と `electron-transport`（ID 2）を特殊化とする。
  tag は名前・特殊化・S1 の物理記述から生成する。条件・受領書・native の観測で
  同じ宣言を照合する。旧 2 名の宣言・条件・承認は再生成が必要。
- `require_quantity(name, quantity, specialization=...)` は名前と特殊化の対で
  引き当てる。H*(10)・飛跡長カーマは local、吸収線量は electron だけ。
  既存の呼出しの既定値は維持（引数なしの declaration は local、名前指定だけは
  electron）。ランタイムの名前指定箇所では特殊化を明示する。
- 光子の相互作用は `chatcarlo/gpu/photon_physics.h` に移した。
  型・補助関数の定義順を保つため、診断／P5／P8f／P8g の節別ガードで読み込む。
  `t2_materials.h` は凍結参照として残し、そのガードでコンパイルを抑止する。
  そこで定義していた電子の補助関数は `t2_tables.h` にそのまま移した。
  `g1_physics.h` は未変更、旧 `t2_physics.h` の光子部分は `#if 0` の凍結参照。
  CPU/CUDA と slab/seed の両駆動が新しいヘッダーを使う。
- 表の数値・配列の形・材料の順序は維持。catalog は同じ登録名に特殊化を付記する。
  既存の外部 sealed bundle は旧宣言なので拒否する。元の入力配列を使って
  新しい bundle と preview/approval を生成する必要がある。
- 試験専用のコンパイル旗は `G5_TEST_LEGACY_LOCAL` と
  `G5_TEST_LEGACY_ELECTRON`。native driver の `_test_legacy_physics=True` で
  特殊化に応じた旗を立てる。シーン・束・通常の入口からは指定できない。
  **S1 は本番も旧物理なので、旗の有無で処理は変わらない。**
  旗あり／なしと移植前のソースとの配列一致を別々に検査する。

## 維持する物理と範囲

診断は K 殻のみ、二次スタックなし、既存の xraylib 表・束縛 Compton・Rayleigh。
治療は Rayleigh なし、束縛 Compton の棄却なし、解析 KN の全断面積、XCOM の光電、
既存の全殻緩和（CK の追加なし）、光子カットは 10 keV のまま。
乱数の消費順と計算式を保持する。

S2 の表の変更・5 チャンネル化・全殻＋CK の統一・二次光子の契約・5 keV 化・
入口の許可範囲拡張は行っていない。NumPy/kernel の本番物理は変更していない。
D14 の全殻化後の説明への変更も行わない。

## 検証

最終実行の結果を以下に記録する。既存の G1/A1〜A3/A2b/MV の記録は上書きしない。
T4 での G2/G4/MV/seed_ratio・D10(1) の速度比 1.0571 は手元では未測定。
S1 全体の GPU 関門合格や S2 への移行を、この手元の結果だけで宣言しない。

### 51 量の CPU の双子

`python -m docs.validation.gpu.g5.a4u_s1_gate1 --out /tmp/chatcarlo-a4u-s1-gate1`
で再実行。既存の G1 のシーン・seed・100 万履歴×10 回×3 シーン×2 エンジン・
判定器と再試行規則を使用した。

- **pass**、K=51、Σz²/K=1.1316442308744354。
- 最大 |z|=2.1058439707858465（`s2:category:1:roi0`）。
- 両側ゼロで除外した量は 10、再試行なし。
- 判定と 61 量の値は [a4u_s1_gate1.json](a4u_s1_gate1.json)。
- 生の個別計算 JSON は `/tmp/chatcarlo-a4u-s1-gate1`（再生成可能な一時出力）。
  小さい判定 JSON と実行スクリプトはリポジトリに保持する。

S1 の小規模の回帰は、固定した移植前コミット
`02bab32af9083897cd255666d91ea7979c6f99af` の native コードと、
移植後・試験旗ありの 3 者を比較する。診断 2048 履歴（履歴ごとの記録を含む）、
治療 32 履歴・3 スロットで全出力配列が厳密一致した。
`seed_ratio` の既存の部品・健全性・収支の試験も別途実行する。

既存の `examples/therapy/prostate_seed_7field_g5_a2b.yaml` は
`physics.set` の名前だけを更新した。線源・幾何・履歴数はそのまま。

### 計画 §5 の全コマンド（実装固定後）

すべて終了コード 0。編集途中の検査結果ではなく、実装固定後の実行結果である。

| コマンド | 結果 | 時間 | 実行ログ |
|---|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **973 passed**、2 warnings | 301.88 秒 | [tests](a4u_s1_tests.log) |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | **1808 passed、17 skipped** | 1251.27 秒 | [prototypes](a4u_s1_prototypes.log) |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_*.py tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | **202 passed** | 84.80 秒 | [G5 指定群](a4u_s1_focused.log) |

`tests/` の 2 warnings は、ゼロ放出条件と SpekPy 不在を意図的に試すテストの警告。
追加の実入口確認:
`.venv/bin/python -m chatcarlo validate examples/therapy/prostate_seed_7field_g5_a2b.yaml`
は終了コード 0、`OK: A2b seed_ratio / 50 batches`。

### 差分とデータの確認

- 変更禁止の CPU 本番 5 ファイル・`tests/test_fluorescence.py`・既存 prototype ファイル・
  `docs/jsmp133/` の tracked diff は空。`scene.py` の 150 keV の検査は未変更。
- 既存のテストが更新した `a2_run/bundle.json`・`a2_run/faults_cpu.json`・
  `mv/p8g1/faults_gate0.json`・`mv/p8g3/fault_observations.json` は実行前の内容へ復元した。
  既存の検証記録の tracked diff は空。
- `git diff --check` は合格。開始時からあったユーザーの未追跡ファイルは保持した。
- 判定 JSON・再実行スクリプト・この報告・最終試験ログはリポジトリに保持する。
  この作業で作った完了済み pytest の一時ディレクトリ 57/58/59 は、ログ保存後に削除。
  以前の作業の pytest 出力・外部の sealed bundle と計算データは変更していない。
- S1 の手元の進む条件（tests・prototypes・51 量）は合格。
  **T4 の G2/G4/MV/seed_ratio と D10(1) の速度回帰は未実施**なので、
  S1 全体の GPU 関門合格を宣言せず、S2 には進めていない。

## Claude の独立検証（S1、2026-11-03）

| 項目 | 結果 |
|---|---|
| 変更禁止の差分 | 空（`transport.py`・`kernel.py`・`physics.py`・`materials.py`・`tally.py`・`g1_physics.h`・`prototypes/`・`data/`・既存の記録） |
| 物理が動いていないこと | `photon_physics.h` 内で治療側は `p5_kn_total`（解析 KN）の 4 チャンネル、診断側は `RAYL` 表・`g1_rayleigh`・K 殻のみ・二次スタックなし。S2 の要素なし |
| 束に載ること | `core.SOURCES` に `photon_physics.h` が入っている（抜けると Colab のビルドだけが落ちる） |
| 登録簿 | 1 名＋特殊化 2 つ、tag に名前・特殊化・段。旧 2 名は理由つきで拒否。`require_quantity` は (名前, 特殊化) |
| 試験専用の旗 | `G5_TEST_LEGACY_LOCAL`／`G5_TEST_LEGACY_ELECTRON` が特殊化ごとにあり、シーン・束・入口から届かない |
| テスト再実行（Claude） | `tests/` **973 passed**（271.91 s）、G5 指定群 **202 passed**（75.41 s）。Codex の報告と一致 |

### 直した点（Claude、1 行）

凍結した `t2_materials.h` の抑止がインクルード順に依存していた（`t2_tables.h` が先に `P8G_MATERIALS_H` を定義することで無効化していたが、`t2_transport.h` に `#include "t2_materials.h"` が残っていた）。
`t2_transport.h` の該当行を `#include "t2_tables.h"` に替え、順序に依存しない形にした。G5 指定群を再実行して 202 passed（71.61 s）。

### 手元で未測定

T4 側（G2・G4・MV・`seed_ratio`）と D10 (1) の速さの回帰。計画どおり H6 の 1 セッションでまとめて測る。

## S2 の実装候補（2026-11-03、受入未達・停止）

ユーザー指定の S2 を追加した。**S2 の受入合格・利用範囲の拡張の完了は宣言しない。**
最初の全体試験の後、診断 Compton の候補抽出が駆動側に残っている未完了箇所を見つけ、
式と乱数順を保って共通ヘッダーへ移した。最終状態の試験結果を下に記録する。
S1 の結果・独立検証の記録は上記のまま保持する。T4 と EGS5 の計算は行っていない。

### 追加したもの

- `chatcarlo/gpu/generate_photon_tables.py`: host 側だけで生成する xraylib 表。
  `CS_Photo`・`CS_Compt`・`CS_Rayl` の絶対値と、正規化した
  `CS_Photo_Partial` の条件付き確率を別の列にする。全元素・量ごとの有効上端を
  実際の API 呼び出しで測定する。必要な下端・全断面積の 5〜150 keV が取れないと
  生成を止める。輸送表の例外をゼロにして継続しない。
- 有効上端を超える輸送表は、既存の SHA256 付き XCOM 応答から読み出した
  **元素ごとの**光電・Rayleigh と解析 KN。新しい XCOM の取得・計算はしていない。
  殻の確率は、その元素で収録した部分断面積に共通する有効上端で固定する。
  sealed bundle の worker は xraylib やローカルキャッシュを必要としない。
  既存材料を直接渡して host で生成する場合は、元の XCOM 応答が必要である。
- 共通の殻・線・CK の表、`p5_compton` 内の S(Z,q)/Z 棄却、表の非干渉性、
  Rayleigh の 5 チャンネル化。slab と seed の駆動を変更した。
- 殻は **K・L1〜L3・M1〜M5** を収録。xraylib の
  `EdgeEnergy`・`FluorYield`・`LineEnergy`・`RadRate`・`CosKronTransProb` を使用。
  5 keV 未満の結合エネルギーの空孔は、輸送できる光子を出せないので局所で終了する。
  残りの非放射遷移後の連鎖・M5 より外の空孔の追跡は省略し、Auger 電子と
  5 keV 未満の線のエネルギーはその場に沈着する。治療光子の打切りも 5 keV。
- 診断に有界の二次光子スタック。履歴の完了数 `completed` と生成光子数 `created`、
  吸収・脱出・検出の光子数 `endings` を分け、
  `completed == N` と `sum(endings) == created` を検査する。
  脱出した全光子を `escaped_spectrum`（1 keV bin）に加える。
  `record=True` の `terminals` は全終端光子を記録し、最後の 1 本で代用しない。
- 検出器は各光子の祖先で分類し、同じ履歴・画素に届く光子のエネルギーを
  合計してから二乗する。G2 の履歴単位の二乗和は全子孫処理後に確定する。
- 治療の入口に `source.energy_keV`（20〜10000 keV）と連続 bin
  `source.spectrum: [{low_keV, high_keV, weight}, ...]`（64 bin 以下）を追加。
  Mohan と 150 keV をまたぐ明示スペクトルも通る。
  単色は A4-D8 の狭い bin を使い、両端は許可範囲へ収める。
  診断の 150 keV 上限には手を入れていない。
- 特殊化ごとの旧物理の試験旗、CK 無効・L1→L3 の誤り・溢れ・取りこぼし・
  スロット再利用の祖先状態の誤りの CPU 故障注入。
  入口・シーン・束から試験旗を指定する機能は追加していない。

### 決定論と小試験

- 表の範囲・線・XCOM との差・継ぎ目は [a4u_s2_h1.json](a4u_s2_h1.json)。
  再生成は `python -m docs.validation.gpu.g5.a4u_h1 --out PATH`。
  比較値の記録であり、MV 線量への影響の関門合格を意味しない。
- 全断面積の実測上端は Compton/Rayleigh が約 800.026 keV、光電は元素により
  約 800.026 または 999.990 keV。部分断面積の共通上端は水素約 60 keV、
  C/N/O 約 100 keV、鉛約 300.006 keV だった。1 MeV 一律の上端にはしていない。
- Compton の継ぎ目の相対差は、水 −0.0718%、肺 −0.0662%、骨 −0.00616%、
  鉛 +1.131%（解析 KN 側との差）。この値だけで D9 の線量許容を満たすとは判断しない。
- A4u の試験群は **16 passed**（最終全体試験にも含まれる）。
  旧物理旗と移植前 native の厳密一致、殻の競合分岐・遷移先・エネルギー予算、
  CK の 2 故障、光子数・エネルギーの収支、全脱出スペクトル、スタックの 3 故障、
  1 履歴 2 光子の別画素／同画素の画像と二乗和、20/150 keV の一次透過、
  20/60 keV の電子駆動とスロット再利用、入口の許可・拒否を検査した。

### 最終の §5 コマンド

実装を固定した後の実行。終了コードとログは成功・失敗をそのまま残す。

| コマンド | 結果 | ログ |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **944 passed、29 failed、9 errors、2 warnings**（307.71 秒、終了コード 1） | [tests](a4u_s2_tests.log) |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | **1808 passed、17 skipped**（1269.24 秒、終了コード 0） | [prototypes](a4u_s2_prototypes.log) |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_*.py tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | **195 passed、10 failed、6 errors**（67.25 秒、終了コード 1） | [G5 指定群](a4u_s2_focused.log) |

### 未達・未検証

- 既存テストには、旧の per-history 配列の厳密値・固定ハッシュ・20 フィールドの診断表・
  33 フィールドの治療表・10 keV の下限を固定した検査があり、新仕様と衝突する。
  新しい `photon` を持たない生材料、`photon` にスカラーを入れる人工材料、
  新しい光子数の契約を満たさない集約入力でも不合格・エラーが出ている。
  **すべてが期待したモデル差だという原因の断定はしない。** 変更を許可されていない
  既存テストは書き換えていない。
- 診断の KN 候補抽出は最後の自己点検で `g5_local_compton` として
  共通ヘッダーへ移した。電子の運動学・診断の KN 提案法・乱数順は維持する。
- H2 の 51 量の測定、H3b の保存 EGS5 との照合、H4/H5、H6、H7 は未完了。
  [a4u_gates.py](a4u_gates.py) は事前の分岐 4 量のハッシュ・帯別の判定・
  K 帯の ON/OFF の定性的判定・D7 の登録・封印照合の枠で、計算を起動しない。
  未取得の値を合格にする処理はない。[未了一覧](a4u_s2_pending.json)。
  分岐する検出器の 4 量は、H3b で独立に検証済みとは呼ばない。
- 合格後の D14（CLAUDE・rules・聞き取り表・STATUS・親計画）は未更新。
  新しい検証済み範囲を宣言していない。外部 sealed bundle の封印し直し・
  preview/approval の再生成も、実利用向けには行っていない。

### データと差分

H1 の数値表・判定の枠・試験ログ・本節をリポジトリに保持する。
完了済み pytest が生成した一時材料束と出力は再生成できるため整理し、
[削除した一時出力の一覧](a4u_s2_cleanup.json) に残す。
Mohan の値はリポジトリへ追加していない。新しい EGS5・T4・長い CPU 関門は実行していない。

最終の差分確認: [a4u_s2_integrity.json](a4u_s2_integrity.json)。CPU 本番 5 ファイル・
NumPy 蛍光試験・凍結ヘッダー・既存 prototype・JSMP の tracked diff は空。
試験が更新した A2 の 2 記録と MV の 2 記録は**開始時のバイト列へ復元**した。
A2 の 2 ファイルには開始時からの差分があったため、HEAD へ戻す操作はしていない。
最終実装のソースハッシュは [a4u_s2_source_hashes.json](a4u_s2_source_hashes.json) と一致。
prototype 群は、変更禁止で差分のない対象に対する並行試験を完了したものである。
最初の未完了状態の試験ログは `a4u_s2_precompletion_*.log` と分離して残す。
**最終状態でも全体試験と G5 指定群は不合格。停止条件を適用し、この後の実装変更・
合格後の文書更新・実利用向けの封印や承認へは進めていない。**

## D15：旧契約のテスト更新（2026-11-03）

ユーザー承認の D15 に従い、旧契約を固定していた検査を更新した。
**C++ と Numba の per-history のバイト一致は緩めていない。**
`kernel.py`、比較関数、比較項目、6 条件各 10,000 履歴、seed、許容誤差を維持し、
C++ を `G5_TEST_LEGACY_LOCAL=1` でビルドするようにした。
本番の全殻＋CK は別の差の向きの試験で制約する。

### 変更ごとの記録

表のファイル名は `tests/` 配下。パラメータ付きの行は全パラメータを含む。
旧テスト全文・変更した関数名・変更前後の SHA256 は
[a4u_d15_test_changes.json](a4u_d15_test_changes.json) に保存した。
これにより旧い固定値も保持する。履歴の A1/A2/G4 等の記録は書き換えない。

| テスト／fixture | 何を固定していたか | 今は何を主張するか・再生成／確認コマンド |
|---|---|---|
| `test_gpu_g1_history.py::test_gpu_g1_0b_history`（6 条件） | 本番 C++ が K 殻のみの Numba と per-history でバイト一致 | **旧物理旗付き C++** と Numba が同じ全項目でバイト一致。旗が実際にコンパイルオプションに入っていることも検査。`pytest tests/test_gpu_g1_history.py -q`（以下すべて `.venv/bin/python -m` を前置） |
| `test_gpu_g1_faults_tally.py::test_gpu_g1_0f_faults` | 旧物理の 4 故障の部品比較 | 旧物理旗で同じ故障・同じ部品比較を維持。`pytest tests/test_gpu_g1_faults_tally.py -q` |
| 同 `test_gpu_g1_0c_detection_first` | Numba の旧 μ と RNG から作った厳密な同着位置で、検出が先 | 同じ旧物理旗で厳密な同着・優先順位を維持。境界条件でゼロにするのは密度欄だけ（旧 `offsets[19]:` は S2 の追加表まで壊すため `offsets[19]:offsets[20]` に限定）。同上 |
| `test_gpu_g2.py::test_null_tally_fixed_g1` | 旧モデル・旧 6 出力の固定ハッシュ | S2 の全出力（全終端光子、生成・完了数、全脱出スペクトルを含む）の固定ハッシュ。旧 6 値は固定値 JSON の `g2.old` に保存。再生成コマンド F（下記） |
| `test_gpu_g4.py::test_gpu_g4_frozen_engine`（2 条件） | 旧 GPU モデルの得点・不確かさ等の fingerprint | 本番 S2 の同じ条件・seed・N の fingerprint。旧値は `g4.old`、新値は `g4.new`。**Numba の `test_gpu_g4_frozen_kernel` は旧値のまま・未変更**。コマンド F |
| 同 `test_gpu_g4_aggregate_independent`（8 条件） | 入射履歴数＝終端光子数、分類別の光子数を持たない人工入力 | `completed=N`、`created=sum(endings)`、分類別の整数検出数と検出数の一致を明示。複数光子／履歴を含む人工入力。40 keV 光子・画素面積 1 cm² としてフルエンスとエネルギーの単位もそろえる。独立ループによる全得点・不等長バッチの二乗和・SEM・ROI の式はそのまま。`pytest tests/test_gpu_g4.py::test_gpu_g4_aggregate_independent -q` |
| 同 `test_gpu_g4_colab_output_copy` | 分類別の光子数を持たないゼロ画像の人工入力 | 完了 1・生成 1・吸収 1、分類別の検出数 0 を持つ入力。NPZ/JSON/PNG のコピーのバイト一致は維持。`pytest tests/test_gpu_g4.py::test_gpu_g4_colab_output_copy -q` |
| `test_gpu_g5_a1_catalog.py::test_original_pack_and_geometry`（4 シーン） | 旧診断の全 blob／20 offsets、G2 込み 26 欄 | **旧 20 欄の prefix は引き続き旧 pack とバイト一致**。追加の offset 21 個目、共通光子表の各材料の範囲・確率の検証、G2 込み 27 欄を検査。幾何のバイト一致・全 slice の SHA 検査は維持。`pytest tests/test_gpu_g5_a1_catalog.py::test_original_pack_and_geometry -q` |
| 同 `test_p8_small_scalar_3d_4d_roundtrip` | 治療の serializer が 33 欄 | 34 欄（`photon` 追加）。全欄の shape・dtype に対応するバイト／復元値の検査を維持。これは schema-only の試験で、人工値での輸送の妥当性を主張しない。`pytest tests/test_gpu_g5_a1_catalog.py::test_p8_small_scalar_3d_4d_roundtrip -q` |
| `test_gpu_g5_a1_drivers.py::test_p8_driver_small` | schema-only の小さい driver が 33 欄 | 同じ小さい roundtrip で 34 欄。`pytest tests/test_gpu_g5_a1_drivers.py::test_p8_driver_small -q` |
| `test_gpu_g5_a2_bundle.py::test_real_roundtrip` | 旧 33 欄・native blob 全体と全 descriptor の一致 | 新しい 34 欄のディスク roundtrip。元の 33 欄は native の各 slice と descriptor prefix まで旧 pack とバイト一致し、追加の光子表は生成値とバイト一致。元の private spectrum 入力・蛍光入力・幾何 ABI・constants・初期化した log 配列の検査もすべて保持。旧手順は凍結し、[a4u_d15_bundle.py](a4u_d15_bundle.py) に拡張版を置いた。`pytest tests/test_gpu_g5_a2_bundle.py::test_real_roundtrip -q` |
| `test_gpu_g5_a2_core.py::test_declaration` | 治療の輸送下限 10 keV | 輸送下限 5 keV（一次線源下限 20 keV とは別）。量の拒否の検査は維持。`pytest tests/test_gpu_g5_a2_core.py::test_declaration -q` |
| 同 `test_cases` | 7 keV を「打切り未満」とする固定 16 状態 | 該当する `low-photon` **入力だけ 4 keV** にし、全 16 状態・18 経路・9 遷移・故障数／未処理エネルギーの検査を維持。他の注入状態、seed、cap、maxsteps は変更しない。凍結した `fixed_cases` の戻り値のコピーをテスト中だけ変更し、元のファイルは未変更。`pytest tests/test_gpu_g5_a2_core.py::test_cases -q` |
| 同 `test_faults` | 正常・故障とも 7 keV の局所沈着を使う 10 故障の witness | 正常と故障をともに 4 keV へそろえる。旧手順の検出器、seed、10 故障と到達数の検査を維持し、許容幅は緩めない。[a4u_d15_faults.py](a4u_d15_faults.py) は凍結した旧手順から、この入力と記録先だけを変更したもの。`pytest tests/test_gpu_g5_a2_core.py::test_faults -q` |
| `test_gpu_g5_a3_native.py::artificial_material`／`native`（native 得点、観測、id/tag、table 変更、実引数変更の各検査） | 全欄をスカラーで埋める人工材料、7 keV は輸送しない | `photon` には生成した有効な He 表を渡し、得点／呼び出し観測の fixture は 4 keV へ移す。元の引数／field／log 等の人工値は維持し、この fixture で物理精度を主張しない。全検出器を維持。`pytest tests/test_gpu_g5_a3_native.py -q` |
| `test_gpu_g5_a2b.py::native`（split、実引数変更、table pointer 変更） | 7 keV の同じ人工材料 | 同じ有効な光子表と 4 keV。実際の seed 呼び出しの観測で下記 NameError が露見し、**3 件は未解決のまま**。`pytest tests/test_gpu_g5_a2b.py -q` |
| 同 `test_native_split_and_faults` が呼ぶ `seed_t4_checks.faults` | 7 keV を全量 1 画素へ局所沈着させる人工条件 | 注入状態、線源、正解の沈着量をすべて 4 keV にそろえた。正常の検査と 4 故障の検出条件を維持。共通の CPU fixture の入力更新であり、T4 の実行・測定はしていない。同上 |
| 同 `test_physics_helpers_and_notebook` | 変更済みの photon `history` まで旧ソースの関数ハッシュと一致 | 変更のない電子・対生成・RNG・弾性等の helper は旧ハッシュとの一致を維持。変わった関数が **`history` だけ**であること、更新した `history` と共通光子関数の固定ハッシュを検査。旧 `history` ハッシュも保存。notebook と zip の検査は維持。コマンド F と `pytest tests/test_gpu_g5_a2b.py::test_physics_helpers_and_notebook -q` |
| 新規 `test_gpu_g5_a4u_s2.py::test_d15_production_fluorescence_direction`（鉛・水） | 本番が旧物理と一致しないことだけでは制約がない | 事前固定した 100 keV・0.05 cm・8192 履歴・seed 9504015 で、鉛の蛍光数・脱出光子数がともに**増加**。水の H/O の輸送する蛍光は両方 0、本番 ON/OFF の得点・光子数・スペクトルが厳密一致。純酸素の殻注入も輸送光子 0・全予算が局所であることを検査。`pytest tests/test_gpu_g5_a4u_s2.py::test_d15_production_fluorescence_direction -q` |
| 新規 同 `test_d15_therapy_photon_cutoff` | 7 keV も打ち切る古い契約 | 薄い水板へ 4 keV／7 keV を各 1 履歴で注入。4 keV は全量局所、7 keV は脱出でき、両方のエネルギー収支が閉じる。`pytest tests/test_gpu_g5_a4u_s2.py::test_d15_therapy_photon_cutoff -q` |

**固定値を作り直すコマンド F**（短い既存条件の再生だけ。長い CPU 関門ではない）:

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_d15_fixtures --regenerate
```

出力は [a4u_d15_fixed.json](a4u_d15_fixed.json)。旧／新の G2 ハッシュ、G4 の
fingerprint、元と変更後の関数ハッシュ、schema の旧／新の欄数・下限、向きの witness
を含む。pytest の中で正解値を生成することはせず、保存した値に照合する。
旧 G4 の `local_baseline.json` と旧 A2 の手順・記録は保存したままである。

### 本番を制約する向き（短い CPU witness）

N=8192、100 keV、厚さ 0.05 cm、seed 9504015。乱数・N・形状は測定前に固定した。

| 材料 | 量 | 旧物理旗 | 本番 S2 | 事前の主張 |
|---|---|---:|---:|---|
| 鉛 | 蛍光数 | 5877 | 9434 | 増える |
| 鉛 | 脱出光子数 | 2443 | 2541 | 増える |
| 水（H/O） | 輸送する蛍光数 | 0 | 0 | 5 keV 未満の線なので増えない |
| 水（H/O） | 脱出光子数 | 8190 | 8190 | 本番の蛍光 ON/OFF でも同一 |

旧物理旗の終端・生成数は旧い「1 履歴 1 レコード」の定義であり、本番の全子孫の
生成数と同じ量として比較しない。蛍光数と脱出数の上記の向きだけを主張する。
この小試験で EGS5 との独立検証や H2 の 51 量の関門を代替しない。

### D15(c)：修正せず残した実装不良

有効な表を持つ fixture に更新したことで、`seed_core.py::photon_digest` まで
実行が進み、`return digest(p[:size])` における **`NameError: name 'digest' is not defined`**
が露見した。`digest` の import は呼出し元 `observation` のローカルだけにあり、
別関数 `photon_digest` には届かない。これは契約変更で説明できないため、
**ユーザーの D15(c) に従い source を修正していない**。skip／xfail にもしない。
`test_native_split_and_faults`・`test_actual_argument_tamper`・`test_table_pointer_tamper`
の 3 件はこの経路で止まる。したがって seed の split／fault の検証済みとも呼ばない。

### D15 最終の §5 コマンド

テストと fixture の更新を終え、ソースを固定した後で全コマンドを実行した。

| コマンド | 結果 | ログ |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **982 passed、3 failed、2 warnings**（338.64 秒、終了コード 1） | [tests](a4u_d15_tests.log) |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | **1808 passed、17 skipped**（1262.12 秒、終了コード 0） | [prototypes](a4u_d15_prototypes.log) |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_*.py tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | **214 passed**（95.95 秒、終了コード 0） | [G5 指定群](a4u_d15_focused.log) |

T4・EGS5・H2 等の長い CPU 関門は実行しない。合格後の D14 は実施しない。
最終ログ・固定値・変更記録・本節を保持し、pytest の再生成できる一時束を整理する。
テストが書き換える歴史的な JSON は、D15 開始時のバイト列に戻す。

D15 最終確認: [結果 JSON](a4u_d15_results.json)、[整合性](a4u_d15_integrity.json)、
[最終ソースの SHA256](a4u_d15_source_hashes.json)、[未修正の不良](a4u_d15_unresolved.json)。
更新対象の契約検査と A4u の **19 試験**、Numba との厳密な 6 条件の比較は通過。
全体の不合格は上記の seed の 3 件で、原因は同じ未定義名である。
これを新契約の期待結果へ書き換えたり、skip／xfail にしたりしていない。
**S2 全体の受入合格は宣言せず、この不良は D15(c) に従って未修正で停止。**

開始時に保存した A2/MV の歴史的な JSON はすべて開始時のバイト列へ復元した。
新しい 4 keV のケースの記録は [a4u_d15_cases_cpu.json](a4u_d15_cases_cpu.json) に分けた。
CPU 本番 5 ファイル・NumPy 蛍光試験・既存 prototype・旧固定値・旧検証手順の tracked diff は空。
最終試験の開始後のソースハッシュの変更は 0。
今回 package 内で変更したのは `seed_t4_checks.py` の人工条件の 7→4 keV だけで、
光子・電子の物理実装と `seed_core.py` は変更していない。
完了した試験の再生成できる一時出力の整理は [一覧](a4u_d15_cleanup.json) に記録した。
既存のキャッシュ・ユーザーの成果物・D15 前の差分は保持し、commit/push はしていない。

補足: 指定群の `test_gpu_g5_a2_*.py` は `test_gpu_g5_a2b.py` を含まない。
このため指定群は 214 passed でも、全体試験には seed の上記 3 不合格が残る。

## Claude の独立検証（S2、2026-11-03）

| 項目 | 結果 |
|---|---|
| `tests/`（Claude 再実行） | **985 passed**（321 s） |
| G5 指定群（同） | **214 passed**（95 s） |
| prototypes（同） | **1808 passed・17 skipped**（1203 s） |
| 変更禁止の差分 | 空（`transport.py`・`kernel.py`・`physics.py`・`materials.py`・`tally.py`・`g1_physics.h`・`prototypes/`・`data/`・`docs/jsmp133/`・既存の記録）。prototypes のテストが書き換える `p8g1/faults_gate0.json`・`p8g3/fault_observations.json` は `git checkout` で復元した（mv.md の決まり） |
| 非干渉性の全断面積 | `c[0]=g5_total(...,1,p5_kn_total(...))` で表由来。有効範囲より上は解析 KN に落ちる |
| S(Z,q) の棄却 | `p5_compton` の中（電子の運動学は保持）。旧物理の旗で消える |
| Rayleigh の 5 本目 | `c[5]`・`z<5`・`process==4` が 6 か所すべてで揃う（`t2_kernel.h:182,191`、`t2_physics.h:264`、`t2_seed_kernel.h:203`、`t2_seed_geometry.h:90`、`t2_seed_probe.h:7`） |
| Coster-Kronig | `generate_photon_tables.py:100-103` で `CosKronTransProb`、収率＋CK の和が 1 を超えないことを生成時に検査 |
| 5 keV | 表の下端 5 keV、線は 5 keV 以上、5〜150 keV が取れない元素は生成を止める |
| 光子数の収支 | `core.py` に脱出スペクトルの保存則と終端光子数の一致。スタック容量・2 光子・故障の試験用の旗あり |
| 入口の範囲 | 単色・連続 bin とも 20〜10000 keV（`therapy_scene.py:17,32`） |
| D15 (b) CPU | `test_gpu_g1_history.py:25` が `_test_legacy_physics=True` でビルドし、バイト一致を維持 |
| D15 (b) 本番の差の向き | `test_d15_production_fluorescence_direction`（鉛 100 keV・0.05 cm・8192 履歴、測定前に固定）: 全殻＋CK で蛍光の発生・脱出光子・生成光子数が増えること、水では両モデルとも 0 で蛍光の ON/OFF がスコアを変えないこと、酸素の空孔は全エネルギーが局所に残ることを確かめる |
| GPU のバイト一致 | `a4u_gates.py` に CUDA 対 CPU のバイト比較はない（ユーザー決定 2026-11-03 のとおり） |

### Claude が直した点（1 行）

`seed_core.py` の `photon_digest` が `digest` を見られず `NameError`（`digest` は `seed_core.py:134` の関数内 import しかなかった）。同じ形の局所 import を 1 行足した。これは契約の変更とは無関係な取りこぼし。

### H1 の数値（私の読み）

| 量 | 水 | 鉛 |
|---|---|---|
| 継ぎ目の段差（光電） | −2.5e-5 | +2.9e-5 |
| 継ぎ目の段差（非干渉性、800 keV） | −7.2e-4 | **+1.13e-2** |
| 継ぎ目の段差（Rayleigh） | −6e-5 | −7e-5 |
| XCOM との光電の相対差（5／20／60 keV） | −5.9e-6／+1.4e-5／… | +1.3e-5／+2.7e-4／… |
| 殻の割合の凍結点 | H 60 keV・O 100 keV | 300 keV |

**未判定**: 鉛の非干渉性の継ぎ目 **+1.13%**（800 keV）だけが大きい。これは 800 keV より上で束縛つき（xraylib）から解析 KN へ切り替わるためで、鉛の束縛の効果そのものの大きさ。MV の線量への寄与が D9 の許容より小さいかを示すまで H1 は合格と呼ばない。

### 未了

H2・H3b・H4・H5・H6・H7（`a4u_s2_pending.json`）。T4・EGS5・長い CPU 関門は未実行。D14 の文書更新も未実施。

## D2 更新：高エネルギー側の非干渉性を XCOM に統一（2026-11-03）

ユーザー決定の D2 に従い、生成器の material / ensure / therapy_material が受け渡す
封印済み XCOM を `(energy_keV, photo, incoherent, coherent)` の 4 列にした。
元素別の全断面積も、それぞれの xraylib 有効上端より上では対応する XCOM 列を補間する。
全断面積の解析 KN は旧物理の試験旗だけに残し、本番の係数から除いた。
コンプトンの角度提案と電子の運動学は今回変更しない。
表の末尾が 3 列から 4 列になるため、治療・seed の photon ハッシュの長さ計算も更新した。

再生成コマンド（既存の封印済み応答を読み、XCOM 実行・新規取得・MC は行わない）:

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_d2_update --out /tmp/chatcarlo-a4u-d2
.venv/bin/python -m docs.validation.gpu.g5.a4u_h1 --out docs/validation/gpu/g5/a4u_d2_h1.json
```

旧い固定値は [開始時の値](a4u_d2_before.json) と既存の S2/D15 記録に残した。
新旧の 5 材料の photon ハッシュ・全 GPU ソースハッシュ・使用した XCOM ファイルの SHA256・
新しい束の manifest は [再生成記録](a4u_d2_rebuild.json) にある。
治療束 `/tmp/chatcarlo-a4u-d2/therapy_bundle` と seed 束 `/tmp/chatcarlo-a4u-d2/seed_bundle`
は生成後に既存の load 検査で検証した。私有スペクトル値はこの外部の束だけに保持する。

### H1 の再測定

今回は実際の本番 C++ `p8g_coefficients` を呼び、元素別に格納表から再構成した係数とも照合した。
旧 H1 は混合物の fallback と解析 KN を用いた Python 再構成なので、旧記録を上書きしない。
混合物の光電・Rayleigh の小差を物理の変更とは解釈しない。
下表は各上端の直下・直上 `E*(1±1e-8)` の段差と、直上の XCOM に対する相対差（ともに %）。
光電の上端が元素で異なる肺・骨では 2 つの継ぎ目を記載した。

| 材料 | 量 | 上端 keV | 段差 % | 直上対 XCOM 相対差 % |
|---|---|---:|---:|---:|
| water | photo | 800.026475 | -0.00243380717 | +1.69126293e-05 |
| water | compt | 800.026475 | -0.00251073545 | +1.07706282e-05 |
| water | rayleigh | 800.026475 | -0.00622054561 | +1.2286193e-05 |
| lung | photo | 800.026475 | -0.00211469358 | +0.00158323782 |
| lung | photo | 999.989542 | -0.00295953259 | -0.000188857869 |
| lung | compt | 800.026475 | -0.002432348 | -4.84683932e-06 |
| lung | rayleigh | 800.026475 | -0.0064351679 | -2.13579122e-06 |
| bone | photo | 800.026475 | -0.000167138898 | +0.00100226977 |
| bone | photo | 999.989542 | -0.00748220243 | -7.930863e-05 |
| bone | compt | 800.026475 | -0.00105769071 | -5.17927812e-08 |
| bone | rayleigh | 800.026475 | -0.00377256697 | +3.09707422e-06 |
| lead | photo | 999.989542 | +0.00287345991 | +0 |
| lead | compt | 800.026475 | -0.0016823273 | +0 |
| lead | rayleigh | 800.026475 | -0.00681308837 | +0 |

鉛の非干渉性の段差は旧記録の **+1.131123485%** から **−0.001682327298%** へ縮小した
（絶対値で約 672 分の 1）。直上の XCOM との相対差は 0。
5・20・60・150・1000・10000 keV の全 3 チャンネルの XCOM 相対差、
殻確率の凍結点と元素別上端は [H1 JSON](a4u_d2_h1.json) に記録した。
これだけで未実行の H2〜H7 の合格は宣言しない。

### 更新した試験と固定値の契約

| 対象 | 以前固定していたこと | 今回主張すること |
|---|---|---|
| `therapy_input_hashes.json` の 5 つの photon 欄 | 3 列の末尾と高エネルギー解析 KN を含む表 | 4 列の末尾と非干渉性 XCOM を含む再生成表。旧値を before/rebuild に保持 |
| `test_gpu_g5_a2b.py::test_physics_helpers_and_notebook` の source hash oracle | D15 時点の物理 helper のハッシュ | D2 の変更後の helper が固定値と一致。旧固定値・旧検査関数は保持し、新しい検査関数へ import のみ変更 |
| 新規 `test_gpu_g5_a4u_d2.py`（6 件） | 該当なし | 封印済み元素応答の第 3 列と高エネルギー非干渉性が一致、実際の native 係数が表と一致、鉛の段差が縮小、全 4 列をハッシュ対象に含める、3 列入力を拒否、全断面積の解析 KN は旧物理のビルドだけに存在 |

固定値の再生成は上記 `a4u_d2_update` コマンドで行い、
旧 source oracle と新 oracle は [固定値記録](a4u_d2_fixed.json) に併記した。
Numba との per-history の厳密一致試験、D15 の本番の差の向きの試験は緩めていない。
独立検証ですでに追加されていた `seed_core.photon_digest` の `digest` import は保持した。

最初の束生成はディスク空き容量の既存の 3 GiB ガードで停止した。
完了・非ロック状態の pytest 一時出力だけを、ファイル名・サイズ・SHA256 を保存して整理し、
再実行した。整理記録（pytest 一時出力のファイル一覧と SHA256）は作ったが、この機械でしか意味のない足場で 5 MB あるため **2026-11-05 にユーザー決定で消した**（`~/.Trash/a4u_cleanup_inventories_deleted_2026-11-05/`）。ガードは変更していない。
T4・EGS5・XCOM 実行・新規取得・長い CPU 関門は実施しない。

### D2 最終の §5 コマンド

実装・固定値更新を完了した状態から 3 コマンドをすべて実行した。結果は下記のとおり。

| コマンド | 結果 | ログ |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **991 passed、2 warnings**（347.81 秒、終了コード 0） | [tests](a4u_d2_tests.log) |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | **1808 passed、17 skipped**（1235.87 秒、終了コード 0） | [prototypes](a4u_d2_prototypes.log) |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_*.py tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | **220 passed**（99.09 秒、終了コード 0） | [G5 指定群](a4u_d2_focused.log) |

全コマンドを実装完了後に実行した。契約変更で説明できない失敗はない。
[結果 JSON](a4u_d2_results.json) にコマンド・終了コード・ログ SHA256 を記録した。
[最終ソースハッシュ](a4u_d2_source_hashes.json) からの試験開始後の変更は 0。
[整合性記録](a4u_d2_integrity.json) のとおり、封印済み XCOM 64 ファイルと両束の全ファイルの
SHA256 を再検証し、既存の検証 JSON 20 ファイルは開始時のバイト列に戻した。
CPU 本体・NumPy 蛍光試験・既存 prototype・既存 JSMP 資料の tracked diff は空。
Numba との per-history 比較は開始時から変更せず、旧物理の旗で厳密一致を維持した。
完了した今回の pytest 一時出力は ファイル一覧と SHA256 を保存して整理し（その目録は 2026-11-05 にユーザー決定で消した。下の「整理記録の扱い」）た。
pytest 自身が開発時の一時出力 `pytest-36` を自動整理したため、これは手動削除していない。
新しい治療束・seed 束と最終ログ、旧固定値、既存キャッシュは保持した。commit/push はしていない。
T4・EGS5・長い CPU 関門は未実行のまま。H2〜H7 や S2 全体の物理受入合格は宣言しない。

## Claude の独立検証（D2 の非干渉性を XCOM へ、2026-11-03）

継ぎ目の段差（相対、`a4u_d2_h1.json`。前の記録 `a4u_s2_h1.json` は残す）:

| 材料 | 光電 | 非干渉性 | Rayleigh |
|---|---|---|---|
| 水 | −2.43e-05 | −2.51e-05 | −6.22e-05 |
| 肺 | −2.96e-05 | −2.43e-05 | −6.44e-05 |
| 骨 | −7.48e-05 | −1.06e-05 | −3.77e-05 |
| **鉛** | +2.87e-05 | **−1.68e-05**（前は +1.13e-02） | −6.81e-05 |

鉛の非干渉性の段差が **+1.13% → −0.0017%** になり、全材料・全チャンネルで **≤ 7.5e-5**。これで「MV の線量への寄与が D9 の許容より小さい」は数値で言える（線量の関門ではない点は変わらない）。

実装の確認:

- `photon_physics.h:270-281`: 本番は `c[0]=g5_total(...,1)`（表由来、上端より上は XCOM）。**解析 KN は `G5_TEST_LEGACY_ELECTRON` のときだけコンパイルされる**。
- `generate_photon_tables.py:119`: `xcom=(keV nodes, photo, incoherent, coherent)` に非干渉性の列が入った。XCOM は実行せず、封印済みの応答を読む。

### テストの再実行（Claude、D2 の変更後）

`tests/` **991 passed**（303.6 s）、G5 指定群 **220 passed**（93.7 s）。Codex の報告と一致。変更禁止の差分は空。

## H2 CPU 分岐測定の起動・再開手順

Claude がリポジトリ直下で次を明示的に起動する。import・`--help`・引数なしでは
build も輸送も始まらず、`--out` が必須。再開は初回と同じ `--out` で下記コマンドを再実行し、完了した run ごとの JSON を再利用する。
シーン・束・実行核のソース・build の旗・seed・履歴数・分岐一覧ハッシュが異なる保存値は拒否する。
常設ルールに従い、起動・測定スクリプト自体のハッシュは関門の照合対象に含めない。

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_h2_gate --out /tmp/chatcarlo-a4u-h2
```

[a4u_h2_gate.py](a4u_h2_gate.py) は `a4u_s1_gate1.inputs` と `validation.quantities` を再利用する。
同じ s1・s2・s3、各 10 run × 1,000,000 履歴、batch 200,000。
旧 build の輸送 seed は 5000〜5009、本番 build は 6000〜6009、
線源 seed は両方 7000〜7009。再試行時はすべて +100。
両 build とも `Core(..., device='cpu')`、worker は 1。
旧 build は `_test_legacy_physics=True`（`G5_TEST_LEGACY_LOCAL=1`）、本番は False。
**診断の旧旗は K 殻のみだが、束縛の棄却・表の非干渉性・Rayleigh を既に含む。**
「棄却なし・解析 KN」は電子輸送版の旧経路であり、この診断版の旗とは異なる。
今回の道具は既存の旗をそのまま用い、物理コードは変更していない。JSON の `model_note` にも残す。

予想する差は事前固定の `DIVERGENT` の 4 量だけ:
`s2:category:3:all`、`s2:category:3:roi0`、`s2:category:3:roi1`、`s2:category:3:roi2`。
これは検出器の蛍光祖先分類のエネルギーフルエンスの全体・ROI 積分であり、
H3b で独立検証済みの量とは呼ばない。一覧は `a4u_gates.DIVERGENCE_SHA256` で固定する。

両側 0 の量も除かず **61 量すべて**について、10 run の平均・SEM、
差（本番−旧）、差の SEM（両側の SEM の二乗和の平方根）、z、検出可能差（3 × 差の SEM）を出す。
SEM=0 で差=0 の z は 0。差≠0 で SEM=0 の z は JSON の null と失敗判定で表す。
`a4u_gates.branches` に全 61 量を渡し、予想 4 量以外で絶対差が検出可能差を超えたものを
`unexpected` と `branch_status=stop` に残す。判定後に対象の集合を広げる処理はない。
加えて予想 4 量を除いた値を既存の `validation.independent_judge` に渡し、
S1 と同じ再試行規則を使う（3<|z|<4 が 1 量だけなら 1 回、|z|≥4・複数の |z|>3・
片側だけ 0 等の失敗なら再試行しない。再試行でも retry なら fail）。
初回の unexpected と判定も `attempts` に保持し、結果を強制的に pass にする処理は入れない。

出力は `OUT/h2.json`（`status / unexpected / divergence_sha256 / quantities / attempts`、
外側 10 run の値と seed を含む）と run ごとの再開用 JSON。
終了コードは判定 pass のときだけ 0、それ以外は 1。T4・EGS5・他の測定を起動する経路はない。
道具の実装時は人工配列・模擬 CPU の検査と §5 の試験だけを行った。H2 の実測は下節の Claude の記録を参照。
道具の実装時の §5 は **995 passed／1808 passed・17 skipped／224 passed**。
[全コマンド・ログの記録](a4u_h2_results.json) を保存した。

## H2（CPU だけの分岐の測定、Claude が実行、2026-11-03）

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_h2_gate --out <DIR>
```

3 シーン × 10 回 × 100 万履歴 × 2 本（`G5_TEST_LEGACY_LOCAL=1` の K 殻のみ ↔ 本番の全殻＋CK）、すべて native の CPU build。
判定は `a4u_gates.py` の予想した分岐の一覧（`s2:category:3:*` の 4 量、ハッシュ `138de34f…`）に対して行う。
結果: **status `pass`、`unexpected` は空**（記録 [a4u_h2.json](a4u_h2.json)）。

Claude が各回の値から計算した統計（対応づいた差も併記。両本は線源の seed を共有するので、対応づいた差のほうが感度が高い）:

| 量 | 旧（K のみ） | 新（全殻＋CK） | 相対差 | z（独立） | z（対応） |
|---|---|---|---|---|---|
| `s2:category:3:all`（蛍光） | 0.096311 | 0.10031 | +4.15% | 1.64 | 1.51 |
| `s2:category:3:roi1` | 0.043098 | 0.045728 | +6.10% | 2.02 | 1.81 |
| `s2:category:3:roi0` | 0.0030136 | 0.0036322 | +20.53% | 1.23 | 1.59 |
| `s2:category:3:roi2` | 0.00021748 | 0.00024852 | +14.27% | 0.34 | 0.26 |
| 予想外の量の最大 | — | — | 6.82%（`s3:category:2:roi2`） | 2.34 | **2.98**（`s3:escape`、+0.37%） |

**読み方（重要）**: この統計量では、**予想した 4 量の動きも検出限界に届いていない**（\|z\| ≤ 1.8）。したがって H2 が示したのは
「予想外の量が検出可能差を超えて動いていないこと」までで、「動くのは 4 量だけだと実証した」ではない。
4 量はいずれも**予想した向き（増える）**に動いており、鉛の L 線の脱出という見積もりと整合する。
物理の向きそのものは `test_d15_production_fluorescence_direction`（鉛 100 keV の箔、測定前に固定）が受け持つ。
予想外で最大の `s3:escape` は \|z（対応）\| 2.98 で、57 量を見ていることを考えれば偶然の範囲（判定の基準では合格）。

### H2 の締め（Claude、2026-11-03）

- 判定の規則は既存の `validation.independent_judge`（検出可能差 = 3×SEM、\|z\|≥4 か \|z\|>3 が 2 件以上で不合格、1 件なら新しい seed で 1 回だけ再試行）。今回の最大は \|z\| 2.34 なので**再試行なしの 1 回で合格**（`attempts` 1、`retry` false、`branch_status` pass、`regression` pass）。
- 記録の `interpretation` は「モデルの変更であり、検出器の ROI の 4 量は H3b で独立に検証されたものではない」。本文の読み方と一致。
- テスト（Claude、H2 の道具を含む状態）: `tests/` **994 passed・1 failed**、G5 指定群 **224 passed**。
  **落ちた 1 件は `tests/test_gpu_g5_a2_bundle.py::test_real_roundtrip` で、原因はコードではなくディスクの見張り**（`therapy.py:24` の `A2 disk space would fall below 3 GiB`）。
  単独で実行すると合格する（6.55 s）。実行時の空きは 5.7 GiB で、束の見込みを引くと 3 GiB の閾値を割るため、意図どおり止まったもの。
- 一時出力: 各回の JSON 60 件（scratchpad、648 KB）は作り直せるので残さない。要約の `a4u_h2.json`（168 KB、各回の値を含む）をリポジトリに置き、表はここから再現できる。

**H2 は閉じる。** 次は H3b（保存済み EGS5 の再解析）以降。**H5・H6 の前にディスクの空きを増やす必要がある**（計画 §6 の「3 GiB を割りそうなら止める」に、今の 5.7 GiB は近い。`~/.cache/chatcarlo` が 25 GB。整理はユーザーの作業）。

## ディスクの整理（2026-11-03、H5・H6 の前に）

空きが 5.7 GiB で、計画 §6 の「3 GiB を割りそうなら止める」に接触していた（`test_real_roundtrip` が実際に `A2 disk space would fall below 3 GiB` で落ちた）。
mv.md の「Drive なしで消してよい」分類だけを対象にし、**消す前に凍結の一覧（`docs/validation/mv/**/*.json` の絶対パス）と照合**した。

| 対象 | 大きさ | 判断 |
|---|---|---|
| `gsdist.dat`・`pgs5job.GSlst` 計 25 件 | 1.77 GiB | **12 件 1.32 GiB は `p9a/protected_start.json` に記録されているので残す**。残り 13 件 0.45 GiB をゴミ箱へ |
| `p7/work/export`（Colab の転送用の束） | 0.68 GiB | ゴミ箱へ。SHA256 の一覧を `archive/removed_exports_20261103/` に保存（2026-10-25 の前例と同じ） |
| `p8g2/export` | 0.82 GiB | **一度ゴミ箱へ移したが、`tests/test_gpu_g5_a2_bundle.py::test_real_roundtrip` が `p8g2/export/p8g2_manifest.json` を読むため戻した**（A4u の fixture 経由。mv.md の「消してよい」分類に入っていたが実際には参照がある） |

結果: 空き **5.7 → 7.5 GiB**（ゴミ箱はユーザーが空にした）。`~/.cache/chatcarlo` は 25 → 23 GB。
整理後に `tests/test_gpu_g5_a2_bundle.py::test_real_roundtrip` 合格、G5 指定群 **224 passed** で、ほかに参照の切れたものはない。

**mv.md に足すべき教訓**: Colab の転送用の束は「消してよい」分類だが、`p8g2/export` は A4u のテストの fixture が読む。
分類だけで消さず、テストを通して参照を確かめる（今回は戻して解決）。

## H3b：保存済み EGS5 との緩和の部品検証の起動道具

実装の詳細と H0 (6)(7)(8) に従う起動道具は [a4u_h3b_gate.py](a4u_h3b_gate.py)。
測定は Claude が次のコマンドを明示的に起動する。再開も同じ `--out` で再実行する。
import・`--help`・引数なしでは build も輸送もしない。

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_h3b_gate --out /tmp/chatcarlo-a4u-h3b
```

結果は実行時に [a4u_h3b.json](a4u_h3b.json) に書く。今回は **測定を起動せず**、実測の結果 JSON は作らない。
`OUT/condition.json` に条件とその SHA256、`OUT/{arm}/batch_*.json` にバッチごとの
`escaped_spectrum` を保存する。束（packed 配列の形・dtype・バイト列）、実行核のソース、
物理の組、CPU の旗、線源・幾何、seed・履歴数・batch 数、帯と閾値、EGS5 の SHA256 を照合し、
再開で異なれば build／輸送前に停止する。起動スクリプト自体を関門のハッシュには含めない。

ChatCarlo は S2 の `core.pack(bake_scene_materials(['air','lead']))` と同じ材料表の組み方を使う。
鉛の寸法 0.05×100×100 cm、中心 (0,0,0)、鉛筆ビームの始点 (−5.025,0,0)、方向 (+1,0,0)、
100 keV。background は air、world の半長は (50.025,100,100) cm とし、
既存 `run_chatcarlo_fluorescence.py` の slab と Geometry の外枠を再現した。
全方向・全子孫の `escaped_spectrum` を使い、履歴の最後の 1 本では代用しない。

| arm | 既存の故障旗 | 履歴数 | batch | seed |
|---|---:|---:|---:|---:|
| production | 0（本番） | 4,000,000 | 200,000 × 20 | 1 |
| CK_off | 1（CK 全停止） | 1,000,000 | 50,000 × 20 | 1 |
| f13_zero | 2（L1→L3 を 0） | 1,000,000 | 50,000 × 20 | 1 |

故障 2 本の履歴数は結果を見る前に 100 万へ固定した。H0 の 441 件／20 万から見積もると
故障後の 10–11 keV は約 1,800〜2,000 件となり、計数の自分側 2σ は約 4.5〜4.7%。
実際のバッチ SEM で 5% 未満に届かなければ `insufficient` とし、自動で履歴数を足さない。

EGS5 は `fluor_on/egs5job.out` の保存済み 120 行だけを読み、
SHA256 `a23ffc257cf53c314d764f2f9581111abbcb253202831a16f26346991f6769b8` を固定する。
1 keV の列をそのまま添字に対応させる。L 帯は **9–10、10–11、11–12、12–13、14–16、L 合計 9–16 keV**、
K 帯は **72–86 keV**。L は線源履歴あたりの光子数、K は既存の観測量 B（全脱出光子に占める割合）。
全帯で **|差| ≤ 2×合成 SEM または |差| ≤ 5%×EGS5 値**を使い、結果を見て閾値や帯を変えない。
ChatCarlo の SEM は外側バッチから求め、二次光子の同一履歴内の相関を含める。
K の SEM は分子と脱出数の共分散を含む比の誤差伝播で求める。
保存済み EGS5 には履歴別の二次モーメントがないので、L は Poisson 計数近似、K は二項近似とし、
その限界も結果 JSON に記載する。EGS5 の CK 使用は H0 (8) で確認済みの事実として記録する。

本番は 7 帯すべての一致、故障は **10–11 keV で不合格になること**と自分側の精度を調べる。
L 合計は識別力の根拠にせず、故障でも合格するという H0 の予想に合うかを別に記録する。
予想に合わない・精度不足のときは、その結果と未実行 arm を残して以後の測定を止める。
故障の帯の不合格を pass に書き換えず、スペクトル比較の判定と故障識別の成否を別欄に保持する。
H3b は緩和の部品検証であり、D13 の検出器 ROI の 4 量を独立に EGS5 で検証済みとは呼ばない。
殻の注入の試験 (c) と故障の仕掛けは作り直していない。T4 の 1 回は H6 に残す。

道具の該当試験 `tests/test_gpu_g5_a4u_h3b.py` は **10 passed**（0.70 秒）。
保存表の SHA256・bin・7 帯、Lα と L 合計の区別、K の観測量 B、誤差と OR 判定、
故障の識別・精度不足での停止、CPU の旗、バッチ再開・条件不一致、起動のガードを
人工データ・模擬 CPU で検査した。物理コード・既存試験・故障の仕掛けは変更していない。

| 計画 §5 のコマンド | 結果 | ログ |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **1005 passed、2 warnings**（341.62 秒、終了コード 0） | [tests](a4u_h3b_tests.log) |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | **1808 passed、17 skipped**（1315.39 秒、終了コード 0） | [prototypes](a4u_h3b_prototypes.log) |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_*.py tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | **234 passed**（88.30 秒、終了コード 0） | [G5 指定群](a4u_h3b_focused.log) |

[試験結果](a4u_h3b_results.json)・[実装のソース SHA256](a4u_h3b_source_hashes.json)・
[整合性](a4u_h3b_integrity.json) に記録した。試験開始後のソース変更は 0。
既存の検証 JSON 20 ファイルは開始時のバイト列へ復元し、Claude の H2 の実測 JSON と保存済み EGS5 は保持した。
完了済みの今回の pytest 一時出力は 一覧・サイズ・SHA256 を残して整理した（その目録は 2026-11-05 にユーザー決定で消した。下の「整理記録の扱い」）。
既存キャッシュ・前回の束・ロックの残る古い出力は削除していない。
**これは起動道具の試験結果であり、H3b の測定結果や合格の記録ではない。**
400 万履歴の本番・故障 2 本・新規 EGS5・T4 は実行していない。commit/push はしていない。

## H3b の測定（Claude が実行、2026-11-03）— **不合格・停止**

事前登録は計画の「H3b 実装の詳細」と [A4U_H0.md](A4U_H0.md) の (6)(7)(8)。判定は見て変えていない。
結果は [a4u_h3b.json](a4u_h3b.json)（`status: fail`、条件 SHA256 `37e4b996…`）。故障の 2 本は
**設計どおり未実行**（本番が落ちたら以後の測定をしない）。

### Claude が直した点（道具の幾何、1 行）

Codex の `make_arrays` は背景の air を実体のまま残していた。H3b の world は半長 50.025 cm なので、
10 keV の L 線は脱出までに約 50 cm の空気を通り、**26% 減る**（`CS_Total_CP(Air)`×0.001205 で
μ=6.2e-3 /cm、exp(−μ×50)=0.735）。EGS5 側は箔の外に媒質がない。S2 の試験
（`tests/test_gpu_g5_a4u_s2.py`）と同じ `a['data'][int(a['offsets'][19])]=0.` を足して真空にした
（`a4u_h3b_gate.py:65`）。該当試験の幾何の検査が `pack` を空の dict に差し替えていたので、
`data`・`offsets` を持つ偽物に直した（`tests/test_gpu_g5_a4u_h3b.py`）。10 passed。

直す前（空気あり）は L 帯が一律に低かった: 10–11 keV −29.7%・12–13 keV −22.7%・L 合計 −27.4%。
直した後が下の表。**空気の吸収が唯一の原因ではなかった**（14–16 keV が残った）。

### 本番（4,000,000 履歴、seed 1、20 バッチ）

| 帯 | ChatCarlo | EGS5 | 相対差 | z | 判定 |
|---|---:|---:|---:|---:|:--|
| 9–10 keV | 9.125e-05 | 8.9e-05 | +2.53% | 0.22 | 合格 |
| 10–11 keV（Lα） | 2.210e-03 | 2.152e-03 | +2.71% | 1.15 | 合格 |
| 11–12 keV | 4.025e-05 | 4.2e-05 | −4.17% | −0.25 | 合格 |
| 12–13 keV | 2.999e-03 | 3.078e-03 | −2.56% | −1.25 | 合格 |
| **14–16 keV** | 3.328e-04 | 4.22e-04 | **−21.15%** | **−3.97** | **不合格** |
| L 合計 9–16 keV | 5.682e-03 | 5.883e-03 | −3.42% | −2.44 | 合格 |
| K 帯 72–86 keV（観測量 B） | 0.773166 | 0.762639 | +1.38% | 12.42 | 合格（5% の側で） |

### 切り分け（計画の停止条件「実装の誤り・データの差・モデルの近似を切り分けて停止する」）

**結論: 表の生成の誤り**（モデルの差でもデータの来歴の差でもない）。**ただしこれを直しても
14–16 keV の不合格は残った**（下の「表の生成の修正と H3b の再測定」の節）。`generate_photon_tables.py:77` の
正規表現

```
(K|L[123]|M[1-5])(L[123]|M[1-5]|N[1-7]|O[1-7]|P[1-5]|Q[1-3])_LINE
```

は遷移先が **1 文字＋1 桁**の名前しか受け付けない。xraylib には遷移先が複数準位をまとめた
`O45`・`P23`・`N67` の線があり、**それが全部落ちている**。Z=82 で落ちている線（別名でなく実体）:

| 線 | エネルギー | その殻の放射の中での割合 |
|---|---:|---:|
| `KO_LINE` | 87.857 keV | 0.00789 |
| `KP_LINE` | 88.001 keV | 0.00026 |
| `L3O45_LINE` | 13.015 keV | 0.01787 |
| `L1P23_LINE` | 15.860 keV | 0.00089 |
| `L1O45_LINE` | 15.840 keV | 0.00056 |
| `L1N67_LINE` | 15.722 keV | 0.00006 |

殻ごとに残った割合の合計は **K 0.9918・L1 0.9985・L2 1.0000・L3 0.9821**。生成器は
`norm=sum(rate)` で残った線を 1 に正規化する（`generate_photon_tables.py:96`）ので、
**落ちた線の重みは残った線（Kα・Kβ1,3 と Lα）に乗る**。観測と量が合う:

| 観測 | 測定値 | 落ちた線からの予想 |
|---|---:|---|
| 86–99 keV（K-O/P の帯） | −9619 件／400 万 | K の 0.82% ≈ −8000 件。**ここが −15% の原因** |
| 72–86 keV（Kα・Kβ1,3） | +1.38% | 落ちた 0.82% が再正規化で乗る |
| 13–14 keV | 32 対 400 | `L3O45`（13.015 keV、L3 の 1.79%）が丸ごと無い |
| 10–11 keV（Lα） | +2.71% | L3 の 1.79% が再正規化で Lα に乗る |
| 15–16 keV | 不足の一部 | `L1P23`・`L1O45`・`L1N67` |

**制動放射で説明が付くのは 16–50 keV だけ**（鉛に線がない帯でこちら 11 件・EGS5 880 件。
診断の特殊化は `local-deposition` で電子を運ばない）。86–99 keV の −15% は制動放射では
説明できない（100 keV の鉛では K 光電子が約 12 keV、L 光電子が約 84–87 keV までで、
連続成分の密度は 16–50 keV の実測から約 25 件/keV、86–99 keV では数百件が上限。
不足の 9600 件に桁が足りない）。

**モデルの近似ではない根拠**: 無衝突透過＋Rayleigh（99–101 keV）が +0.02%、
強い線（10–11・12–13・72–86 keV）が 3% 以内。ずれているのは**落ちた線の帯と、
その重みが乗った帯だけ**で、符号も量も上の表で一致する。

EGS5 側のデータの差（f12=0.12 対 0.066、f13=0.584 対 0.62、ωL1=0.112 対 0.128）は別に存在するが、
今回の不合格の説明には要らない。

### 止めた（実装は直していない）

計画の停止条件どおり、基準を見て変えず、**生成器も直していない**。EGS5 は新規に回していない。
H3b は**不合格のまま**。これは H3b だけの問題ではなく、**本番の物理の表の誤り**なので、
直す範囲（生成器の正規表現、再生成、影響を受ける固定値のテスト、H1・H2 の再測定の要否）は
ユーザーの判断を仰ぐ。

**先に訂正**: この節の最初の版では「実装の誤りではない」「86–99 keV は制動放射で説明が付く」と
書いた。どちらも誤り。落ちた線を数えて取り消した。

### 落ちた線が「別名の重複」ではないことの確認と、影響する元素

`KO_LINE`・`L3O45_LINE` が個別の線の合計（別名）なら落として正しいが、Z=82 では
**個別の線が xraylib に無い**（`KO1`・`KO2`・`KO3`・`L3O4`・`L3O5` はいずれも
`Invalid line for this atomic number`）。つまりこの 2 つが唯一の表現で、落とすと線そのものが消える。
`KN2`（87.241）・`KN3`（87.360）は regex に通るので残っており、二重計上もない。

正規表現が落とす実体の名前（Siegbahn の別名を除く）: `KO`・`KP`・`L1N67`・`L1O45`・`L1P23`・
`L2P23`・`L3O45`・`L3P23`・`L3P45`。殻ごとの放射の重みのうち落ちる割合:

| Z | K | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 8（酸素）・20・26・29（銅）・47 | 0 | 0 | 0 | 0 |
| 74（タングステン） | 0.58% | 0.01% | — | 0.38% |
| 82（鉛） | 0.82% | 0.15% | 0.00% | 1.79% |

**水・骨・軟部組織は影響を受けない**（遷移先をまとめた線を持たない）。影響するのは高 Z の
鉛とタングステン、つまり遮蔽材と X 線管のターゲットで、A4u が足した全殻の緩和がいちばん効く所。
N・O・P 殻から出る線は束縛エネルギーが 5 keV 未満で追跡対象外なので関係ない。

## 表の生成の修正と H3b の再測定（Claude、2026-11-03。ユーザー指示「正規表現を直す」）

### 直したもの（`chatcarlo/gpu/generate_photon_tables.py`）

1. 行 81 の正規表現に、遷移先をまとめた準位を足した:
   `...|N[1-7]|N67|O[1-7]|O45|P[1-5]|P23|P45|Q[1-3]|O|P)_LINE`。
   拾うようになった名前は **`KO`・`KP`・`L1N67`・`L1O45`・`L1P23`・`L2P23`・`L3O45`・`L3P23`・`L3P45` の 9 つだけ**で、
   Siegbahn の別名（`KA1`・`LB1` など）は 1 つも入らない（Z=5〜92 で確認）。
2. 行 104 に**放射の重みの予算の検査**を足した: 1 つの空孔の線の割合は xraylib では 1 に足されるので、
   `abs(norm-1)>1e-3` で止める。修正前は鉛の K が 0.9918 で**この検査に引っかかる**。
   修正後は Z=5〜92・K/L1/L2/L3/M1–M5 の全殻で最大のずれが **2e-5**。

今回の欠陥は 1005 件のテストを 1 つも落とさなかった（鉛の線の組を数値で固定した試験が無い）。
この予算の検査が同じ種類の取りこぼしの歯止めになる。

### 直った量（再測定、同じ条件・4,000,000 履歴）

| 区間 | 修正前 ChatCarlo/EGS5 | 修正後 | 予想どおりか |
|---|---:|---:|:--|
| 86–99 keV（K-O/P の帯） | 0.847 | **0.995** | はい（`KO`・`KP` の復活） |
| 13–14 keV | 0.080 | **0.958** | はい（`L3O45` の復活） |
| 10–11 keV（Lα） | +2.71% | **+0.41%** | はい（再正規化の解消） |
| 72–86 keV（K 線） | +1.38% | **+0.45%** | はい |
| 9–10 keV | +2.53% | +0.28% | はい |
| 12–13 keV | −2.56% | −3.58% | — |
| 99–101 keV | +0.02% | +0.00% | — |

### 残った不合格: 14–16 keV が −21.4%（z=−3.99）

内訳は 14–15 keV が 0.852、15–16 keV が 0.635。L 合計 9–16 keV は −3.36% で合格。
`status: fail`、条件 SHA256 `f3aec122…`。故障の 2 本は設計どおり未実行。

**緩和の実装にこれ以上の誤りは無い**。殻に空孔を直接注入して（`g5_relax_probe`、各 20 万回）
出る光子を数えると、表の値どおり:

| 空孔 | 帯 | 実測／空孔 | xraylib からの期待 |
|---|---|---:|---:|
| L1 | 15–16 keV | 0.03005 | ωL1 0.128 × 0.2334 = **0.0299** |
| L2 | 14–15 keV | 0.06234 | ωL2 0.373 × 0.167 = **0.0623** |

**EGS5 側のデータも読んだ**（「粗い組」という当て推量を避けるため）:
- L 線のエネルギーは**同じ値**（`ELX1/2/3`、Z=82 で 12.794・15.097・15.216・15.757・15.775・14.765・15.178・13.015 …）。
  ただし組は **L1 が 8 本・L2 が 5 本・L3 が 7 本の計 20 本**で、こちらの鉛 62 本より少なく、
  残した線に再正規化される（xraylib の割合で数えると L1 の 8 本は 0.962 ぶん＝約 +4%）。
- ωL1 は **0.112 対 0.128**（EGS5 対 xraylib）、f12 0.12 対 0.066、f13 0.584 対 0.62。
- K から L1 への供給は EGS5 にもあるが（`egs5_kxray.f:51` の `ik=3`）、その線は 72.144 keV で
  累積確率の差が **0.00045**。無視できる。

**ここが説明できていない（第 2 の生成の欠陥を 1 つ見つけた）**: 互いのデータで数えると
15–16 keV は **こちらが約 13% 多い**はずなのに（L1 からの 0.0299 対 EGS5 の 0.112×0.236=0.0264）、
実測は 37% 少ない。向きが逆なので緩和のデータの差では説明が付かない。そこで表の吸収係数を
線のエネンルギーごとに xraylib と比べたところ、**1 本だけ食い違う**:

| 線 | エネルギー | 表の光電 | xraylib | 比 |
|---|---:|---:|---:|---:|
| L3M1・L3M5・L2M4・L3O45・L2N4・L1N2・L1N3・L1O2・L1O45・L1P23 | 9.18〜15.86 keV | — | — | **1.000** |
| **L2O4** | **15.178 keV** | 131.4 | 104.9 | **1.253** |

原因: 節の作り方（`generate_photon_tables.py:63`）が吸収端の両側に `E×(1±1e-7)` を置く。
15 keV では ±1.5 meV だが、**xraylib の `CS_Photo` の跳びは `EdgeEnergy` と一致しない** ——
鉛の L2 は `EdgeEnergy=15.200000` に対し跳びが **15.199989 keV**（11 meV 下）。
そのため「端の下」として置いた節が実際には跳びの上にあり、端の下側の節が存在しない。
結果、15.130〜15.200 の区間が跳びをまたいで内挿され、**端のすぐ下の μ が最大 25% 高く**なる。
K も同じ（跳びは `EdgeEnergy` より 3.8 meV 下）、L3 も同じ（0.24 meV 下）、L1 だけは跳びが
`EdgeEnergy` の上（+0.22 meV）なので正しく挟めている。線が区間に入るのは L2O4 だけ。

**効き**: L2 の空孔は L1 の約 9.4 倍あるので L2O4 はこの帯の約 2/3 を占める。μ が 25% 高いと
薄い箔の表面層からの脱出は約 2 割落ち、帯としては **−14% 程度**。実測の −37% の
**3 分の 1 から半分**を説明する。残りは未説明。

**制動放射は残りの説明にならない**: 鉛に線がない 16–50 keV での EGS5 の連続成分は
6.5 件/keV/100 万で、14–16 keV に必要な約 45 件/keV/100 万に桁が足りない。しかも μ が下がる
15→30 keV では脱出が**増える**はずで、向きも合わない。

### 止めた状態

- 正規表現と予算の検査は入れた。全テスト **1005 passed**（305 秒）。生成器の SHA256 は `f2e41991…`（**この後に `bracket_edge` の無限ループを直したため、H4・H5・H3a が使った生成器は `982870d9…`**。表そのものは H4 の `a4u_h4_table_hash_fix.json` が `prepare_cpu(20)` 対 `therapy.pack` の 136 鍵・4 材料を照合して一致（生成器 982870d9）。独立な作り直しではない）。
- **端の節の欠陥は直していない**（指示は正規表現。これは別の欠陥なので範囲をユーザーに確認する）。
- 封印済みの治療の束に photon の表は入っていない（`p8e/materials/*_bundle.npz`・`p9c2/export/material_*.npz` は
  `lines` を持つ別系統で、元素は水 (1,8) と種 (22,47)。**どれも今回の修正で割合が変わらない**元素）。
  表は実行時に xraylib から組み直すので、再生成の要る封印データは無い。
- **H1 は物理の結論が動かない**: 比べているのは全体の光電・非干渉性・干渉性と XCOM の継ぎ目で、
  線の組は入らない（`a4u_h1.py:70-86`）。変わるのは `generator_sha256` と `photon_sha256` だけ。
- **H2 は再測定が要る**: シーンの材料は s1 水・s2 **鉛**＋アルミ・骨・銅・軟部組織・s3 アルミで、
  今回の修正で割合が変わるのは**鉛だけ**。分岐の 4 量（`s2:category:3:*`）は s2 の蛍光の区分なので
  まさにここが動く。
- H3b は不合格のまま。14–16 keV の残りは、端の節を直してから測り直すのが先。

## 吸収端の節の修正と H3b の 3 回目の測定（Claude、2026-11-03。ユーザー指示「端の節も直す」）

### 直したもの（`chatcarlo/gpu/generate_photon_tables.py`）

1. **`bracket_edge`（新規、行 40）**: 吸収端の両側の節を `EdgeEnergy×(1±1e-7)` で決めるのをやめ、
   `CS_Photo` の**跳びの位置を二分探索**して挟む。下端が跳びの上に来てしまう場合は窓を 4 倍ずつ
   広げる（ウランの L3 は跳びが `EdgeEnergy` より 1.7 eV 以上下で、端から決めた窓が丸ごと跳びの上に
   入っていた）。`EdgeEnergy` は挟んだ対の中に残すので、殻の条件つき確率との整合は崩さない。
2. **線の位置の μ の検査（行 150 付近）**: 5 keV 以上の全部の線について、表の内挿した光電吸収が
   xraylib と 3e-3 以内で一致することを確かめ、外れたら止める（3e-3 は 1025 点の対数格子の
   内挿誤差の上。実測の最大は 2.6e-4）。挟んだ対の**中に入る線**は xraylib 自身が解像できない
   境目なので、止めずに `report['lines_at_edges']` に残す。
3. **エネルギーが引けない線の扱い（行 110 付近）**: xraylib が割合は持つのにエネルギーを持たない線が
   少数ある。その重みを黙って残りに再正規化しないよう、`report['rates_without_energy']` に記録し、
   予算の検査を `norm + 不明分 ≈ 1` に直した（Z=57 の L2 で 0.12%、Z=76 の L2 で 0.77%)。

**Z=5〜92 の全数走査（2026-11-04）**: 88 元素すべてで表が生成でき（合計 **0.5 秒**）、
線の位置での μ のずれは最大 **4.63e-4**（Z=70、9.9992 keV）。挟んだ対の中に入る線は
全元素で 103 本（記録するだけで止めない）。鉛では `KP`（88.0014 keV、割合 0.00026）の 1 本だけ。
全テスト **1005 passed**（310 秒）。

**走査で私の入れた不具合が 1 つ出た（修正済み）**: 窓を広げるループを
`while ... and edge-low<limit` で書いていたため、`edge-(edge-limit)` が打ち消しで
`limit` をわずかに下回ると `low` が更新されずに**無限ループ**になった。
Z=84・86・87・88（Po・Rn・Fr・Ra）で発生し、CPU を 6 時間以上空回りさせた
（その 2 プロセスは落とした）。回数を数える `for _ in range(12)` に直して全元素が 0.5 秒で通った。
**鉛は窓を広げる経路に入らないので表は変わらない**（走査の μ の検査が最大 4.6e-4 で通っており、
H3b の測定はやり直し不要）。この不具合は、ユーザーの「全数走査が終わってから」という指示と
「操作のコードは確認した？」という問いで走査を先に完走させたことで見つかった。

### 直った量（3 回目、同じ条件）

| 区間 | 修正前 | 正規表現のみ | 端の節も |
|---|---:|---:|---:|
| 15–16 keV | — | 0.635 | **0.682** |
| 86–99 keV | 0.847 | 0.995 | **0.998** |
| 13–14 keV | 0.080 | 0.958 | 0.965 |
| 14–16 keV の帯 | — | −21.4% | **−19.96%**（不合格） |

μ の誤りの効き（L2O4 の 1.25 倍）は **15–16 keV の小区間で 4.7 ポイント**（0.635→0.682）、
**事前登録の 14–16 keV の帯では 1.4 ポイント**（−21.39%→−19.96%）だった。
先に帯で −14% と見積もったのは過大で、実測で置き換える。鉛の K 吸収端のすぐ下（Kβ2 87.36・KO 87.86 keV）も修正前は μ が
4.7 倍（1.579 対 7.3）だったのが直り、86–99 keV が 0.998 になった。

### 残った −20% は参照データの差（EGS5 を回さずに切り分け済み）

1. **断面積は両コードで一致している**。保存済みの PEGS5 の一覧
   （`fluorescence_lead_150kev/egs5_on/pgs5job.pegs5lst`、既存の `extract_pegs5_photo_xs.py` で抽出）と
   xraylib を鉛で比べると、10.5・12.6・14.765・15.097・15.5・15.9・20・60・80 keV すべて **比 1.000**。
   例外は 15.178 keV で PEGS5 が **1.098 倍**（PEGS5 の区分線形フィットも L2 端を同じようににじませる）。
   つまり EGS5 側も L2O4 を余分に吸うので、差を**広げる**向きで、残りの説明にはならない。
2. **こちらの発光は表に忠実**（空孔の直接注入で確認済み、上の節）。
3. **EGS5 自身の L 線の分岐が xraylib より 14–16 keV に厚い**。EGS5 の `DFLX1/2/3` から読んだ
   L X 線 1 本あたりの確率:

| 殻 | 14–16 keV の合計（EGS5） | 同（xraylib） | 比 |
|---|---:|---:|---:|
| L1 | 0.2794（Lγ3 0.1348・Lγ2 0.1071 ほか） | 0.2334（L1N3 0.1033・L1N2 0.0868） | **1.20** |
| L2 | 0.1982（L2O4 **0.0286**・L2N4 0.1641） | 0.1869（L2O4 **0.0175**・L2N4 0.1615） | **1.06** |

### 小区間に割ると、未説明はどこか

事前登録の帯を 1 keV に割って、上の分岐データから予想した比と実測を並べる:

| 小区間 | 主な線 | データの差から予想 | 実測 | 状態 |
|---|---|---:|---:|:--|
| 15–16 keV | L2O4（EGS5 0.0286 対 0.0175）・Lγ2/Lγ3（1.2〜1.3 倍） | 0.76〜0.81 | 0.682 | **おおむね説明できる**（脱出確率で重みを付けると EGS5 の組は L2O4 寄りで、より脱出しやすい側に寄る） |
| 14–15 keV | L2N4（14.765 keV） | **0.97** | **0.852** | **未説明** |

14–15 keV では、こちら側で確かめられるものが全部合っている:

- μ は xraylib と **1.000**、保存済みの PEGS5 とも **1.000**。
- 分岐は EGS5 0.1641 対 xraylib 0.1615（1.6% 差）。
- L2 の空孔数も合う。12–13 keV は同じ L2 空孔からの Lβ1 が約 2/3 を占めるが −3.6% で、
  14–15 keV を説明できるほど L2 空孔が違えば 12–13 keV は −8% 程度になる。
- 13–14 keV が 0.965 なので、なめらかな連続成分がここに隠れてもいない。
- **輸送も清い**: 箔の中の深さ 5 µm・25 µm に 14.765 keV の光子を等方に 20 万本入れて脱出を数えると、
  無衝突の解析解 ½[E₂(μd)+E₂(μ(t−d))] の **1.03〜1.08 倍**（Rayleigh が同じ 1 keV の桶に戻る分で、
  深いほど比が上がるのも整合）。対照の 12.614 keV・15.178 keV も同じ。第 3 の欠陥は無い。

つまり **14–15 keV の約 12 ポイントは、こちら側では説明が尽きている**。残る候補は EGS5 側
（分岐データのさらに奥、あるいは電子衝撃イオン化のように電子を運ぶ側だけが持つ 14.765 keV の源）で、
保存データからは切り分けられない。

### 現状

- 生成の欠陥は **2 つとも直した**（落ちた線・端の節）。歯止めの検査を 3 つ入れた
  （重みの予算・線の位置の μ・エネルギー不明の重み）。全テスト 1005 passed。
- H3b は `status: fail`（条件 SHA256 `caebfcb1…`）。14–16 keV のみ不合格、ほかの 6 帯は合格。
  故障の 2 本は未実行。
- **H2 は再測定が要る**（鉛の表が 2 回変わった）。H1 は識別子だけが変わる。
- EGS5 は 1 本も新規に回していない。

## H2 の再測定（表の修正後。Claude、2026-11-04）

表が 2 回変わったので、計画の約束どおり H2 をやり直した。**条件は前回と完全に同じ**
（3 シーン × 10 回 × 100 万履歴 × 2 腕＝6000 万履歴、seed 5000/6000/7000 系列、CPU のみ、
分岐の一覧のハッシュ `138de34f…` も固定）。結果は [a4u_h2_after_table_fix.json](a4u_h2_after_table_fix.json)
（前回の [a4u_h2.json](a4u_h2.json) は**書き換えずに残す**）。

### 判定

| 項目 | 前回（2026-11-03） | 今回（表の修正後） |
|---|---|---|
| `status` | pass | **pass** |
| `unexpected`（予想外に動いた量） | なし | **なし** |
| 分岐を除いた回帰 | pass | **pass** |
| 再試行 | なし | なし |
| 全 61 量の最大 \|z\| | — | **2.341**（\|z\|>3 は 0 件） |

### 動いた量は予想どおり鉛のシーンだけ

61 量のうち**値が変わったのは 15 量、全部 s2**（鉛＋アルミ・骨・銅・軟部組織）。
s1（水）と s3（アルミ）は **1 量も動かなかった** —— 今回の修正で放射の割合が変わるのは
鉛とタングステンだけという H0 の予想と一致する。

**旧物理（K 殻のみ）の腕は 1 量も動かなかった**。`G5_TEST_LEGACY_LOCAL=1` の経路は
A4u の線の表を読まず、元の診断の K 殻の模型を保っている（先に「両腕が動く」と書いたのは誤りで、
実測で取り消す）。つまり**参照側が固定されたまま本番側だけが動いた**ので、前回との比較が素直に読める。

本番の腕の相対変化（大きい順）:

| 量 | 変化 |
|---|---:|
| `s2:category:3:roi0`（蛍光・ROI 0） | **+2.594%** |
| `s2:category:3:roi2` | +1.082% |
| `s2:category:3:roi1` | +0.761% |
| `s2:category:3:all` | +0.490% |
| `s2:category:2:roi1` | +0.096% |
| `s2:deposit:lead` | −0.042% |
| 残り 9 量 | \|変化\| < 0.03% |

動いたのは**蛍光の区分（category 3）が主**で、付与エネルギーは鉛で −0.042% しか動いていない。
落ちていた線（鉛の K で 0.82%・L3 で 1.79%）が戻り、再正規化が解けた分だけ蛍光の光子が増えた向きで、
符号も大きさも整合する。

### 分岐の 4 量（事前に予想して固定した量）

| 量 | K 殻のみ | 全殻（前回） | 全殻（今回） | z 前回 → 今回 |
|---|---:|---:|---:|---:|
| `s2:category:3:all` | 0.096311 | 0.100305 | 0.100796 | 1.64 → **1.89** |
| `s2:category:3:roi1` | 0.043098 | 0.045728 | 0.046077 | 2.02 → **2.24** |
| `s2:category:3:roi0` | 0.0030136 | 0.0036322 | 0.0037264 | 1.23 → **1.34** |
| `s2:category:3:roi2` | 0.00021748 | 0.00024852 | 0.00025121 | 0.34 → **0.37** |

表の修正で**分岐が少し大きく見えるようになった**（全殻の値が上がった）が、
この統計量（100 万履歴 × 10 回）ではまだ \|z\| ≤ 2.24 で解像しない。
前回記録した検出力の限界はそのまま有効で、**「予想外のものは動いていない」ことだけが言える**。

## H3b の判定：EGS5 新規 1 本による起源の切り分け（2026-11-04）

事前登録 [A4U_H3B_EGS5_PREREG.md](A4U_H3B_EGS5_PREREG.md) の規則で判定した（規則は走る前に固定し、変えていない）。
EGS5 側は `docs/egs5_crosscheck/fluorescence_lead_100kev_origin/`（保存済みの `fluorescence/` は無変更）。
採点だけを足した走行で、**1 keV の 120 帯すべてと総脱出数 309706 件が保存済みと完全一致**（健全性の検査に合格）。

### 起源の内訳（EGS5、100 万履歴）

R=緩和由来、B=制動放射由来とその子孫、P=一次とその散乱。LATCH は制動放射に新しい桁（`iarg=7`）を立て、
**B を R より先に判定**して、光電子に継承される印による混ざりを避けている。

| 帯 | ChatCarlo /履歴 | EGS5 R /履歴 | EGS5 B（件） | B の帯内割合 | ours/R |
|---|---:|---:|---:|---:|---:|
| 9–10 keV | 8.925e-05 | 8.300e-05 | 6 | 6.7% | 1.075 |
| 10–11 keV | 2.1605e-03 | 2.1390e-03 | 13 | 0.6% | **1.010** |
| 11–12 keV | 4.000e-05 | 3.200e-05 | 10 | 23.8% | 1.250 |
| 12–13 keV | 2.9677e-03 | 3.0570e-03 | 21 | 0.7% | **0.971** |
| 13–14 keV | 9.650e-05 | 9.500e-05 | 5 | 5.0% | 1.016 |
| **14–15 keV** | 2.5050e-04 | 2.9100e-04 | **3** | **1.0%** | **0.861** |
| **15–16 keV** | 8.725e-05 | 1.2600e-04 | 2 | 1.6% | **0.692** |
| L 合計 9–16 | 5.6917e-03 | 5.8230e-03 | 60 | 1.0% | 0.977 |

全体では R=256258・B=413・P=53035。**P は 1〜50 keV のどの帯にも 0 件**（100 keV の 1 回 Compton は
72 keV 以上、Rayleigh はエネルギー不変なので整合）。

### 判定：事前登録の (2) に該当。H3b は**不合格のまま閉じる**

- **制動放射は原因ではない**。14–15 keV の B は **3 件＝帯の 1.0%**（事前登録の足切り 5% 未満）。
  この 1 本で「電子を運ばない模型の差」という仮説は**量で否定された**。
  制動放射は実在するが、**鉛に線がない 16–50 keV に閉じている**（EGS5 は R=1・B=219、こちらは 11 件）。
  先の記録で「残りの候補は制動放射か EII」と書いたのは外れで、ここで取り消す。
- **R だけで帯を取り直しても合格しない**。14–15 keV は \|z\|=2.15・−13.9%、15–16 keV は \|z\|=3.19・−30.8%。
  2σ と 5% の**どちらも**超える。したがって事前登録の (1) は成立せず、H3b は不合格。
- 事前登録の (2) は「B が 5% 未満 → 残差は L 線の分岐データの差」と書いていた。**前半（B<5%）は満たすが、
  後半の解釈は 14–15 keV には当てはまらない**（EGS5 の分岐との差は 1.6% しかない。下の切り分け）。
  15–16 keV は分岐データの差でおおむね説明が付く。**14–15 keV は未説明のまま不合格として残す** ——
  事前登録の分類に合わないことを、合うように言い換えない。

**操作のコードも読んだ**: `fluor_on.f` と `fluor_origin.f` の差分は 123 行で、中身は
`iarg=7` の印（1 粒子 1 回の番人つき）・`iausfl(8)=1`・旧 jj 判定の `mod(latch,10000000)` 化・注記だけ。
物理・幾何・旗・seed は触られていない。印が効いていることは、鉛に線が無い 16–50 keV で
**R=1・B=219・P=0** になることでも裏が取れている。

### 残りの切り分け（EGS5 は増やさず、手元の保存データだけで）

**15–16 keV（−30.8%）はおおむね説明が付く**。EGS5 自身の分岐 `DFLX` はこの帯に xraylib より厚い
（L2O4 が **0.0286 対 0.0175**、Lγ2/Lγ3 が 1.2〜1.3 倍）。ここから予想される比は **0.76〜0.81** で、
実測 0.692 はその少し下。しかも**この走行そのものの PEGS5** では 15.178 keV の μ が xraylib の
**1.15 倍**（下表）で、EGS5 側は L2O4 を余分に吸う＝比を**上げる**向きなので、
残差はこの分岐データの差の範囲に収まる。

**14–15 keV（−13.9%）は説明が尽きた**。この帯は L2N4（14.765 keV）だけで、L2 空孔のみに由来する。

| 確かめたこと | 結果 |
|---|---|
| **この走行そのものの PEGS5 の μ**（`run_fluor_pb100_origin/pgs5job.pegs5lst`）対 xraylib、14.765 keV | **1.0000** |
| 同、9.184・10.551・12.614・13.015・14.309・15.216 keV | 0.9999〜1.0001 |
| 同、15.178 keV（L2 端のすぐ下） | 1.1499（EGS5 側のフィットのにじみ。上記のとおり差を**狭める**向き） |
| 分岐（EGS5 `DFLX2` の 14.765 keV 対 xraylib） | 0.1641 対 0.1615（**1.6%**） |
| L2 空孔の数 | 合う（12–13 keV は同じ L2 空孔からの Lβ1 が約 2/3 で **0.971**） |
| 空孔への直接注入での発光（L2 の 14–15 keV） | 0.0623 対期待 0.0623 |
| こちらの輸送（14.765 keV を深さ 5・25 µm に等方注入 対 無衝突の解析解） | 1.03〜1.08 倍（Rayleigh の戻り分） |
| 制動放射 | 帯の **1.0%**（3 件） |

**断面積・分岐・空孔数・発光・輸送・制動放射のすべてが合うのに 13.9% 足りない。**
手元の保存データで当たれる筋は尽きた。14–15 keV は **未説明のまま**とする。

**先に書いた「遷移先が N・O 殻の線だけ足りない」という筋は取り下げる**。13–14 keV は
L3O45（13.015 keV、遷移先 O45、生成器では `child=9`）が主だが **1.016 で合っている**ので、
N・O 行きの線が一律に足りないわけではない。`child=9` の扱いを疑う次の一手も、同じ理由で取り下げる。

### 私の誤りを 1 つ訂正

EGS5 を回す指示の中で、照合用に「14–15 keV が 1176 件・15–16 keV が 512 件」と渡したが、
これは**保存済みファイルの生の値ではなく 400 万履歴に換算した値**だった（ファイルは 294・128）。
操作側がファイルの生の値で照合し、この混同を指摘した。他の 6 個（89・2152・42・3078・100・236194）は
生の値と一致していた。判定は生の値で行っている。

### H3b を閉じる（ユーザー決定 2026-11-04「未説明のまま記録して先へ」）

| 帯 | 判定 |
|---|:--|
| 9–10・10–11・11–12・12–13・72–86 keV | 合格 |
| L 合計 9–16 keV | 合格（−2.25%） |
| **14–16 keV** | **不合格**（15–16 keV は EGS5 の分岐データの差で説明、**14–15 keV の −13.9% は未説明**） |

H3b は**不合格のまま閉じる**。基準は結果を見て変えず、実装も合わせていない。
A4u の受入条件には「H3b は緩和の部品検証として不合格のまま、未説明の残差つきで閉じた」と書く。
分岐した 4 量を「EGS5 で検証済み」とは呼ばない（D13 (ii)）。

**この段で得たもの**（H3b が落ちたことの価値）: 表の生成の欠陥 2 つ
（遷移先をまとめた線の取りこぼし・吸収端の節の置き方）を見つけて直し、歯止めの検査を 3 つ入れた。
どちらも 1005 件のテストを 1 つも落とさずに本番の物理に入っていた欠陥で、
H3b の帯別の判定でしか表に出なかった。

## H4 試し計算：EGS5 側（Claude、2026-11-04）

D6 の 2 点で EGS5（物理 ON）を小さい N で回した。`a4_egs.py` の既存の起動役をそのまま使い、
新しい道具は書いていない。出力は `a4_run/egs5/h4_pilot_{20,60}kev/`。

| 点 | N | seed | 時間 | **1 履歴あたり** | 付与／入射 | 脱出光子 | 健全性の検査 |
|---|---:|---:|---:|---:|---:|---:|:--|
| 20 keV | 1,000,000 | 20001 | 88.86 s | **88.9 µs** | 19057/20000 = 95.3% | 49,956 | **19 項目すべて合格** |
| 60 keV | 1,000,000 | 60001 | 94.50 s | **94.5 µs** | 44496/60000 = 74.2% | 328,220 | **19 項目すべて合格** |

物理の旗は両方 `IBOUND=1,INCOH=1,IRAYL=1`、`iedgfl=0,iauger=0`（水は 5 keV 以上に線がないので影響なし）、
電子の運動エネルギーの打ち切り 10.001 keV（ユーザーの常設決定どおり）。どちらも 100 バッチで出ているので、
本番の N の算出にはこのバッチ統計をそのまま使える。

**計画 F13 の空欄が埋まった**: 「低エネルギーの EGS5 の 1 履歴あたりは未測定」→ **約 90 µs/履歴**。
MV の実績 127 µs/履歴より安い。§8 の見積もり（4,000 万履歴で 1.4 時間／点）は **1.0 時間／点**に下がる。

### PEGS5 の断面積は低エネルギーでも使える（確認済み）

この走行そのものの PEGS5 の一覧（`~/.cache/chatcarlo/g5/a4/egs5/h4_pilot_20kev/pgs5job.pegs5lst`）を
xraylib の水と比べた。PWLF の格子は MV 用に 0.01〜21 MeV に張られているので、低エネルギーで
荒くなっていないかを先に確かめた:

| E | 10 keV | 15 keV | **20 keV** | 30 keV | **60 keV** | 100 keV |
|---|---:|---:|---:|---:|---:|---:|
| PEGS5 / xraylib（全断面積） | 1.0000 | 0.9933 | **0.9988** | 0.9988 | **0.9994** | 0.9989 |

**判定する 2 点ではどちらも 0.12% 以内**。最大のずれは 15 keV の 0.67% で、D7 の許容（2%）の中。
参照側の不確かさとして記録する。

### 進む前に決めること：光子の打ち切りが両コードで違う

- **EGS5**: `AP=0.01 MeV` ＝ **10 keV** で光子を打ち切り、その場に沈着（PEGS5 入力は MV の設定のまま）。
- **ChatCarlo の治療の特殊化**: **7 keV まで運び**、4 keV はその場
  （`tests/test_gpu_g5_a4u_s2.py::test_d15_therapy_photon_cutoff` で固定）。

20 keV の点では、7〜10 keV の散乱光子を**こちらは運び、EGS5 は殺す**。水の 9 keV の平均自由行程は
約 2.4 cm で、`fine` の 0.1 cm の bin から見れば局所ではないので、PDD の形に効きうる。
60 keV では 1 回散乱の最大損失が約 11 keV なので、10 keV を下回るには多重散乱が要り、効きは小さい。

**H5 の本番の前に決める**（本番の EGS5 はどちらにせよ新規に回すので、追加の計算は増えない）:
(a) H5 の EGS5 の PEGS5 入力の `AP` を 0.007 MeV にしてこちらに合わせる、
(b) 食い違いを量で測って記録したうえでそのまま比べる。
**どちらも事前登録の変更なので、私が勝手に選ばない。**

## H4：低エネルギー試し計算の起動道具（2026-11-04、測定は Claude が実行）

[a4u_h4_gate.py](a4u_h4_gate.py) を追加した。起動・再開はリポジトリ直下で同じコマンドを使う。
import・`--help`・引数なしでは build／輸送をしない。

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_h4_gate --out /tmp/chatcarlo-a4u-h4
```

Claude が起動済みの `a4_run/egs5/h4_pilot_20kev`（100 万履歴、seed 20001）と
`h4_pilot_60kev`（同、60001）の `config.json / summary.json / result.npz` を読むだけ。
両方が完了・健全性合格・物理 ON であることと、P7 regions、half=5、SSD=100、N、seed、
スペクトル配列のハッシュを先に検査し、不足があれば CPU build 前に停止する。
`config.E=10` は旧コードの仮値なので線源エネルギーに使わない。
両側とも `a4_egs.mono_edges` の同じ数値（20 keV は 19.9–20.1、60 keV は 59.9–60.1 keV の一様な 1 bin）を使う。

双子は本番 `chatcarlo-photon-electron-v1 / electron-transport`、水 30 cm、P7、N=100 万、
100 バッチ、seed は 20011／60011、slots=32。旧物理の試験旗は指定しない。
元の封印済み A2 入力を検証して読み、たとえ photon 表が付いていてもそれを引き継がず、
**現在の `generate_photon_tables.py` から全材料の photon を再生成**する。
H3b で修正した 2 欠陥を含む生成器の SHA256 と再生成表の SHA256 を条件に保存する。
`chatcarlo/` の物理コードや H3b の不合格という結論を変更していない。

D7 の PDD は EGS5 の **`fine`（600 bin、0.05 cm、全横方向を積分した Gy cm²/history）**を用い、
P7 の `axis_fine`（0.1 cm、局所領域）とは混同しない。
EGS5 の fine は既に正規化済みであり、再び N で割らない。
OCR は `profile_fine` の最浅層 5 cm（0.125 cm、x/y 各方向）、0.25 g の水 voxel の Gy/history。
両側の raw MeV を同じ体積・バッチ履歴数で換算し、100 個の外側バッチから平均と SEM を求める。

事前登録の **提案値**はコードの定数とコメントに置いた（最終承認は Claude、H5 の許可ではない）。

| 項目 | 提案 | 選び方 |
|---|---|---|
| 対象 PDD | EGS5 pilot の fine 平均がその最大の 5% 以上の bin | D7 の指定。双子の値で選別しない |
| 対象 OCR | 深さ 5 cm、中心 ±0.25 cm、x/y 各 4 bin | 保存済み P7 にこれより浅い OCR はない。場の端・尾部に計算を費やさない |
| 必須 PDD | 対象内の入口 0–0.5 cm、最低 4 bin | 足した低エネルギー物理に最も敏感な入口を落とさない |
| 必須 OCR | 中心 ±0.125 cm、x/y 各 2 bin | 両軸の中心を判定に必ず残す |
| 判定可能数・割合 | 必須すべて、かつ全対象の 80% 以上、最低 8 bin | 非必須の尾部だけは統計不足を許す。空集合で満たさない |
| 本番 N の相談上限 | 各側・各エネルギー 2.4 億履歴 | 計画の上限見積もりを提案値として使用。超えたら止め、勝手に N を増やさない |

対象・必須 bin の実際の添字は **双子を実行する前に**参照から固定し、条件の封印に含める。
多重性の family はエネルギーごと。K はその対象 PDD/OCR の総数、
z_crit = Φ⁻¹(1−0.05/(2K))。差 Δ=CPU−EGS5、合成 SEM=sqrt(SEM_EGS5²+SEM_CPU²)。
D7 の OR を T=max(z_crit×SEM, 0.02×EGS5) として実装し、
**|Δ|−2×SEM>T の bin が 1 つでもあれば全件を `cutoff_bins` に挙げ、停止して本番 N を出さない。**
SEM>1%×EGS5 は統計不足、歴史的 2σ の件数も併記する。
判定可能な不合格 bin は `known_failure_bins` に名前つきで残し、原因不明でも対象から除かない。

足切りに掛からなければ、合成 SEM≤1%×EGS5 を満たす精度予算を両側へ等分する。
各側の必要数は N_pilot×[SEM_side / (0.01×EGS5/√2)]²（最低 100 履歴）。
各側について対象 bin の最大を取り **×1.5、100 バッチで割り切れる整数へ切り上げ**て報告する。
これは判定に必要な統計精度の推定であり、一致や H5 合格を保証する N ではない。
pilot 自体が統計不足でも N の推定は報告するが、参照 0 等で推定できない・必須領域の不足・上限超えは停止する。
結果に合わせた ROI の縮小・基準の変更・N の自動増加・再試行・H5 の起動はしない。

EGS5 の WALL/CPU 秒と両者の秒/履歴を保存出力から取り、双子は `CPU.histories` の呼び出し時間を測る。
CPU の値には呼び出しの配列確保と native 観測を含み、build・表生成・出力 I/O は含めない。
EGS5 の既存 WALL/CPU の定義との差を記録し、起動時間を推測で引かない。
再開時も元のバッチの所要時間を再利用し、測っていない時間を埋めない。

`OUT/{20,60}/condition.json` とバッチの npz／照合 JSON を保存し、どちらかの既存条件が変われば
**どちらの CPU も実行する前に**停止する。再開では npz の SHA256 も照合する。
結果は実行時に `a4u_h4.json` に出す。今回は道具の pytest だけを行い、EGS5・CPU pilot・H5・T4/CUDA は起動していない。

道具の該当試験 `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_h4.py -q` は
**11 passed**（1.01 秒）。人工の外側バッチ・模擬 CPU だけで、正規化と単色 bin の一致、
Bonferroni/OR、足切り全件の列挙、既知の不合格と歴史的 2σ、統計不足、両側の N と 1.5 倍、
上限停止、最新 photon の再生成、2 ケースの事前照合、バッチ再開、起動ガードを検査した。
[ログ](a4u_h4_tests.log) と [ソース・テストのハッシュ／実行範囲](a4u_h4_tool_tests.json) を保存した。
現在の表生成器は開始時と同一で、H3b で修正されたものを保持している。
作業中に `photon_physics.h / t2_kernel.h / t2_physics.h` の開始時からのハッシュ変更を検出した。
**本作業ではこの 3 ファイルを書き換えておらず、変更元は未確認。復元せず保持し、新旧ハッシュを記録した。**
この該当 pytest は物理の実測や C++ の検証ではない。実行時にはその時点の核のソースを条件に封印する。
今回の指示に従い該当 pytest のみを実行し、§5 全体・EGS5・CPU pilot・H5・T4/CUDA は実行していない。

### 治療側の光子の打ち切りを 10 keV に戻した（ユーザー決定 2026-11-04「10 keV で合わせる」）

**訂正（2026-11-04）: 当初これを「計画 D3 への適合」と書いたのは私の誤読だった。**
D3 の「線の下限は 5 keV に**そろえる**（診断の帯は 5 keV から。治療側は**今** 10 keV 未満をその場）」の
「今」は変更前の状態を指し、「そろえる」は治療側を 5 keV に下げる意味だった。実際、
S2 は本番を意図して 5 keV にし、D15 の表（上記）で **7 keV を打ち切る旧契約を 4 keV へ移した**のは
そのためである（`test_d15_therapy_photon_cutoff` の欄「7 keV も打ち切る古い契約」）。
つまり**今回の 10 keV はユーザーの決定による計画の変更**であり、適合ではない。
決定そのものは有効（EGS5 の `AP=0.01 MeV` と一致させるため）。実装は

```
if(E<(G5_TEST_LEGACY_ELECTRON?.01:.005))   // 本番が 5 keV になっていた
```

と、本番の治療の特殊化を 5 keV にしていた（`t2_physics.h:256`・`t2_kernel.h:167`）。
EGS5 は保存済みの MV・低エネルギーのどの走行でも `AP=0.01 MeV`＝10 keV で光子を殺すので、
**両コードが同じエネルギーで止まる**ようになった。

| 変更 | 旧 | 新 |
|---|---|---|
| 治療の特殊化の光子の打ち切り | 5 keV（本番）/ 10 keV（旧物理の旗） | **10 keV（共通）**。`photon_physics.h` の `G5_ELECTRON_PHOTON_CUT` に名前をつけて由来を書いた |
| 診断の特殊化 | 5 keV の帯・線の下限 5 keV | **変更なし**（この 2 か所は治療の核だけが使う） |

**D15 に従って記録した旧→新**:

| 対象 | 旧 | 新 |
|---|---|---|
| `t2_physics.h` の `history` の関数ハッシュ（`a4u_d2_fixed.json`） | `849612e9b0eeb5cc…` | `bc5ec31ec9bbaed8…` |
| `test_d15_therapy_photon_cutoff` の探り | その場 4 keV・運ぶ 7 keV | **その場 7 keV・運ぶ 12 keV** |

`a4u_d2_fixed.json` には `history_of_changes` に日付・理由・旧新のハッシュを残した。
全テスト **1016 passed**（309 秒。Codex が H4 の道具の試験 11 件を足したので 1005 → 1016）。

**H1・H2・H3b には影響しない**（どれも診断の特殊化＝`local-deposition` の経路で、この 2 か所を通らない）。
**H6 の MV には影響する**: 旧物理の旗の値と同じになったので、保存済みの P7／A3／A2b の参照
（EGS5 側も 10 keV）との打ち切りの食い違いが無くなった。速さの比較（D10）は H6 で測り直す。

### H4 の native 表ハッシュ照合の修正（2026-11-04）

最初の実行は `H4 native CPU tables` で停止した。実材料の回帰試験を先に追加し、
修正前に `0.line_counts`〜`3.line_counts` の 4 鍵だけが不一致になることを再現した。
現在の `therapy.pack` は各材料の nz と shape を保って、全フィールドを float64 として格納する。
元の検証用 export の `line_counts` は int64 のため、その生のバイト列のハッシュは packed と異なる。
本番は `catalog.p8_material` が float64 の blob を作り、`therapy_bundle.load` が復元した材料を
`therapy_inputs.load` がハッシュするので一致する。

H4 の期待値を `catalog_table_hashes` から作るようにした。この関数は本番と同じ
`catalog.p8_material` を呼び、カタログの各フィールドの `data_sha256` を使う。
個別の鍵を手で合わせたり、観測側の `therapy_observation.table_hashes` を変えたりしていない。
条件に残す元の材料属性のハッシュは同じ入力の識別用として保持する。

`.venv/bin/python -m pytest tests/test_gpu_g5_a4u_h4.py -q` は **12 passed**（4.21 秒）。
追加試験は mock を使わず実際の `prepare_cpu(20)` と `therapy.pack(materials)` を通し、
観測側 `table_hashes` と **4 材料・全 136 鍵**の集合と値が完全一致することを検査する。
既存の再生成の小試験は単純な架空材料での再生成だけを検査し、全カタログの照合はこの実材料試験が受け持つ。
[修正前の失敗](a4u_h4_table_hash_fix_before.log)・[修正後のログ](a4u_h4_table_hash_fix_tests.log)・
[ハッシュと実行範囲](a4u_h4_table_hash_fix.json) を保存した。
本番の入力経路・観測側・物理ヘッダー・10 keV の打ち切り・D2 のピン・既存の打ち切り試験は
開始時のバイト列を保持した。EGS5・CPU pilot・輸送の測定は実行していない。

## H4 の判定（Claude、2026-11-04）: `stop_budget` —— 足切りではなく N が非現実的

結果は [a4u_h4.json](a4u_h4.json)（20 keV の条件 SHA256 `ed2b6a1f…`）。60 keV は設計どおり未実行
（20 keV で止まったため）。**物理の食い違いは出ていない**:

| 項目 | 20 keV |
|---|---|
| 足切り（`\|Δ\|−2×SEM(Δ)` が基準超） | **0 件** |
| D7 の不合格 | **0 件** |
| 歴史的 2σ の不合格 | **0 件** |
| 判定可能な bin | 17 / 96（必須 14 のうち 10） |
| K（対象 bin 数）・多重性の閾値 | 96・z_crit = 3.470 |

止まった理由は **D7 が判定できる状態に届かないこと**だけ。必要な履歴数（×1.5 後）は
**EGS5 7.29 億・CPU 9.74 億**で、EGS5 の時間に直すと **18.0 時間／点**。
§8 の見積もり（4,000 万履歴で 1.0 時間／点）の 18 倍で、計画の計算量の上限に収まらない。

### 原因は OCR の深さ 5 cm が 20 keV では「浅く」ないこと

必要な履歴数は bin の群でまったく違う:

| 群 | bin 数 | 判定可能 | 必要 N の最大 |
|---|---:|---:|---:|
| `fine`（PDD、入口 0〜0.5 cm が必須） | 88 | 17 | **1,229 万** |
| `ocr_x`（深さ 5 cm） | 4 | 0 | **4.86 億** |
| `ocr_y`（深さ 5 cm） | 4 | 0 | 4.20 億 |

入口の PDD は 1 本あたり 49〜90 万履歴で足りている（10 本すべて判定可能）。
行き詰まりは**深さ 5 cm の OCR** で、20 keV の水では μ ≈ 0.81 /cm なので 5 cm で e⁻⁴ ≈ 1.8% まで落ち、
入口の約 1/2600。100 万履歴での相対 SEM が 13〜16% ある。

P7 の保存している OCR の深さは **5・10・20 cm** だけで、これより浅い採点は無い（EGS5 を採点ごと
回し直す以外に手がない）。D7 自身も「30〜60 keV では 5 cm より深いブロックの線量がほぼ 0 なので
P7 の 11 量は使わない」と同じ理屈で深い量を外している。

### 選べる登録の範囲（計画の「収まらなければ ROI を狭め、履歴数を増やさない」に従う）

必須の OCR 4 bin を両方の点で同じ式で見積もった（保存済みの EGS5 の試し計算から、EGS5 側のみ）:

| 登録の範囲 | 20 keV の N（×1.5） | EGS5 の時間 | 60 keV の N（×1.5） | EGS5 の時間 |
|---|---:|---:|---:|---:|
| 現在の提案（PDD＋OCR 5 cm） | 7.29 億 | **18.0 時間** | 約 1.31 億 | 3.4 時間 |
| **PDD だけ**（OCR を外す） | **1,844 万** | **27 分** | 同程度 | 約 30 分 |
| PDD＋OCR は 60 keV だけ | 1,844 万 | 27 分 | 1.31 億 | 3.4 時間 |

60 keV の OCR は μ ≈ 0.206 /cm で 5 cm の減弱が e⁻¹ なので、20 keV の約 1/3 の履歴数で届く。

**私の推し**: **PDD は 2 点とも、OCR は 60 keV だけ**を必須に登録する。20 keV の OCR は
「測れないので判定しない」と明記する（測れないものを測れたことにしない）。合計の EGS5 は
27 分＋3.4 時間で、§8 の見積もりに収まる。

**事前登録の変更なので、私が勝手に選ばない。** Codex の提案値
（`PDD_FRACTION=.05`・`OCR_DEPTH_CM=5.`・`MIN_MANDATORY_BINS`・`MIN_EVALUABLE_COUNT=8`・`minimum_evaluable=77`）も
この決定に合わせて直す。

### 測った時間（計画 F13 を埋める）

| | 1 履歴あたり（wall） | 備考 |
|---|---:|---|
| EGS5 20 keV | 88.9 µs | 保存済みの native のタイマー。起動の差し引きなし |
| EGS5 60 keV | 94.5 µs | 同 |
| CPU の双子 20 keV | **26.5 µs** | `CPU.histories` の合計。build・表の生成・ファイル I/O を除く |

CPU の双子は EGS5 の **3.4 倍速い**ので、H5 で双子側が N に届くかは履歴数ではなく時間で決まる
（9.74 億履歴なら 7.2 時間）。

## H4 完走（登録を PDD だけに。ユーザー決定 2026-11-04）

事前登録の必須 ROI を**深さ方向の線量曲線だけ**にした（OCR は測るが判定しない）。
変更したのは `a4u_h4_gate.py` の `selected`（OCR を判定の集合から外す）と
`MIN_MANDATORY_BINS={'fine':4}` の 2 か所で、決定の理由と数値を定数の直前に書いた。
道具の試験 4 件は**旧登録を固定していたので更新**し（K 608→600 ほか、旧→新をコメントに記録）、
`stop_unestimable` の試験は OCR に頼れなくなったので「登録のあとで判定対象の bin を 0 にする」形に書き直した。12 passed。

| | 20 keV | 60 keV |
|---|---:|---:|
| `status` | **pilot_complete** | **pilot_complete** |
| 対象 bin 数 K・多重性の閾値 | 88・3.446 | 531・3.905 |
| 足切り | **0 件** | **0 件** |
| D7 の不合格・歴史的 2σ の不合格 | **0・0** | **0・0** |
| 本番の N（×1.5 後）EGS5／CPU | 1,844 万／2,013 万 | 1 億 2,845 万／1 億 1,358 万 |
| 推定時間 EGS5／CPU | **0.46 h／0.14 h** | **3.37 h／1.22 h** |

**物理の食い違いは 2 点とも 0 件**。止まる理由は無くなった。

## 1 時間を超える計算の洗い出しと条件の見直し（ユーザー指示 2026-11-04）

| 段 | 計算 | 現状の見積もり | 判定 |
|---|---|---:|:--|
| H5 | EGS5 20 keV | 0.46 h | そのまま |
| H5 | **EGS5 60 keV** | **3.37 h** | **見直す** |
| H5 | CPU の双子 2 点 | 0.14＋1.22 h | 60 keV は上に連動 |
| H5 | 入口経由の計算 2 本 | A3 の実績 581.8 秒／本 | そのまま |
| H6 | **`seed_ratio` の入口 1 本（T4）** | **116.2 分 = 1.94 h**（A2b の実績） | **見直す** |
| H6 | MV 10 MV・4 MV | A3 の実績から 1 本 10 分弱 | そのまま |
| H3a | 51 量の CPU の双子 | 約 8 分 | そのまま |

### 60 keV が 3.37 時間になる理由と、縮める案

20 keV は μ ≈ 0.81 /cm なので 5% 以上の深さが 88 bin で済むが、60 keV は μ ≈ 0.206 /cm で
**531 bin（深さ 26.9 cm まで）**になる。多重性の閾値が 3.905 に上がり、80% の bin を
判定可能にする要求が深い裾まで及ぶため N が 1.28 億に膨らむ。

| 判定する深さの上限 | bin 数 | N（×1.5） | EGS5 |
|---|---:|---:|---:|
| 〜5 cm | 100 | 817 万 | 0.21 h |
| 〜10 cm | 200 | 1,283 万 | 0.34 h |
| **〜15 cm** | **300** | **2,322 万** | **0.61 h** |
| 〜20 cm | 400 | 4,341 万 | 1.14 h |
| 〜30 cm（現状） | 531 | 1 億 2,845 万 | 3.37 h |

**推し: 60 keV の判定を深さ 15 cm までにする**（300 bin・0.61 時間）。60 keV の水では 15 cm が
入口の約 5%（e⁻³）で、足した物理（Rayleigh・束縛コンプトン・緩和）はこの範囲で十分に効く。
bin が 300 あるので多重性の補正も成り立つ。これで **H5 の全部が 1 時間未満／本**になる
（EGS5 0.46＋0.61 h、CPU 0.14＋約 0.2 h）。

### H6 の `seed_ratio` が 1.94 時間

A2b の実績（`A2B_LOCAL.md`: old_cold 117.4 分・new_cold 116.2 分）。H6 では
「新しい核で回して保存済みの EGS5 参照に元の基準で当てる」ことが目的なので、
**履歴数を減らすと基準が成り立たなくなる**（A2b の判定はこの履歴数で設計されている）。
縮める案は 2 つ: (i) そのまま 1.94 時間を 1 回だけ払う（T4 は 1 セッションなので合計には収まる）、
(ii) `seed_ratio` を H6 の対象から外し、「A2b の判定は A4u の表の変更後に取り直していない」と
明記する。**(i) を推す** —— 鉛の表が変わったので種（Ti・Ag）は影響しないが、
`seed_ratio` の幾何には鉛が入らないことを確かめたうえで判断したい。

### 打ち切りの 3 か所目（seed の核）を直した

1 時間超の計算の洗い出しの途中で見つけた。`t2_seed_kernel.h:196` だけ
`(G5_TEST_LEGACY_ELECTRON?.01:.005)` が残っており、**seed の経路の光子だけ 5 keV**で切れていた
（同じ `electron-transport` の特殊化なのに治療の 2 か所と食い違う）。`G5_ELECTRON_PHOTON_CUT` に統一した。
周りの電子の打ち切りは同ファイルで一貫して `.01` なので、これで経路全体が 10 keV にそろった。
全テスト **1017 passed**。

**`seed_ratio` への影響**: シーンの材料は `ag_ti_mixture`（Ag 47・Ti 22）と水（1・8）だけで、
**表の修正では放射の割合が 1 つも変わらない元素**（変わるのは鉛とタングステンのみ）。
したがって H6 で `seed_ratio` を回す理由は**表ではなく打ち切りの変更**（5→10 keV）だけになる。
I-125 の線（約 27〜35 keV）から 5〜10 keV に落ちるには多重散乱が要り、Ag の K 線は約 22 keV、
Ti の K 線は約 4.5 keV（どちらの打ち切りでも運ばない）なので効きは小さいと見込まれるが、
**見込みで省かず H6 で 1 回測る**ことを推す（1.94 時間、T4 の 1 セッションの中）。

## H5 の事前登録（ユーザー承認 2026-11-04）と本番の起動

判定の規則は計画 D7 のまま。**ユーザーが決めた 2 点**を登録に入れ、`a4u_h4_gate.py` の
`REGISTRATION_PROPOSAL['status']` を `approved_by_user_2026-11-04: PDD only, judged depth <= 15 cm` にした。

| 登録した項目 | 値 | 決めた理由 |
|---|---|---|
| 判定する量 | **深さ方向の線量曲線（`fine`）だけ** | OCR の唯一の浅い保存深さ 5 cm が 20 keV では入口の 1/2600 で、判定可能にするのに 4.9 億履歴＝18 時間かかる。P7 はこれより浅い OCR を保存していない |
| 深さの上限 | **15 cm** | 60 keV は 5% の規則だけだと 26.9 cm・531 bin に広がり 1.28 億履歴＝3.37 時間。15 cm なら 300 bin で多重性の補正も成り立ち、0.61 時間 |
| 深さの下限 | 最大の 5% 以上（D7 のまま） | 変更なし |
| 必須 ROI | 入口 0〜0.5 cm の `fine` を 4 本以上 | 足した物理がいちばん効く所 |
| 判定可能の最低数 | K の 80%（かつ 8 本以上） | 変更なし |
| OCR・15 cm より深い bin | **測るが判定しない** | 測れないものを測れたことにしない |

### 本番の N（H4 の実測から `required`×1.5）

| 点 | K | 多重性の閾値 | N（EGS5） | N（CPU） | 推定時間 EGS5／CPU |
|---|---:|---:|---:|---:|---:|
| 20 keV | 88 | 3.446 | **18,438,400** | 20,127,000 | 0.46 h／0.14 h |
| 60 keV | 300 | 3.765 | **23,216,500** | 23,164,000 | 0.61 h／0.24 h |
| 合計 | | | | | **1.06 h／0.38 h** |

**1 時間超の計算は無くなった**（当初の見積もり EGS5 21.4 時間 → 1.06 時間）。

### 起動した計算

`a4_egs.py` で EGS5 の本番 2 本（`h5_main_20kev` seed 20101・`h5_main_60kev` seed 60101）。
試し計算とは**別の seed** にして、N を決めるのに使ったデータで判定しないようにした。
残りは CPU の双子 2 本と、入口経由の計算 2 本（preview → ユーザーの 3D 承認 → approve → run）。

## ⚠️ 訂正: EGS5 の 1 履歴あたりの時間を 9〜15 倍に見誤っていた（2026-11-04）

H5 の本番 2 本が終わって分かった。**判定はまだしていない**（下記の理由で止めている）。

| 走行 | N | 時間 | 1 履歴あたり（割り算） |
|---|---:|---:|---:|
| 試し計算 20 keV | 1,000,000 | 88.9 s | 88.9 µs |
| **本番 20 keV** | 18,438,400 | **193.7 s** | **10.5 µs** |
| 試し計算 60 keV | 1,000,000 | 94.5 s | 94.5 µs |
| **本番 60 keV** | 23,216,500 | **333.1 s** | **14.3 µs** |

2 点から固定費と周辺費用を分けると:

| | 周辺の費用 | 起動などの固定費 |
|---|---:|---:|
| 20 keV | **6.01 µs/履歴** | 82.8 s |
| 60 keV | **10.74 µs/履歴** | 83.8 s |

つまり試し計算の「88.9 µs/履歴」は **約 93% が起動の固定費**で、
1 履歴あたりの費用として読んではいけない値だった（関門の時間の欄は
`wall_seconds/N` をそのまま出し、`scope` に「起動の差し引きなし」と書いてあったが、
**私がそれを N の見積もりにそのまま使った**）。

### 直した見積もりでは、狭めた 2 つはどちらも 1 時間未満だった

| 当時の判断材料 | 私が示した時間 | **実際の時間** |
|---|---:|---:|
| 20 keV の OCR を判定に入れる（4.86 億履歴） | 18.0 時間 | **0.81 時間** |
| 60 keV を深さ 30 cm まで判定（1.28 億履歴） | 3.37 時間 | **0.38 時間** |
| 採用した狭い登録（1,844 万＋2,322 万） | 1.06 時間 | **0.15 時間**（実測 8.8 分） |

**ユーザーの 2 つの決定（「PDD だけ」「15 cm まで」）は、9〜15 倍に膨らんだ時間を根拠に示した表の上で行われた。**
正しい時間ならどちらも「1 時間を超える計算」に当たらない。

### だから判定を止めている

登録は**どの bin を判定するか**を決めるものなので、本番のデータを見たあとで広げると
「データを見てから判定範囲を選んだ」ことになる。いまは**まだ CPU の双子を回しておらず、
D7 の判定もしていない**ので、選び直す余地が残っている。

- **(a) いまの狭い登録のまま判定する** —— 事前登録は守られており、走り終わった 2 本をそのまま使える。
  ただし「OCR と 15 cm より深い所は判定していない」という範囲の狭さが最後まで残る。
- **(b) 登録を広げてから、新しい seed で EGS5 を回し直す** —— 先に登録を書き換え、
  それから走らせるので事前登録の順序は保てる。追加の時間は、深さ 30 cm まで 0.38 時間、
  20 keV の OCR も入れて 0.81 時間。CPU の双子は 26.5 µs/履歴なので
  4.86 億履歴だと 3.6 時間かかる（**こちらが律速になる**）。

**私が勝手に選ばない。** 時間の見積もりを誤ったのは私なので、選び直す機会を出すのが筋だと考える。

### 併せて直す事実

- 計画 F13 の「低エネルギーの EGS5 の 1 履歴あたり」は **約 90 µs ではなく 6〜11 µs**（起動は別に約 83 s）。
  MV の 127 µs/履歴という実績も、同じ割り算で出ているなら起動を含んでいる可能性があり、
  **MV の値も確かめ直さないと比較に使えない**。
- 「CPU の双子は EGS5 の 3.4 倍速い」も誤り。周辺の費用で比べると
  **EGS5 6.0〜10.7 µs 対 CPU の双子 26.5 µs** で、**EGS5 のほうが 2.5〜4.4 倍速い**。

## H5：低エネルギー本番の CPU 判定道具（測定は Claude が実行）

[a4u_h5_gate.py](a4u_h5_gate.py) を追加した。初回・中断再開は同じコマンドを使う。
import・`--help`・引数なしでは build／輸送をしない。

```bash
.venv/bin/python -m docs.validation.gpu.g5.a4u_h5_gate --out /tmp/chatcarlo-a4u-h5
```

結果は実行時に `a4u_h5.json` に書く。今回は本番も再試行も実行していない。
`a4u_h4_gate` の **registration・analyse・moments・candidate_samples を直接 import** して使い、
判定式を書き写していない。H5 の EGS5 参照から registration を作り、承認済みの
「PDD のみ・中心深さ ≤15 cm・最大の 5% 以上」を判定する。
OCR と深い PDD は全 100 バッチの npz および結果の `measured` の平均・SEM に残すが、合否に使わない。
必須 ROI・最低判定可能数と割合・多重性の K と閾値も H4 の確定済み関数に従う。
H4 の `analyse()['pilot_D7']` を H5 の `D7` として使う。両側の N が異なるため、
H4 の pilot 用本番 N 見積もりと足切り欄は H5 の判定に流用しない。

| エネルギー | 保存 EGS5 の N / seed | CPU 本番 N / seed | 人が判断する再試行用 CPU seed |
|---|---|---|---|
| 20 keV | 18,438,400 / 20101 | 20,127,000 / 20111 | 20121 |
| 60 keV | 23,216,500 / 60101 | 23,164,000 / 60111 | 60121 |

EGS5 の `h5_main_20kev`・`h5_main_60kev` の config／summary／result を読むだけ。
19 項目の健全性、物理 ON、N、seed、P7、half=5・SSD=100、`mono_edges` の一致を検査し、
ファイルの SHA256 を条件に封印する。再開前には両エネルギーの条件を先に照合し、異なれば CPU build 前に停止。
双子は H4 の `prepare_cpu` を使い、修正済みの現行生成器による photon と本番の
`chatcarlo-photon-electron-v1 / electron-transport` を使用する。旧物理の試験旗は指定しない。

100 外側バッチを採る。1 外側バッチは 20 keV が 201,270、60 keV が 231,640 履歴で、
`CPU.histories` の 200,000 上限を超えるので、H4 と同じ **10,000 履歴以下の呼び出し**に分割する。
各区間の native 観測と健全性を照合し、raw MeV を足してから外側バッチの履歴数で 1 回だけ正規化する。
`OUT/{20,60}/batch_*.npz` と受領 JSON に条件ハッシュ、全体の start/count/seed、
物理の組、区間ごとの start/count と観測のハッシュ、npz の SHA256、呼び出し時間を残す。
再開ではこれらを検査し、完了したバッチの再測定・build はしない。

結果の D7 には補正閾値での合格・不合格、統計不足、必須 ROI が判定可能か、
歴史的 2σ の不合格件数を併記する。原因未特定の不合格も `known_failure_bins` と全 bin の値に残す。
不合格なら `retry_required` を出して停止し、同じ N・条件・参照と新 CPU seed の計画を出すだけ。
統計不足は `insufficient` として停止する。**再試行は自動実行せず、実行用の retry スイッチも追加していない。**
EGS5 の新規走行を計画・起動しない。

人が再試行を決めて別途作った結果は、読取り専用の `judge_retry(first_case, second_case)` で照合できる。
第 2 ケースは同じ条件を持ち、`attempt='retry'`・`cpu_seed=retry_cpu_seed` として
共通の `evaluate_case`（内部は H4 の analyse）で判定したものを渡す。
同じ bin が再び落ちれば fail。第 2 回が合格なら pass。別の bin に新たな不合格が残る場合は
除外せず `review_required`、統計不足は insufficient とし、3 回目を許可・起動しない。
条件・参照・登録・N が違う第 2 回は拒否する。

時間は EGS5 の保存済み pilot/main の 2 点から
marginal=(T_main−T_pilot)/(N_main−N_pilot)、fixed=T_main−marginal×N_main として分けて記録する。
校正に用いた pilot の config／summary の SHA256 も封印する。
`wall_seconds/N` を周辺費用には使わない。この 2 点の分解は推定であり、
周辺費用には履歴処理・周辺 I/O 等を含み、I/O 単独を測ったものとはしない。
CPU は native 呼び出しの時間を記録し、build・表生成・検査・出力 I/O を含めない。

該当試験 `.venv/bin/python -m pytest tests/test_gpu_g5_a4u_h5.py -q` は **14 passed**（0.95 秒）。
H4 の 4 関数の同一性、狭い登録、両側の N の相違、OCR・深部の非判定、補正/2σ と統計不足、
再試行の報告だけで止まること、読取り専用の第 2 回の判定、時間の固定/周辺分離、
分割バッチと受領・再開照合、起動のガードを人工データ・模擬 CPU で検査した。
[試験ログ](a4u_h5_tests.log)・[ハッシュと範囲](a4u_h5_tool_tests.json) を保存。
H4 の関数・登録、物理コード、表生成器、10 keV の打ち切りは変更していない。
これは CPU 双子の統計判定の道具であり、入口の preview／承認／実行や H6 の検証は別に残る。
H3b は不合格・未説明の残差つきで閉じたまま。EGS5・CPU 本番・再試行・CUDA は起動していない。

## H5 低エネルギー本番の判定（Claude が実行、2026-11-04）: **合格**

> **⚠️ 2026-11-05 追記（判定そのものは変えない）**: ここで「深さ方向の線量曲線」と呼んで
> 判定した量は、`a4u_h4_gate.py:260` の `groups[:,450:1050]`、すなわち**横方向に積分した
> スラブ全体の深さ線量**である。製品が PDD として出す**中心軸（2×2 cm² の柱）ではない**。
> 下の「H5 が判定したのは中心軸の PDD ではなかった」を参照。合否と数値は有効で、
> **指している量の名前が違っていた**。

ユーザーの決定「狭いまま判定」に従い、承認済みの登録（PDD だけ・深さ 15 cm まで・下限は最大の 5% 以上）で
判定した。規則は `a4u_h4_gate` の `registration`・`analyse`・`moments`・`candidate_samples` を
**import してそのまま使っている**（書き写していない）。結果は [a4u_h5.json](a4u_h5.json)。

| | 20 keV | 60 keV |
|---|---:|---:|
| 判定した bin 数 K | 88（深さ 0.025〜4.38 cm） | 300（深さ 0.025〜14.98 cm） |
| 多重性を補正した閾値 | 3.446 | 3.765 |
| **D7 の合格／不合格** | **88 / 0** | **300 / 0** |
| **歴史的 2σ の不合格（併記）** | **0** | **0** |
| 統計不足 | **0** | **0** |
| 必須 ROI が判定可能か | **はい** | **はい** |
| 原因不明の「既知の不合格」 | なし | なし |
| やり直しの要否 | 不要 | 不要 |

**全部の bin が、補正した閾値と歴史的 2σ の両方を通った。** 統計不足が 0 件なので、H4 で決めた N は足りていた。

### 数値

| | 20 keV | 60 keV |
|---|---:|---:|
| \|z\| の最大 | **2.74**（閾値 3.446） | **3.09**（閾値 3.765） |
| \|z\|>2 の件数 | 1 / 88 | 15 / 300 |
| \|z\|>3 の件数 | 0 | 3 |
| 相対差の平均 | −0.053% | −0.083% |
| 相対差の絶対値の平均 | **0.298%** | **0.410%** |
| 相対差の最大 | 1.363% | 1.658% |
| 最大 \|z\| の bin | `fine:37` 深さ 1.875 cm（EGS5 5.9920e-13 / CPU 5.9398e-13） | `fine:28` 深さ 1.425 cm（EGS5 6.0574e-13 / CPU 5.9871e-13） |

\|z\|>3 が 60 keV で 3 件あるが、どれも多重性を補正した閾値 3.765 の下。

**歴史的 2σ が 0 件である理由を正確に書く**: 歴史的な基準は「2σ **または** 2%」の OR なので、
相対差の最大が 1.658%（< 2%）である以上、0 件は **2% の枝で保証されていた**。
\|z\|>2 の 16 件（20 keV で 1・60 keV で 15）は 2σ の枝では落ちており、2% の枝で通っている。
「補正に頼らず通った」という読み方は**できない**。

**\|z\|>2 の bin が固まっていないことの確認**（H0 (5) で 60 keV の G2 のデータに対して行ったのと同じ見方）:

| | 件数 | 符号 | 深さの範囲 |
|---|---:|---|---|
| 20 keV | 1 / 88 | 負 1 | 1.88 cm |
| 60 keV | 15 / 300 | **負 10・正 5** | 0.28〜14.53 cm |

深さ帯ごとの平均相対差（60 keV）: 0–0.5 cm −0.092%・0.5–1 cm −0.396%・1–2 cm −0.270%・
2–4 cm −0.067%・4–8 cm −0.157%・8–15 cm +0.004%。**どの帯も ±0.4% 以内で、符号も入れ替わる。**
20 keV も +0.1% から −0.15% の間で振れる。つまり**特定の深さに固まった一貫したずれは無く**、
300 回の検定で \|z\|>2 が 15 件というのは偶然の範囲（2σ で期待 14 件）。
\|z\|>3 の 3 件は期待 0.8 件より多いが、隣り合う深さ bin は相関があり、符号も両方に出ている。

### 実測した時間（固定費と周辺費用を分けて）

| | EGS5 | CPU の双子 |
|---|---:|---:|
| 20 keV | 193.7 s（周辺 6.01 µs/履歴＋固定 82.8 s） | 500.4 s（24.9 µs/履歴。build・表の生成・I/O を除く） |
| 60 keV | 333.1 s（周辺 10.74 µs/履歴＋固定 83.8 s） | 880.0 s（38.0 µs/履歴） |
| 合計 | **8.8 分** | **23.0 分** |

**H5 の計算は合計 32 分**で終わった（当初の見積もり 21.4 時間、狭めた後の見積もり 1.06 時間に対して）。
`seconds_per_history` を出すときは必ず固定費を分けること、という教訓はこの段で得たもの。

### まだ残っている H5 の項目

計画 H5 は「EGS5 2 点＋**入口経由の計算 2 本**（preview → ユーザーの 3D 承認 → approve → run）」。
**判定の部分（EGS5 対 CPU の双子）が合格**した。入口経由の 2 本はユーザーの 3D 承認の関門が入るので未着手。

### H5 の「入口経由の計算 2 本」の下調べ（2026-11-04、計算なし）

計画 H5 の残り。**判定の統計はすでに合格しているので、この 2 本は製品の経路が
低エネルギーの点で端から端まで通ることを示すためのもの**。入口は
`chatcarlo {validate|preview|approve|run} <scene>` で、`domain: therapy_mv`・`scoring: pdd_ocr` なら
`chatcarlo/gpu/therapy_entry.py` に入る（`__main__.py:514-519`）。

| 段 | 何が起きるか | 出るもの |
|---|---|---|
| `validate` | シーンの検査だけ | 「水 30 cm、10×10 cm²、SSD 100 cm、採点 ±20 cm」の確認文（`therapy_entry.py:98`） |
| `preview` | 条件を固めて 3D 表示を書く | `preview.html`＋`generation.json`。表示の見出しに線源・幾何・履歴数・種・engine・**条件の SHA256**（`therapy_entry.py:54`） |
| **ユーザーの承認** | 3D 表示を見て可否を判断 | —— |
| `approve --preview DIR` | preview の条件と今の条件が一致するか照合 | `approval.json` |
| `run` | 実行 | `results/dose.npz` |

**使える線源**: `source.energy_keV` が 20〜10000 keV で通る（`therapy_scene.py:15-20`）。
幾何は H5 と同一（水 30 cm・10×10 cm²・SSD 100 cm・P7 の ±20 cm 採点）。

#### 先に片付けないといけないこと 3 件

1. **20 keV では線源の bin が H5 と違う**。`a4_egs.mono_edges(20)` は
   **[19.9, 20.1] keV**（中心 20.0・幅 0.2）だが、入口は下端を 20 keV の床で切るので
   **[20.0, 20.1] keV**（中心 20.05・幅 0.1）になる（`therapy_scene.py:20` の `max(20.,e-.1)`）。
   60 keV は [59.9, 60.1] で**完全に一致**する。つまり 20 keV の入口の 1 本は
   **H5 の参照と同じ線源にならない**。扱いは 3 通り: (a) 入口の床を下げる、
   (b) H5 の点を 20.05 keV として扱い直す、(c) 20 keV の入口の 1 本は
   「経路が通ることの確認」だけを主張し、D7 の比較には使わないと明記する。
2. **既定の束が古い**。`therapy_inputs.DEFAULT_BUNDLE`
   （`~/.cache/chatcarlo/g5/a3/therapy_bundle`）は **2026-11-01** のもので、
   A4u の S2・表の 2 つの修正・打ち切りの 10 keV より前。入口はこれを読むので、
   **書き出し直さないと古い表と古い契約で走る**。H6 の T4 の前にも同じ作業が要る。
3. **`cpu_local` は履歴数が 20 万で上限**（`therapy_scene.py:65`）。H5 の N（2,013 万・2,316 万）は
   `colab_t4`（上限 2.4 億）でないと通らない。したがって**本番の N で入口を回すなら
   T4 の 1 セッション（H6）の中**になる。手元でできるのは 20 万履歴の経路の確認まで。

#### 床の挙動は (b) 受理したまま記録（ユーザー決定 2026-11-04）

`source.energy_keV` に**ちょうど 20 または ちょうど 10000** を書いたときだけ、
`max(20.,e-.1)`／`min(10000.,e+.1)` が bin を半分にし、中心を ±0.05 keV 動かす:

| 指定 | 実際の bin | 幅 | 中心のずれ |
|---|---|---:|---:|
| **20.0 keV** | 20.00〜20.10 keV | 0.10 keV | **+0.050 keV** |
| 20.1 keV 〜 9999.9 keV | 指定 ±0.1 keV | 0.20 keV | **±0** |
| **10000.0 keV** | 9999.90〜10000.00 keV | 0.10 keV | **−0.050 keV** |

**挙動は変えない**（ユーザー決定）。`therapy_scene.py:20` の直前にこの事実を書き、
A4u の 20 keV の判定は入口ではなく `a4u_h5_gate`（bin は 19.9〜20.1 keV）で行ったことも明記した。

**入口経由の計算は 60 keV の 1 本だけにする**（ユーザー決定）。20 keV が入口の経路で固有に踏むのは
この床の分岐だけで、それは上の表のとおり計算で特定済み。`cpu_local` の上限 20 万履歴を回しても
これ以上は分からない。20 keV の物理の検証は H5 で本番の N で合格している。

## H3a 診断側・CPU（`transport.py` 参照）の判定（Claude が実行、2026-11-04）: **合格**

計算は既存の 51 量の関門 `a4u_s1_gate1` をそのまま回した（3 シーン × 2 経路 × 10 回 × 100 万履歴。
s1 = 60 keV 単色の水、s2 = 120 kVp、s3 = 70 kVp）。**関門は `a4u_s1_gate1.json` を上書きするので、
S1 の記録は先に退避し、判定の後で元の名前へ戻した**。H3a の生の 61 量は
[a4u_h3a_gate1.json](a4u_h3a_gate1.json)、判定は [a4u_h3a.json](a4u_h3a.json)。

### (1) 51 量から D13 の 4 量を除いた判定 —— 合格

| 項目 | 値 |
|---|---|
| 記録した量 | 61 |
| z が定義できない量（両方とも 0） | 10（`s1:category:0:roi1`・`roi2`、`s1:category:3:*`、`s3:category:3:*`） |
| D13 の分岐の 4 量（除外） | `s2:category:3:all`／`roi0`／`roi1`／`roi2`。一覧のハッシュ `138de34f…` |
| **判定した量 K** | **47** |
| 最大 \|z\| | **2.108** |
| Σz²/K | 1.2147 |
| \|z\|>3 の量 | **0 件** |

D13 の 4 量の z は 1.49・0.24・0.89・1.55 だった。**これは「合格」と数えない**（D13 (ii) の決定どおり、
記録だけ）。この関門は同じ物理の組どうし（NumPy の `transport.py` 対 本番の双子）を比べるので、
D13 の分岐（K 殻のみ 対 全殻）はそもそもこの軸には現れない。分岐は H2 の旧 build 対 新 build の軸で見る。

**表の修正と 10 keV 打ち切りが効いていることの確認**: 参照側（`transport.py`）の平均は S1 の記録と
**61/61 が完全一致**（同じ seed・同じ実装なので動かないのが正しい）。一方、双子側は **61 量のうち 39 量が動いた**
（一致は 22 量）。つまり本番の物理は確かに変わっており、それでも独立な NumPy 参照と \|z\|≤3 で合っている。

### (2) 20／150 keV の一次透過率（D8 (4)） —— 合格

水 0.1 cm、平行ビーム、蛍光なし、100 万履歴、seed 9504006。解析解 exp(−μx) と比べた。

| エネルギー | μ（/cm） | 解析解 | 実測 | SEM | z | 相対差 |
|---|---|---|---|---|---|---|
| 20 keV | 0.80983 | 0.9222096 | 0.922168 | 2.678e-4 | **−0.16** | −0.0045% |
| 150 keV | 0.15052 | 0.9850603 | 0.984836 | 1.213e-4 | **−1.85** | −0.023% |

事前にあった基準（S2 の試験の 5σ+0.0003、および H3a の \|z\|≤3）の両方を満たす。
**ただしこれは輸送の仕組みの確認であって、断面積の確認ではない** —— 解析解の μ は表と同じ xraylib から取っている。

### (3) 二次スタックの故障注入 3 種（CPU） —— 既存の試験で達成済み

`tests/test_gpu_g5_a4u_s2.py::test_photon_ledger_escape_spectrum_and_faults` が
溢れ（fault 3）・二次粒子の取りこぼし（fault 4）・スロット再利用での状態の消え残り（fault 5）を
検出する。再実行して合格を確認した。T4 での 1 回は H6 に残る。

### (4) 保存済み 60 keV 水の EGS5 比較の分類（D8 (3)） —— **未了。並べる相手がまだ無い**

D8 (3) の「今の 37・6・4」は **GPU エンジンの** 60 keV 水 PDD/OCR の分類であって、
`transport.py`（NumPy）の 2026-10-07 の再検証とは別の記録である。混同しやすいので両方を並べる。

| 記録 | 経路 | 従来 2σ・2% 合格／不合格／SEM<1% 未達 | Bonferroni 補助 |
|---|---|---|---|
| [revalidation_2026_10_07](../../../egs5_crosscheck/revalidation_2026_10_07/comparison.json) | NumPy `transport.py`、カーマ近似・K 殻のみ | **42／1／4** | 43／0／4 |
| [a1_run/g2gate3](a1_run/g2gate3/RESULTS.md)（G5-A1、2026-10-30） | GPU エンジン（A1 の物理） | **37／6／4** | 42／1／4 |

**この 2 つはどちらも A4u より前の記録で、表の修正（2 件）と 10 keV 打ち切りを含まない。**
D8 (3) が求めているのは「今の物理で 60 keV 水の PDD/OCR をもう一度測り、旧 `compare.py` で分類して
A1 の 37・6・4 の隣に置く」ことであり、**それはまだやっていない**。既存の記録を引くだけでは満たせない。

必要な計算は `a1_g2gate3.py` の 3 条件（`pdd` 500 万履歴 seed 3101、`thin` 50 万 seed 3102、
`phantom` 50 万 seed 3103。合計 600 万履歴）で、EGS5 側は封印済みのため回さない。
**ユーザー決定 2026-11-04: H6 の T4 セッションに同梱する**（A1 の記録が T4 なので装置を揃える）。
計画の H6 の行にも書いた。手元の CPU 双子では回さない。

### 範囲

CPU だけ。NVRTC のコンパイル誤りは検出しない（H6 の T4 セッション）。D13 の 4 量を
「EGS5 で検証済み」とは呼ばない。


## 治療の束の書き出し直しが止まった（Claude、2026-11-04）

H6 の前提として `~/.cache/chatcarlo/g5/a3/therapy_bundle` を書き出し直そうとして、
**検証の段で止まった**。束そのものは書かれたが**検証を通っていない**ので、既定の場所には置かず
`therapy_bundle_unverified_2026-11-04` へ退避した（2026-11-01 の束は
`therapy_bundle_2026-11-01` に退避済み。既定の場所は空のまま——古い束は引退した名前
`therapy-electron-v1` を持つので、どちらにしても大きな音を立てて落ちる）。

### 止まった場所

`a2_bundle.run` の 45 行目:

```python
for f in catalog.P8_DEFINITIONS:rows[name+'.'+f]=equal(np.asarray(getattr(l,f),np.float64),getattr(r,f),name+'.'+f)
```

`AttributeError: 'types.SimpleNamespace' object has no attribute 'photon'`。
A4u が `P8_DEFINITIONS` に **34 番目の欄 `photon`**（統合した光子の表）を足したが、
左辺の `l` は旧 P8g2 の書き出しから読んだ材料で、その欄を持たない。
本番の `catalog.p8_material` は 297〜300 行で**この欄が無い材料を適合させる作り**になっているのに、
検証の輪だけが取り残されていた。`run` の報告は今も `fields_per_material=33` を書き込む。

**なぜテストが捕まえなかったか**: A2 の実データの束の往復の検査は、A3_LOCAL.md の
「G5 の 7 件の除外」に入っている（cache を読むためガードされている）。つまり**この経路は
`photon` が入って以降、一度も通っていない**。`a2_run/bundle.json` の差分も物理の組の名前を
書き換えただけで、`run()` が成功した形跡ではない。

### 全部の不一致を 1 回で洗い出した

`equal` を「例外を投げず記録する」形に差し替え、左辺を `catalog.p8_material` と同じ手順で
適合させて通した（記録は [a4u_bundle_reexport_probe.json](a4u_bundle_reexport_probe.json)）。

| 結果 | 件数 |
|---|---:|
| 一致した照合 | **212**（34 欄 × 4 材料、`photon` の往復を含む。入力の配列すべて、native の場・定数・ログ） |
| **不一致** | **2** |

不一致の 2 件はどちらも**旧 P8g2 試作との比較**で、A4u の意図した変更そのものだった:

| 照合 | 旧試作 | 新 | 差 |
|---|---:|---:|---:|
| `native.data`（packed blob） | 36,474,851 | 37,105,385 | **+630,534** |
| `native.meta`（記述子 4×136） | — | — | 上の分だけ offset がずれる |

**+630,534 は 4 材料の `photon` 表の大きさの合計（52,085＋324,378＋225,351＋28,720）と
ちょうど一致する**。つまり余分な数値の変化は無く、増えた分は統合した光子の表だけである。

### 置き換える契約（ユーザー指示 2026-11-04「バイト一致は不要、必要なものを満たしながら効率性を重視」）

バイト一致そのものを条件にするのはやめる。代わりに、**この照合が本当に守りたかったもの**を
名前で挙げて、それぞれを塊の切り出しで確かめる。守りたいのは
「移植で電子・MV の側を変えていない」ことであって、「blob 全体が同じ」ことではない。

`therapy.pack` の並びは `meta` に全部書いてある: `meta[mid,0:132]` が 33 欄の
(offset, size, 次元0, 次元1)、`meta[mid,132:134]` がログ領域、その後ろに 37 要素の詰め物と光子の表。
**33 欄は全材料ぶんが先に連続して置かれ、光子の塊は後ろにまとめて付く**ので、
前半の offset は光子を足しても動かない。だから次の 5 つで足りる。

| # | 契約 | 計算 |
|---|---|---|
| 1 | 33 欄の領域（先頭 **36,364,709** 要素）が旧 P8g2 と**バイト一致** | memcmp 1 回。電子・MV のデータが 1 バイトも動いていないことの証明 |
| 2 | `meta[:,:132]` が旧とバイト一致 | memcmp 1 回。33 欄の offset・大きさ・次元が不変 |
| 3 | 材料ごとにログ領域が全 NaN、37 要素の詰め物が全 0 | 走査のみ |
| 4 | 光子の塊が `ensure(material)` とバイト一致（**往復の検査**。表の中身の検証は H1 の仕事） | 既に手元にある配列の比較 |
| 5 | `meta[:,132:135]` が閉じた式に一致: `meta[mid+1,132] = meta[mid,133] + 37 + len(photon[mid])` | 整数の算術 |

**どれも新しい生成も輸送も要らない**（すでに作った配列を切って比べるだけ）。しかも
「blob が違う」ではなく**どの欄が違うか**が出るので、診断の役に立つ。

実測（[a4u_bundle_contract_probe.json](a4u_bundle_contract_probe.json)。**読み戻した束に対して**測った。
契約の 19 件はすべて `pass`、ただし最後の受け入れの関門だけ不合格——下の「封印の 3 件」）:

- 33 欄の領域: 新 36,364,709 = 旧 36,364,709、**バイト一致**
- `meta[:,:132]`: **バイト一致**
- 光子の往復: 4 材料すべて一致（52,085／324,378／225,351／28,720）
- offset の式: 材料 1〜3 の予測が実測と一致
- 全体の増分 630,534 = 光子の表の合計
- 追加で閉じた欄: ログの大きさが旧と同じ、`meta[:,134]` が 0,1,2,3、`meta[0,135]` の加数が旧と同じ 28,417

**契約 4 は最初、循環していた**。`therapy.pack` は光子の塊を `ensure(m)` で埋めるので、
それを `ensure(m)` と比べても必ず一致する。`ensure` は 234 行で
**`hasattr(m,'photon')` なら `validate(m.photon)` を返す**ので、読み戻した材料では
**封印された表がそのまま使われる**（docstring も「worker は束の中の `photon` を受け取り、
この経路に入らない」と書いている）。したがって T4 で xraylib は要らない。
契約 4 は「`therapy.pack(restored)` の光子の塊が `restored.photon` と一致」に直し、
測り直して 4 材料すべて一致した。

### 封印の 3 件が古い（`a3_export` の最後の関門が不合格）

`a3_export` は最後に `therapy_inputs.load(out, 'mohan10'/'mohan4')` を呼び、
`therapy_input_hashes.json` のピンと照合する。**ここが落ちた**。

| 結果 | 件数 |
|---|---:|
| 照合したピン | 184 |
| 一致 | **181** |
| **不一致** | **3** |

| ピン | 形（ピン） | 形（今） | 中身 |
|---|---:|---:|---|
| `lead.photon` | 28,688 | **28,720**（+32） | 変化 |
| `bone.photon` | 225,351 | 225,351 | 変化 |
| `lung.photon` | 324,378 | 324,378 | 変化 |
| `water.photon` | — | — | **一致（変化なし）** |

これは**表が壊れたのではなく、ピンが古い**。ピンは A2b（commit `ffe20b9`）の時点のもので、
H3b で直した表の 2 つの欠陥（まとめた準位の正規表現・吸収端の節）より前である。
内訳も物理と合っている: **鉛は両方の欠陥がまさに刺さった材料**で要素数が増え（節が増えた形）、
骨と肺は吸収端の節が動いた分だけ中身が変わり、**水は K・L 吸収端も 5 keV 以上の線も無いので
どちらの修正も触らず、一致したまま**。鉛の +32 が節の増加だけで説明できるかは数えていない
（形の向きが一致する、までが言えること）。

ピンは封印の記録なので、更新は Codex に回す（計画の範囲外の修正の表に含める）。

あわせて直すもの: `a2_bundle.run` の 45 行目は `catalog.p8_material` と同じ手順で左辺を
適合させる（34 欄すべてを比べられるようにする）。報告の `fields_per_material=33` は 34 にし、
「34 番目の `photon` は往復のみ」と書く。

### 位置づけ

「packed blob が旧 P8g2 試作とバイト一致」という契約は、**A4u では成り立たないのが正しい**
（光子の表を足したのだから）。上の 5 つはその置き換えで、守る対象を狭めるのではなく
**名前を付けて切り分けた**ものである。表の中身そのものの検証は H1 と
`a4u_h4_table_hash_fix.json`（`prepare_cpu(20)` 対 `therapy.pack` の 136 鍵・4 材料）が受け持ち、
ここは「封印された入力が 目録 → ディスク → native まで壊れずに届くか」だけを見る。

`a2_bundle.py` は「束が何に対して検証されるか」を決める場所なので、**実装は Codex に回す**
（計画の範囲外の修正として表に足す。Claude は契約の設計と独立検証に回る）。


## 束の照合の作り直しと書き出し直し（Codex 実装・Claude 独立検証、2026-11-04）

### Codex が入れた変更（2 ファイル）

| ファイル | 変更 |
|---|---|
| `docs/validation/gpu/g5/a2_bundle.py` | +21 −3。45 行目の左辺を適合させて 34 欄を照合。`native.data`／`native.meta` のバイト一致 2 件を**18 件の切り分けた照合**に置換。報告を `fields_per_material=34` ＋「34 番目は往復のみ、中身は H1」に |
| `chatcarlo/gpu/therapy_input_hashes.json` | `lead.photon`（形も 28,688→28,720）・`bone.photon`・`lung.photon` の 3 件を更新。`water.photon`・`seed_agti.photon` は触っていない |

`chatcarlo/gpu/therapy.py` の `pack` には触れていない（物理にも束の中身にも変更なし）。

### Claude の独立検証

**(1) 受入条件の再実行**（Codex の報告をなぞらず、自分で走らせた）:

| 項目 | 結果 |
|---|---|
| 行の照合 | **215 件すべて一致**（34 欄×4 材料、入力の配列、native 18 件、ログ） |
| 封印のピン | **184 件すべて一致** |
| `therapy_inputs.load` | `mohan10`・`mohan4` とも成功 |

**(2) 誤りで本当に落ちるかの試験**（新しい照合に的を絞り、`therapy.pack` の出力を 1 か所ずつ壊した）:

| 壊した所 | 検出 |
|---|---|
| 33 欄の領域の 1 要素を 1e-12 相対で動かす | **検出**「33-field data region」 |
| 肺の光子の塊の 1 要素に 1e-9 足す | **検出**「native.lung sealed photon roundtrip」 |
| 37 要素の詰め物に 1 を書く | **検出**「native.lung zero padding」 |
| `meta[:,134]` を壊す | **未試験**（空き容量の番人に当たった。`arange` との比較なので危険は小さいが、確かめていない） |

束を読み戻す側を壊した試験（光子・`stop_T`）も、34 欄の輪が先に検出した。

### 書き出し直した束

検証を通した束をそのまま既定の場所へ移した（同じ装置内の移動なので書き直していない）。
置いた後にもう一度確かめた:

- `manifest.json` の 4 ファイルのハッシュ: **一致**
- 物理の組: `chatcarlo-photon-electron-v1` / `electron-transport`（引退した `therapy-electron-v1` ではない）
- `therapy_inputs.load`: `mohan10`・`mohan4` とも成功、ピン 184 件一致
- 光子の表の大きさ: 水 52,085／肺 324,378／骨 225,351／**鉛 28,720**
- `manifest.json` の**ファイルのバイトの** SHA256: `20f3e762…`
- 入口の条件に載る `bundle_manifest_sha256`（`contracts.sha256` による**辞書の正規形の**digest）:
  `9f54db5a996dc6822611d7fd53de9b21da8fd4b1296c3f6f25b0eea9e49dc119`。
  **H7 が照合するのはこちら**（2 つは別の値で、同じ束を指す）
- 物理の組の tag: **6882153074982124678**

**識別子のずれは無い**: H4（`a4u_h4.json`）・H5（`a4u_h5.json`）・`a4u_d2_rebuild.json` も
同じ tag **6882153074982124678** を持つ。追跡下の `a2_run/bundle.json` だけが
古い tag `11068429372807448304` を載せている（A2 の記録なので作り直さない。
**古いことをここに記録して済ませる**）。H7 の識別子の照合はこの束で通る。

合格した受入の記録は [a4u_bundle_reexport_acceptance.json](a4u_bundle_reexport_acceptance.json)（215 行）。
[a4u_bundle_contract_probe.json](a4u_bundle_contract_probe.json) は**修正前**の探りで、
`status: fail` は封印のピンが古かったことを記録したものである（契約の 19 件自体は合格）。

2026-11-01 の束は `therapy_bundle_2026-11-01` に残してある。検証を通らなかった束と
確認用の束は `~/.Trash/a4u_scratch_bundles_2026-11-04/` へ移した（空にするのはユーザー）。
**空き容量が 3.0 GiB しかない**ので、次に束を作る作業の前にゴミ箱を空にしてもらう必要がある。


### 全テストの再実行（Claude、2026-11-04〜05）: **1031 passed、回帰 0 件**

1 回目は空き容量 3.0 GiB で走らせたため **992 passed、25 failed、14 errors**（283 秒）になった。
39 件の内訳は:

| 分類 | 件数 | 中身 |
|---|---:|---|
| **空き容量の番人** | **38** | `A2 disk space would fall below 3 GiB`／`A1 disk space below 3 GiB`。束を作る検査（a1_drivers・a2_bundle・a2_core・a2b・a2_judge・a3_native・a4u_s1/s2 の一部） |
| **時間に敏感な検査のぶれ** | **1** | `test_fake_colab_timeout_retry`（0.7 秒の間に 3 回再試行するはずが 2 回。`--timeout -29.3` と期限切れ）。**単独では合格**。Colab の再試行の検査で今回の変更とは無関係 |
| **本当の回帰** | **0** | |

992+39 = 1031 で件数は増減していない。ただし**番人に当たった 38 件は「落ちた」のではなく
「走っていない」**ので、合否は保留にした。

ユーザーがゴミ箱を空にして空き容量が **6.3 GiB** になった後に走らせ直した結果:

```
1031 passed, 2 warnings in 328.46s (0:05:28)
```

**`disk space` の文字列は 0 件**、`FAILED`／`ERROR` も 0 件。
束を作る 38 件が今度は実際に走って合格した。Codex の報告（1031 passed）と一致する。
**今回の変更による回帰は無い。**


## 入口経由の 60 keV（1 本だけ。ユーザー決定 2026-11-04、承認・実行 2026-11-05）

シーンは [examples/therapy/water_60kev_10x10.yaml](../../../../examples/therapy/water_60kev_10x10.yaml)。

| 項目 | 値 |
|---|---|
| 線源 | 単色 60 keV → bin **[59.9, 60.1] keV**。`a4_egs.mono_edges(60)` と**完全に一致**（20 keV は床で中心がずれるので入口では回さない） |
| 幾何 | 水 30 cm、10×10 cm²、SSD 100 cm、採点 ±20 cm（H5 と同一） |
| 履歴数 | 200,000（`cpu_local` の上限）。**統計の判定ではなく経路の確認** |
| seed | 20261104 |
| 物理の組 | `chatcarlo-photon-electron-v1` / `electron-transport`、tag **6882153074982124678**（H4・H5 と同じ） |
| 束 | `bundle_manifest_sha256` **9f54db5a…**（今日書き出し直して検証した束） |

`validate`・`preview` を経てユーザーが 3D 表示を見て**承認（2026-11-05）**、`approve`・`run` を実行した。
28 秒で完走。条件 SHA256 と数値は [a4u_entry60.json](a4u_entry60.json)。

### 結果: **経路は端から端まで通った**

比べたのは **field 2**（offset 420、120 bin × 0.25 cm、x,y が ±∞ = **スラブ全体**の深さ曲線）。
H5 の gate の `fine`（600 bin × 0.05 cm、スラブ全体）を 5 bin ずつまとめて 0.25 cm に揃えた。

| 量 | 入口（20 万履歴） | H5 CPU（2,316 万） | H5 EGS5 | 入口/CPU | z |
|---|---|---|---|---|---|
| ∫5–15 cm / ∫0–5 cm | **1.13134 ± 0.00506** | 1.12590 | 1.12519 | **+0.48%** | **+1.08** |
| ∫15–30 cm / ∫0–5 cm | **0.36663 ± 0.00266** | 0.36587 | 0.36544 | **+0.21%** | **+0.29** |

絶対値の比（入口/H5 CPU）は 0〜15 cm で **1.9% 以内に一定**で、これは単位の違い
（MeV 対 Gy cm²/history）による定数倍である。bin ごとの相対 SEM は中央値 1.64%・最大 2.42%。
**20 万履歴でも深さ依存性を 0.5% 程度で再現**した（スラブ全体の柱なので統計が効く）。

### 採点器をコードで確定させた（数値を見てから選んだのではない）

同じ 2,000 履歴を 1 回走らせ、**同じ履歴を 2 通りに採点した結果**を比べた:

| 比較 | 最大 \|比−1\| | 合計 |
|---|---|---|
| `groups[:,450:1050]` を 5 bin ずつ ÷ **field 2**（スラブ全体） | **2.8e-15**（和の丸めだけ） | 88.453099 / 88.453099 |
| `groups[:,450:1050]` を 2 bin ずつ ÷ **field 1**（2×2 cm² の柱） | 2.1e+03（比の平均 129） | 88.453099 / 1.815139 |

したがって **H5 の `fine` はスラブ全体を横方向に積分した深さ曲線**であることが確定した
（`a4u_h4_gate.py:260`。gate の幾何は `faces=[0,30]` の 1 次元スラブで横方向の境界が無い）。
中心の柱が受け取るのは全体の 2.1% にすぎない。

### 絶対値の規格化も合っている（窓の比だけでは履歴数の取り違えを検出できない）

- 利用者が読む `absorbed_dose_Gy` は `field_MeV.sum(0)×1.602176634e-10/質量/N` から
  **ビット一致で再現**した（最大相対差 0）。
- 入口の MeV/batch と H5 の Gy cm²/history の間の定数倍は、観測 **3.12617e12** に対し
  予測 **3.12075e12**（= 2,000 履歴/batch × 0.25 g ÷ 1.602176634e-10）で、
  **+0.173%、z=+0.70**。比だけでなく絶対値で合っている。
- なお **field 2 の 120 bin は横方向に境界が無く、使う質量が 0.25 g/cm² なので
  実際の単位は Gy cm²/history** である。metadata は `absorbed_dose_Gy` 全体を
  Gy/history と書いており、ここだけ食い違う。A3 からの持ち越しで、今回は**記録だけ**とする。

### ⚠️ 最初の対応づけを間違えた

はじめ **field 1**（offset 120、2×2 cm² の細い中心の柱、300 bin × 0.1 cm）を H5 の `fine` に
対応させ、「∫5–15/∫0–5 が **−31%**（z=−16）ずれる」と出した。**採点器の取り違えである。**
H5 の gate の `fine` は `r['groups'][:,450:1050]`（[a4u_h4_gate.py:260](a4u_h4_gate.py)）、
すなわち**スラブ全体**を横方向に積分した深さ曲線で、入口が書き出す `r['field']` の
P7 の 3420 bin とは**別の採点器**である。細い柱は散乱を取りこぼすので深さ方向の落ち方が
速くなるのが当然で、−31% は欠陥ではなく別の量を比べていたことの帰結だった。
採点器を field 2 に取り替えて一致した。

なお 0.12 cm → 0.88 cm で線量が上がる（立ち上がり）のは入口の作り出したものではない:
H5 の CPU・EGS5 の曲線も同じ形（5.52 → 5.95 ×10⁻¹³）を持つ。

### 範囲

**経路の確認のみ**。20 万履歴は `cpu_local` の上限で、本番の N（2,316 万）での入口は
H6 の T4 セッション。**H5 の判定を置き換えるものではない**。20 keV は入口では回していない。


## ⚠️ H5 が判定したのは中心軸の PDD ではなかった（2026-11-05 判明）

上の採点器の確定から出てきた、**言い直しでは済まない話**。

ユーザーの指示は「PDD だけ」だった。しかし H5 が EGS5 と照合した「深さ方向の線量曲線」は
`groups[:,450:1050]`、すなわち**横方向に積分したスラブ全体の深さ線量**である。
製品が PDD として出すのは**中心軸（2×2 cm² の柱、field 0・field 1）**で、これは別の量である。
60 keV で両者の ∫5–15 cm / ∫0–5 cm は **1.126（スラブ全体）対 0.789（中心軸）**と
**約 30% 違う**。つまり**利用者が読む中心軸の PDD は、低エネルギーでは EGS5 と照合されていない**。

### ただし EGS5 の再計算は要らない

保存済みの EGS5 の結果（`a4_run/egs5/h4_pilot_60kev/result.npz`）には
**`axis_fine` (100, 300)**（2×2 cm² の柱・0.1 cm bin）と **`axis_coarse` (100, 120)** が
**すでに入っている**。EGS5 は `regions=h.layout('p7')['regions']` で全領域を採点していた
（`a4_egs.py:94`）。だから中心軸の判定に**新しい EGS5 は 1 本も要らない**。

### EGS5 側も名前と形で決めず、npz の内部整合で確かめた

| 検査 | 結果 |
|---|---|
| `axis_fine` を 0.5 cm に足した値 ÷ `axis_coarse` を 0.5 cm に足した値 | **1.00000000（最大 \|比−1\| 2e-15）** → 同じ柱の、bin 幅だけ違う採点 |
| `fine` を 5 bin 足した値 ÷ `slab_coarse` | **定数**（std/mean 4.9e-14）→ どちらもスラブ全体の同じ量、違いは規格化だけ |
| `axis_coarse` ÷ `slab_coarse`（60 keV） | 0–5 cm **0.0290**、5–10 cm 0.0217、10–15 cm **0.0167** → 細い柱で、深さとともに割合が減る（散乱が広がる） |

ChatCarlo 側で測った「中心の柱はスラブ全体の 2.1%」と整合する。新しい計算は要らなかった。

### 今ある数字での最初の照合（入口 20 万 対 EGS5 の試し計算 100 万）

| 量 | 入口（20 万） | EGS5（100 万） | 差 | z |
|---|---|---|---|---|
| ∫5–15 cm / ∫0–5 cm | 0.77600 ± 0.02130 | 0.78872 ± 0.00931 | −1.61% | **−0.55** |
| ∫15–30 cm / ∫0–5 cm | 0.17092 ± 0.00825 | 0.15365 ± 0.00316 | +11.24% | **+1.95** |

5–15 cm は食い違いが見えないが、**15–30 cm の z=+1.95 は「見えない」と言える値ではない**。
どちらも統計が薄い（入口の bin ごと相対 SEM は中央値 17%、EGS5 の**試し計算**は 7.7%）ので、
これは合格でも不合格でもなく**判定保留**である。本番の統計で判定するまで結論は出せない。
絶対値の比 0.1988 は履歴数の比 0.2 に対し −0.6% で、こちらは整合する。

### 本番の EGS5 で中心軸を判定したら何 bin 使えるか（実測）

H5 が参照にした**本番の** EGS5（`h5_main_20kev` N=18,438,400／`h5_main_60kev` N=23,216,500）にも
`axis_fine`・`axis_coarse` が入っている。D7 の「両コードの相対 SEM ≤1%」を EGS5 側に当てた数:

| 配列 | bin 幅 | 20 keV（≤15 cm） | 60 keV（≤15 cm） |
|---|---|---|---|
| `axis_fine`（中心軸） | 0.1 cm | 中央値 6.38%、**24/150** | 中央値 1.61%、**4/150** |
| **`axis_coarse`（中心軸）** | 0.25 cm | 中央値 4.34%、**15/60** | 中央値 1.01%、**29/60** |
| `fine`（スラブ全体＝H5 が判定した量） | 0.05 cm | 中央値 1.91%、119/300 | 中央値 0.33%、**300/300** |

**これが「なぜスラブ全体で判定したのか」の答えでもある**: 中心の柱は全体の 2〜3% しか
受け取らないので、同じ N では SEM が約 7 倍悪い。60 keV のスラブ全体は 300/300 が
評価可能だったが、中心軸の細い bin は 4/150 しか評価できない。

したがって**中心軸を判定するなら 0.25 cm の `axis_coarse`**で、
**60 keV は約 29 bin、20 keV は約 15 bin**（≤15 cm）。EGS5 の追加は **0 本**。

### 直した記述

- [.claude/rules/chatcarlo.md](../../../../.claude/rules/chatcarlo.md)
- [vive-interview/SKILL.md](../../../../.claude/skills/vive-interview/SKILL.md)
- 計画の「範囲外で入れた修正」の「PDD だけ」の行
- 下の H5 の節と `a4u_entry60.json` の `scope`

### ChatCarlo 側の費用（実測から）

H5 の gate は `fine` と `profile_fine` だけ保存したので、中心軸には **CPU の回し直しが 1 回**要る。
H5 の実測は native の実時間で **20 keV 500.4 秒（N=20,127,000）・60 keV 880.0 秒（N=23,164,000）**、
合わせて **約 23 分**。同じ seed・同じ N で `field` も保存する形にすれば、
**スラブ全体の値が H5 と完全に一致するはず**で、再現性の検査が無料で付く。

ChatCarlo 側の中心軸の SEM も、入口の 20 万履歴の実測（field 1 で 17%）から
23,164,000 履歴へ伸ばすと 0.1 cm で約 1.6%、0.25 cm で約 1.0% になり、EGS5 側と同程度である。

### 選べる道

| | 内容 | 費用 |
|---|---|---|
| **(a)** | 中心軸の PDD を**本番の EGS5（保存済み）＋ChatCarlo の CPU 1 回**で判定する。0.25 cm bin、60 keV 約 29 bin・20 keV 約 15 bin（≤15 cm） | **EGS5 0 本＋CPU 約 23 分**。H5 のスラブ全体の値との完全一致も確認できる |
| **(b)** | 中心軸の PDD は「低エネルギーでは未照合」と記録に残し、文言を直すだけにする | **0** |

**判断はユーザー**。(a) は安く、利用者が読む量そのものを押さえられる。


### 入口経由 60 keV の計算データの扱い（段階を閉じるときの表）

| 出力 | 大きさ | 行き先 |
|---|---:|---|
| `condition.json`・`approval.json`・`generation.json` | 20 KiB | **リポジトリ**（[a4u_entry60_run/](a4u_entry60_run/)）。条件 SHA256 `a1a86f6a…` の裏付け |
| 深さ曲線 3 本＋`absorbed_dose_Gy`（`depth_curves.npz`） | 248 KiB | **リポジトリ**。中心軸（粗・細）とスラブ全体、記録が引く数値はすべてここから出る |
| batch 単位の生データ 100 本（`results/batch_*.npz`） | 68 MB | **消す**（G5-A3 でユーザーが決めた「生データは複製せず消す」に従う）。再計算は 28 秒 |
| 束の複製（`bundle/`） | 282 MB | **消す**。正典は `~/.cache/chatcarlo/g5/a3/therapy_bundle`、`bundle_manifest_sha256` で同一性が追える |
| `preview.html` | 20 KiB | 消す（承認の事実は `approval.json` に残る） |


## H6（T4 の 1 セッション）の内訳と時間（1 時間超なので先に出す。2026-11-05）

**実測の数字だけを使う**（試し計算の `wall/N` を周辺費用ごと掛け算して 9〜15 倍外した
2026-11-04 の失敗をくり返さないため）。

| 項目 | 根拠（実測） | 時間 |
|---|---|---:|
| 10 MV 入口（4,600 万履歴） | A3 で **581.8 秒**（キャッシュあり 579.1 秒）。T4、12.6 µs/履歴（周辺費用込み） | **9.7 分** |
| 4 MV 入口（4,600 万履歴） | 同じ経路 | **約 9.7 分** |
| `seed_ratio` | A2b で **116.2 分** | **1.94 時間** |
| 低エネルギーの小試験（20・60 keV 各 10 万、CPU 対 T4） | CPU 側は H5 の実測 24.9／38.0 µs/履歴 → 2.5／3.8 秒。T4 側は更に小さい | **約 1 分** |
| **実測で積める小計** | | **約 2.3 時間** |
| D8 (3)（`pdd` 500 万・`thin` 50 万・`phantom` 50 万） | T4 の診断は 0.59〜2.33 µs/履歴 → 核の時間は 3.5〜14 秒。**支配するのは 3 本ぶんの固定費で、それは測っていない** | 核は無視できる。固定費**未測定** |
| G4 の画像・G2 のボクセル（CPU 対 T4） | 既存の関門。**別立てで測っていない** | **未測定**（小さい） |
| 故障注入 1 回・D10 の速さの段 | 同上 | **未測定**（小さい） |

**つまり約 2.3 時間＋（未測定の小さい項目と VM の立ち上げ）**。
`seed_ratio` の 1.94 時間が実測分の **84%** を占める。これは D9（MV の作り直しは
A3 の 10 MV・4 MV と A2b の `seed_ratio`）で要求されている本数なので、削るなら
計画の対象範囲に戻る話になる。

固定費を測っていない項目は、**最初に D8 (3) の `thin`（50 万履歴）を 1 本流して
VM 1 本あたりの固定費を実測し**、それを使って残りを見積もり直すのが安い。



### 訂正（2026-11-05、着手前の前提確認で発覚）: MV 入口の行は履歴数を 10 倍間違えていた

上の表の「10 MV 入口（4,600 万履歴）581.8 秒・12.6 µs/履歴」は**誤り**。原本を読み直した。

| 事実 | 出典 |
|---|---|
| 581.8 秒 が測ったのは **4,600,000 履歴**（460 万） | `a3_entry_speed.py:12` の `HISTORIES = 4_600_000`、`a3_run/t4/f/entry_speed.json` の `new_cold.histories = 4600000`、`main_s = 581.675` |
| したがって **126.4 µs/履歴**（周辺費用込み）。12.6 µs は 10 倍の誤り | 同上 |
| H6 で回すのは**速さの 460 万ではなく本番の N**: 10 MV **46,000,000** seed 20261101、4 MV **26,000,000** seed 20261102 | `a3_judge.py:15` の `EXPECTED` |
| A3 の「H5 統計」は MV 本番ではなく**診断の 51 量（100 万履歴 × 10 回）** | `a3_run/t4/d/H5_g1.json` の `n_histories: 1000000`, `runs: 10` |

126.4 µs/履歴で換算すると **10 MV 46,000,000 → 約 1.61 時間、4 MV 26,000,000 → 約 0.91 時間**。
`seed_ratio` の 1.94 時間と足すと **H6 の実測分は約 2.3 時間ではなく約 4.5 時間**になる。

**ただしこの 126.4 µs/履歴は入口 1 本ぜんぶ（コンパイル約 16 秒・通信・採点を含む）を 460 万で割った値**なので、
履歴数が 10 倍になれば固定費の割合は下がる。つまり 4.5 時間は**上限側の見積もり**である。
T4 で 46,000,000・26,000,000 を実測した記録は見つかっていない（A3 では回していない）ため、
**核の時間と固定費を分けた実測はまだ無い**。
したがって「`thin` 50 万を 1 本流して VM 1 本あたりの固定費を測る」案は、
**MV の時間の下限を決めるためにも必要**だった（当初は D8 (3) のためだけと書いていた）。

この誤りは 2026-11-05 の教訓 1390（精度・費用の条件を、届くか見積もらずに書いた）と同型である
——**実測の秒数を、それが何履歴に対する秒数かを確かめずに別の履歴数の行に書いた**。

## 中心軸 PDD の判定（ユーザー決定 (a)、2026-11-05）: **統計不足**。不一致は検出されず

事前登録は [A4U_PDD_PREREG.md](A4U_PDD_PREREG.md)（ChatCarlo 側の値を 1 つも計算する前に書いた）。
道具は Codex 実装の [a4u_pdd_run.py](a4u_pdd_run.py)（H5 の `run_cpu` に `raw_field[:540]` の保存を足しただけ）と
[a4u_pdd_gate.py](a4u_pdd_gate.py)（120 bin の registration のみ新規、判定は `a4u_h4_gate.analyse` をそのまま呼ぶ）。
結果は [a4u_pdd.json](a4u_pdd.json)。

### 無料の検査は通った: スラブ全体が H5 とビット一致

同じ seed・N・chunk で回し直した `fine`（スラブ全体）が、**両エネルギーで H5 の記録と完全一致**
（`fine_matches_H5: true`）。実装も環境も H5 当時から変わっていない。

### 判定

| | 20 keV | 60 keV |
|---|---|---|
| K（山の 5% 以上・≤15 cm） | **17** | **60** |
| z_crit = Φ⁻¹(1−0.05/2K) | 2.9738 | 3.3415 |
| 合格 | 11 | 8 |
| **不合格** | **0** | **0** |
| 統計不足 | 6 | 52 |
| 必要な評価可能数 max(8, 0.8K) | 14 | 48 |
| 必須 bin が全部評価可能 | はい | **いいえ** |
| 打ち切り bin | 0 | 0 |
| 歴史的 2σ の不合格 | 0 | 0 |
| 最大 \|z\| | 2.01 | 2.75 |
| **判定** | **統計不足** | **統計不足** |

**不一致は 1 本も検出されていない**（最大 \|z\| が閾値の 2/3〜4/5、歴史的 2σ でも 0 件）。
しかし事前登録した網羅の条件（評価可能が max(8, 0.8K) 以上、必須 bin は全部評価可能）を
満たさないので、**合格とは呼ばない**。事前登録どおり**履歴数を増やさず、ROI も狭めない**。

### なぜ足りなかったか —— 私の見積もりが間違っていた

| | 20 keV | 60 keV |
|---|---|---|
| EGS5 側の相対 SEM（中央値） | 0.583% | **1.013%** |
| ChatCarlo 側（中央値） | 0.600% | 1.002% |
| **合成**（hypot） | 0.821% | **1.407%** |
| 合成 ≤1% の本数 | 11/17 | **8/60** |

**⚠️ 私は (a) の費用を「EGS5 0 本＋CPU 約 23 分で 60 keV 約 29 bin」と見積もったが、これは誤り。**
EGS5 側**だけ**の相対 SEM ≤1% の本数（29/60）を引いたが、D7 の 1% は
**合成 SEM に掛かる**（`a4u_h4_gate.analyse` の `enough=(r>0)&(sem<=SEM_LIMIT*r)`、`sem=hypot(sr,st)`）。
60 keV は **EGS5 側だけで中央値 1.013%** なので、合成が 1% に届く余地がほとんど無かった。
このコードは事前登録を書く前に読んでいた。片側の本数をそのまま使ったのが誤りである。

### 本当に判定するのに要る計算（実測から換算）

合成を K の 80% の bin で ≤1% にするには、各辺を 0.707% 以下にする必要がある。
F13 の実測（EGS5 6.0／10.7 µs/履歴＋固定 83 秒、CPU 24.86／37.99 µs/履歴）で換算:

| | 必要な倍率 | 必要な履歴数 | 時間 |
|---|---|---|---|
| 20 keV EGS5 | 1.91 | 35,151,790（今 18,438,400） | 約 0.1 時間 |
| 20 keV CPU | 1.59 | 31,968,835（今 20,127,000） | 約 0.2 時間 |
| 60 keV EGS5 | 4.33 | 100,511,582（今 23,216,500） | 約 0.3 時間 |
| 60 keV CPU | 4.22 | 97,815,246（今 23,164,000） | 約 1.0 時間 |
| **合計** | | | **EGS5 約 0.4＋CPU 約 1.3 = 約 1.7 時間** |

**ただしこれは EGS5 を 2 本増やす**ので、D8 の「新規の EGS5 は回さない」からは外れる
（H5 の参照も作り直しになる。保存済みの参照は残し、別の封印として足す形になる）。

### 補助の検査（事前登録に無い。合否は変えない）

bin ごとの検定は合成 SEM 約 1.4%・z_crit 3.34 なので、**1 本あたり約 4% 未満の差は検出できない**。
そこで窓の積分と bin をまとめた z で、系統的なずれが無いかを見た。

| | 20 keV | 60 keV |
|---|---|---|
| ∫5–15/∫0–5（EGS5 / CPU） | 0.02589 / 0.02590 | 0.77704 / 0.78119 |
| 　相対差・z | +0.04%・**+0.04** | +0.53%・**+1.57** |
| ∫15–30/∫0–5 の相対差・z | （線量ほぼ 0 で無意味） | +0.36%・**+0.51** |
| bin ごと z の平均 | −0.139 | +0.090 |
| **まとめた z**（平均×√K） | **−0.57** | **+0.70** |
| 符号の内訳 | +9／−8 | +33／−27 |
| 深さに対する z の傾き | −0.118 /cm | **+0.024 /cm** |

**系統差は +0.054% ± 0.165%（1σ）、2σ で ±0.33% の範囲内**である（60 keV、pooled z=+0.70、符号も半々、傾きも無い。導出は下の「補助の検査」の訂正を参照）。
bin ごとの「不合格 0」より強い言い方ができる。

**入口の 20 万履歴で判定保留にした 15–30 cm の +11.24%（z=+1.95）は、本番の統計では
+0.36%（z=+0.51）だった** —— 雑音だったことが確認できた。

### 時間の実測（見積もりを手元の走りで検算した）

| | 今回 | H5 の実測 | 比 | µs/履歴 |
|---|---|---|---|---|
| 20 keV | **488.4 秒** | 500.4 秒 | 0.976 | 24.27 |
| 60 keV | **843.4 秒** | 880.0 秒 | 0.958 | 36.41 |

この実測で換算し直すと、判定に要る CPU は **20 keV 0.22＋60 keV 0.99 = 約 1.21 時間**
（F13 からの見積もり 1.3 時間とほぼ一致）。EGS5 は約 0.4 時間で、**合計 約 1.6 時間**。

### 条件のハッシュ（H7 のため）

この走りの条件 SHA256 は 20 keV `88cd9887…`、60 keV `12b32104…` で、**H5 の封印とは 1 欄だけ違う**:
`sources.photon_physics.h`（H5 `a9c89e4b…` / 今 `161bdd9b…`）。前日にそのファイルの**コメントを
直した**ためで、batch 0 の再計算が H5 の batch 0 と**ビット一致**し、100 batch 全体でも `fine` が
H5 と完全一致した。**物理は変わっていない。**

### 計算データの扱い

| 出力 | 大きさ | 行き先 |
|---|---:|---|
| `raw_field[0:540]`＋`fine` の 100 batch 分（2 エネルギー） | 1.3 MB | **リポジトリ**（[a4u_pdd_run/](a4u_pdd_run/)）。値の保全用。**⚠️ 訂正: これだけでは判定し直せない** —— `read_cpu` は `condition.json`＋`batch_NNN.npz`＋receipt を要求し、この圧縮 npz を読めない |
| batch の npz と receipt（`pdd_out` 6.9 MB＋`pdd2_out` 3.6 MB） | 10.5 MB | **消す**（ユーザー決定 2026-11-05「消して再計算」。`~/.Trash/a4u_pdd_batches_deleted_2026-11-05/` へ退避、空にするのはユーザー）。**判定し直すには再計算が要る**: 60 keV は `a4u_pdd_run`（H5 と同じ N）で約 14 分、20 keV は `--cases pdd --energy 20` で約 19 分。どちらも seed・N は記録済みなので同じ値が出る |
| EGS5 `pdd_60kev`（150,767,400 履歴） | 3.4 MB | **消す**（ユーザー決定 2026-11-05。`~/.Trash/a4u_pdd_60kev_deleted_2026-11-05/` へ退避、空にするのはユーザー）。1.0 cm の判定には使わなかった。60 keV を 0.25 cm で判定し直すなら EGS5 から回し直す |
| 条件の封印 2 本 | 15 KiB | **リポジトリ**（同上） |
| batch の npz と receipt 200 組 | 6.9 MB | **消す**（G5 の前例「生データは複製せず消す」。上の npz に中身は入っている） |

### 言えること・言えないこと

- **言える**: 中心軸 PDD で ChatCarlo と EGS5 の食い違いは、この精度では検出されない
  （20 keV で 11 bin、60 keV で 8 bin が D7 の精度で合格、不合格 0）。
  スラブ全体の判定（H5 合格）は有効のまま。
- **言えない**: 「中心軸 PDD は EGS5 と照合して合格」。網羅の条件を満たしていない。
  利用者が読む PDD は**依然として低エネルギーでは未照合**である。


## 中心軸 PDD の判定（最終、2026-11-05）: 20 keV は合格、60 keV は粗くすれば合格

記録は [a4u_pdd_final.json](a4u_pdd_final.json)。事前登録は
[A4U_PDD_PREREG.md](A4U_PDD_PREREG.md)（本文＋追補 1＋追補 2＋矛盾の訂正）。
**3 つを必ず並べて読む。**

| | 20 keV | 60 keV（0.25 cm） | 60 keV（1.0 cm） |
|---|---|---|---|
| bin 幅 | 0.25 cm | 0.25 cm | **1.0 cm** |
| データ | **履歴数を増やした**（EGS5 52,727,700／CPU 47,953,300） | H5 と同じ N | H5 と同じ N（**追加計算 0**） |
| 判定する深さ | 0.12〜**4.12 cm** | 0.12〜14.88 cm | 0.50〜**14.50 cm** |
| K | 17 | 60 | 15 |
| z_crit | 2.9738 | 3.3415 | 2.9352 |
| 合格 | **15** | 8 | **12** |
| **不合格** | **0** | **0** | **0** |
| 統計不足 | 2 | 52 | 3 |
| 必要な評価可能数 | 14 | 48 | 12（**ちょうど**） |
| 歴史的 2σ の不合格 | 0 | 0 | **1 件** |
| 最大 \|z\| | 2.07 | 2.75 | 2.19 |
| 最大の相対差 | 1.792% | 5.527% | 2.056% |
| **判定** | **合格** | **統計不足** | **合格（後出し）** |
| 性質 | **事前登録どおり。後出しなし** | 事前登録どおり | **順序は後出し**だが、bin 幅は 2026-10-07 の正典と同じ既存の標準 |

### 20 keV（合格）は後出しの要素が無い

履歴数を増やす判断は**値を見る前**に追補 1 として登録した。
合成 SEM は中央値 0.494%（EGS5 側 0.309%、CPU 側 0.353%）で、1% の条件に余裕がある。
15/17 が評価可能、不合格 0、歴史的 2σ でも 0 件。
**ただし覆うのは深さ 4.12 cm まで** —— 20 keV は 4.25 cm で線量が山の 5% を切るので、
ROI 自体がそこで終わる。

新しい N・新しい seed なので、H5 とのビット一致の検査は**適用外**（`fine_matches_H5=false`）。
H5 との一致は、同じ N で回した 2026-11-05 の走りで既に確認済み。

### ⚠️ そもそも 0.25 cm が設計の誤りだった（ユーザー指摘 2026-11-05）

プロジェクトの正典の PDD 照合
（[revalidation_2026_10_07/comparison.csv](../../../egs5_crosscheck/revalidation_2026_10_07/comparison.csv)。
CLAUDE.md が参照として挙げるもの）の中心軸 PDD の bin は
`pdd_z0-1`〜`pdd_z14-15` の **1 cm × 15 本 × 0〜15 cm** である。
**私が 1.0 cm で出した判定（K=15、0.50〜14.50 cm）は、これと同じ設計**になっている。

**0.25 cm という 4 倍細かい bin を、理由を書かずに自分で持ち込んだのが誤り。**
`axis_coarse` が 0.25 cm の配列だったのでそのまま使っただけで、
**目的（利用者が読む PDD の照合）に必要な分解能を考えていない。** その帰結として:

- 合成 SEM が 1.407% になり、D7 の 1% に構造的に届かなくなった
- 「統計不足」を履歴数の問題と誤認し、**EGS5 を 2 本増やす計画変更**を提案した
- 1.0 cm に戻すことを「後出しの救済」と記録した。**実際は標準に戻しただけ**

**「後出し」の評価を訂正する**: 残る弱点は**順序**（先に 1.0 cm を登録しなかったこと）だけで、
「通るように恣意的な幅を選んだ」という弱点は**無い**。幅は既存の標準である。

**20 keV は 0.25 cm で正しかった。** 20 keV は 4.25 cm で山の 5% を切り ROI が約 4 cm しかなく、
1.0 cm では bin が 4〜5 本で `MIN_EVALUABLE_COUNT`=8 の床を下回り判定できない。
つまり誤りは「0.25 cm を選んだこと」ではなく、**ROI の広さが 3.5 倍違う 2 つのエネルギーに
1 つの bin 幅を当てたこと**である。

**無駄にした計算**: この誤りのため EGS5 `pdd_60kev`（150,767,400 履歴、約 0.47 時間）を回した。
1.0 cm なら既存データで足りたので**不要だった**。20 keV の分（約 0.43 時間）は無駄ではない。

### 1.0 cm の合格に残る留保

- **順序は後出し**（0.25 cm の失敗後に走らせた）。先に 1.0 cm を登録していれば弱点は無かった
- **評価可能が必要数ちょうど 12/12 で余裕ゼロ**。1 本でも欠ければ統計不足になる
- **歴史的 2σ の不合格が 1 件**ある（D7 の合否には使わないが、必ず併記する。最大の相対差 2.056% が
  2% の OR の枝をわずかに超え、z の枝で合格になった bin）
- 深さ分解能は 1 cm に落ちる

### 補助の pooled 検査（事前登録に無い）

60 keV の系統差は **+0.054% ± 0.165%（1σ）、2σ で −0.276〜+0.384%** の範囲内である
（判定した K=60 本を batch ごとに合計し、全体比 1.000538 の SEM を batch 統計から取った。
bin どうしの相関もこの取り方なら自動的に入る）。pooled z=+0.70、符号 33/27、
深さに対する傾き +0.024/cm、窓の比は +0.53%・+0.36%。

**⚠️ 訂正（振り返り 2026-11-05）**: 当初「0.2〜0.3% の水準で否定」と 4 か所に書いたが、
あれは**点推定**（平均 z 0.09 × 1.4% ≈ 0.13%）を限界値として書いたもので導出が誤り。
正しい 2σ の範囲は **±0.33%** である。値は近いが、やや狭く書いていた。
bin ごとの判定より強い言い方で、1.0 cm の判定を支持する独立な材料になる。

### 回したが使っていない計算

60 keV の EGS5 `pdd_60kev`（150,767,400 履歴、seed 60201）は完走したが、
1.0 cm の判定には**使っていない**（既存データで足りたため）。
**ユーザー決定 2026-11-05: 消す**（`~/.Trash/a4u_pdd_60kev_deleted_2026-11-05/` へ退避。
空にするのはユーザー）。1.0 cm で足りたので保持の理由が無くなった。
後で 60 keV を 0.25 cm で判定するなら、EGS5 から回し直すことになる（約 0.5 時間）。

### 言えること

- **20 keV の中心軸 PDD は、深さ 4.12 cm まで EGS5 と照合して合格**（事前登録どおり）
- **60 keV の中心軸 PDD は、1.0 cm の深さ分解能で 14.50 cm まで合格**。
  これは 2026-10-07 の正典と同じ bin 幅であり、**判定の順序だけが後出し**である。
  0.25 cm の分解能では統計不足（ただしその 0.25 cm 自体が目的以上に細かい設計の誤り）
- どちらも**不合格は 0 本**で、60 keV の系統差は **±0.33%（2σ）の範囲内**

### 言えないこと

- 「中心軸 PDD は 0.25 cm の分解能で EGS5 と照合して合格」—— 60 keV は統計不足
- 20 keV について 4.12 cm より深い所
- OCR（判定していない）、15 cm より深い bin（判定していない）


## 封印の不一致の扱い（ユーザー決定 2026-11-05: (A) を採用）

**コメントだけのソース変更による封印の不一致は、1 batch のビット比較で確認して非検出とする。**

`docs/ai/PLAN_TEMPLATE.md:66` は既に「凍結・ハッシュの対象に計画ファイルと報告ファイルを
入れない（結果に影響しない変更で照合が止まる）」と決めている。封印はソースの**バイト**を
見るので、同じ問題がソースファイルの**中身**のレベルでも起きる。

### 手順

1. 封印の不一致が出たら、**条件を 1 欄ずつ比べてどの欄が違うかを特定する**
   （全体のハッシュだけ見て「物理が変わった」と判断しない）
2. 差が **1 ファイルのソースハッシュだけ**なら、そのファイルの変更がコメントのみか調べる
3. **1 batch を同じ seed・同じ start/count で再計算し、保存済みの batch とビット比較する**
   （全体を回し直す必要はない。実測 231,640 履歴で 8.0 秒）
4. ビット一致なら**物理は変わっていない**と結論し、**その事実を記録に残す**
   （残さないと後から「同じ走りなのに条件ハッシュが 2 つある」と見えて物理の変更を疑われる）
5. ビット一致しなければ**停止して相談する**

### 適用の実例（2026-11-05）

H5 と同じ条件で回し直す前の照合で、条件 SHA256 が不一致。差は
`sources.photon_physics.h` の 1 欄だけ（H5 `a9c89e4b…` / 当時 `161bdd9b…`）で、
前日に `G5_ELECTRON_PHOTON_CUT` の USER DECISION のコメントを直したのが原因だった。
batch 0 を再計算して H5 の batch 0 と**ビット一致**を確認し、
100 batch 全体でも `fine` が H5 と完全一致した。

### しないこと

- コメントを落としてからハッシュを取る（複雑で、別の取り違えを生みやすい）
- 不一致を「たぶんコメントだろう」で見逃す（**必ず 1 batch を回して確かめる**）


## 整理記録（pytest 一時出力の目録）の扱い（ユーザー決定 2026-11-05）

D2・H2・H3b の各段で、pytest の一時ディレクトリを片付けるときに
「どのファイルがあったか」の目録（パス・バイト数・SHA256）を JSON で残していた。
5 件・**計 22 MB**（`a4u_d2_cleanup`・`a4u_d2_test_cleanup`・`a4u_h2_cleanup`・
`a4u_h2_test_cleanup`・`a4u_h3b_cleanup`）。

**消した**（`~/.Trash/a4u_cleanup_inventories_deleted_2026-11-05/` へ退避。空にするのはユーザー）。

**Why**: 中身は `pytest-of-oishifamily/pytest-NN` 配下の足場（.so・build.log・symlink）の
目録で、**この機械でしか意味を持たない**。SHA256 が 14,399 件あるが、照合の相手になる
一時ディレクトリ自体はもう無い。リポジトリは公開なので、22 MB を置く価値が無い。

**How to apply**: 一時出力を片付けるとき、**目録を JSON に残すのは「その場で照合する」ときだけ**に
する。片付けが終わってから残す目録は、照合の相手が消えているので価値が無い。
残すなら件数と合計サイズの 1 行でよい（例:「pytest-38 の 1,842 ファイル・181 MB を整理」）。
