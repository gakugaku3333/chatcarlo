# G1 実装・手元検査（2026-10-21）

承認済み G1 の実装を追加した。**手元の関門 0 は合格。関門 1・2 および実際の Colab は未実行**で、GPU の数値的な合否・速さを示す結果はまだない。Claude が [HANDOFF.md](HANDOFF.md) の順で実行する。関門 1/2 の合格記録は作っていない。

## 成果物

- `chatcarlo/gpu/`: 共通 C++ の物理・幾何・履歴関数、Philox/検査専用 MT、CPU の双子、CuPy RawKernel、束の SHA256 照合、VM 用 NumPy 線源、既存の結果クラスへのアダプター、Colab 転送・回収・再開、依頼 D 互換の画像保存。
- `run_gates.py`: 固定した S1/S2/S3・種・履歴数・ROI・判定と再試行で関門 1 を実行、関門 2 の束と job を作成し、回収結果を `gate2.json` に統合する。共通の判定・固定入力・GPU 比較は `chatcarlo/gpu/validation.py`。
- `baseline_s1.npz` / `baseline_s2.npz`: 実装前に作った NumPy の全項目の固定値。実装後の完全一致は `test_gpu_g1_regression.py`。
- `tests/test_gpu_g1_*.py`: 70 件の GPU 関連検査（パラメーター展開後）。

既存ファイルの変更は `transport.py` の 7 行の差分だけ（キーワード専用引数 2 個と先頭の遅延 import・早期 return）。既存の NumPy/kernel 経路は変更していない。指定された変更禁止ファイルには差分がない。作業開始時から未追跡だった `data/` は操作していない。commit・ステージングはしていない。

## 関門 0h の基準

基準コミットは指定の `636b2cffd3c8839e7d0f50b73f5496289d6db18a`。そのコミットの `chatcarlo/` を `git archive` で一時フォルダへ展開し、実際の import 元も確認して、実装に先立って固定値を作成した。条件は各 200,000 履歴、seed 42、workers 1、batch_size 100,000、指定の 60×60 検出器、track_uncertainty=True。

固定値に含めたものは、全 `TransportResult` 項目と、検出器の全配列・スカラー項目（4 画像、category_sum2、total_sum2、n_batches_hit、スナップショットを含む）。環境と各ファイルの SHA256 は `baseline_manifest.json`。確認用の材料への付与エネルギー（全履歴の和、MeV）:

| シーン | 材料 | 固定値 |
|---|---|---:|
| S1 | water | 7415.759111377638 |
| S1 | air | 54.80171233868075 |
| S2 | soft_tissue | 3632.729121357589 |
| S2 | bone | 2328.82987658692 |
| S2 | copper | 441.67407640964353 |
| S2 | lead | 146.36194707082427 |
| S2 | aluminum | 88.68283775896701 |
| S2 | air | 82.19883758578104 |

これらだけでなく、付与・SEM・R・割合・散乱・蛍光・検出器の全固定値が実装後とバイト一致した。

## 指定コマンドの結果

Python/Numba/Matplotlib/pytest のキャッシュと計算・ビルドの一時出力は一時フォルダを使った。`~/.cache` は使わない。コマンドと出力は `tests_all.log` / `tests_gpu.log` / `validate.log`。

| 計画のコマンド | 結果 |
|---|---|
| `.venv/bin/python -m pytest tests/ -q` | 490 passed、2 warnings、終了コード 0 |
| `.venv/bin/python -m pytest tests/ -q -k gpu` | 70 passed、420 deselected、終了コード 0 |
| `.venv/bin/python -m chatcarlo validate examples/chest_room.yaml` | 有効、物体 6 個、終了コード 0 |

全件実行の警告は、既存の SpekPy のゼロ放出条件の検査と、SpekPy 不在時のフォールバックを意図的に確認する検査によるもの。

関門 0a: 各関数 10⁵ 入力と端の入力、戻り値・有効な作業配列・MT 624 語・index・後続 16 個が一致。0b: 指定の 6 ケースそれぞれ 10⁴ 履歴の 6 出力が一致。0c: 指定の 10⁵ 線分と端、傾けた検出面、境界/相互作用との同着で検出優先を確認。0c': 人工履歴の画像・材料付与・終了数と、不揃い/ゼロ寄与/単一バッチ/不確かさ無効を確認。0d: S1/S2/S3、明示スペクトルと kVp、斜めと基底の境目、5 バッチと再開の配列・Generator 状態が一致。0e: 検出器を含む世界の箱が一致。0f: 4 種の故障を検出。0g: xraylib/SciPy/SpekPy を import 禁止にして CPU 1,000 履歴を実行。0h: 上記の固定値と一致。

偽の Colab 実行ファイルでは分割・組立て、照合失敗での停止、セッション消失後の回収済み保持、再開時の回収済み輸送省略と線源状態の継続を確認した。依頼 D 互換ランナーは CPU の双子で 10⁴ 履歴、元と同じ 10 個の npz キー・配列形状と追加メタデータを確認した。

環境は `environment.json`。CUDA への展開ソースはホストの構文検査も通したが、NVRTC/T4 のコンパイル・実行の検査ではない。

## 計画との差・解釈した点

対象範囲・物理・判定閾値・履歴数・種・ROI の変更はない。関門 1/2 と Colab を未実行としたのは使用者の指定どおり。

- 受入条件の例は `device='cpu'`、設計 §1-2 は `gpu_device='cpu'` と表記が異なる。具体的な API の設計 §1-2 を採用し、`run_transport` には `gpu_device` を追加した。ランナーの `--device` と `Core(device=...)` はそれぞれの入口の引数。
- 関門 2c の丸め上限は「GPU−CPU」を作る両側の加算回数と寄与の絶対値の和に適用する。区分ごとの検出件数も装置で数え、ROI を先に合計してから比較する。加算数・上限を記録し、上限外の比較には固定の対応バッチ差の 3 SEM を使う。浮動小数の一致率から合格を決めない。
- CPU の製品名を取得する `sysctl` は sandbox が拒否した。arm64、OS、Python、NumPy、Numba、LLVM、clang とフラグは記録した。
- 関門の共通処理は許可された新規 `chatcarlo/gpu/validation.py`、手元の実行入口は許可された `docs/validation/gpu/g1/run_gates.py` に置いた。

## Claude の独立検証と関門 1（2026-10-21）

- 変更範囲: 既存ファイルの差分は `chatcarlo/transport.py` の 7 行だけ（先頭の遅延 import と早期 return、キーワード専用引数 2 個）。変更禁止のファイルに差分なし。
- テストを Claude が再実行: `pytest tests/ -q` 490 passed（160 s）、`-k gpu` 70 passed、`validate examples/chest_room.yaml` 有効。
- 関門 0h の固定値を独立に作り直した: 基準コミット 636b2cf の `chatcarlo/` を `git archive` で別の一時フォルダに展開し、そこから import したことを確かめて S1・S2 を同条件（2×10⁵ 履歴、seed 42、workers 1、batch 10⁵）で計算。材料ごとの付与は上の表と全桁一致し、検出器の全配列は `baseline_s*.npz` とバイト一致した。
- 関門 0b は `kernel.run_batch(n_chunks=1)` を参照に取り、6 ケースの 6 出力をバイト比較していることをテストのコードで確かめた（自己複製との比較ではない）。

**関門 1: 合格（初回、再試行なし）。** `run_gates.py gate1`、19:30〜19:38（約 8 分）。比較した 51 量すべて |z|≤3、Σz²/K = 1.13（K = 51）。|z| の大きい順に S2 一次散乱 ROI0 −2.11、S2 一次 ROI1 −2.05、S1 多重散乱 ROI0 +1.94。両側とも厳密に 0 の 10 量（`both_zero`）は判定から除いた: S1・S3 の蛍光区分（水の O K 線と Al K 線は 5 keV 未満で局所吸収されるため、輸送される蛍光がない）と、S1 の照射野外の一次 ROI。輸送される蛍光は S2（Cu・Pb）で比べ、全体 z = +0.47、ROI 0.06〜−0.22。結果は [gate1.json](gate1.json)。この精度では差を検出しなかったという意味で、同等性の証明ではない。

## 関門 2 初回起動の NVRTC コンパイル修正（2026-10-21）
Colab T4 の初回起動で `g1_histories_kernel` の NVRTC コンパイルが `identifier "INFINITY" is undefined` で失敗した。
原因は NVRTC がホストの `<cmath>` の `INFINITY` マクロを提供せず、`g1_geometry.h` の参照が未定義になったこと。
`g1_common.h` の GPU 分岐に P8f と同じ未定義時の定義を追加し、CPU 分岐は変更せず、展開した CUDA ソースで最初の使用より前に定義があることを回帰テストにした。

## 関門 2a の格子番号の扱いの変更（2026-10-21、結果を見たあとの判定基準の変更）

2 回目の T4 起動（NVRTC 修正後）で、関門 2a は 15 検査のうち `physics0`（`g1_mu`）だけ不合格になり、2b・2c には進まなかった。`gpu_probe` が格子番号 `idx`（出力の列 8+3·me〜8+4·me）を完全一致で比べていたため。Claude が VM で調べた結果:

- 277,156 入力のうち 31 件で `idx` が 1 ずれた。31 件すべてが `physics_inputs` が足した格子点の真上の入力で、ランダムな 100,000 件には 0 件。
- 31 件すべてで、log(E) が CPU と GPU で 1 ulp 違っていた（全入力では 240 件で違う）。
- 断面積（列 0〜3）の相対差は最大 1.6×10⁻¹⁵。同じ点で隣の区間を選んだだけで、値は同じ。
- 他の 14 検査（physics1〜6、人工の蛍光表、direction0/1、geometry0〜2、detector、傾けた detector）は整数・乱数の消費数とも完全一致。

計画の 2a が完全一致を求める整数は「画素番号、乱数の消費数、元素の選択」で、格子番号は含まない。格子番号は数学関数（log）に左右される出力として扱うことを、**結果を見たあとに、ユーザーの承認を得て**決めた。変更後の `physics0` の判定（`validation.physics0_judge`）: 格子番号のずれは件数（全体・ランダム部分）を記録するだけにし、代わりに列 0〜3 と元素ごとの p・c・r（列 8〜8+3·me）が、CPU 側が 0 の要素は完全一致、それ以外は相対差 ≤ 10⁻¹² であることを合否の条件にする。乱数の消費数の比較と、他の検査の判定は変えていない。2c の合否の基準（丸めの上限、対応するバッチ差の 3 SEM）も変えていない。この変更は Codex の使用量の上限のため、ユーザーの指示で Claude が実装した（テスト `test_gpu_g1_2a_physics0_grid_index_tolerated`、全件 492 passed）。
