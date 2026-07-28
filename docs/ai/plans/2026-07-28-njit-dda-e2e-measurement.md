# 計画: njitスカラーDDA、end-to-end実測（予測2.721倍の検証）

- 状態: approved
- 作成者: claude
- 実装担当: codex
- 日付: 2026-07-28（Codex計画レビューを反映し改訂）

## 前計画との関係（優先順位の明示）

本計画は [docs/ai/plans/2026-07-28-njit-dda-premise-check.md](2026-07-28-njit-dda-premise-check.md)
（`状態: approved`、判定 DO_NOT_PROCEED）の**後継計画**である。

前計画の判定（ゲートB FAIL、3倍化プロジェクトには進まない）は**取り消さない**。
本計画は「ユーザー判断により、3倍化という一括の目標とは切り離し、**着実な高速化として採用できるか**を
実測で確かめる」ものであり、前計画の対象範囲を置き換える。実施時は本計画の対象範囲が優先する。

判定の考え方も前計画から変わる。前計画は「3倍に届くか」という一括目標のゲートだったが、
本計画は「実際に速くなったか」を問う。倍率の閾値は設けない。

## 背景

前計画で、njitスカラーDDAが既存のベクトル化numpy DDAより **DDA単体で6.516倍高速**
（water60_free、N=2e5、resolution=2cm、1,022,763区間、AB/BA交互8反復の中央値:
numpy 0.379355秒 / njit 0.058222秒）であることを実測した。正しさもボクセル単位で完全一致
（境界ケース・6000-ray fuzz・chest_room実輸送由来103,336区間、最大絶対/相対誤差ともに0）。

一方、end-to-endの **2.721倍は Amdahl則による予測値であって実測ではない**
（`0.538 / (0.136 + 0.402 / 6.516)`）。採用を判断する前に、予測を実測で置き換えるのが本計画の目的。

## 目的（1〜3行）

njitスカラーDDAをkernelの実経路へ差し込み、water60_freeでのend-to-end時間を実測して、
**実際に高速化するかどうか**を確かめる。同時にアーム内のDDA時間・残余時間も計測し、
置換後に何が新たなボトルネックになるか（＝次の高速化の入口）を特定する。

## 今回、競合回避の設計は不要（重要）

`chatcarlo/kernel.py` の `run_batch_with_tally` は、prange並列の輸送（`_run_batch_scalar_tally`）が
**完了した後**に、チャンクごとの区間バッファを連結し、`accumulate_track_length_multi` を
**シリアルで1回だけ**呼ぶ設計（`kernel.py:1084-1099`）。DDAは既にprange領域の外にあり、
共有グリッドへの並列書き込みは発生しない。

したがって本計画の「統合」はこの1箇所の呼び出しを差し替えるだけで成立し、
前計画が「将来統合時の制約」として挙げたデータ競合は**今回一切扱わない**
（それはDDAをprange領域の**内側**へ移す場合の話であり、本計画のスコープ外）。

## 測定できない範囲（制約の明示）

- `examples/chest_room.yaml` は cylinder / sphere を含むが、kernelは box専用
  （cylinder/sphereはB-1c未対応、`kernel.py:19`, `kernel.py:244`）。
  **chest_roomでのend-to-end測定は原理的に不可能**。前計画で得たchest_roomのDDA単体速度比
  4.374倍は参考値として残るが、本計画の実測は **water60_freeのみ**。
- 本計画で言う "end-to-end" は **kernelのdose-grid経路（`run_dose_grid`）全体**を指し、
  CLI起動・シーン読込・断面積テーブル構築（`bake_scene_materials`）等は含まない。
  この定義を結果ドキュメントにも明記すること。

## 対象範囲

- 変更してよい:
  - `docs/speedup_baseline/dda_njit_e2e_benchmark.py`（新規。end-to-end A/B計測スクリプト）
  - `docs/speedup_baseline/dda_njit_e2e_result.txt`（新規。生出力の記録先。再実行時は上書きしてよい）
  - `docs/plan_dda_njit_premise_check.md`（**既存ファイルへの追記のみ**。「## end-to-end実測（追記）」節を末尾に追加。既存の記述・判定（ゲートB FAIL、DO_NOT_PROCEED）は**書き換えない**。
    既に同名の節が存在する場合は**追記せず停止して報告する**（再実行による重複を防ぐ））
- 変更禁止:
  - `chatcarlo/tally.py`・`chatcarlo/kernel.py`・`chatcarlo/transport.py` 等、本番コード一式
  - `tests/` 配下（既存・新規とも）
  - `docs/plan_chatcarlo_speedup_post_egs5.md`
  - `docs/speedup_baseline/dda_njit_prototype_benchmark.py`（前計画の成果物。`dda_njit` は**import して再利用**し、コピーしない — バグを踏みやすいDDAを複製しないため）

## 受入条件（検証可能な形で列挙）

### 1. 統合方式（本番コードを変更しない）

- [ ] `docs/speedup_baseline/dda_njit_prototype_benchmark.py` から `dda_njit` を import し、
      `accumulate_track_length_multi(pairs, grid, o, d, ds)` と同じシグネチャのアダプタでラップする。
- [ ] `chatcarlo.kernel` のモジュール大域名 `accumulate_track_length_multi` をそのアダプタで
      モンキーパッチして計測する（前計画の `water_segments()` が同手法で動作実績あり）。
      **本番コードのファイルは編集しない。**
- [ ] アダプタは呼び出し契約を実行時にassertする: `pairs` がちょうど2要素であること、
      各targetの `shape` が `grid.shape` と一致すること、重み配列長が区間数と一致すること。
      （プロトタイプの `dda_njit` は2target固定であり、汎用の
      `accumulate_track_length_multi` と完全な互換ではないため、暗黙に壊れないよう明示的に落とす）
- [ ] アダプタは **上書きではなく積算**すること（既存の `accumulate_track_length_multi` と同じ意味論）。
      事前に非ゼロ値を入れたtargetへ加算されることを小さな単体ケースで確認する。
- [ ] 計測終了後に必ず元の関数へ戻す（`try/finally`）。アダプタから意図的に例外を送出した場合にも
      復元されることをテストする。

### 2. 正しさ（実測より先に完了させる）

アダプタの配線ミス（引数の取り違え、バッチの取りこぼし、グリッド属性の誤り）を検出するため、
**速度測定の前に**、統合経路の最終出力を突き合わせる。

- [ ] **1バッチ条件**: water60_free、N=200,000、resolution=2cm、`n_chunks=8`、
      `batch_size=200_000`、`fluorescence_enabled=True`、同一seed。
      アームA=現行 `accumulate_track_length_multi` / アームB=njit DDAアダプタで
      `kernel.run_dose_grid` を実行。
- [ ] **複数バッチ・端数バッチ条件**: 同じN=200,000に対し `batch_size=60_000`
      （→ 60k×3 + 20k の4バッチ、端数あり）で同様にA/B比較する。
      **前バージョンの本計画は1バッチ条件しか検証しておらず、「複数バッチでも正しく積算される」と
      書いていたが根拠がなかった。この条件は必須とする。**
- [ ] 両条件とも、最終 `grid.kerma_keV` / `grid.h10_track_pSv_cm3` を**ボクセル単位**で比較し、
      非ゼロボクセル集合の一致・最大絶対誤差・最大相対誤差を報告する。
      許容は前計画の集約許容と同じ `atol=2e-10, rtol=2e-12`。
- [ ] **`KernelBatchResult` の一致確認**: `n_scatter` / `absorbed` / `escaped` / `final_energy` /
      `energy_deposited` / `n_fluorescence` がA/Bで一致することを確認する
      （DDAの正しさではなく、両アームが同一の区間母集団を生成したことの健全性確認）。
- [ ] 空入力（区間数0）をアダプタへ直接渡しても既存関数同様に安全であることを確認する。
- [ ] いずれか一致しない場合は速度測定へ進まず、原因を報告して停止する。

### 3. end-to-end 実測（モデルの仮定も同時に検査する）

- [ ] `run_dose_grid` 全体を `perf_counter` で計測する。JITウォームアップのため、計測前に
      小さな `n_histories`（例: 2,000）で両アームを1回ずつ空実行する。
- [ ] **アーム内のDDA時間も同時に計測する**: 両アームともアダプタ（アームAは現行関数をラップした
      計時専用アダプタ）内でDDA所要時間を累積し、各反復について以下を記録する:
      - `E2E_A`、`DDA_in_A`、`E2E_B`、`DDA_in_B`、`residual_A = E2E_A - DDA_in_A`、
        `residual_B = E2E_B - DDA_in_B`
      これにより Amdahl則の仮定「DDAだけが置き換わり、残余は不変」を直接検査できる。
      計時呼び出しのオーバーヘッド（DDA呼び出しはバッチあたり1回のため無視できる水準）は報告に明記する。
- [ ] 条件は既報0.538秒と比較可能な `batch_size=200_000`・`n_chunks=8`・N=200,000 とする。
- [ ] **AB 4回・BA 4回の計8反復**（順序を固定して再現性を確保）。各反復で `VoxelGrid` を新規作成し、
      前反復の積算が残らないようにする。
- [ ] `S_E2E_measured = median(E2E_A) / median(E2E_B)` を算出する。
      あわせて各アームの min / median / max、および**反復ごとの速度比**を記録する（ばらつきの提示）。
- [ ] 固定式の予測（2.721倍）に加え、**同時測定した値から算出した予測**
      `median(E2E_A) / (median(residual_A) + median(DDA_in_A) / (median(DDA_in_A)/median(DDA_in_B)))`
      も併記する（現環境の実測分解に基づく予測）。
- [ ] アームAの実測値と既報0.538秒の比較を健全性確認として1行報告する
      （前計画の統合再現値は0.526749秒だった）。
- [ ] メモリは **プロセス全体の `ru_maxrss` を参考値として1つ報告するのみ**とする。
      `ru_maxrss` はプロセス開始以来の累積最大値であり、同一プロセス内のAB/BAでは
      アーム別のピークメモリを比較できない。**アーム間のメモリ差は主張しない。**

### 4. 事前登録する判定基準（結果に合わせて後から変更しない）

**方針**: 特定の倍率に届くことを条件にしない。**実際に高速化していれば採用に値する**と判断する
（ユーザー方針: 「少しずつでも着実に高速化を進める」）。3倍という旧基準は Phase A 当時の前提に
基づくものであり、本計画の採否判定には用いない。

- [ ] **採用判定（これが唯一の合否ゲート）**: 以下を両方満たせば「高速化を達成」と判定する。
      - 8反復**すべて**で `E2E_B < E2E_A`（測定ばらつきによる偶然でないこと）
      - `S_E2E_measured = median(E2E_A) / median(E2E_B) > 1.0`
      いずれかを満たさない場合は「高速化を確認できず」と判定する。
      **注**: 正しさ（受入条件2）が全て一致していることが前提。速いだけで正しくない結果は採用しない。

- [ ] **以下はすべて「診断情報」であり、合否には用いない**（乖離があっても採用は妨げない）。
      ただし次の高速化ステップを見つける手がかりになるため、必ず報告すること。
      - 予測値 2.721倍 との比較。乖離した場合はその倍率差を報告する
        （±15% = 2.313〜3.129倍 を目安の帯として併記してよいが、判定には使わない）
      - `median(residual_A)` と `median(residual_B)` の比較。有意に異なる場合
        （差が `residual_A` の10%超）は「DDA置換以外の相互作用がある」として報告する
      - 予測から乖離した場合の推定要因（重み計算コスト、区間バッファ確保、メモリ帯域、
        バッチ分割のオーバーヘッド等）。プロファイルまでは求めない、観察の記述で足りる
      - **置換後に新たなボトルネックとなった処理**（`residual_B` の内訳で最大の項）。
        これが次の高速化計画の入口になるため、可能な範囲で特定して報告する

### 5. 記録

- [ ] 結果（正しさの一致（1バッチ・複数バッチ両方）、`S_E2E_measured`、採用判定、
      診断情報（予測との乖離・残余の変化・次のボトルネック候補）、
      chest_roomが測定不能である旨、"end-to-end"の定義）を
      `docs/plan_dda_njit_premise_check.md` の末尾に「## end-to-end実測（追記）」節として追記する。
- [ ] **既存の記述は書き換えない**。ゲートBがFAIL・決定がDO_NOT_PROCEEDだった事実はそのまま残し、
      追記節で「前計画は3倍という一括目標に対する判定であり、本追記は
      『着実な高速化として採用できるか』という別の問いに対する実測である」と位置づけを明記する
      （このリポジトリの「基準を測定に合わせて動かさない」規律を守るため、
      前計画の基準を書き換えるのではなく、別基準の測定として並置する）。
- [ ] 生出力は `docs/speedup_baseline/dda_njit_e2e_result.txt` に記録する。

## テストコマンド（実装完了の定義）

```bash
# 正しさ（1バッチ・複数バッチ両条件で、統合経路の最終グリッドをボクセル単位で突き合わせ）
PYTHONPATH=. .venv/bin/python docs/speedup_baseline/dda_njit_e2e_benchmark.py --mode correctness

# end-to-end 実測（AB 4回・BA 4回、同一プロセス内、DDA時間・残余も同時計測）
PYTHONPATH=. .venv/bin/python docs/speedup_baseline/dda_njit_e2e_benchmark.py --mode timing

# 既存テストスイートに影響がないことの回帰確認（本番コード未変更のため補助的。
# 合否の中心は上記スクリプト自身の自己検証とする）
.venv/bin/python -m pytest tests/ -q
```

**注**: `tests/test_parallel_transport.py` と `tests/test_uncertainty_transport.py` の計8件は、
Codexのサンドボックスが `os.sysconf("SC_SEM_NSEMS_MAX")` を `PermissionError` で拒否するため
失敗する（前計画で確認済みの環境要因、サンドボックス外では334件全通過）。この8件が同じ
`PermissionError` で失敗している場合は環境要因として報告してよい。**それ以外の失敗は環境要因として
扱わず、原因を報告すること。**

## 実装方針

- 統合はモンキーパッチで行い、本番コードは1行も変更しない。前計画で `dda_njit` の正しさは
  ボクセル単位で確認済みなので、今回新たに検証すべきは「アダプタの配線が正しいか」に絞れる。
- `run_dose_grid` は `batch_size` ごとに `run_batch_with_tally` を呼び、その中でDDAが1回呼ばれる
  （`kernel.py:1096`）。N=2e5・batch_size=200,000 では **1バッチ＝DDA呼び出し1回**にしかならないため、
  複数バッチの積算は `batch_size=60_000` の条件で別途検証する（受入条件2）。
- kernel経路には不確かさ推定（snapshot-diff、`kerma_sum2`等）が実装されていないため、
  その相互作用は考慮不要。

## 書かなかったこと（スコープ外を明示）

- `chatcarlo/kernel.py` への恒久的な組み込み・切り替えスイッチの追加は行わない
  （実測で予測が整合し、ユーザーが採用を決めた後の、次の計画で扱う）。
- DDAをprange領域の内側へ移す設計・データ競合の回避策（専用グリッド+reduce、atomic加算等）は
  一切扱わない。今回の測定はDDAがprange外にある現行構成のままで行う。
- アーム別のピークメモリ測定（別プロセス実行による切り分け）は行わない。
  メモリはプロセス全体の参考値のみ。
- chest_room等、box以外の形状を含むシーンでのend-to-end測定は行わない（kernel未対応のため不可能）。
- EGS5相互検証・`tests/` への追加は行わない。
