# 計画: njitスカラーDDAのkernel.py恒久組み込み

- 状態: implemented
- 作成者: claude
- 実装担当: codex
- 日付: 2026-07-29

## 目的（1〜3行）

`docs/ai/plans/2026-07-28-njit-dda-e2e-measurement.md`（実装済み・検証済み）で、
njitスカラーDDAはkernel経路のend-to-endで中央値2.570倍・8反復全てで高速化を
実測済み（`docs/plan_dda_njit_premise_check.md`「### 修正後」節）。この結果は
これまで`docs/speedup_baseline/dda_njit_prototype_benchmark.py`の**モンキーパッチ**
経由の測定にとどまっており、本番`chatcarlo/kernel.py`には未組込。本計画はこの
検証済み実装を本番へ恒久的に組み込む（採用の実行）。

## 対象範囲

- 変更してよい:
  - 新規 `chatcarlo/tally_njit.py`（`dda_njit`スカラー本体を
    `docs/speedup_baseline/dda_njit_prototype_benchmark.py`から移設・整理し、
    既存`accumulate_track_length_multi`と**同一シグネチャ**の公開ラッパー
    `accumulate_track_length_multi_njit(target_weight_pairs, grid, origin,
    direction, length_cm)`を提供する。scalar本体はこのラッパー内部の詳細とし、
    `kernel.py`からは直接`dda_njit`を呼ばせない）
  - `chatcarlo/kernel.py`（`run_batch_with_tally`/`run_dose_grid`に
    `use_njit_dda: bool = True`引数を追加し、呼び出すcallableを
    `accumulate_track_length_multi_njit` / `accumulate_track_length_multi`
    の間で切り替えるだけにする）
  - 新規 `tests/test_tally_njit.py`
  - `docs/speedup_baseline/dda_njit_e2e_benchmark.py`（モンキーパッチではなく
    `use_njit_dda`引数経由で本番経路を叩くよう更新）
  - `CLAUDE.md`（Architecture節の「既存の監査済み`tally.accumulate_track_length_multi`
    にそのまま委ねる」という記述を実情に合わせて更新。あわせて「setup」の
    pip installコマンド一覧にnumbaを追記する——`kernel.py`が既にhard depとして
    使っているにもかかわらず現状リストから漏れている）
  - `docs/plan_chatcarlo_speedup_post_egs5.md`（設計(b)の記述に、後日njit版へ
    切り替えた旨の追記。既存記述は書き換えず追記する）
  - `docs/plan_dda_njit_premise_check.md`（本計画の実装結果を追記。既存の
    「### 初回」「### 修正後」節は書き換えない）
- 変更禁止:
  - `chatcarlo/tally.py`・`chatcarlo/transport.py`・既存の`tests/`ファイル
    （`tests/test_tally_njit.py`以外）は一切変更しない。numpy版
    `accumulate_track_length_multi`は参照実装として現状のまま残す。
  - `docs/speedup_baseline/dda_njit_prototype_benchmark.py`は削除・改変しない
    （測定の再現性のため過去の記録として残す）。

## 受入条件（検証可能な形で列挙）

- [ ] `chatcarlo/tally_njit.py`の`dda_njit`スカラー本体は、境界セマンティクス
      修正後のプロトタイプ（面除外、中点分類、範囲外discard、厳密同着のみ排除）
      と同一ロジックである。`_EPS_PLANE`は**njit関数の引数として毎回渡す**
      （numbaはモジュールグローバルをコンパイル時に焼き込むため、
      `from chatcarlo.tally import _EPS_PLANE`をnjit本体内で参照する実装は
      禁止——将来`tally.py`側の値だけ変更された場合に`.nbi`キャッシュ経由で
      古い値のコンパイル済みコードが使われ続ける危険を、importではなく
      引数化によって構造的に排除する）。公開ラッパー
      `accumulate_track_length_multi_njit`が`chatcarlo.tally._EPS_PLANE`を
      読み取ってscalar本体へ渡す。`tests/test_tally_njit.py`には
      `monkeypatch.setattr(chatcarlo.tally, "_EPS_PLANE", changed_value)`で
      `chatcarlo.tally`モジュールの値そのものを変更した上で固定の境界近傍
      区間を実行し、njit版の結果が変更後の値に追随すること（可能であれば
      同じ変更後値を読むnumpy参照版とも一致すること）を確認するテストを
      含める（固定値`1e-7`との一致チェックだけでは不十分——値を実際に
      変えて動作が変わることまで見る）。
- [ ] `accumulate_track_length_multi_njit(target_weight_pairs, grid, origin,
      direction, length_cm)`は既存`accumulate_track_length_multi`と同一の
      呼び出しシグネチャを持つ。ラッパー内部で以下を明示的に検査し、
      違反時は例外を送出する（黙って誤った結果を返さない）:
      `target_weight_pairs`がちょうど2要素であること、各targetのshapeが
      `grid.shape`と一致すること、`origin.shape == (n, 3)`・
      `direction.shape == (n, 3)`・`length_cm.shape == (n,)`・各weightの
      shapeが`(n,)`であること（先頭次元の長さだけでなく次元数・列数まで
      検査する——`(n, 2)`のような欠損shapeが素通りしてnjit本体側で範囲外
      アクセスするのを防ぐ）。float32・非連続配列・読み取り専用配列など
      想定外dtypeの扱いは既存`accumulate_track_length_multi`と同じ前提
      （float64・C-contiguous前提、それ以外は未定義）であることを
      docstringに明記する（既存関数が現状これらを検査していないため、
      本計画で新たに厳格化はしないが、前提を文書化する）。
- [ ] `ts`一時配列のサイズは`shape[0]+shape[1]+shape[2]+5`ちょうどではなく、
      安全マージンを持たせ（例: `+16`）。交点を`ts`へ書き込む**直前**に
      `n_t >= ts.size`を検査し、超える場合は例外を送出する（書き込んでから
      検査するのではなく、書き込み前にガードする——`kernel.py:1066`の
      「欠落を検知せず一部だけ積算しない」という既存の設計方針と同じ理由）。
      このテストを発火させるため、交点列挙を小さなnjitヘルパーへ分離し、
      意図的に小さいscratch配列を渡すテスト専用の経路を用意する。
- [ ] `run_batch_with_tally`・`run_dose_grid`はどちらも`use_njit_dda: bool = True`
      引数を持つ。`True`なら`accumulate_track_length_multi_njit`、`False`なら
      現行`accumulate_track_length_multi`を呼ぶcallableを選択するだけの
      分岐にする（呼び出し箇所自体のロジックは変えない）。デフォルトは`True`
      （採用の実行が本計画の目的のため）。`run_dose_grid`が複数バッチを
      回す際、`use_njit_dda`の値が全バッチに一貫して伝播することをテストで
      確認する。
- [ ] 直接A/B: water60_free条件（N=200,000、resolution=2cm）で、
      `use_njit_dda=True`と`False`のグリッド出力が`np.array_equal`で完全一致
      （kerma・H\*(10)とも）。1バッチ（`batch_size=200_000`）と複数・端数
      バッチ（`batch_size=60_000`、60k×3+20k）の両方で確認する。
      `KernelBatchResult`の6配列全てもビット一致することを確認する。
      **`tests/test_kernel.py::test_tally_variant_matches_reference_variant`は
      本条件の代替にならない**（`transport.py`とのRNG差異を前提にした
      統計的検証であり、DDA実装間の厳密A/Bではないため）。
- [ ] `tests/test_tally_njit.py`は以下を含む（フル区間セット・fuzzは既存の
      `docs/speedup_baseline/dda_njit_prototype_benchmark.py`側に残し、テストには
      コストの低い決定論的ケースのみ移植する）:
      - `boundary_cases()`由来の境界ケース一式（軸並行・外側・内側・端点等）
      - `endpoint_regression()`のwater60_free回帰区間（既知の境界近傍バグの
        固定回帰ケース、輸送不要で最も価値が高い）
      - `boundary_near_fuzz()`相当の境界近傍fuzzのうち、代表的な組合せを
        数十件程度に絞ったもの（フル数千件は速度実測スクリプト側に残す）
      - 上記`ts`配列サイズのオーバーフロー検査（意図的に小さいscratch配列で発火）
      - 空区間入力（`n=0`）: targetが変更されず、例外にもならない
      - 非ゼロtargetへの積算: 既存値を上書きせず加算されること
      - `use_njit_dda=False`が確実に既存`accumulate_track_length_multi`
        （numpy版）を呼ぶこと。`use_njit_dda`の値をkernel.py内で実引数として
        渡すこと自体に加え、`unittest.mock.patch`または`monkeypatch`で
        `chatcarlo.tally.accumulate_track_length_multi`と
        `chatcarlo.tally_njit.accumulate_track_length_multi_njit`の双方に
        spyを仕掛け、`use_njit_dda=True/False`それぞれで意図した側だけが
        実際に呼ばれたことを確認する（出力比較だけでは、誤って逆の実装を
        呼んでも境界修正後は値が一致するため検知できない）
      - 不正なペア数（1個または3個）、target shape不一致（`(n,2)`等の
        欠損shape）、weight長不一致を与えた際に例外になること
- [ ] pytest全体を実行し、既知の8件（`os.sysconf("SC_SEM_NSEMS_MAX")`を
      `PermissionError: [Errno 1] Operation not permitted`として拒否する
      並列テスト、`docs/plan_dda_njit_premise_check.md`に記載済み）**以外の
      failureが1件でもあれば停止**する（件数の閾値ではなく、既知の8件と
      完全に一致するテスト名・エラー内容であることを確認する）。
- [ ] 統合後の実装（モンキーパッチではなく`use_njit_dda=True`の実経路）で
      end-to-endタイミングを1回、water60_free・N=200,000・`batch_size=200_000`
      ・8反復（AB4回/BA4回交互）で再測定する。既に登録済みの採用ゲート
      （`docs/ai/plans/2026-07-28-njit-dda-e2e-measurement.md`）と同じ基準
      ——8反復全てで`E2E_(use_njit_dda=True) < E2E_(use_njit_dda=False)`、
      かつ中央値比>1.0——を適用する。モンキーパッチ経由の既報測定値
      （中央値2.570倍）と大きく乖離する場合（ラッパー自体のオーバーヘッドの
      有無等）は乖離の理由を報告に明記する。乖離があっても本計画のゲートは
      「実装後に実際に速いか」のみで判定し、既報の2.570倍という数値そのものを
      新たな基準に置き換えない。

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_tally_njit.py -v
.venv/bin/python -m pytest tests/test_kernel.py -v
# 既存のdda_njit_e2e_benchmark.pyをuse_njit_dda引数経由の実装呼び出しに更新
# （モンキーパッチを撤去）してから、既存の--modeフラグ規約のまま再実行する。
# 現行スクリプトのcorrectness判定はnp.allcloseだが、境界修正後は最大誤差0が
# 既報のため、この更新時にnp.array_equalへ厳格化する。
.venv/bin/python -m docs.speedup_baseline.dda_njit_e2e_benchmark --mode correctness
.venv/bin/python -m docs.speedup_baseline.dda_njit_e2e_benchmark --mode timing
```

## 実装方針

- `tally_njit.py`は`chatcarlo/`直下の新規モジュールとし、`tally.py`を変更しない
  ことで参照実装への影響ゼロを保つ。numbaは既に`kernel.py`が使用している
  依存（`docs/plan_chatcarlo_speedup_post_egs5.md`のB-1相当、CLAUDE.mdの
  pip installリストにはないが`kernel.py`が既にhard depとして使っている）
  なので、新規依存の追加ではない。CLAUDE.mdのsetupコマンド一覧に
  numbaを追記し、この既存の記載漏れも合わせて直す。
- `tally_njit.py`は2層構成にする: 非公開の`@njit`スカラー本体（引数として
  `eps_plane`を受け取り、モジュールグローバルは一切参照しない）と、
  既存`accumulate_track_length_multi`と同一シグネチャの公開ラッパー
  `accumulate_track_length_multi_njit(target_weight_pairs, grid, origin,
  direction, length_cm)`。ラッパーが`chatcarlo.tally._EPS_PLANE`を読み取って
  スカラー本体へ引数として渡す。`kernel.py`はラッパーとnumpy版のどちらの
  callableを呼ぶかを選ぶだけにし、呼び出し側のロジック（ペアの組み方、
  weight計算等）は一切変えない。
- numbaはグローバル値をコンパイル時に焼き込むため、モジュールグローバル
  import経由の値参照は将来のサイレントな乖離要因になる——`_EPS_PLANE`を
  スカラー本体の引数にすることで、キャッシュの陳腐化を構造的に排除する
  （importして固定値と一致するかテストするより根本的な対策）。
- `use_njit_dda`はデフォルト`True`とする。理由: 本計画は「採用の実行」であり、
  既存呼び出し元（`chatcarlo/__main__.py`等、kernel.py経由のCLIパスがあれば）
  を明示的に更新しなくても新しい経路が使われるようにする。ただし
  `False`で確実に現行numpy経路へフォールバックできることを維持し、
  将来問題が見つかった場合の切り戻しを容易にする。
- 差分レビュー（Claude）では、`accumulate_track_length_multi`呼び出し箇所
  （`kernel.py:1096`, `kernel.py`内`run_batch_with_tally`）が`use_njit_dda`に
  応じて正しく分岐しているか、既存の`seg_overflow_per_chunk`等のエラー処理経路
  に影響がないか、`accumulate_track_length_multi_njit`のシグネチャが
  既存関数と完全に一致しているかを重点的に確認する。

## 書かなかったこと（スコープ外を明示）

- `transport.py`（既存の逐次numpy輸送経路）への同様の統合は行わない。
  `transport.py`は参照実装として現状維持する。
- `residual_B`（置換後の新ボトルネック、transport+バッファ確保+重み計算）の
  プロファイルは別計画とする。
- prange並列領域内でDDAを実行する設計（データ競合回避方式の選定）は
  引き続き未着手・未設計のまま。本計画のDDAはこれまでと同じく
  prange領域の外側・逐次実行のまま統合する。
- `chest_room`（cylinder/sphere含む、box専用kernelでは非対応）でのend-to-end
  測定は引き続き対象外。
