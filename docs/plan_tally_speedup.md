# 計画: タリー精密化の性能回帰を解消する

作成日: 2026-07-27 / ステータス: **Step 0-3完了、Step 4は不要と判断（下記参照）。
`/furikaeri`指摘によりStep 2のメモリ受入基準未達（測定漏れ）が発覚し追加対応済み
（チャンクごとに`np.add.at`まで完結させる設計に変更）。回帰条件の壁時間を
28.4s→5.55s、ピークメモリを旧方式2.31GBに対し1.67GB（旧方式より約28%少ない）に
短縮。事前登録した数値目標（旧方式の1.5倍以内）を達成・超過。詳細な実測データは
docs/speedup_baseline/tally_exact_resolution_growth.txt「性能回帰の解消」節参照。**
/ 実行担当: Claude

親コンテキスト: [[future-directions]]。`accumulate_track_length`を解析的重なり長方式へ
置き換えた際（コミット`0c97ab4`、CLAUDE.md「Dose/H\*(10)タリー」節参照）に生じた性能回帰。
`/furikaeri`での自己レビューで発覚し、CLAUDE.mdとauto memoryに「次回着手」として記録済み。

この計画書は設計判断と実施順序を事前に固定してある。実行者は各ステップを上から順に
実施し、**受入基準を通してから次へ進む**こと。判断に迷う箇所は勝手に設計変更せず
「設計判断（確定事項）」に戻るか、ユーザーに確認する。

## 背景

`chatcarlo/tally.py`の`_segment_grid_traversal`（サブステップ方式を置き換えた解析的
重なり長方式）は、git worktree A/B測定（`0157179`=旧サブステップ方式 vs `0c97ab4`=現行、
インターリーブ3反復）で以下の回帰が確認されている（`docs/speedup_baseline/tally_exact_resolution_growth.txt`
セクション4）:

| 条件 | 旧(3反復) | 新(3反復) | 倍率 |
|---|---|---|---|
| res=5cm(既定), n=1e5, batch=2e5 | 3.01/2.99/2.99s | 4.89/4.96/4.85s | 約+63% |
| res=2cm, n=5e4, batch=2e5 | 1.95/2.05s | 5.04/5.07s | 約2.5倍 |
| res=1cm, n=2e4, batch=2e5 | 3.23/2.75s | 5.58/5.52s | 約2倍 |
| res=2cm, n=2e5, batch=2e5(バッチ満杯) | 6.38s / 1.63GB | 29.85s / 2.94GB | **約4.7倍** |

物理結果（総カーマ・材料別付与エネルギー）は新旧でビット一致することは確認済み——
今回の回帰は数値ではなく速度・メモリのみの問題。

**2026-07-27追記（メモリ値の訂正）**: 上表のメモリ値（1.63GB/2.94GB）は、後日
`docs/speedup_baseline/tally_speedup_timing.py`で同一手法・同一条件で3アーム
（旧サブステップ/この時点の新方式/最終修正後）を測り直したところ**再現しなかった**
（旧2.31GB、この時点の新方式4.59GB——測定方法が途中で変わっていたと見られるが
特定できず）。以降のメモリに関する数値目標・受入基準の判定は、この訂正後の
一貫した測定値を基準にする（下記「性能回帰の解消」節参照）。速度の数値
（6.38s/29.85s等）は本セッションで同条件を再測定し概ね再現することを確認済みで
訂正不要。

CLAUDE.mdは現時点でこの回帰の原因を「`_segment_grid_traversal`内の`np.lexsort`の
O(N log N)コストではないか（未確認）」と推測で記載している。**この推測は算数が合わない**:
res=2cmでhistory当たりコストが40µs→149µs（約3.7倍）に悪化したが、batch=5e4→2e5は
交差点総数Nを4倍にするだけで、lexsortのO(N log N)由来ならコスト増は高々1.1倍程度の
はずである。したがって別の要因（メモリ帯域/キャッシュ）を疑うべきで、この推測を
検証せずに最適化へ進むと、CLAUDE.mdに「原因の推測が推測のまま何年も残る」
（`docs/lessons_learned.md`に記録済みの教訓そのもの）を繰り返すことになる。

## 設計判断（確定事項）

1. **最適化の前に必ずフェーズ別プロファイルを取る**（Step 0）。原因を確認してから
   手を打つ。「lexsortが原因」という仮説は他の仮説（メモリ確保・fancy-index gatherの
   キャッシュミス）と並べて検証し、勝手にlexsort置換から着手しない。
2. **ビット一致を保てる変更を優先する。** Step 1（kerma/h10でtraversal共有）とStep 2
   （チャンク化）は`0c97ab4`とビット一致するはずで、これらを先に実施する。ソート方式の
   変更（Step 3）や`np.add.at`→`np.bincount`（Step 4）は加算順序が変わり得るため、
   数値的な影響を明示的に評価してからのみ着手する。
3. **旧サブステップ方式へのフォールバックオプションは追加しない。** 旧方式の
   `field.shape: parallel`ビームでの境界位相の過小スコアは実バグとして修正済みであり、
   速度フラグとして再導入すると同じバグを再び踏むリスクとテスト面積の倍増を招く。
   速度が必要な場合のノブは`--resolution`を粗くすることに限定する。
4. **測定は必ずgit worktreeインターリーブA/B、3反復以上、アーム順を交互に。**
   単発測定・固定アーム順は`docs/lessons_learned.md`に記録済みの失敗パターン
   （Phase 4、およびタリー精密化の`/furikaeri`で判明した失敗モード）そのものであり、
   このプロジェクトでは繰り返さない。

## 実施ステップ

### Step 0: フェーズ別プロファイル（前提・最優先）

回帰が最も顕著な条件（`examples/chest_room.yaml`、`--dose-grid`、`--resolution 2`、
`-n 2e5`、`--batch-size 200000`、single worker、`--no-uncertainty`なし）で
`_segment_grid_traversal`を以下の区間に分けて`time.perf_counter`で計測する:

1. AABBクリップ（t_enter/t_exitの計算、`active`の抽出まで）
2. 軸ごとのcounts計算（`m_lo_all`/`d_safe_all`の構築まで）
3. ラグド配列構築＋fancy-index gather（`t_parts`/`seg_parts`の`np.concatenate`まで）
4. `np.lexsort`本体
5. 中点計算＋`grid.voxel_index`
6. `accumulate_track_length`側の`np.add.at`

**受入基準**: 6区間それぞれの所要時間（%内訳）が判明し、どこが支配的かが数値で
言える状態になっていること。「lexsortが支配的」という仮説が正しいか誤りかを
ここで決着させ、CLAUDE.mdの該当記述をこの結果で更新する（速度改善の実装を待たずに、
まず原因の記述を訂正する）。

### Step 1: kermaとh10でtraversalを共有

[chatcarlo/transport.py:141-143](../chatcarlo/transport.py)で同一の`(origin, direction,
length_cm)`に対し`accumulate_track_length`（内部で`_segment_grid_traversal`を呼ぶ）が
2回（kerma用・h10用）呼ばれている。交差分解（`seg_id`/`idx`/`overlap`）を1回だけ計算し、
2つの`target`配列へそれぞれ異なる`weight_per_cm`で加算する形に変更する。

**受入基準**:
- chest_room.yamlで新旧（このステップ前後）の`kerma_keV`/`h10_track_pSv_cm3`が
  ビット一致すること（`np.array_equal`）。
- git worktreeインターリーブA/B（3反復、アーム順交互）で、Step 0の回帰条件において
  有意な高速化が確認できること。

### Step 2: 交差点数ベースのチャンク化

`counts.sum(axis=0) + 2`（各軸の内部境界面通過回数＋始点終点2点）でセグメントごとの
交差点数が事前に分かるので、その累積和が目標値（暫定N_target=1e6、要調整）を超える
たびにセグメント列を分割し、各チャンクを独立に`_segment_grid_traversal`の後半処理
（ラグド構築〜lexsort〜中点評価）にかける。チャンク境界をまたいでもセグメント単位の
処理内容は変わらないため、セグメント順序を保つ限りビット一致するはず。

**受入基準**:
- ビット一致（Step 1と同様の検証）。
- `--workers 4`時のピークメモリが、ワーカー1つあたりStep 0条件の対応するピーク値の
  半分以下に収まること（目標N_targetの妥当性を実測で確認）。
- 事前登録した数値目標: **res=2cm・batch満杯条件（29.85s）を旧方式6.38sの1.5倍
  （9.57s）以内に短縮**、かつ「batch=5e4→2e5でhistory当たりコストがほぼ一定」に
  なること（現状40µs→149µsの悪化が解消される）。
- git worktreeインターリーブA/B（`0157179`＝旧 / `0c97ab4`＝現行 / 本ステップ後の
  3アーム、各3反復以上、アーム順交互）でこの目標を確認する。

**2026-07-27追記（メモリ受入基準の未測定が`/furikaeri`で発覚・対応済み）**:
Step 2実装当初、上記メモリ受入基準を**測らずに素通りしていた**（速度の受入基準だけ
確認して「達成」と報告）。`/furikaeri`で指摘を受け実測したところ、チャンク化は
np.lexsortのキャッシュ崖を避ける効果はあったが、各チャンクの結果
（seg_id/idx/overlap）を`seg_chunks`/`idx_chunks`/`overlap_chunks`に溜めてから
最後に`np.concatenate`＋`np.add.at`していたため、**ピークメモリはチャンク化前と
実質変わっていなかった**（基準は「半分以下」、実測はむしろ増加）。
対処: チャンクごとに`np.add.at`まで完結させ、次のチャンクへ進む前に中間配列を
破棄する設計に変更（`_segment_grid_traversal_accumulate`、旧`_segment_grid_traversal`
から改名）。ビット一致再確認済み（加算順序はチャンク→区間の処理順のまま変えて
いないため）、全308テストpass。結果はStep 2-3全体の最終数値としてこの計画書末尾の
ステータス・`tally_exact_resolution_growth.txt`「性能回帰の解消」節に記載。

### Step 3: ソートの単一キー化（Step 2で目標未達の場合のみ）

`seg + (t - t_enter) / (t_exit - t_enter)`のような単一の正規化キーを構築し、
`np.lexsort((all_t, all_seg))`を`np.argsort(single_key)`に置き換える案。segが
batch_size程度（〜18bit）に収まるならfloat64の残り仮数部（〜34bit）で時刻の分解能は
足りるはずだが、これは数値的な変更であり、既存の収束オラクルテスト
（`tests/test_tally.py::test_matches_fine_substep_reference_voxel_by_voxel`）に加えて、
**タイ（同一キーになる境界ケース）を明示的に作る専用テスト**を新規に書いてから着手する。

**受入基準**: 新規タイテスト含め全テスト green、かつStep 2までで届かなかった分の
追加高速化が数値目標に届くこと。Step 2で目標達成済みなら本ステップは実施しない。

### Step 4: `np.add.at` → `np.bincount`（最後の手段・見送り）

**2026-07-27追記**: Step 3までで事前登録した数値目標（旧サブステップ方式の1.5倍
以内）を達成・超過した（旧方式より約3%高速）ため、本ステップは実施しなかった。
`np.add.at`はStep 0のプロファイルで全体の1.5%しか占めておらず、加算順序が変わる
リスク（ビット一致保証を壊す）に見合わない。将来さらなる高速化が必要になった
場合のみ、その時点の数値目標と照らして再検討する。

`accumulate_track_length`最後の`np.add.at(target.reshape(-1), flat_idx, weight_flat)`を
`np.bincount(flat_idx, weights=weight_flat, minlength=target.size)`+加算に置き換える案。
浮動小数点の加算順序が変わり得るため、**着手前に**以下を確認する:
- `tests/test_uncertainty.py`等に「`track_uncertainty`のON/OFFでビット一致する」という
  スナップショット差分の保証を検証するテストがあるか（CLAUDE.md該当節参照）を
  grepし、この変更で壊れないか検討する。
- 壊れる場合は、Step 1〜3で目標が達成できていれば本ステップは見送る
  （数値保証を犠牲にしてまで追う優先度ではない）。

**受入基準**: 実施する場合は、数値保証（ビット一致 or 定量化された許容誤差）と
その根拠をこの節に追記してから着手する。

## 測定方法（全ステップ共通）

- git worktree A/B: 対象コミットを`git worktree add --detach <path> <commit>`で作業ツリー外に
  退避し、現行コードと交互に（アーム順を固定しない）3反復以上インターリーブ実行する。
- 条件: `examples/chest_room.yaml`、`--dose-grid`、single worker、Step 0の回帰条件
  （res=2cm, n=2e5, batch=2e5）を主指標とし、CLAUDE.md記載の他の条件（res=5cm既定、
  res=1cm）でも回帰していないことを確認する。
- 各反復で壁時間とピークメモリ（`resource.getrusage(RUSAGE_SELF).ru_maxrss`等）を記録し、
  平均・SD・反復間の符号一致を報告する（単発測定・固定アーム順の報告は禁止——
  `docs/lessons_learned.md`参照）。

## 完了時にやること

- CLAUDE.mdの「Dose/H\*(10)タリー」節にある性能回帰の記述を、実測結果（Step 0の
  原因確認結果、Step 1-2以降の改善後の数値）で更新する。「`np.lexsort`が原因では
  ないか（未確認）」という推測の記述は、Step 0の結果に基づく確定的な記述に置き換える。
- `docs/speedup_baseline/tally_exact_resolution_growth.txt`に本計画の実施結果への
  参照を追記する（新しい実験ログファイルを作る場合はそのパスを記す）。
- auto memory（`future_directions.md`/`project_status.md`）の「性能回帰が未解決」の
  記述を「解決済み・数値X」に更新する。
