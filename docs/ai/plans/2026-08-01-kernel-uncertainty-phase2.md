# 計画: kernel経路に統計不確かさ（R・SEM）を統合する（Phase 2）

- 状態: implemented
- 作成者: claude
- 実装担当: codex
- 日付: 2026-08-01（Codexレビュー2往復→人間承認→Codex実装→Claude独立検証、完了）

## 実施結果（Claudeの独立検証、2026-08-01）

- 全テスト **382 passed / 0 failed**（Codexは自環境で5件failedと報告したが、
  Phase 1と同じ`ProcessPoolExecutor`のサンドボックス権限制限が原因でコード起因では
  ないことを再実行で確認）。
- **変更禁止範囲は完全に無傷**: `git diff --numstat`で`chatcarlo/tally.py`・
  `chatcarlo/kernel.py`とも変更行数0を確認（Phase 1をコミット済みにしたため
  差分ベース検証が正確に効いた）。実装は`transport.py`に9行追加・
  `__main__.py`から3行削除という最小変更。
- **R/SEMの値をブルートフォース照合して合格**: n=6,000・batch=1,000（M=6）で
  kernel経路のバッチ寄与S_bを同じシード規約（SeedSequence spawn木）で外部から
  再現し、(a)推定式 σ̂²=(Q−T²/N)/(M−1), SEM=√(σ̂²/N) と (b)等バッチ限定の
  素朴式 std(ddof=1)/√M/n_b の**2通りの独立な導出**が、報告値と
  `rtol=1e-9`で完全一致した（water 2.778502e-04、air 2.546802e-05 MeV）。
- **絶対制約（統計ON/OFFのビット一致）を最も厳しい条件で確認**: 端数バッチ
  （n=2,500=1,000+1,000+500）＋`n_chunks=4`＋dose-gridありで、材料別エネルギー・
  `kerma_keV`・`h10_track_pSv_cm3`がすべてビット一致。`n_batches=3`・
  `n_histories=2,500`も期待通り。
- **ミューテーション検証を追試して両方とも検出を確認**: 「`end_batch`呼び出しを削除」
  →2件failed、「`add_batch`に渡すnを1に固定」→端数バッチ検証がfailed。
- 実CLI: n=5e5・batch=5e4で最大線量R=0.009（寄与バッチ10/10）、
  グリッド信頼性サマリ、`.npz`の統計キー（`rel_err_dose`/`rel_err_h10`/
  `sem_dose_per_history_Gy`/`sem_h10_per_history_pSv`/`n_batches`/`n_batches_hit`）を確認。
- 性能（Codex計測、ON/OFF交互3反復・中央値）: ON 2.6130s / OFF 2.6054s、+0.30%
  ——**反復ばらつき以下のため「測定分解能以下」**（`docs/plan_statistical_uncertainty.md`
  Phase 4のnumpy経路での結論と整合）。

## 目的（1〜3行）

Phase 1（`docs/ai/plans/2026-08-01-kernel-cli-wiring-phase1.md`、コミット`1872be4`）で
`chatcarlo run --engine kernel`を接続したが、統計不確かさ追跡を強制無効化したままにした。
高速化の当初の動機は「鉛遮蔽背後でも相対誤差数%が現実的な時間で出る」ことであり、
**Rを出せないエンジンはその動機を満たさない**。本計画でkernel経路にR・SEMを配線する。

## 前提（実装者はここを読んでから着手すること）

- 統計機構の設計・不偏推定器の導出は`docs/plan_statistical_uncertainty.md`（Phase 0-4完了）
  にある。**本計画は新しい推定器を作らない**——既存の監査済み機構を配線するだけ。
- `VoxelGrid.end_batch(n)`は**スナップショット差分方式**で、`kerma_keV`/
  `h10_track_pSv_cm3`そのものを一切書き換えない（`chatcarlo/tally.py`のdocstring参照）。
  これが「統計ON/OFFでtotalがビット一致する」という絶対制約の根拠であり、
  kernel経路でも同じ制約を守る（受入条件に明記）。
- `ScalarMoments.add_batch(batch_values, n)`は「材料名 -> このバッチでの寄与和 S_b」を
  受け取る。**kernel側は既にバッチごとの`result.energy_deposited_by_material`を
  返しているので、S_bはそのまま手に入る**（`transport.py`の`_run_kernel_batches`が
  現在これを累積しているだけ）。新規に集計ロジックを書く必要はない。
- numpy経路の呼び出し方（`_run_batches`内）が参照実装。**実際の順序は
  `energy_moments.add_batch(result.energy_deposited, n)` → `grid.end_batch(n)`**
  （Codexレビューでの指摘。初版は逆順に書いていた誤記。両者に相互依存はないので
  結果は変わらないが、参照実装と同じ順に揃える）。ガード条件も参照実装に合わせる:
  `add_batch`は`if track_uncertainty:`、`end_batch`は`if grid is not None:`
  （`end_batch`は`track_uncertainty=False`なら内部でno-opになる設計）。

## 対象範囲

- 変更してよい:
  - `chatcarlo/transport.py`（`_run_kernel_batches`へのend_batch/add_batch配線、
    `run_transport`のkernel分岐から強制無効化を除去）
  - `chatcarlo/__main__.py`（kernel指定時の強制無効化・警告の除去）
  - `tests/test_kernel_cli_dispatch.py`（または新規テストファイル）
  - `docs/plan_chatcarlo_speedup_post_egs5.md`（Phase 2の実施記録を追記）
  - `CLAUDE.md`（kernel経路がR非対応という記述の更新）
  - `docs/ai/plans/2026-08-01-kernel-cli-wiring-phase1.md`（Phase 1の「R非対応」
    記述に、Phase 2で解消した旨の追記のみ。**過去の判定・実測値は書き換えない**）

- 変更禁止:
  - `chatcarlo/tally.py`（`VoxelGrid.end_batch`・`ScalarMoments`・`relative_error`等の
    推定器機構。**本計画は配線のみで、推定器そのものには一切触らない**）
  - `chatcarlo/kernel.py`（Phase 1で追加した関数を含め、今回は変更不要のはず。
    もし変更が必要になったら、それは設計の読み違いなので**手を広げず停止して報告する**）
  - `chatcarlo/source.py`, `chatcarlo/physics.py`, `chatcarlo/spectrum.py`,
    `chatcarlo/tally_njit.py`, `chatcarlo/geometry.py`
  - Phase 1で確立した互換性判定の条件（box限定・単色・parallel・worker 1）——
    今回スコープを広げない
  - `docs/egs5_crosscheck/`配下の既存結果

## 受入条件（検証可能な形で列挙）

- [ ] `--engine kernel`指定時の`track_uncertainty`強制無効化と警告を除去する
      （`transport.py`・`__main__.py`の両方）。`--no-uncertainty`は従来通り効く。
- [ ] **【最重要・絶対制約】統計ON/OFFでtotalがビット一致すること**:
      同一`(seed, n_chunks, batch_size, n_histories)`で
      `track_uncertainty=True`と`False`を実行し、材料別吸収エネルギー・
      `grid.kerma_keV`・`grid.h10_track_pSv_cm3`が**ビット一致**することをテストで確認する
      （`np.array_equal`。統計機構がtotalに一切影響しないというスナップショット差分方式の
      設計制約をkernel経路でもコードに固定する）。
- [ ] `--engine kernel --dose-grid`でR/SEMマップが出力される:
      - CLI標準出力に最大線量・最大H\*(10)の隣のR・寄与バッチ数、グリッド全体の
        信頼性サマリが表示される（numpy経路と同じ表示経路を通る）。
      - `--dose-out`の`.npz`に`rel_err_dose`/`rel_err_h10`/`sem_*`/`n_batches`/
        `n_batches_hit`が含まれる。
      - `plot --quantity relerr-dose`が実際に描画できる（生成した.npzで確認）。
- [ ] `--engine kernel`（`--dose-grid`なし）で材料別SEM・Rが出力される。
- [ ] **推定器の値が正しいことをブルートフォースで検証する**（配線ミスで
      「それらしいが誤ったR」が出る事故を防ぐ）: 小規模条件（例 n=6,000,
      batch_size=1,000でM=6）でkernel経路を走らせ、各バッチの材料別寄与を
      独立に記録して`std(ddof=1)/sqrt(M)`相当を手計算した値と、
      `ScalarMoments.standard_errors()`が返す値が一致することをテストで確認する。
- [ ] R自体がkernel/numpy間で統計的に整合すること: 同一シーン・同一
      `(n_histories, batch_size)`で両エンジンを独立6シードで走らせ、
      材料別Rの平均が結合4σ以内で一致することを確認する
      （Rは分散の推定量なのでばらつきが大きい。4σで有意差が出た場合は
      **合格に倒さず停止して報告する**）。
      なお`n_histories`はPhase 1の`_CROSSCHECK_HISTORIES`（50,000）と同じ規律で選び、
      検出力が不足していないかミューテーションで確認すること（下記）。
- [ ] **テストの検出力をミューテーションで確認する**（Phase 1の教訓、
      同計画の「受入テストの検出力について」節参照）: 少なくとも
      「`end_batch`の呼び出しを削除する」「`add_batch`に渡す`n`を誤る（例 n→1）」の
      2つの変異を注入し、追加したテストが実際に失敗することを確認して報告に含める。
      検出できない変異があれば、テスト条件（n_histories・バッチ数）を調整するか、
      **検出できない旨をコメントに正直に書く**（Phase 1の`_CROSSCHECK_HISTORIES`の
      コメントが手本）。
- [ ] **端数バッチの検証**（Codexレビュー指摘1、推定式は可変`n_b`を明示的に扱うため
      ここを通さないと最重要の配線誤りを見逃す）: `n_histories=2,500`・
      `batch_size=1,000`で実行し、最終バッチの`n_b=500`が`add_batch`・`end_batch`の
      **両方**に正しく渡ること、結果として`n_batches=3`・`n_histories=2,500`に
      なることをテストで確認する。
- [ ] **`kernel_chunks>1`＋端数バッチ＋dose-gridでの統計配線の検証**
      （Codexレビュー指摘2、Phase 1は複数chunkの物理値のみ確認済みで統計配線込みは未確認）:
      `n_chunks=4`・端数バッチあり・`--dose-grid`ありの条件で、
      (a) 統計ON/OFFのtotalビット一致、(b) 同一seedでの`kerma_sum2`/`h10_sum2`の
      再現性、をテストで確認する。
- [ ] **kernel経路での`--no-uncertainty`のCLI検証**（Codexレビュー指摘3、既存の
      `tests/test_cli_uncertainty.py`はnumpy既定経路のみ）: `--engine kernel`＋
      `--no-uncertainty`でR表示が消え、`.npz`から統計キー（`rel_err_*`/`sem_*`/
      `n_batches*`）が消えることをテストで確認する。
- [ ] M=1（バッチが1つしかない）条件でkernel経路が、NaNを黙って出すのではなく
      numpy経路と同じアクショナブルなメッセージ（`batch_shortage_message`）を
      出すことをテストで確認する。
- [ ] 既存の全テストが引き続き全通過する（Phase 1で追加した15件を含む）。
      特にPhase 1の`test_kernel_numpy_six_seed_crosscheck_*`が、統計ONが既定に
      戻ったあとも通ること（これらは`track_uncertainty=False`を明示しているので
      影響を受けないはずだが確認する）。
- [ ] 統計ONによるkernel経路の壁時間オーバーヘッドを1回計測して記録する
      （`docs/plan_statistical_uncertainty.md` Phase 4の教訓に従い、
      **同一セッション内でON/OFFを交互に3反復以上**し、中央値で比較する。
      1回測定の差を報告しない。差が測定分解能以下なら「分解能以下」と正直に書く）。

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest tests/ -q
# R付きで実行できること（M>=2になるようbatch-sizeを設定）
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel \
    -n 1e6 --batch-size 50000 --seed 42 --dose-grid --resolution 2 --dose-out /tmp/k.npz
.venv/bin/python -m chatcarlo plot /tmp/k.npz --scene examples/water_phantom_pdd_ocr.yaml \
    --quantity relerr-dose -o /tmp/k_relerr.png
# --no-uncertainty が従来通り効くこと
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel \
    -n 1e6 --batch-size 50000 --seed 42 --no-uncertainty
```

## 実装方針

- `_run_kernel_batches`のシグネチャに`energy_moments: ScalarMoments | None`を追加し、
  バッチループ内で以下を**numpy経路（`_run_batches`）と同じ順序**で呼ぶ:
  1. kernelでバッチを実行（既存）
  2. `energy_moments.add_batch({material_name: S_b}, n)`（`track_uncertainty`が
     真のとき。S_bは`result.energy_deposited_by_material`を材料名へzipしたもの
     ——**バッチ累積後の値ではなくそのバッチ単体の寄与**を渡すこと。
     `energy += result...`の累積とは別に、渡す辞書はバッチ単体の値であること）
  3. `grid.end_batch(n)`（`grid is not None`のとき）
  `n`は**そのバッチの実際のhistory数**（最終バッチでは端数になる）を渡すこと
  ——ここを`batch_size`で固定すると端数バッチで統計が静かに誤る（受入条件参照）。
- `run_transport`のkernel分岐から`warnings.warn`＋`track_uncertainty = False`を削除。
  `energy_moments`は既存の`ScalarMoments() if track_uncertainty else None`が
  そのまま使える（numpy経路と共有）。
- `__main__.py`のkernel用強制無効化ブロックを削除。`kernel chunks:`の表示は残す。

## 書かなかったこと（スコープ外を明示）

- cylinder/sphere形状のkernel対応（実シーン解禁の本丸。別計画）
- 分光スペクトル（`source.kvp`）・rect/cone照射野・mas/ctdi校正のkernel対応
- `--workers`並列とkernel経路の統合（並列時のモーメント合成`combine_moments`は
  既存だが、kernel経路は依然worker 1限定のまま）
- 推定器そのものの改良（`docs/plan_statistical_uncertainty.md`のスコープ）
- kernel.pyのさらなる高速化・DDA再実装
