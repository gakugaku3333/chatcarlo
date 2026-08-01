# 計画: kernel.py（Numbaカーネル）をchatcarlo run CLIへ接続する（Phase 1: 単色平行ビーム限定）

- 状態: implemented
- 作成者: claude
- 実装担当: codex
- 日付: 2026-08-01（Codexレビュー5回→人間承認→Codex実装2回→Claude独立検証、完了）

## 実施結果（Claudeの独立検証、2026-08-01）

実測値・検証方法の詳細は
[docs/plan_chatcarlo_speedup_post_egs5.md](../../plan_chatcarlo_speedup_post_egs5.md)
の「Phase 1 CLI接続【完了・2026-08-01】」節に記録した。要点:

- 全テスト **374 passed / 0 failed**（新規`tests/test_kernel_cli_dispatch.py` 15件を含む）。
  Codexは自環境で8件failedと報告したが、`ProcessPoolExecutor`のサンドボックス権限制限が
  原因でコード起因ではないことを別環境での再実行で確認した。
- `git diff`で`kernel.py`の**削除行数ゼロ**を確認——既存公開APIの非破壊制約は守られている。
- 速度: kernel 1.32秒 vs numpy 3.17秒（water_phantom_pdd_ocr, n=1e6, JITウォームアップ後）。
- 物理の一致: n=1e6・独立6シードで水+0.038%（0.53σ）・鉛−0.047%（0.88σ）・
  空気−0.398%（0.38σ）。系統差なし。
- **受入テストの検出力に問題を発見し修正した**: クロスチェックの`n_histories`が2,000では
  「6シード・4σ」を形式的に満たしつつ検出限界が水8.9%・鉛5.9%しかなく、
  コンプトン沈着1%過小の変異が素通りした（ミューテーション検証で判明）。
  50,000へ引き上げ検出限界0.7〜1.2%に改善。配線バグ類（per-chunk集約の破壊・
  kernel経路だけ蛍光を握り潰す変異）は確実に検出できることを確認済み。
  1%粒度の系統差は単体テストでは捕捉不能（n=1e6規模が必要）でありEGS5相互検証の
  役割、という分担を`_CROSSCHECK_HISTORIES`のコメントに明記した。
- Codexの実装1回目では受入条件の大半が未テストのまま「実装完了」と報告されたため、
  2回目の委任で不足分を補完させた（＋CLI表示式のバグ1件を修正させた）。

## 目的（1〜3行）

`chatcarlo/kernel.py`（Numba per-historyスカラーカーネル、
`docs/plan_chatcarlo_speedup_post_egs5.md` Phase B完了）は現状`chatcarlo run` CLIから
一切呼べない、実験実装のまま孤立している。本計画は、kernel.pyが対応できる
機能範囲に限定したシーンについてのみ、`chatcarlo run`からopt-inで呼べるようにし、
将来の機能拡張の土台を作る。

## 改訂の経緯（初版からの変更点、実装者は読むこと）

初版はCodexのread-onlyレビューで「現状のままでは実装不可」と判定された。
指摘は全て実物のコード・ファイルで裏取りし、正しいと確認した。以下が確定した事実:

1. **`field.shape: parallel`は単一originではない**。`source.py`の
   `sample_source_photons`は、parallelでも方向`d_a`は全光子共通だが、
   **発生位置は`size_cm`の面上に一様分布**させる
   （`source.py`該当箇所、`su[:,None]*u_a + sv[:,None]*v_a`）。
   `kernel.run_batch`等は単一`origin`しか受け取らないため、初版の
   「kernel.pyを変えずディスパッチだけ追加する」方針では
   `water_phantom_pdd_ocr.yaml`（10×10cm²照射野）を正しく再現できない。
2. **kernelは材料別エネルギー内訳を返さない**。`_transport_one`が積算する
   `energy_deposited`は全材料合算の単一値。`TransportResult`は材料名ごとの
   辞書を返す契約なので、CLI出力互換性のためにはkernel側に材料別集計の
   追加が要る。
3. 初版が参照した`docs/egs5_crosscheck/run_vivemonte_pdd60.py`は存在しない
   （正しくは`run_chatcarlo_pdd60.py`）。しかもこのスクリプトは`run_transport`
   ではなく`transport_photons`を直接呼ぶ独自スクリプトで、**チェックイン済みの
   結果自体が現行物理で再現しないと明記済み**（同ファイル冒頭の警告、
   現行版で再実行すると平均線量で最大2.6%動く、2026-07-26実測）。
   この既知の非対応な基準値をkernel/numpy比較の合否ラインに使うのは無効。
4. B-2（既存のkernelタリー統合検証）は「同一seedで統計誤差内」ではなく、
   **独立6シード・結合4σ**という設計で相互作用エネルギー等を検証していた
   （`docs/plan_chatcarlo_speedup_post_egs5.md`のB-2節）。本計画のクロス
   チェックも同じ設計に揃える。
5. `n_chunks=1`（kernel既定）は実測でシングルスレッド0.51x
   （EGS5比、`docs/plan_chatcarlo_speedup_post_egs5.md`のB-1b節）。
   CLI側で何もしなければ「速くなったつもりが実は遅い」経路になりうる。

これらを踏まえ、**採用する設計をCodexの代替案に切り替える**:
「単色・parallel・box限定のsource adapter」として、`source.py`の
`sample_source_photons`をそのまま呼んでバッチごとの`origins (N,3)`配列を得て、
それをkernelに渡す（方向・エネルギーはバッチ内で単色parallel前提により共通なので
配列化不要）。kernelの物理サンプリングアルゴリズム自体（相互作用選択・角度分布・
蛍光等）は変更しない——変更するのは「入出力のインターフェース」（origin配列受理、
材料別内訳の返却）のみ。

### 改訂の経緯（続き、Codexレビュー2回目を反映）

第2版はCodexの2回目read-onlyレビューで「前回5指摘は解消済みだが、実装方針が
未確定な箇所が4つあり承認保留」と判定された。以下、指摘を検証の上すべて計画に
確定事項として反映した:

6. **`run_dose_grid`とバッチ外側での線源サンプリングの責務が衝突していた**。
   kernel既定の`run_dose_grid`（`kernel.py`）は自分でバッチ分割・シード生成・
   単一origin配布を行う高レベルAPIで、これと「外側が`sample_source_photons`を
   バッチごとに呼ぶ」という新方針は責務が二重になる。**確定: kernel既存の
   `run_dose_grid`/`run_batch`/`run_batch_with_tally`という高レベルAPIは
   kernel経路では使わない。** `transport.py`側に新規のバッチループ
   （`_run_batches`と同様の構造）を書き、各バッチで
   (a) そのバッチ用のRNG（`np.random.SeedSequence(seed).spawn(n_batches)`と
   同じ階層的シード設計を流用）で`sample_source_photons`を呼んでorigin配列を得て、
   (b) kernelの**低レベル関数**（`_run_batch_scalar`/`_run_batch_scalar_tally`、
   または新規origin配列対応版）を直接呼ぶ。バッチseedの所有者は常に外側の
   `transport.py`側バッチループに統一する。
7. **`--dose-grid`なしでもCLI既定`track_uncertainty=True`と矛盾していた**。
   `__main__.py`の`--no-uncertainty`は既定offなので、`--engine kernel`かつ
   `--dose-grid`未指定でも`track_uncertainty=True`のまま呼ばれうる。
   **確定: `--engine kernel`を指定した時点で、`--dose-grid`の有無によらず
   常に不確かさ追跡を強制無効化・警告する**（受入条件を修正、下記）。
8. **Sceneからkernel用材料表を作る手順が計画になかった**。
   **確定: 材料表は「geometry内box物体の材料を出現順で列挙し、background材料を
   末尾に追加、重複除去」という規則で構築する
   （`bake_scene_materials`への入力リスト構築規則として明記）。線源は
   backgroundから始まりうるため、boxに含まれない場合でもbackground材料コードが
   必ず材料表に含まれることをテストで確認する。材料コード→材料名への対応表
   （`SceneMaterialTables.material_names`、既存）をそのまま`TransportResult`の
   辞書キーに使う**。
9. **`n_chunks`が再現性の一部なのにCLIで指定・記録できなかった**。
   **確定: `chatcarlo run --engine kernel`に`--kernel-chunks N`
   （既定`0`=自動、`min(numba.get_num_threads(), 8)`に展開）を追加する。
   実効`n_chunks`の値は`--kernel-chunks`の指定有無によらず必ずCLI標準出力に
   表示する**（受入条件を修正、下記）。
10. **材料別配列の型（`(N, n_materials)` vs 集約済み`(n_materials,)`）が
    「実装時に決める」のまま先送りされていた**。CLI用途は個々のhistoryの
    内訳を必要としないため、メモリ効率の観点から**確定:
    材料別集計は**バッチ単位で集約済みの`(n_materials,)` float64配列**とする**
    （`(N, n_materials)`のようなhistory×材料の巨大配列は作らない）。

### 改訂の経緯（続き、Codexレビュー3回目を反映）

第3版はCodexの3回目read-onlyレビューで「前回指摘1〜3は解消済みだが、4〜5に
実装契約として残る曖昧さがあり、新たにCLIで実行不能なエラー回復手段の欠落も
見つかった」と判定された（承認保留）。特に重大だったのは、**「材料別配列を
`(n_materials,)`に変更する」という決定が、`KernelBatchResult.energy_deposited`の
既存契約（historyごとの配列）を暗黙に破壊する内容になっていたこと**——
`run_batch`/`run_water_slab_probe`と、それらに依存する既存テスト
（`test_energy_conservation_per_history_water`等、`tests/test_kernel.py`に
計5箇所）は`r.energy_deposited[r.absorbed]`のようにhistory単位の配列である
ことを前提にしている。実物のコード・テストで確認し、指摘は正しいと確認した。

11. **`KernelBatchResult.energy_deposited`の移行方法が未決だった**。
    **確定: 既存の`run_batch`/`run_batch_with_tally`/`run_dose_grid`・
    `KernelBatchResult`・`_transport_one`/`_transport_one_tally`・
    `_run_batch_scalar`/`_run_batch_scalar_tally`は**一切変更しない**
    （シグネチャ・戻り値・意味論すべて現状維持、既存テストは無変更で
    全通過する）。** kernel経路のCLI接続専用に、**新規の**低レベル関数
    （例: `_run_batch_scalar_origins_tally`）と**新規の**戻り値型
    （例: `KernelOriginBatchResult`、材料別集計`(n_materials,)`を持つ）を
    追加する。新規関数は`_transport_one`/`_transport_one_tally`と同じ相互作用
    ループ構造を複製することになるが（numba jit関数のため、既存関数への
    追加パラメータで両立させるより、既存契約を一切壊さないことを優先する）、
    複製元は`_transport_one_tally`（最新・タリー対応版）1つに限定し、
    差分は「単一originでなくorigins配列を読む」「材料別配列へ加算する」の
    2点のみとし、それ以外の物理サンプリングロジックは**1文字も変えず
    コピーする**（レビュー時に既存関数とのdiffがこの2点だけであることを
    確認できるようにする）。将来的な保守コスト（同じロジックが2箇所に
    存在する）は認識した上で、Phase 1では「既存の検証済みAPIを絶対に
    壊さない」ことを優先する——コード重複の解消は将来のPhase 2以降で
    検討する（本計画のスコープ外として明示）。
12. **`sample_source_photons`用RNGとkernel物理RNGの二段階分離方法が未定義
    だった**。**確定**: 各バッチについて、外側の`transport.py`バッチループが
    まず`np.random.SeedSequence(top_level_seed).spawn(n_batches)[b]`で
    そのバッチのSeedSequenceを得る。そこから**さらに`spawn(2)`**し、
    `child[0]`を`sample_source_photons`用RNG
    （`np.random.default_rng(child[0])`）に、`child[1]`をkernel低レベル関数へ
    渡す整数シード（`int(child[1].generate_state(1)[0])`、既存
    `run_dose_grid`と同じ導出パターン）に使う。この2つの乱数源は完全に独立
    （spawnの木構造上、兄弟ノードは独立ストリームを生成する設計、numpyの
    `SeedSequence`の仕様）であり、線源サンプリングと物理輸送の乱数消費が
    互いに影響しないことを保証する。
13. **`n_chunks`の「実効値」の定義が欠けていた**。kernel既存の`_chunk_plan`は
    `n_chunks = max(1, min(n_chunks, n_histories))`という丸めを行うため
    （`kernel.py`該当箇所）、最終バッチのhistory数が要求`--kernel-chunks`
    より少なければ実効値は自動的に切り下がる。**確定: CLI出力には
    「設定値（`--kernel-chunks`の指定値または自動決定値）」と
    「最終バッチでの実効値（`min(設定値, 最終バッチのhistory数)`）」の
    両方を表示する**。`--kernel-chunks`に負値または0未満相当の不正値が
    渡された場合はCLIレベルで明示的にエラーにする（`0`は「自動」の意味で
    予約済みなので、負値のみ拒否すればよい）。
14. **区間バッファ溢れ（`max_segments_per_history`超過）時に、CLIから
    調整できる手段がなかった**。**確定: `chatcarlo run --engine kernel`に
    `--kernel-max-segments-per-history N`（既定16、kernel.pyの既定値と同じ）を
    追加する。** エラーメッセージにはこのCLIオプション名を含める
    （kernel.py内部のPython引数名ではなくユーザーが実際に打てるCLIフラグ名を
    案内する）。
15. **「同一seedで入口面分布が一致する」という受入条件の表現が、二段階シード
    導出（指摘12）を踏まえると不正確だった**。kernel経路はバッチごとに
    `SeedSequence`から子を導出するのに対し、numpy経路（`transport.py`従来パス）
    は単一のRNGを逐次消費する設計のため、たとえ同じ`--seed`を与えても
    サンプル列は座標単位では一致しない。**確定: 入口面分布の検証は
    「同一の`sample_source_photons`呼び出しを両経路の共通コードパスとして
    直接比較する単体テスト（同一RNGオブジェクトを渡して完全一致を確認、
    kernel経路が本当に`sample_source_photons`をそのまま呼んでいることの
    確認が目的）」と「独立6シードでのモーメント一致（B-2と同じ統計的検証、
    実際のCLI実行を通した総合確認が目的）」の2段構えにする**（受入条件を修正、
    下記）。

### 改訂の経緯（続き、Codexレビュー4回目を反映）

第4版はCodexの4回目read-onlyレビューで「前回6点は解消済みだが、新たに
2点の実現可能性上の問題が見つかり承認保留」と判定された。実物のコードで
検証し、指摘は正しいと確認した。

16. **`physics.fluorescence`（K殻蛍光X線on/off）の新関数への伝播が
    計画に書かれていなかった**。既存`run_transport`は`scene.raw`から
    `physics.fluorescence`を読み取り輸送関数へ渡す（`transport.py`該当箇所）。
    新規追加するkernel低レベル関数は`_run_batch_scalar_tally`と同様に
    `fluorescence_enabled`引数を持つ必要があり、これをscene設定から読んで
    渡し忘れると、蛍光を無効にしたシーンでkernel経路だけが既定`True`のまま
    走り、numpy経路と物理結果が食い違う。**確定: 新規低レベル関数・
    アダプター関数のシグネチャに`fluorescence_enabled: bool`を明示的に含め、
    `scene.raw`の`physics.fluorescence`（既定`True`、既存の解釈規則を流用）
    から導出して渡す。受入条件のクロスチェックは`fluorescence_enabled=True`
    ・`False`の両方で実施する**（受入条件を修正、下記）。
17. **`prange`並列下での材料別`(n_materials,)`配列への加算が競合状態になる**。
    既存`_run_batch_scalar_tally`は`for c in prange(n_chunks):`で複数チャンクを
    並列実行する設計（`kernel.py`該当箇所、B-0で確定した「チャンクごとに
    独立領域へ書き込みnumpyのprangeでデータ競合が原理的に起きない」設計方針）。
    新規関数で相互作用のたびに共有の`(n_materials,)`配列へ直接加算する設計は
    この前提を破り、複数スレッドから同じメモリへの非アトミックな`+=`となって
    値が不定になる（結果の合計もビット再現性も壊れる）。**確定: 各chunk専用の
    `(n_chunks, n_materials)`累積配列をNumba側（`prange`ループ内）で書き込み、
    Python側でチャンク軸を固定順（`c=0,1,...,n_chunks-1`）に`sum(axis=0)`して
    初めて`(n_materials,)`にする**（既存の区間バッファが`(n_chunks,
    seg_capacity_per_chunk, ...)`という同じ「チャンクごとに独立領域」設計を
    既に使っているのと同型のパターン）。受入条件に「`n_chunks=1`と
    複数chunk（例: 4）で同一シードから得られる材料別合計が一致すること」
    「同一`(seed, n_chunks)`の2回実行が完全に同一の結果になること（ビット再現性）」
    を追加する（受入条件を修正、下記）。

補足（軽微、指摘への対応）:
- `--dose-grid`なしのkernel経路がどの新関数を使うか（タリー版と共用するか、
  区間バッファを一切確保しない専用の軽量版を別途用意するか）を明記する
  （受入条件・実装方針に追記、下記）。
- RNG兄弟ストリームの独立性検証は「先頭値が異なる」という弱い基準ではなく、
  「指定したspawn木構造（`spawn(n_batches)`→各要素をさらに`spawn(2)`）を
  そのまま使っていること」「線源側の乱数消費量（`sample_source_photons`が
  内部で消費する乱数の個数）を変えてもkernel側シードの値が変わらないこと」を
  検証対象にする（受入条件を修正、下記）。
- 区間バッファ溢れは既存kernelでは`ValueError`。CLIがPython tracebackの
  まま露出するのではなく、終了コード1・`--kernel-max-segments-per-history`
  を含む整形済みメッセージまでCLI層で変換することを明記する（受入条件を修正、
  下記）。

## 対象範囲

- 変更してよい:
  - `chatcarlo/transport.py`（`run_transport(..., engine=...)`ディスパッチの追加。
    既存numpyベクトル化パスの計算ロジックは変更しない）
  - `chatcarlo/__main__.py`（`chatcarlo run`への`--engine`オプション追加）
  - `chatcarlo/kernel.py`への**追加**（既存コードの変更は不可、詳細は下記）:
    - **既存の`run_batch`/`run_batch_with_tally`/`run_dose_grid`・
      `KernelBatchResult`・`_transport_one`/`_transport_one_tally`・
      `_run_batch_scalar`/`_run_batch_scalar_tally`・`_chunk_plan`は
      シグネチャ・戻り値・意味論とも一切変更しない**（既存テストが無変更で
      全通過することがこの制約の担保）
    - 新規に、origins配列を受け取り材料別集計を返す低レベル関数を追加する。
      `--dose-grid`ありの場合用（`_transport_one_tally_origins`、
      `_run_batch_scalar_tally_origins`、コピー元`_transport_one_tally`）と
      `--dose-grid`なしの場合用の軽量版（`_transport_one_origins`、
      `_run_batch_scalar_origins`、コピー元`_transport_one`、区間バッファを
      確保しない）の両方を用意し、新規戻り値型（例: `KernelOriginBatchResult`）を
      共通で使う。実装は各コピー元の相互作用ループ構造をコピーし、差分を
      「単一originでなくorigins配列の該当要素を読む」「材料別配列へ加算する」
      「`fluorescence_enabled`引数をそのまま受け取る（既存同様、既に
      パラメータとして存在するので追加ロジック不要）」の3点のみに限定する
      （改訂の経緯・指摘11・16参照）
    - `_run_batch_scalar_tally_origins`（`prange`で複数chunkを並列実行する
      版）は、材料別集計を**chunkごとに独立した`(n_chunks, n_materials)`
      累積配列**へ書き込む（改訂の経緯・指摘17、`prange`のデータ競合回避が
      目的）。この配列をPython側`run_batch`相当の新規ラッパー関数で
      `sum(axis=0)`して`(n_materials,)`にする。既存の区間バッファが
      `(n_chunks, seg_capacity_per_chunk, ...)`という同型の設計を既に
      使っていることと平仄を合わせる。
    - 相互作用選択・角度分布・自由行程サンプリング・蛍光判定など、
      **物理サンプリングのアルゴリズム自体は既存関数からコピーする際も
      1行も変えない**（変えるのは「どの配列を読み書きするか」だけ）
  - `tests/test_kernel.py`, `tests/test_transport.py`, または新規
    `tests/test_kernel_cli_dispatch.py`
  - `docs/plan_chatcarlo_speedup_post_egs5.md`（Phase 1接続の実施記録を追記）
  - `CLAUDE.md`（kernel.pyの説明を更新）

- 変更禁止:
  - `chatcarlo/source.py`, `chatcarlo/physics.py`, `chatcarlo/spectrum.py`
    （既存の線源サンプリング・物理サンプリングロジック。kernel側から
    `sample_source_photons`を**呼び出す**のは許可、中身の改変は禁止）
  - `chatcarlo/geometry.py`, `chatcarlo/tally.py`, `chatcarlo/tally_njit.py`の
    既存関数のシグネチャ・計算ロジック
  - 既存の`transport.py`従来パス（`--engine numpy`が既定、非対応シーンは
    今まで通り従来パスで動く。既存呼び出し元との後方互換を壊さない）
  - `examples/`配下の既存シーンファイル
  - `docs/egs5_crosscheck/`配下の既存結果ファイル・スクリプト
    （`run_chatcarlo_pdd60.py`等、新規追記はOKだが実行方式の変更・
    既存記載数値の書き換えは不可）
  - `chatcarlo.tally`/`chatcarlo.tally_njit`のDDA実装自体（材料別集計は
    kernel側の相互作用ループで行い、タリーDDAには手を入れない）

## 受入条件（検証可能な形で列挙）

- [ ] `chatcarlo run <scene> --engine {numpy,kernel}`を追加。既定は`numpy`
      （ユーザーが明示するまでkernel経路は有効化しない）。
- [ ] シーンが以下の**すべて**を満たす場合のみ`--engine kernel`が動作する:
      - `geometry`が全てbox形状
      - `source.spectrum`が要素数1（単色）、`source.kvp`は使用不可
      - `source.field.shape == "parallel"`
      - `source.mas`, `source.ctdi_vol_mGy`, `source.heel_effect`,
        `source.rotation`のいずれも未指定
      - 実効worker数（`--workers 0`はCPU数に展開された**後**の値で判定）が1
- [ ] 非対応シーン・非対応オプションで`--engine kernel`を指定した場合、
      **`sample_source_photons`もkernel実行もNumbaコンパイルも一切呼ばれる前に**
      具体的な理由を含むエラーメッセージを表示し終了コード1で停止する
      （monkeypatchで両経路とも未呼び出しであることをテストで確認）。
      黙って`--engine numpy`にフォールバックしない。
- [ ] `--engine kernel`は`sample_source_photons`（既存関数、変更しない）を
      呼んでバッチごとのorigin配列を得る。以下の2段構えで検証する
      （改訂の経緯・指摘15、シード導出方式の違いにより「同一seedで座標が
      一致する」という単純な主張は成立しないため）:
      1. `sample_source_photons`を同一のRNGオブジェクトを渡して直接呼び出し、
         kernel経路の呼び出しコード（transport.py側の新規アダプター部分）が
         numpy経路と全く同じ引数・同じ結果になることを単体テストで確認する
         （kernel経路が本当に既存関数をそのまま呼んでいることの直接確認）。
      2. CLI実行を通した独立6シード・結合4σでの入口面分布のモーメント一致
         （下記クロスチェック項目と統合して実施してよい）。
- [ ] `scene.raw`の`physics.fluorescence`（既定`True`、既存の解釈規則を流用）を
      新規kernel関数の`fluorescence_enabled`引数へ明示的に伝播する。
      `fluorescence_enabled=True`・`False`の**両方**でkernel/numpyクロス
      チェック（下記）を実施する（改訂の経緯・指摘16）。
- [ ] 材料別集計はchunkごとに独立した`(n_chunks, n_materials)`累積配列へ
      `prange`ループ内で書き込み、Python側で`sum(axis=0)`して初めて
      `(n_materials,)`にする（改訂の経緯・指摘17）。これを検証するため:
      - `n_chunks=1`と`n_chunks=4`は乱数ストリームが異なる（`_chunk_plan`が
        `SeedSequence(seed).spawn(n_chunks)`でチャンク数ごとに異なる履歴列を
        作るため、既存`test_chunk_count_changes_stream_but_not_statistics`と
        同じ既知の性質）。したがって**独立複数シード・結合誤差（既存B-2と
        同じ6シード/4σ）で材料別合計が統計的に一致すること**を確認する。
        完全一致を要求するのは同一`(seed, n_chunks, batch_size)`の
        再実行のみ（次項）。
      - 同一`(seed, n_chunks, batch_size)`の2回実行が材料別集計を含め
        完全に同一の結果になる（ビット再現性）ことをテストで確認する。
- [ ] `--engine kernel`は**`--dose-grid`の有無によらず常に**不確かさ追跡を
      強制無効化する（`track_uncertainty=True`のまま`--engine kernel`が
      指定された場合は警告を出して自動的にoffにする。`grid.end_batch()`は
      本計画では呼ばない）。`--dose-grid`ありの場合はCLI表示・`--dose-out`の
      `.npz`のいずれにも`rel_err_*`/`sem_*`/`n_batches*`が出力されないこと、
      `--dose-grid`なしの場合も材料別SEM等の統計出力が出ないことをテストで確認する。
- [ ] kernel/numpy間のクロスチェックはB-2と同じ設計に揃える:
      **独立6シード実行・結合4σ**で以下が一致することを確認する
      （`docs/plan_chatcarlo_speedup_post_egs5.md`のB-2節の許容基準を踏襲）:
      - 材料別吸収エネルギー合計（複数box・異材料・重なり後勝ちを含むシーンで）
      - `--dose-grid`のグリッド合計カーマ・H\*(10)
      - 蛍光X線放出イベント数（**銅または鉛を含むシーンで**——水では蛍光が
        ほぼ出ないため検証にならない、とのCodex指摘を反映）
      ビット一致は要求しない（乱数消費順序が異なるため原理的に不可能）。
- [ ] `--engine kernel --workers 0`（CPU数への展開後、実効値>1になりうる）は
      互換性判定でエラーになることをテストで確認する。
- [ ] `chatcarlo run --engine kernel`に`--kernel-max-segments-per-history N`
      （既定16）を追加する。`--dose-grid`使用時、区間バッファ溢れ
      （このオプション値の超過）が、**このCLIオプション名を含む**
      アクショナブルなエラーメッセージとしてCLIレベルまで伝播することを
      確認する（改訂の経緯・指摘14、CLIから調整不能だった問題への対応）。
      既存kernelは`ValueError`を送出するのみなので、`__main__.py`側で
      これを捕捉し、Python tracebackをそのまま露出せず終了コード1・
      `--kernel-max-segments-per-history`という具体的なCLIフラグ名を含む
      整形済みメッセージへ変換することをテストで確認する。
- [ ] `--engine kernel --dose-grid`が複数バッチにまたがる条件
      （`n_histories > batch_size`）で、parallel面分布込みの6シード/4σ比較を行う
      （B-2の既存検証は鉛筆ビーム限定だったため、面分布adapterを通した
      複数バッチ条件は本計画で新規に検証が必要、とのCodex指摘を反映）。
- [ ] `water_phantom_pdd_ocr.yaml`のように、線源の発生面がbox入口面と
      ちょうど一致するシーンで、境界包含規則（`_material_at_scalar`の
      境界上判定）による系統的な過不足がnumpy経路と比べて生じないことを確認する。
- [ ] kernel経路の速度計測方法を固定する:
      - `chatcarlo run --engine kernel`に`--kernel-chunks N`を追加する
        （既定`0`=自動、`min(numba.get_num_threads(), 8)`に展開。負値はCLIで
        明示的にエラーにする）。CLI標準出力には**「設定値」と「最終バッチでの
        実効値（`_chunk_plan`の丸め`min(設定値, 最終バッチのhistory数)`適用後の
        値、改訂の経緯・指摘13）」の両方を必ず表示する**
        （`(seed, n_chunks設定値, batch_size)`の組でユーザーが再現できるように）。
        `n_chunks`が変わると乱数ストリームが変わり結果が変わりうることを
        `tests/test_kernel.py`の既存`test_chunk_count_changes_stream_but_not_statistics`
        と同じ扱いでドキュメント化する。
      - Numba初回JITコンパイル時間を計測から除外する（ウォームアップ実行を
        1回挟む、または計測結果に「初回コンパイル時間別記」を明記する）
      - `water_phantom_pdd_ocr.yaml`で`--engine kernel`が`--engine numpy`より
        実測で速いことを1回記録する（具体的な倍率目標は設けない。未達でも
        本計画は不合格にしない——主目的は配線の正しさ）
      - `--kernel-chunks`を同一`(seed, batch_size, kernel_chunks)`で2回実行し、
        完全に同一の結果になることをテストで確認する（再現性の担保）。
- [ ] `run_dose_grid`/`run_batch`/`run_batch_with_tally`というkernel既存の
      高レベルAPIはkernel経路のCLI接続では使わない（内部で呼ばれてもいない）。
      バッチ分割・シード生成は`transport.py`側の新規バッチループが担い、
      各バッチで`sample_source_photons`と新規追加した低レベル関数
      （例: `_run_batch_scalar_tally_origins`、改訂の経緯・指摘11参照）を
      呼ぶ構造になっていることをコードレビューで確認する（テスト化しにくい
      構造的受入条件のため、差分レビュー時に対象範囲・実装方針との適合として確認）。
- [ ] バッチごとの乱数源が二段階に分離されていることを確認する
      （改訂の経緯・指摘12: `SeedSequence(top_seed).spawn(n_batches)[b]`から
      さらに`spawn(2)`し、`child[0]`を`sample_source_photons`用、`child[1]`を
      kernel低レベル関数の整数シードに使う）。「先頭値が異なる」という弱い
      基準ではなく、以下をテストで確認する（改訂の経緯・指摘17の補足）:
      - 指定した通りのspawn木構造（`spawn(n_batches)`→各要素をさらに
        `spawn(2)`）がコード上そのまま使われていること。
      - `sample_source_photons`が内部で消費する乱数の個数を変えても
        （例: monkeypatchで線源RNGの消費回数を変える）、kernel側へ渡される
        整数シードの値が変わらないこと（2つの乱数源が独立であることの
        直接検証）。
- [ ] `--dose-grid`なしのkernel経路がどの新関数を使うかを明記する:
      区間バッファを確保しない軽量版（`_run_batch_scalar_origins`、
      `_transport_one_origins`——タリーなし版のコピー元は既存`_transport_one`）を
      別途用意し、`--dose-grid`ありの場合のみタリー版
      （`_run_batch_scalar_tally_origins`）を使う。両者の材料別集計・
      物理結果が同一条件下で一致することをテストで確認する。
- [ ] kernel用材料表は「geometry内box物体の材料を出現順で列挙、background材料を
      末尾に追加、重複除去」の規則で構築されることをテストで確認する
      （線源がbackground材料中から始まり、boxがそのbackground材料を1つも
      含まないシーンでも材料表にbackgroundが含まれることを含む）。
- [ ] 既存の全テスト（`pytest tests/ -q`）が引き続き全通過する。
- [ ] `--engine numpy`（既定パス）が、本計画の変更前後で完全に同一の出力を返す
      回帰テストを追加する（`--engine`引数追加がexisting呼び出しに影響しないことの確認）。

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel -n 1e6 --seed 42
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel --kernel-chunks 4 -n 1e6 --seed 42
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine kernel --kernel-max-segments-per-history 32 -n 1e6 --seed 42 --dose-grid --resolution 2
.venv/bin/python -m chatcarlo run examples/water_phantom_pdd_ocr.yaml --engine numpy -n 1e6 --seed 42
.venv/bin/python -m chatcarlo run examples/chest_room.yaml --engine kernel -n 1e5 --seed 42
# ↑ chest_room.yamlはkvp/rect/heel_effect等を使うため互換性エラーで終了コード1になることを確認
```

## 実装方針

- ディスパッチは`transport.py`の`run_transport`冒頭に集約する。互換性判定は
  独立関数（例: `kernel_engine_compatible(scene, n_workers) -> tuple[bool, str]`、
  非対応理由の文字列を返す）として切り出し、`__main__.py`とテストの両方から
  再利用できるようにする。
- kernel経路のバッチループは`transport.py`側の新規コードとして実装する。
  各バッチについて: (1) そのバッチの`SeedSequence`を`top_seed`から`spawn`し、
  さらに`spawn(2)`で線源用/kernel用の2子に分ける（改訂の経緯・指摘12）、
  (2) 線源用子から作った`np.random.default_rng`で`sample_source_photons`
  （既存、変更禁止）を呼んでenergies/origins/dirsを得る、(3) 単色parallel
  互換判定によりenergies/dirsがバッチ内で定数であることを表明（assertまたは
  検証）した上でスカラー`energy0_kev`/`direction`とorigin配列に分解し、
  kernel用子から導出した整数シードとともに新規低レベル関数へ渡す。
  「線源分布の解釈」自体は`source.py`に委譲し続けるため、二重実装は生じない。
- kernelの材料別内訳は、`kernel.py`に**新規追加**する低レベル関数
  （`_transport_one_tally`をコピーして作る、既存関数は変更しない、
  改訂の経緯・指摘11）の相互作用ループ内で「エネルギー付与が起きた時点の
  mat_idx」に対応する`(n_materials,)`配列の要素へ加算する形にする
  （コピー元の`mat_idx`変数はループ内で既に追跡されているため、新規に
  材料判定ロジックを足す必要はない——加算先を単一スカラーから配列要素に
  変えるだけがコピーとの差分になる）。

## 書かなかったこと（スコープ外を明示）

- 分光スペクトル（`source.kvp`）のper-historyサンプリングをkernel.pyに追加すること
- rect/cone等の発散照射野（origin固定・方向可変）のkernel対応
- CT回転・ヘリカル・ヒール効果・mAs/CTDIvol校正のkernel対応
- cylinder/sphere形状のkernel対応
- `--workers`マルチプロセスとkernel経路の統合
- `--engine kernel`での統計不確かさ（R・SEM）の実際の出力
  （`grid.end_batch()`統合は将来のPhase 2）
- `--engine auto`のような自動選択モード
- kernel.py自体のさらなる高速化（DDA再実装等）
