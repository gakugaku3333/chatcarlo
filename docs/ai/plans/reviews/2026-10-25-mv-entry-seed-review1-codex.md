計画は実現可能ですが、**現状のままの承認は勧めません。致命は0件、実質9件、軽微2件、参考3件**です。隔離された研究・教育用試作という枠組みで評価しました。ファイルは変更せず、Colab・計算・pytestは実行していません。

まず§0の照合結果です。

| 項目 | 実ファイルでの確認結果 |
|---|---|
| 1・2：既存入口・承認 | 記載と一致。`__main__.py:487` の振り分け、承認前の実行停止も確認しました。 |
| 3：描画できる形状 | **不一致**。球の描画もあります。さらに描画器の入力は最終的には頂点・辺・面なので、楕円形状を描くために箱へ近似する必要はありません。 |
| 4：P9cの関門 | **一部不一致**。productionでも保存済み `gates.json` があれば再利用し、V1c・V2bを毎回再計算しません。 |
| 5：束・T0 | ファイルは存在。manifestの全SHA256、`input.sources` と現在のfingerprint、束内の対応コード、T0のsourcesは一致。`require_t0` も通りました。ただし同関数は記載された条件以外も検査します。 |
| 6：速度・容量 | 空き11 GiB・95%は一致。36〜38 µs/履歴は指定のRESULTS本文では裏づけられず、既存 `prod_prostate` の保存タイミングからの集計は約44.1 µs/履歴でした。 |
| 7：図・数値の関数 | **不一致**。図作成側はトップレベルで実行するスクリプトです。`combined.py` は `P9C_RUNS` のみを読み、固定ディレクトリを前置します。`P9C_OUT` は図作成側が使います。 |

以下が修正すべき指摘の全件です。

1. **【実質】`ssd_cm: 100` で承認する条件が、実計算ではSAD 100 cmになる。**  
   [計画§1](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-25-mv-entry-seed.md:33)に対し、[実装の線源位置](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_core.py:79)は原点から100 cmです。0°の中心線では体表まで88 cm、90°では82 cmで、SSD 100 cmではありません。新sceneを `sad_cm: 100` に変更し、プレビューにもSADと表示してください。7門の線源位置・方向・体表までの距離を検査するテストが必要です。

2. **【実質】既存図スクリプトをそのまま使う案は、許可された変更範囲では成立しない。**  
   [combined.py](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/p9c_figures/combined.py:3)に絶対パスを渡しても、`~/.cache/chatcarlo/p9c2/` が前置されます。[make_top3.py](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/p9c_figures/make_top3.py:8)は過去の `prod_prostate/prostate_summary.json` も固定で読みます。新出力だけを使った解析にならず、過去データがない環境では失敗します。  
   **簡素案:** 今回は1チェックポイントだけなので合算処理を外し、`seed_run.py` から明示的な入力パスを受ける解析関数を使う。既存図コードを関数化して再利用するなら、`p9c_figures/` を変更許可範囲へ追加してください。「旧キャッシュなし・任意の `--out`」で図とJSONまで完成するテストを追加すべきです。

3. **【実質】光子数・種は保存されている。照合先を `job.json` だけにすると保証が抜ける。**  
   [チェックポイントのidentity](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_colab.py:191)には `n`・`seed`・`angles`・`grids`・`beam`・`sources`・`bundles` があり、`completed` にバッチ番号があります。`job.json` には光子数・種がありませんが、それを理由に持ち越す必要はありません。  
   `prostate.npz` のidentity全項目と `completed == list(range(100))` を照合し、summaryの `checkpoint_sha256` も結び付けてください。種・光子数・格子・角度・資材の異なるチェックポイント、欠番、重複、古いsummaryを拒否するテストが必要です。

4. **【実質】「productionの関門がすべて合格」の定義が実際の出力と合わない。**  
   [validate_gates](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_colab.py:25)はV2bの `insufficient-statistics` ＋ `fallback: V2a` を許容します。既存結果もこの状態です。またV2a・V6・V6bは束の `input.local`、GPU側の関門は `job.gates`、本計算の `V3_CUDA_acceptance` はsummaryに分かれます。  
   必須キー・参照ファイル・許容状態を列挙してください。既存の `validate_gates(..., required=True)` と束検査を再利用し、summaryのV3も別途確認する形が単純です。欠落・fail・正当なV2b代替を区別するテストが必要です。

5. **【実質】T0の承認対象が、実際に送る3ファイルを覆っていない。**  
   計画は `job.json` だけをハッシュしますが、[アップロード処理](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_cli.py:145)は `gates.json`・`rectangle_summary.json` も送ります。現在のpartial-accepted T0では、それらも存在・内容検査に関係します。  
   3ファイルを承認ハッシュに含め、実行直前に再照合してください。各ファイルの変更・欠落と、partial-acceptedの理由欠落を試験対象に加えるべきです。今回のT0がpartial-acceptedであること自体は問題ではありません。

6. **【実質】プレビューと計算の座標対応が未定義で、寸法一致テストだけでは取り違えを検出できない。**  
   sceneの前立腺寸法 `[4.5, 3.0, 3.5]` は左右・前後・頭尾と読めますが、[計算のxyz軸](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_core.py:13)では全径 `[4.5, 3.5, 3.0]`、頭尾はyです。直腸の評価範囲も[計算側](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_colab.py:137)では中心z=3、頭尾長3.5 cmです。  
   軸順を明記し、シードは規則を再実装せず `core.seeds()` を直接使ってください。前立腺・直腸は水中の**評価領域**であることも表示する。軸順、直腸中心・長さ、シードの半径・長さ・方向、7門すべてをテストする必要があります。

7. **【実質】共通セッション名 `p9cent` と任意の `--out` に、同時実行・再開の規則がない。**  
   [既存driver](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c_cli.py:149)は出力内のファイルを再送し、セッション名には再試行番号だけを付けます。別条件の入口実行でもVM名が衝突します。これはmv.mdに記録された既知の事故と同型です。  
   セッション接頭辞を条件ハッシュから作り、出力には条件との一致確認と実行ロックを設けてください。承認HTML等の置き場とGPU結果ディレクトリも分けるのが単純です。二重実行、別条件の既存出力、同条件の部分再開を試験してください。

8. **【実質】低履歴数で未定義になる比・SEMと、距離ごとの集計範囲が未定義。**  
   最小値では各門・各バッチ1光子です。ゼロ分母によりNaNが出ても不思議ではなく、単に「SEMが大きければ書く」ではJSON出力の仕様になりません。既存 [make_radial.py](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/p9c_figures/make_radial.py:20)も、0.6/0.9/1.4/2 mmは丸めた区間中心であり、`3 mm以上` は別の集計です。  
   各値に実際の距離区間・セル数・比・SEMの定義を持たせ、未定義は `null` と理由を保存してください。全ゼロ・ゼロ分母・有効セルなしでも報告まで完了するテストが必要です。

9. **【実質】目的の「入口からColabまで通した実績」が受入条件にない。**  
   §5は偽クライアント・validate・ブラウザ表示までで満たせます。実GPU実行が§8でCodexの作業外なのは妥当ですが、計画全体の受入からも抜けています。  
   「Codexの実装完了」と「Claudeによる実証完了」を分け、後者に使用者の3D承認、実Colab完走、回収物照合、数値・図の提示、RESULTS記録を追加してください。前者だけで計画全体を完了扱いにしない条件が必要です。

10. **【軽微】承認後変更テストの期待エラーが、既定出力先の仕様と矛盾する。**  
    種・履歴数を変えると出力先ハッシュも変わるため、通常は旧承認とのmismatchより先に「承認なし」になります。§5(c)は同じ `--out` を指定した場合と、既定出力先の場合を分けてください。HTML・snapshotの承認後改変、見えるハッシュの削除も追加すると、§3の要件を直接検証できます。

11. **【軽微】型・境界テストと変更範囲の判定方法が不足する。**  
    上限超過、seedの両境界、bool、浮動小数の履歴数、null、mappingでないroot、角度の重複・順序、各階層の未知キーも必要です。`git diff --stat` は未追跡の新ファイルを拾わず、「旧経路を1文字も変更しない」ことも証明しません。今回の計画自体も未追跡でした。  
    テストパスを `prototypes/mv_electron/tests/test_entry_seed.py` に統一し、変更範囲は未追跡を含むファイル一覧とdiffで確認してください。

12. **【参考】箱24枚の近似は不要に複雑。**  
    [描画JSON](/Users/oishifamily/Projects/viveMonte/chatcarlo/preview.py:90)へ楕円柱の辺・面を直接作り、前立腺は球のワイヤーを軸別に拡大すれば、本体変更なしで表示できます。箱を積む方式でも試作として許容できますが、24枚という数だけでは表示精度を測れません。直接メッシュ化した方が内部境界と大量ラベルも減らせます。

13. **【参考】既存の統計近似・物理上の制約は、この計画の実質欠陥にはしない。**  
    セル間相関を無視した距離別SEM、少ない光子数、近傍EGS5照合なし、P9b不合格の継続は、今回の枠組みで許容されています。ただし `run_record.json` と図にも「セル独立近似」と既知の制約を残すべきです。チェックポイントの種・履歴数はメタデータであり、既存入口のような実際のGPU引数の独立観測と同じ強さではないため、保証範囲の表現も区別してください。

14. **【参考】速度・seed上限の説明を正確にする。**  
    36〜38 µs/履歴は測定条件と出典を付け、所要時間の確約には使わないでください。`seed < 2⁶¹/8` は[実装の許容範囲](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p9c.py:38)である `seed < 2⁶¹` より8倍厳しい制限です。安全側なので支障はありませんが、意図的な入口制限と明記すれば十分です。

枠組み自体への異議はありません。隔離された研究・教育用の入口実証に、臨床用途の検証水準を要求する必要はありません。今回の実質指摘は、主として「承認した条件との一致」と「指定した入力だけで処理を完了できること」という、この計画自身の約束に関するものです。