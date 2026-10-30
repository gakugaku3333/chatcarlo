**実現可能な構成ですが、§5以降には、そのまま実装すると判定や検査を誤る契約のずれが残っています。致命0件・実質10件・軽微3件です。** 前回解消済みの論点と、承認済みのE2は指摘から除外しました。

AGENTS.md・CLAUDE.md・関連ルール、前回レビュー、指定関数と呼び出し側を確認しました。ファイルは変更していません。関数署名、配列寸法、固定ケース数、判定の短絡動作は、書き込みを伴わないPythonでも確認しました。GPU計算と全テストは実行していません。

1. **【実質】P7は「配列の形をそろえる」だけでは判定関数を呼べません。**

   [計画§5.1・作業3](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:152)の説明には、付与エネルギーから線量への変換が抜けています。[`analysis_p7.dose`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/analysis_p7.py:55)は、各配列を **`MEV_TO_GY / ((N/100) × VOLUMES[key])`** で正規化します。`compare`のDrefはGy/historyなので、生の和を渡すと単位が合いません。

   また、既存の本番解析は`compare`の直後に`stop_on_missing`を呼び、bootstrapで50%交点が一度でも欠けたら停止します。作業3にはこの停止契約がありません。

   **修正案:** 新出力→既存の5領域配列→`dose`→`compare`→`stop_on_missing`→既存の分類、という薄いアダプターを明記する。`symmetry`にも正規化後のprofileを渡す。Nの誤り、体積の取り違え、交点なし、bootstrapの一部だけ交点なしをテスト対象にしてください。

2. **【実質】積層`judge`の入力は100×19ではなく、100×220です。`dref`も必須です。**

   [計画§5.1](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:157)の「100×ブロック数」は実装と異なります。[実際の署名](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_core.py:80)は`judge(prototype, reference, grid, *, dref)`で、デフォルト値はありません。入力は既存gridの **220採点binの線量（MeV cm²/g/history）** で、内部で19ブロックに変換します。`paired_blocks`も同じ220binを要求します。

   `load_egs_result`の再利用自体は成立します。ただし実体は[`run_p8e.load_egs_result`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/run_p8e.py:424)で、引数は`out, cache, grid, densities`、戻り値は`(dose, report)`です。指定キャッシュに加え、リポジトリの`egs_production.json/.npz`と封印対象ファイル群を読みます。

   **修正案:** 生の付与和を、各外側バッチの履歴数と`mass_g_cm2`で割って100×220にする。`dref=core.DREF`を明示し、100×19の誤入力・質量の誤り・異なるDrefの拒否を確認する。19ブロックの再実装は不要です。

3. **【実質】積層の戻り値だけでは、E7の「失敗が一項目だけ」を判定できません。**

   [`classify_blocks`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/run_p8e.py:393)は、最初の`fail`以降をすべて`not-judged`にします。実際に二ブロックを閾値超過にしても、戻り値は`['fail', 'not-judged', …]`でした。`status.count('fail') == 1`では複数失敗を救済してしまいます。

   **修正案:** 既存の正式な判定結果は維持し、再試行資格の確認だけ、返された全19個の`difference/SEM`に同じ分類関数を一項目ずつ適用して数える。単独失敗、複数失敗、統計不足との混在、再試行済みの再失敗をテストする。これはE7方針の変更ではなく、その実装条件です。

4. **【実質】故障を「バイト差以外で検出する」ための判定器が未定義です。**

   [作業3](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:225)は検出を要求していますが、既存検査をそのまま使えるとは限りません。[P8g-2の既存テスト](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/tests/test_p8g2_gates.py:77)では、g1とg2の拒否手段が履歴のバイト比較です。P7の1・3・8も旧来は完全一致比較で拒否しています。

   特に、正方形照射野のx/y交換は分布を変えないので、本番の対称性検査だけでは検出できません。

   **修正案:** 故障ごとに「固定入力・到達記録・拒否する検査」を表にする。例えばx/y交換は固定seedの線源座標と方向の対応、g1は飛行中消滅後の二重消滅を検出するイベント条件、g2は実際に同じスロットを再使用させた低エネルギー光子の付与量と出力収支で検出する。正常版通過と故障箇所への到達も受入条件にしてください。

5. **【実質】`fixed_cases()`の契約は「18ケース」ではなく、16ケースで18経路を覆うものです。T4での再生も受入条件から抜けています。**

   [`fixed_cases()`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8g2_core.py:32)は16件を返し、`PATHS`は18件、`TRANSITIONS`は9件です。再生には状態だけでなく、`SEED2`、global history番号としての`start`、ケース別`cap/maxsteps`、**スペクトル無効**という条件が必要です。容量不足・maxsteps不足を意図したケースも含みます。

   [§7](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:245)ではCPUしか要求していないため、CUDA固有の経路不具合を固定例で確認できません。「最大の深さ・容量の失敗は登録どおり」も、現在のケース辞書には期待最大深さの登録がありません。

   **修正案:** 16ケースを上記条件でCPU/T4双方に再生し、ケース別targetsと全9遷移を確認する。意図した失敗ケースの期待カウンター・未解決エネルギーの扱いを明記し、通常本番の「失敗0」と分けてください。

6. **【実質】`P8_DEFINITIONS`の再利用だけでは、共通形式の治療bundleを作成・検証できません。**

   [`catalog.p8_material`](/Users/oishifamily/Projects/viveMonte/chatcarlo/gpu/catalog.py:290)は既にありますが、返すのは**一材料分の目録とblob**です。各呼び出しでblob名は`p8`、offsetは0から始まります。四材料の目録を単純に連結すると参照位置が衝突します。

   また、[`bundle.verify/load`](/Users/oishifamily/Projects/viveMonte/chatcarlo/gpu/bundle.py:17)は現在、診断の宣言・構成ファイル・目録に固定されています。[作業1](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:189)には、治療bundleの構成と読み込み経路がありません。

   **修正案:** 既存`p8_material`を使い、四材料を別blob名で保持するのが単純です。治療用の薄いcreate/verify/loadを定義し、同じ`chatcarlo-g5-v1`形式で識別子を検証する。永続化→再読込→native入力までを照合し、材料交換、欠落、offset破損、偽装した組タグを拒否するテストを足してください。

7. **【実質】保存済みP7の`analysis.json`だけでは、指定された新旧比較を実行できません。**

   [§7の比較対象](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:263)は`analysis.json`ですが、そこに旧GPUの100バッチ配列はありません。J1/J2のSEMは旧GPUとEGS5を合成した差のSEMなので、新GPU対旧GPUのSEMとしてそのまま使えません。`analysis_p7.compare`にも直接渡せません。

   さらに、§5.2はP7 workを「A2では読まない」としています。

   **修正案:** 封印済みの旧GPU結果から、比較に必要な100バッチ配列だけを読み出す工程を追加する。積層もP8f/P8g-3のどの成果物を使うか、ハッシュと必要な復元範囲を固定する。P8g-3をP8g-2核の結果として使う対応関係も一行明記してください。

8. **【実質】本番駆動には、バッチの独立性・回収・メモリー容量を守る具体的契約が不足しています。**

   [作業2・3](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:208)では、既存核の「一呼び出し内100群」と、統計の「外側100バッチ」の対応が未定義です。旧P7は内側の群を合計して一つの外側バッチにします。ここを取り違えるとSEMが変わります。

   また、条件Bの一外側バッチ138万履歴について3,420binの履歴別float64配列を確保すると、**それだけで約37.8 GB**になります。「VM内で検査して捨てる」だけではT4に収まりません。条件完了時だけの回収では、長い条件Bの途中成果も失います。

   **修正案:** 外側バッチごとの`N/seed/start/タグ/健全性/ハッシュ`を保存し、履歴区間の重複・欠落を拒否する。履歴単位の健全性はデバイス上で検査・集約し、必要な履歴記録は小分けに処理する。10分ごとに完成バッチを回収する。重複start、欠落バッチ、途中保存、再試行seedとの混在をテストしてください。

9. **【実質】移植の中心である採点・横位置の接続について、正常系の直接テストが抜けています。**

   [作業2](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:203)はP7の採点をP8g-2輸送に接続しますが、受入条件は故障検出と本番統計が中心です。binの和やエネルギー収支が正しくても、横位置の保存・復元や付与位置の誤りは残り得ます。

   **追加すべきケース:** 点付与と線分付与、bin/箱の境界、±20 cm外端、軸に平行な線分、複数binをまたぐ線分、二次粒子の生成位置、hard/brems中断後の親粒子の横位置復元、補充後の領域配列の初期化。

   **単純化案:** [`p7.scorer_vectors/scorer_fixture`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p7.py:125)の入力を再利用し、幾何から求めた期待値と数値許容で比較する。新しい輸送参照核をもう一つ作る必要はありません。

10. **【実質】§9の保存方針が親計画・常設ルールと逆です。平均とSEMだけでは判定を再生成できません。**

    [§9](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:285)は100バッチ表をDriveへ送り、リポジトリには平均・SEM等だけを残します。しかし[常設ルール](/Users/oishifamily/Projects/viveMonte/.claude/rules/mv.md:134)と親計画は、**判定に使う外側バッチ表と再生成スクリプトをリポジトリに残す**としています。

    P7の交点bootstrapや対称性の対応差は、平均と各binのSEMだけから再構成できません。

    **修正案:** 判定に必要な100バッチ表をリポジトリへ残し、大きな生データをDriveへ送る。RESULTS・判定JSONの再生成を受入条件に加えてください。

11. **【軽微】作業4の記録先が、§3の変更許可と衝突しています。**

    [作業4](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:233)は`A2.md`、既存`RESULTS.md`、`mv/STATUS.md`、親計画を変更しますが、許可は新規`a2_*`等に限定され、`docs/validation/mv/`は禁止です。§9で残す表の置き場も必要です。

    **修正案:** 記録用の例外を対象ファイル単位で列挙し、「既存の記録に変更がない」という受入条件にもその例外を反映する。

12. **【軽微】付録のg5は、既存故障の内容と違います。**

    [付録の16番](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:306)は「群の最大値の破損」ですが、既存[`P8G2_FAULT == 5`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8g2_gpu.cu:44)は**集計先を次の群へずらす故障**です。

    **修正案:** g5を「群の集計先のずれ」に直す。最大値の検査も維持するなら、最大値だけを壊す別の検査例を設ける。全体の和だけではg5を見逃すため、群別照合が必要です。

13. **【軽微】「全故障をCPUとT4で実行」「有限・非負」の対象を絞る必要があります。**

    [作業3](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-30-chatcarlo-gpu-g5-a2-therapy-set.md:224)は全故障をCPU/T4双方に要求しますが、§7はP7の故障をCPU用とT4用に分けています。GPU限定マクロの故障は、そのままではCPUで発動しません。

    また、`brems`には符号付き差などがあり、全出力への非負要求は既存契約と合いません。

    **修正案:** 故障×実行デバイス×検出器の表を一つにして受入条件から参照する。非負は付与・線量等に限定し、符号付き診断量は有限性と収支・集計の許容で検査する。

実装を単純化するなら、**新しく書く解析は二つの入力アダプターに限定する**のがよいです。P7は領域の展開と既存`dose`への接続、積層は220binの質量正規化までを新規実装とし、その先は既存関数を呼ぶ。bundleも既存の一材料目録生成を組み合わせ、故障・固定例・期待失敗・デバイスの対応を一つの表に集約すると、作業と受入条件の重複を減らせます。

方針自体への異議はありません。G5の採否は既存の統計判定と同一セッションの速度で決め、参照実装をバイトで守る方針の中で評価しました。新しい履歴単位のバイト一致関門や、統計的同等性検定の追加は求めていません。