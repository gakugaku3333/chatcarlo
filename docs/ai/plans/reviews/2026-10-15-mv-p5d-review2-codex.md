**詳細版は実装可能な方向ですが、「前回指摘をすべて解消」とは判定できません。致命0・実質7・軽微4・参考2です。** 特に、CPU出力と検査関数の契約、`c3`の適用範囲、測定条件と停止条件の確定が必要です。

ファイルは変更していません。以下は実装・計画の静的照合であり、計算やGPU測定は実行していません。

**前回8件の判定**

| 前回 | 判定 | 詳細版での状況 |
|---|---|---|
| 1 CPU入口 | 解消 | 集計済み出力の`p5a_run`と、新規g++用ローダーに変更されている。 |
| 2 正しさの関門 | 一部解消 | C4再判定と回帰確認は分離された。ただし「全出力一致」、CPU出力の形式、`c3`の適用に問題が残る。 |
| 3 全計時実行の健全性 | 一部解消 | 各回の検査は明記されたが、A〜Cの出力をそのまま`core.health`には渡せない。実行条件の記録も未完。 |
| 4 D2の解釈・規模 | 解消 | 2万履歴に限定し、原子的加算の単独費用と解釈しない旨が明記された。 |
| 5 原因分離の限界 | 一部解消 | 第3節は原因候補の観察に限定されたが、目的の「分けて測る」と占有率の扱いに曖昧さが残る。 |
| 6 計時境界 | 一部解消 | 定常GPU計時の順序は明確。一度きりの費用とCPU/GPUの比の境界は未確定。 |
| 7 D3の照合 | 解消 | 128版との照合、不一致時の診断値不採用が明記された。 |
| 8 反復・機種差 | 一部解消 | 中央値・範囲、再測定、機種差は反映。CPUの測定n、反復のseed/start、例外処理に不足がある。 |

**【実質】**

1. **A〜Cの出力は、`core.health`／`core.c3`の入力契約を満たしません。**  
   [計画41行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:41)は全構成に`core.health()`を要求しますが、[CPU.run](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5a.py:160)と[p4_run](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5a.py:467)は`nan_count`を返しません。[core.health](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_core.py:226)はそれを必須参照するため、直接呼ぶと`KeyError`です。さらにCPU出力には`seconds`があり、`c3`内部の`validate_output`は余分なキーも拒否します。

   **修正案：** CPUの物理出力の抽出と健全性検査を新規アダプターとして定義する。`nan_count=0`を補う場合は「CPUで計測した値」ではないことを明示し、非有限値検査は実出力に対して行う。正常なCPU出力が検査を通るテストも必要です。

2. **「全出力`np.array_equal`」は、文字どおりでは正常実行でも成立しません。**  
   [計画36〜37行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:36)と43行の一致条件には、除外フィールドがありません。`CPU.run`と`p4_run`は実測時間`seconds`も返すので、全キー比較では毎回不一致になります。保存npzを直接比較するならメタデータも混ざります。

   **修正案：** 比較対象を`p5a.OUTPUTS`に固定し、保存結果は既存の`load_local_inputs`／`saved_batch`相当で検証・抽出する。byte一致を要件にするなら、`np.array_equal`ではなく既存`compare_outputs`／ビット比較を指定してください。

3. **M3参照とx86／GPUの比較に、`core.c3`の幅を流用する根拠がありません。**  
   [計画38〜39行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:38)は、束のM3由来`history_*`とColabの本番出力を`c3`で比較します。しかし[c3](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_core.py:363)の幅は、同じ履歴出力を異なる順序で加算する際の丸め誤差に基づきます。数学関数の機種差・コンパイラ差・輸送経路の差は含みません。

   既存P5cは[GPU自身の履歴出力とGPU本番出力](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_colab.py:171)をC3で比較しています。今回の接続は別の検査です。実際に不合格になるとは断定しませんが、正常な移植差を拒否し得ます。

   **修正案：** Dの集計検査は既存と同じGPU履歴出力対GPU本番出力にする。M3対Colabの回帰条件は別に定義し、許容幅の根拠を固定する。Cもクロスプラットフォーム回帰として扱い、加算誤差だけの幅と区別する必要があります。

4. **本測定の条件が、関門の条件と区別して確定されていません。**  
   [計画50行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:50)はGPUのnだけを固定しています。A/B/Cは関門表の10万／2万履歴をそのまま3回測るのか、共通nを別に設けるのか不明です。反復ごとのseed/start、D3の比較相手も明示されていません。

   またAの`history_rng=True`は、従来の284 µs/履歴を得た[5万履歴・既定`history_rng=False`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5a.py:938)とは異なります。履歴ごとのMT再初期化費用が加わります。

   **修正案：** 関門用・本測定用・診断用の実行表を分け、n/seed/start/history_rng/maxsteps/ブロック寸法を固定し、各回の記録に残す。旧速度との設定差も報告してください。

5. **100万履歴の失敗について、停止条件が矛盾しています。**  
   [計画84行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:84)は健全性不合格なら停止、86行は100万履歴の失敗なら継続としています。100万履歴でstep limit、非有限値、整数不一致が出た場合、両方に該当します。

   **修正案：** 継続可能なのはOOM・セッション切断・時間切れなど「検証済みの出力が得られなかった実行」に限定する。出力が得られて検査不合格なら停止、と優先順位を明記する。CUDAエラー後に継続する場合の再初期化条件も必要です。

6. **一度きりの費用を「それぞれ別に記録する」ための測定手順がありません。**  
   [計画48行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:48)について、どのAPI呼び出しを文脈生成として計るか、コンパイルキャッシュがある場合をどう呼ぶか、表転送の終了同期をどこに置くかが未定です。既存コードは`kernel.compile()`を呼びますが、その時間を常にNVRTCの新規コンパイル時間とは扱えません。

   **修正案：** 計時する操作と同期点を明記し、キャッシュ状態を記録する。削除禁止の範囲では「既存キャッシュ込みのコンパイル要求時間」と報告して構いません。ノートブックを再実行した場合も冷間値と区別してください。

7. **テスト計画が、測定ドライバーの主要な失敗を捕捉できません。**  
   [計画80行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:80)のキー・単位・人工不一致だけでは、次が抜けています。

   - 実CPU出力の正常系、`seconds`除外、欠落・余分なキー／shape／dtypeの拒否。
   - ctypes契約の`start`、seed、maxstepsの受け渡し。n=2000・start=0だけではstartの無視を見逃す。
   - 束・固定ハッシュ・コンパイルオプション不一致の拒否。
   - 偽GPU／イベントを用いた、同期→確保→開始イベント→起動→終了イベント→downloadの順序検証。
   - 健全性不合格と100万履歴の資源不足で、停止／継続が分かれること。
   - **Colab用モジュールがP5a・Numbaを読み込まないこと。** 手元でCuPyを読み込まない検査とは別です。
   - Colab上で新規g++ローダーとGPU計時経路を確認する具体的なスモーク実行。

**【軽微】**

8. **`deep_blocks`の呼び方と相対差の定義を固定してください。**  
   [計画39行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:39)の正しい接続は、[bundleでの生成方法](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b.py:196)に合わせて次です。

   ```python
   core.deep_blocks(
       output["batches"].sum(axis=0),
       n,
       float(reference["conversion"]),
   )
   ```

   `batches`をそのまま渡すと100×5になり、参照の5要素とは別物です。相対差には既存`core.relative`を使うか、分母・ゼロ時の扱いを明記し、5要素すべてで判定すると指定してください。

9. **`core.health()`の「0件」は戻り値の契約と一致しません。**  
   戻り値は件数ではなく、`status`と各違反数を持つdictです。`core.require(core.health(r), ...)`など、成功判定を明示してください。D2の履歴別出力には`nspec`がなく`nspec_index`なので、全計時実行のスペクトル検査も形式別に定義する必要があります。

10. **D2自身の正しさの関門と、診断失敗の適用範囲が不足しています。**  
    [計画56行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:56)にはD2の両出力の照合条件がありません。そこで得た履歴出力と本番出力の`c3`を指定すれば自然です。またD2不合格時もD3同様に診断のみ不採用なのか、停止なのかを明記してください。

11. **速度比の向きと時計の組合せが曖昧です。**  
    [計画61行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:61)のB/A、D/C、D/Bは、µs/履歴の比なら小さいほど高速、履歴/秒の比なら逆です。CPU内側時間とGPUイベント時間、呼び出し全体時間同士の比を区別して列名に式を入れてください。D/CにもCPUコンパイラと実行実装の違いが含まれます。

**【参考】**

12. **新規ファイルだけで実装する方針に、根本的な障害はありません。**  
    実シグネチャは以下です。

    ```python
    CPU(work, data, rng="mt", fault=0, cap=64)
    CPU.run(n, seed, start=0, history_rng=False,
            diagnostic=False, maxsteps=10000000)

    p4_run(data, n, seed, history_rng=False,
           maxstack=4096, maxsteps=10000000)

    core.c3(histories, production, nspec)
    core.health(r)
    core.deep_blocks(coarse, histories, conversion)
    core.load_bundle(directory)  # → data, reference, manifest

    p5b.bundle(work, saved, drive=None)
    ```

    計画中の`CPU(rng=...)`と`p5a_run(20000,start=0)`は略記としてなら理解できますが、後者は実際にはTables・seed・history_rng・maxsteps・出力ポインター等を渡すC ABIです。新規ローダーはその契約全体を再現する必要があります。

    `load_bundle`は内容とマニフェストの整合を検査しますが、承認済みの束そのものを外部から固定する役割はノートブック側の期待ハッシュにあります。何のsha256を固定するかも明記すると確実です。

13. **目的と診断の表現を統一すると、完了条件が明確になります。**  
    [計画19行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5d-speed.md:19)の「原因候補を分けて測る」は、第3節の「原因の特定は目的にしない」より強く読めます。「資源量・出力方式差・起動寸法感度を観察する」に揃えるのが適切です。D1の列挙値だけで厳密な理論占有率を出すこともできないため、今回は属性値の報告だけに限定すれば、追加調査をスコープに持ち込まず完了できます。