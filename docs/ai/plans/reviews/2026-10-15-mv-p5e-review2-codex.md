**実装は可能ですが、詳細版をそのまま実装へ渡すには修正が必要です。致命0件・実質7件・軽微4件・参考2件です。** 指定の4ファイルと関連する呼出し・判定コードを照合しました。ファイル変更、GPU実行はしていません。

**前回の各指摘の判定**

| 前回の指摘 | 判定 | 詳細版での対応・残件 |
|---|---|---|
| 実質1：乱数の不変と履歴出力の不変を分ける | 一部解消 | E1のGPU同士の直接比較は適切。ただし`np.array_equal`はビット比較ではない |
| 実質2：0.272と動的割り当ての因果を断定しない | 解消 | 目的で指標の意味と因果の限界を明記 |
| 実質3：資源調整を動的割り当てに従属させない | 解消 | 2×2×3の全12変種で独立に比較 |
| 実質4：履歴ごとの入口も動的にし、境界を検査する | 一部解消 | 両入口の動的化は明記。E3の検査対象と実入口との接続が不足 |
| 実質5：新構成の指紋・保存・再開 | 解消 | 必要な構成要素と別構成の拒否を明記。探索結果の証拠保存には残件 |
| 実質6：比較条件と選択規則を固定する | 一部解消 | 同一セッション・3巡・中央値は明記。選択規則に誤採用の余地 |
| 軽微1：入口ごとに常駐数を再取得 | 一部解消 | attributes記録は追加。占有率の代替計算法が不正確 |
| 軽微2：`top+2`も存在、あふれ0の範囲を限定 | 一部解消 | 範囲限定は反映。`top+2`の記載漏れは残る |
| 軽微3：原子的集計単独の費用と解釈しない | 解消 | 差0.1%の意味を適切に限定 |
| 参考：理想順序と実用高速化を分ける | 解消 | 診断限定、選択対象外、置換検査を明記 |

**致命：0件**

`p5_kernel.h`・`p5b_wrap.h`を変更せず、新しい入口とコンパイル指定で実装できます。CuPyの制約によって計画全体が成立しなくなる問題は見つかりませんでした。

**実質：7件**

1. **占有率APIは使えますが、代替計算法は直してください。**  
   [計画38行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:38)

   CuPyには低水準の`occupancyMaxActiveBlocksPerMultiprocessor(func, blockSize, dynamicSMemSize)`が実装されています。`RawKernel.kernel.ptr`から関数ハンドルを取得する経路もあります。したがって「APIがあれば」から、使用バージョンと呼出し経路を固定する段階へ進められます。ただし低水準APIなので、Colab冒頭で実際に呼べることを確認する必要があります。[CuPy driver実装](https://raw.githubusercontent.com/cupy/cupy/v13.6.0/cupy_backends/cuda/api/driver.pyx)、[RawKernel実装](https://raw.githubusercontent.com/cupy/cupy/v14.0.1/cupy/_core/raw.pyx)

   フォールバックの「`num_regs`と`local`・スレッド上限から計算」は不十分です。`local_size_bytes`はSMの共有メモリ容量のように割って常駐数を求める値ではありません。レジスタ割当粒度、ブロック数・ワープ数の上限、共有メモリ、関数の最大ブロックサイズも関係します。

   **本番入口・履歴入口それぞれについてAPI値を取得し、その入口自身のTを固定してください。** E3の軽い核の占有率から本番のTを決めるのも避けるべきです。APIが使えなければ停止するか、T4限定の明示した見積式に落とす必要があります。

2. **E3は作れますが、独立した訪問ループだけでは変更対象を検査できません。**  
   [計画47行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:47)

   凍結側の`p5b_history`は履歴計算、`p5b_store`は保存を担い、割り当てループは凍結側にありません。新規`p5e_gpu.cu`内で割り当てループを共通化し、輸送・訪問を切り替えれば、凍結ファイルを変えずにE3を作れます。[p5b_wrap.h:44](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_wrap.h:44)

   ただし、訪問だけでは以下を確かめられません。

   - 非ゼロstartで、本当に`start+k`を輸送に渡したか
   - 本番集計が`p5b_row(k,n)`を使ったか
   - 履歴出力を位置kに保存したか
   - 両入口が同じ割り当て処理を使っているか

   **共通ループの使用を要件化し、非ゼロstartで`(k, global_h, row)`の検査を追加してください。** 小さい実輸送でも、動的入口と静的入口の履歴出力を比較する必要があります。Pythonのカウンタ模擬だけではCUDA入口の誤りを捕まえられません。

3. **候補を単純な構成へ変更した後、V0との分離条件を再検査していません。**  
   [計画54行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:54)

   最速候補AがV0より明確に速くても、Aと範囲が重なる単純な候補Bは、V0とも範囲が重なり得ます。

   例：V0＝10〜11、A＝8〜9、B＝8.5〜10.5。Aは分離しますが、Bに変更すると「V0と範囲が重なるなら採用なし」に反します。

   **先に`max(candidate) < min(V0)`を満たす集合を作り、その集合内だけで単純さを比較してください。** また、区間の重なりは推移的ではないため、「最初の最速候補と重なる集合を一度だけ作る」など選択手順も固定する必要があります。対応する反例テストが抜けています。

4. **E1の「ビット一致」と`np.array_equal`が一致していません。**  
   [計画45行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:45)

   `np.array_equal`は`+0.0`と`-0.0`を等しいとします。計画どおり「全出力のビット一致」を要求するなら、既存の`core.bits`を使ってください。[p5b_core.py:199](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_core.py:199)

   テストには通常の1ビット差に加えて、符号付きゼロ、shape・dtype差、非有限値の拒否を入れるべきです。V0自身とのE1は自明なので、V0の妥当性を検査したことにはならない点も区別してください。

5. **探索結果を手元で再判定するための保存契約がありません。**  
   [計画61行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:61)

   保存先はありますが、何を保存するかが未定です。合否JSONだけでは、手元でE1〜E3を独立に再計算できません。

   最低限、V0と各変種のE1出力、E2出力、E3の訪問・履歴番号・行の配列、3回の生の計時、構成指紋と採用判定入力を指定してください。既存P5dは計測出力をnpzで保存しています。[p5d_colab.py:132](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5d_colab.py:132)

   履歴出力は1変種約131 MB、12変種で約1.57 GBです。保存容量として現実的ですが、全変種をGPU・ホストに保持し続けず、逐次保存・解放する設計が必要です。

6. **変種の不合格、コンパイル失敗、CUDAエラーの扱いが整合していません。**  
   [計画44行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:44)、[83行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:83)

   「変種不合格は除外して継続」「E3が動かないなら停止」「CUDAエラー後は再起動」の境界が不足しています。レジスタ制限でコンパイル・起動できない変種が出た場合、除外か全停止かを実装者が決めることになります。

   既存`timed_gpu`は健全性失敗を`core.Stop`で送出します。そのまま流用すると、変種だけを落とすつもりでも全体が停止します。[p5d_colab.py:39](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5d_colab.py:39)

   **数値不合格・コンパイル不可・検査実装の故障・CUDA実行エラーを分けて処理を固定してください。** 不合格変種は3巡から除くこと、候補ゼロの場合の終了処理もテスト対象です。

7. **スタック16のCPU照合は可能ですが、既存CPUクラスの直接流用では実装できません。**  
   [計画49行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:49)

   `p5d_core.CPU`は64固定の`FLAGS`以外を拒否し、その固定値でビルドします。[p5d_core.py:47](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5d_core.py:47)

   凍結`p5_cpu.cpp`を16指定で新規ビルドすること自体は可能です。ただし新規`p5e.py`側にビルド経路が必要で、旧クラスの固定値を書き換える方式は変更禁止に反します。

   「全出力」の照合対象も固定してください。既存CPU出力には実行時間`seconds`があり、これはバイト一致の対象にできません。**物理出力10フィールド、比較元100バッチの所在・指紋、コンパイラと指定、健全性検査**を明示し、小規模の実ビルド検査を追加する必要があります。

**軽微：4件**

1. **CuPy禁止のテストは、遅延importを許す表現にする必要があります。**  
   [計画79行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:79)の「`p5e_colab`が`cupy`を読み込まない」は、「モジュールをimportした時点では読み込まない」と明記してください。実行時のCuPy importまで禁止するとGPU実行と矛盾します。既存P5dも関数内で読み込みます。

2. **C1の再実行条件が二重です。**  
   [計画49行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:49)は「C0〜C4を再実行」と「指定が違うならC1も再実行」が併記されています。採用構成では常にC1を行う、と一本化すれば明確です。C3の故障注入版も、新しい本番入口に適用することを明記してください。

3. **行番号の説明とスタック検査の説明に誤記が残っています。**  
   [計画21行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:21)の`p5b_row(h,n)`のhは、全体履歴番号ではなく呼出し内のkです。実装は`int(h*100/n)`なので、非ゼロstartを含む全体番号を渡してはいけません。スタック検査には`top+2`もあります。[p5b_wrap.h:43](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_wrap.h:43)、[p5_kernel.h:434](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5_kernel.h:434)

4. **診断の並べ替えについて、基準・正しさ・計時境界が未指定です。**  
   [計画40行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-15-mv-p5e-gpu-speedup.md:40)は、どの資源構成で走らせるか、昇順・同値時の順序、反復数、順序生成・転送の費用、V0との履歴出力比較を固定してください。置換検査だけでは、乱数・保存先の取り違えを検査できません。

**参考：2件**

1. **`--maxrregcount`はCuPy＋NVRTCで指定可能です。**

   既存コードは`RawKernel(..., options=..., backend='nvrtc')`を使っています。CuPyはoptionsをバックエンドへ渡し、NVRTCは`--maxrregcount=<N>`を正式に受け付けます。NVCCへ切り替える必要はありません。[CuPy公式資料](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html)、[NVRTC公式資料](https://docs.nvidia.com/cuda/archive/12.9.2/nvrtc/index.html)

   指定値は要求なので、コンパイル後の`num_regs`を記録して実効値を確かめてください。レジスタ数が減っても、spillとローカルメモリ増加で遅くなる可能性はあり、速さは実測事項です。

2. **12変種×3巡は、既存速度ならColabの時間に十分収まる規模です。ただし1回完走の保証はできません。**

   P5dの実測から、全変種がV0と同速と仮定すると次の概算です。

   | 処理 | 概算 |
   |---|---:|
   | 12変種×3巡×10万履歴 | 約285秒 |
   | E2：12変種×10万履歴 | 約95秒 |
   | E1：12変種×2万履歴 | 約20秒 |
   | 候補とV0：各100万履歴 | 約153秒 |
   | 合計 | **約9.2分＋コンパイル・保存・その他検査** |

   採用後の1,000万履歴もV0相当なら約13.2分です。[P5d実測]( /Users/oishifamily/Projects/viveMonte/docs/validation/mv/p5/RESULTS_P5D.md:26)

   したがって、**計画の規模自体を時間超過として却下する理由はありません。** 一方、P5dのコンパイル時間はキャッシュ済みの値で、新変種の冷コンパイル時間には使えません。spillによる悪化も未測定です。無料Colabは最大12時間で、資源・継続時間は保証されません。[Colab公式FAQ](https://research.google.com/colaboratory/faq.html)

   「探索は同一セッションで完了した場合だけ採用判定可能」と明記してください。切断前後の計時を混ぜると、今回追加した比較条件を満たせません。完走できなければ、保存済みの正しさ検査は再利用可能でも、採用用の3巡は同一セッションで取り直す扱いが妥当です。