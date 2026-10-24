**A1 は実現可能ですが、「同じ端末で同じ `log` を使えば履歴ごとのバイト一致が自動的に成り立つ」という断定は強すぎます。致命の指摘はありません。実質7件・軽微2件・参考2件です。**

親・当該の AGENTS.md、CLAUDE.md、mv.md、指定の詳細計画・上位計画§8・前回レビューを読み、P8f の輸送経路、CPU/GPU の入口、比較関数、P5i の実装と検証を照合しました。ファイルは変更していません。輸送計算・テストは実行せず、コンパイラの小例だけを標準入力から LLVM IR に出力しました。

ユーザー決定への見解を別に述べます。**A1→B→A2 の順番と、ビット一致に縛られない検証への変更に異議はありません。** A1 でバイト一致を安価な強い検査として試すことも妥当です。ただし、成立を事前の保証とせず、この構成で検証する仮説として扱うべきです。

1. **【実質】同一端末・同一関数名だけでは、元の式と初期化の生成コードの一致を保証できません。**

   [詳細計画16行](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md:16)の断定が対象です。現在の CPU 指定は `-O2 -ffp-contract=off`、GPU は `--fmad=false` です。これらは積和の融合を止めますが、定数畳み込みまで止めません。[Clangの仕様](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior)、[NVRTCの仕様](https://docs.nvidia.com/cuda/archive/12.9.0/nvrtc/index.html)

   実際、この端末の Apple clang 21 に小例を与えると、`log(.01)` は `0xC0126BB1BBB55515` という定数になり、配列から読んだ値の `log` は `llvm.log.f64` 呼び出しとして残りました。**不一致を実証したわけではありませんが、同じ綴りでも評価経路が違うことは確認できました。**

   一方、現在の指定でコンパイラが任意に `log(a/b)` を `log(a)-log(b)` に変える、と決めつける根拠もありません。計画の「割ってから log」「`(q*q)*q` を保持」は適切です。CPU と GPU の log の違いも、端末内で新旧を比較する限り、それ自体は障害ではありません。

   **修正案:** 「一致するはずなので不要」を「同じ式・指定を保持し、関門で一致を確認する」に変更する。初期化と輸送を同じ翻訳単位／NVRTC ソース・指定でコンパイルし、不一致時には商・`q`・`q³`・log引数まで比較できるようにしてください。

2. **【実質】§2 に `p5_shape` の材料定数 log が抜けています。**

   到達経路は `p8f_history → p8f_sample_k → p5_shape` です。[呼び出し元](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_materials.h:85)、[対象の式](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5_kernel.h:233)

   次の3つはすべて、粒子エネルギー・乱数によらず、元素の `z` だけで決まります。

   - `log(1194.*pow(z,-2./3.))`
   - `log(184.15*pow(z,-1./3.))`
   - `log(z)`

   最初の2つは `z!=1` の枝だけで使います。前計算でもこの枝を保持し、`fc`、`xi`、`xl` 全体の前計算へ無断で広げず、まず log の項だけを置き換えるのが D1 に沿います。`log(d+.952)` は粒子状態に依存するため対象外です。

3. **【実質】P5i を根拠に部品比較を省いていますが、その前例には部品比較が含まれています。**

   [P5i の正式な関門0](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5i.py:87)は、凍結した元関数を直接呼んだ結果と新関数の結果を比較しています。初期化式を書き直した補助診断は、正式な合否条件ではありません。[P5iの記録](/Users/oishifamily/Projects/viveMonte/docs/validation/mv/p5/RESULTS_P5I.md)

   今回の0bは配列値の一致だけなので、**読み出す添字・探索・分岐の取り違えを検査できません**。特に次が必要です。

   - `p8f_log_interp` は `l<log(x[j])` で右側を探索し、`lo-1` により**最後の同値節点**を選ぶ。
   - `p5_pair_interp` は `log(es[j])<x` の探索で、**最初の同値節点**を選ぶ。
   - 吸収端の `i` は元エネルギー上の `p8f_upper` で決める。log格子上の探索結果で代用しない。
   - 吸収端の `log(photo[right]/photo[left])` は、通常補間の `log(photo[right])-log(photo[left])` と共有しない。

   [P8fの探索と吸収端処理](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_materials.h:37)、[対生成の探索](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5_kernel.h:78)

   **追加する関門:** CPU・T4それぞれで、変更した関数を元関数と直接比較する。元節点と `nextafter` 両隣、重複節点、異なる入力が同じlogに丸まる合成例、吸収端の両側、対生成しきい値、`quantities` の `i==0` を含める。これは再格子化の精度検証ではなく、A1 の置換が元の処理を保つ検査です。

4. **【実質】f3 は、記載どおりの変更では無効な log 読み出しが起きない場合があります。**

   [故障注入f3](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md:67)の具体化が必要です。例えば `p5_pair_interp` の `if(E<=th)return 0.` だけを外しても、後続の `start` 探索は `es[start]>th` を維持するため、しきい値以下の NaN スロットを読みません。係数がおかしくなっても、0cの無効読み出し数は0のままになり得ます。

   **修正案:** 配列ID・材料・無効添字・到達入力を固定し、通常の読み出し関数を確実に通す注入にする。f1も変更する要素と独立した比較相手、f2も変更する材料・配列を固定してください。配列種類全体の参照数だけでは、その故障要素への到達証拠になりません。

   また、読み出し検査は NaN スロットだけでなく、`0<=j<length` を**配列アクセス前に**確認する必要があります。GPU側の配置・読み出し経路も対象なので、代表的な注入はT4でも実行するのが適切です。

5. **【実質】検査ビルドと通常ビルドの出力一致が関門にありません。**

   [0c・2e](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md:63)は無効読み出し数だけを確認します。一方、履歴の一致と計時には通常ビルドを使う想定です。`P8G_CHECK` による条件分岐・カウンター追加を含む構成について、通常ビルドと同じ出力であることが未確認です。

   **追加する条件:** CPU・T4とも、検査用2,000履歴で通常版／検査版／同端末P8fの履歴出力を比較する。これにより、カウンターで確認した経路と、採用する計算経路を結び付けられます。P5iにも通常版・検査版双方の一致確認があります。

6. **【実質】2c の `check_aggregation` では「全出力の集計」「整数は完全一致」を実施できません。**

   [既存関数](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_core.py:68)が確認するのは、積層binの付与量とその加算数を使った浮動小数点の集計だけです。coarse／fine、収支、brems、各カウンターなどは検査しません。

   **修正案:** 出力ごとに集計規則を明記する。加算項目は履歴和、`maxima`・`p8_max`・`maxdepth` は最大値、診断など非集計項目は対象外、と定義してください。整数は対応するGPU群出力と厳密比較し、符号付きのbrems項目を含む浮動小数点加算には絶対値和と加算回数の上限を使います。既存の [`check_resume`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_core.py:174)は再利用の材料になります。

7. **【実質】2d の比較器では、追加した起動構成の誤りを十分に拒否できません。**

   [`compare_histories`](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_core.py:111)の合否対象は健全性、線源の一致、最初の最大2診断点の乱数消費位置です。線量・境界カウンターなどの一致は報告だけで、履歴全体の `calls` の一致も要求しません。[乱数診断の判定](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p5b_core.py:315)

   したがって、40×128で2bに合格しても、40×256や非ゼロstartの不具合は、2dの比較だけでは残り得ます。

   **修正案:** 2dでは、同じGPUのP8fとの全履歴出力のバイト比較、または既に確認したP8gの同じ履歴番号の出力とのバイト比較を使う。新しい統計基準は不要です。既存のCPU–GPU比較規則自体への異議ではなく、同端末で変更した起動経路を検査する用途への指摘です。

8. **【軽微】§2 の到達性と、純定数logの扱いを整理してください。**

   [§2の一覧](/Users/oishifamily/Projects/viveMonte/docs/ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md:32)にある `p5_sample_k` は、P8fの履歴からは呼ばれません。呼ぶのは `p8f_sample_k` です。

   逆に `p8f_hard → p5_primitive0/1` では、上端引数 `.5` によって `log(.5/(1-.5))` と `log(.5)` が純定数になります。下端の `.01/T` については変数依存です。[呼び出し](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f_materials.h:51)

   `log_sites.md` は関数内の字面だけでなく、こうした呼び出し条件も区別してください。定数引数の `log1p(-.5)` は、計画が `log1p` を対象外と明記しているので変更不要です。

9. **【軽微】実行手順・比較対象に、小さな矛盾があります。**

   - 0dのf2は関門1で拒否する仕様ですが、「関門0を完了してから関門1」の順序と、Codexが関門0まで担当する区切りに合いません。注入版の作成と実際の拒否確認を分けて記載してください。
   - `p8f_ref_gate2.npz` は履歴別出力、`p8f_ref_batch_000/050/099.npz` は各10万履歴の集計出力です。1bではそれぞれ対応するAPIと比較キーを固定し、`seconds` を除外する必要があります。[参照生成](/Users/oishifamily/Projects/viveMonte/prototypes/mv_electron/gpu/p8f.py:258)
   - 保存バッチと比較するとき、履歴レコードを後から合算すると、輸送途中で加算する `brems[1]` の順序が変わり得ます。ここは元のCPU集計経路を保つ必要があります。
   - §6の許可パターン `p8g_*` には、§3・§7で必要な `p8g.py` が入りません。
   - P8fのGPUドライバーは40×128と`next=5120`を固定しています。40×256の比較用には、新規側に起動ラッパーを用意する、と明記すると変更禁止範囲と両立します。

10. **【参考】純定数logを配列化しない方が実装を単純にできます。**

    `log(.01)` や硬衝突上端の純定数式は、そのまま残せばコンパイラによる定数評価を利用できます。P5iも `log(.01)` は残しています。配列化による削減効果が乏しく、定数評価と実行時評価の差を持ち込む理由にもなります。

    D1に「コンパイル時に決まる純定数式は列挙するが、置換対象から除く」と明記する案を勧めます。また、P5iの9配列を丸ごと転用するより、必要な読み出し関数と比較器だけを再利用し、P8f固有の右側探索・材料別配列を保つ方が明快です。

11. **【参考】有限の履歴一致から、全履歴の同一性までは証明できません。**

    計画の固定した履歴でのバイト一致は強い証拠ですが、全入力・全履歴についての証明ではありません。新しい本計算を必須にする指摘ではなく、報告では「検査した履歴で一致し、採用方針に従ってP8fの判定を継承した」と範囲を明示すれば十分です。P8fの合格範囲を、他の積層・照射野・境界近傍のbinなどへ広げない扱いも、そのまま引き継げます。