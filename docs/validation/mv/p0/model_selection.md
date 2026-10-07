# T6 電子・陽電子輸送モデルの候補（選定は次の子計画で承認）

これは本番実装の変更・承認ではない。期間は1人が継続して実装と独立検証に取り組む場合の
**工学的な見積もり**であり、計算や文献から取得した測定値ではない。
診断領域の本番NumPy経路を保存して新しい連成輸送経路を作ることを前提とする。

|対象|候補・推奨案|出典|最短／標準／難航（作業週）|幅を左右する未検証事項|
|---|---|---|---|---|
|多重散乱|Goudsmit–Saunderson分布表＋condensed history。EGS5のdual random hingeを推奨候補。Molièreも比較候補|[EGS5 SLAC-R-730](https://rcwww.kek.jp/research/egs/egs5.html) §2、installed `pegs/gsdist.f`, `egs/egs5_electr.f`|2–3／4–6／8–12|角度分布表の生成、運動エネルギー依存、横変位と散乱角の結合、表の精度を独立に検証できるか|
|境界横断|解析面距離＋境界でステップを制限。dual random hingeの境界処理を候補とし、境界近傍のsingle scatteringも比較する|同マニュアル、`docs/Writing_HOWFAR.pdf`, `egs/egs5_electr.f`|1–2／3–5／6–10|薄層・斜入射・同一媒体境界での系統差、ゼロ距離ループ、幾何による電子ステップ増加|
|平均エネルギー損失と揺らぎ|AE依存の制限衝突阻止能＋明示Møller/Bhabhaによるhard loss。continuous soft lossから始め、薄層でのstragglingは別に検証|EGS5 §2、`pegs/pegs5.f` SPIONB/SPTOTE, `egs/egs5_moller.f`, `egs/egs5_bhabha.f`|1–2／2–4／4–7|restricted/unrestrictedの取り違え、step size依存、soft-loss分布を追加する必要がある条件|
|制動放射|カットオフ以下を制限radiative stopping power、以上を明示二次光子。ICRU補正を含むEGS5方式を推奨候補|EGS5 §2、`egs/egs5_brems.f`, PEGS IAPRIM=1|1／2–3／4–5|角度とエネルギーの同時分布、二次光子スタック、低エネルギーカットオフの収支|
|対生成|核場・電子場の部分係数をXCOMから取得し、運動学・energy sharing・反跳の扱いをEGS5資料にそろえる案|[NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html)、EGS5 `egs/egs5_pair.f`|1／2–3／4–5|電子場対生成(triplet)の独立扱い、閾値直上、電子/陽電子エネルギー和と角度、資料と輸送実装のモデル差|
|陽電子|Bhabha・飛行中消滅・停止後2光子消滅。静止質量のledgerを必須とする|EGS5 `egs/egs5_bhabha.f`, `egs/egs5_annih.f`, `egs/egs5_electr.f` §15–16|1／2–3／3–5|媒質電子の静止質量、停止陽電子と脱出陽電子の扱い、収支の二重計上|
|連成スタック・採点・独立検証|固定容量スタック＋overflow明示停止、history統計、細かい線量採点。Fano/PDD/OCR等は次計画で受入条件を固定|EGS5 user manual B、既存ChatCarlo統計API（読み取りのみ）|2–3／4–6／8–12|二次粒子の深い分岐、SEMの裾、実装したモデルをEGS5以外の解析基準でも点検できるか|

行間は依存しており単純加算しない。重複を考慮した全体の目安は最短8–12週、標準16–24週、
難航32–48週。最短は水・箱・単一線源に限定しても検証が順調である場合。
2026-10-07から2027年3月までの約5か月に標準上限は収まり切らない可能性があり、
多重散乱と境界処理の小さな独立試作を先に行って幅を狭めることが必要。
この表だけで2027年3月の完成、2026-10-16の水PDD/OCR合格を予測しない。

P0では上記モデル、Fano試験、MVの本番コードのいずれも実装していない。
EGS5内部ではpair生成とtripletのモデルがXCOMの2チャネルに一対一対応するとは確認できていない。
選定案を次の子計画で人間が承認する際の未解決項目とする。
