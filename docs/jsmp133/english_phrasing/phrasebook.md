# v5 の語句ごとの英語の候補（2026-10-28、Claude 作成）

番号は [sources.md](sources.md) の番号。「推奨」は Claude の案で、まだ決定ではない。**未決**は英文を書く前に使用者と決める。

## 全体の形

- 見出しは [Purpose] [Methods] [Results] [Conclusion]（v5 と同じ。1 の DosimeTron も同じ形の構造化抄録）。
- 本文は 320〜500 words、題名は 20 words 以内。旧英語案（v3 の訳）は 409 words だった。v5 は v3 より短いので、400 words 前後になる見込み（見込みであり、実測ではない）。
- 時制: Methods と Results は過去形。システムの動作の説明（AI が聞き取る、条件から入力を作る）は現在形でもよい。旧英語案は現在形。
- 主語は "we" を使ってよい（1・2 でも使われている）。

## 題名（v5: AI対話型モンテカルロシミュレーションシステムの開発）

| 候補 | 語数 | 備考 |
|---|---|---|
| Development of an AI-Driven Conversational Monte Carlo Simulation System | 9 | 旧英語案 |
| Development of a Conversational AI Agent System for Monte Carlo Radiation Transport Simulation | 13 | "radiation transport" を足すと分野がはっきりする（2・3 の言い方） |

**決定 1（使用者、2026-10-28）**: 英語題名に "radiation transport" を足す。たたき台は 2 行目の案（13 words）。細かい語順は英文作成のときに詰める。

## Purpose

| 日本語 | 英語の候補 | 根拠 |
|---|---|---|
| 専門的な入力形式の習得 | learning specialized input formats / the steep learning curve of code-specific input formats | 4 |
| 十分な統計精度を得るための計算資源 | computational resources required to achieve adequate statistical precision | 一般的な言い方 |
| 条件設定を AI との対話で行い | specifying simulation conditions through conversation with an AI agent / through natural-language interaction | 1・3 |
| 計算をクラウド上で実行する | running calculations on cloud GPUs | — |
| 容易に実行できる | make Monte Carlo simulations accessible to non-experts / easy to perform | 一般的な言い方 |

## Methods

| 日本語 | 英語の候補 | 根拠・注意 |
|---|---|---|
| 最大 10 MeV の光子・電子・陽電子輸送 | coupled photon–electron–positron transport up to 10 MeV | 5・7（"coupled electron-photon transport"）。8 は範囲で "from 0.01 to 20 MeV" と書く |
| Python で記述し、GPU（CUDA）で実行 | written in Python and executed on GPUs using CUDA | — |
| 光子の断面積には NIST の公開データなどの評価済みデータ | evaluated photon cross-section data, including NIST data | 「EPDL と書かない」決定に従う（abstract_plan.md） |
| 電子の弾性散乱には公開コード ELSEPA | electron elastic-scattering cross sections calculated with the publicly available code ELSEPA | — |
| 対話型の AI エージェントで操作する | operated by conversational AI agents | 1（agentic）・2（AI agents） |
| 汎用の手順書形式 Agent Skills | Agent Skills, a general-purpose format for agent instructions | "open standard" と書くなら一次資料で確かめる（sources.md の 12） |
| Claude Code や Codex（OpenAI）等で利用可能 | usable with Claude Code, Codex (OpenAI), and other agents | — |
| 計算目的の聞き取り | eliciting the purpose of the simulation | — |
| 計算方法の検討 | discussing (or proposing) feasible calculation methods | — |
| 確定した条件から計算の入力を自動生成 | automatically generates the simulation input from the agreed conditions | 3（input generation） |
| 使用者が 3D 表示で条件を承認した後 | after the user approves the geometry and conditions in a 3D view | — |
| 汎用コード EGS5 | the general-purpose Monte Carlo code EGS5 | EGS5 は電子・光子の汎用コード（KEK/SLAC） |
| 水・肺・骨・鉛の積層ファントム | a multilayer slab phantom of water, lung, bone, and lead | 5・7 は "water–lung–water slab phantom" の形。順番まで書くなら water–lung–water–bone–water–lead |
| PDD | percentage depth dose (PDD) | 一般的 |
| OCR | **決定（使用者、2026-10-28）: off-axis ratio (OAR)** | 下の注を参照 |
| 差は各点の EGS5 の値に対する局所の割合 | differences were expressed as local percentage differences relative to EGS5 at each point | 7 は最大線量に対する割合なので、定義を明記する意味がある |
| ビルドアップを含む全域 | over the entire depth range, including the buildup region | 9 |
| 照射野内と照射野端から外側 3 cm まで | within the field and up to 3 cm outside the geometric field edge | 9（領域の呼び方） |
| 入射光子数は、いずれも両コードで 1.0×10⁸ | 1.0 × 10⁸ histories were simulated with both codes | MC では "histories" が一般的。"incident photons" でもよい |
| 性質の異なる 2 件の計算依頼 | two simulation requests of different natures | — |
| 入射光子数を変えた X 線画像 | X-ray imaging with varying numbers of histories | — |
| 前立腺内の I-125 シードが 10 MV の外照射の線量分布に与える影響 | the dose perturbation caused by intraprostatic I-125 seeds in 10 MV external-beam radiotherapy | 10・11（"dose perturbation"） |
| 10⁴ から 10⁸ まで 5 段階 | five levels from 10⁴ to 10⁸ | — |
| シードの有無の各 3.0×10⁸ | 3.0 × 10⁸ histories each with and without seeds | — |

**OCR の注（決定 2: OAR を使う。使用者 2026-10-28）**: OCR（off-center ratio）は日本でよく使う呼び方。英語の論文では off-axis ratio（OAR）や lateral (dose) profile のほうが多い（5・9 と、検索で見た論文）。発表は日本の学会なので OCR でも通じる。国外の読者も考えるなら "off-axis ratio (OAR)"。図や公開ページの表記とそろえるかも考えて決める。

## Results

| 日本語 | 英語の候補 | 根拠 |
|---|---|---|
| 差は平均 X%、最大 Y% | mean and maximum local differences were X% and Y%, respectively | 一般的 |
| 照射野端のすぐ外側 | just outside the field edge | 9 |
| 鉛の直前 | immediately upstream of the lead layer / just proximal to the lead layer | — |
| 画像ノイズが低下していく過程が視認できた | image noise visibly decreased as the number of histories increased | — |
| シードの表面から 1 mm 以内で線量が最大 3.3% 増加 | the dose increased by up to 3.3% within 1 mm of the seed surface | 10・11（dose enhancement） |
| 表面から約 2 mm でほぼ元に戻った | and returned to nearly baseline at about 2 mm from the surface | 10（spatially limited） |
| 前立腺全体の平均では +0.5% | the mean dose over the whole prostate increased by 0.5% | — |

## Conclusion

| 日本語 | 英語の候補 | 根拠 |
|---|---|---|
| 検証した条件で | under the tested conditions | 2（条件つきで述べる） |
| EGS5 との差は平均 0.4% 未満、最大 1.3% | mean differences from EGS5 were below 0.4%, with a maximum of 1.3% | — |
| 2 件の計算依頼を対話で実施できた | two different simulation requests were completed through conversation | 1 は "demonstrating the feasibility of …" でまとめる |

## 英文を書くときの注意（v5 の決まりから）

- 60 keV の OCR は 1 cm 区間の平均。"accurately reproduced the penumbra" のような、ペナンブラの細かい形を確かめたと読める言い方をしない（abstract_plan.md 2026-10-26）。
- ビルドアップは主に 10 MV の積層の話。60 keV にもビルドアップがあると読める書き方をしない。
- 「別の AI でも再現した」とは書かない（決定済み）。Codex は「利用可能」の範囲。
- A1（シードの計算は入口を通していない）と A2（コードは 2 系統）は日本語 v5 でも未対応。英文で「対話で実施した」「1 本のコード」と強めて書かない。
