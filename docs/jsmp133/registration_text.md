# 登録画面に貼る文（UMIN 用、2026-10-29）

元: 英語は [abstract_en_v5.md](abstract_en_v5.md)、日本語は [abstract_ja_draft.md](abstract_ja_draft.md) の ②⁺（v5）。この 2 つが正典で、ここはその変換結果。正典を直したら、ここも作り直す。
変換（UMIN の入力規則に合わせた。規則: <https://www.umin.ac.jp/elbis/endai/help/special_symbol/>、上付き・改行は登録画面の説明）:
- × → `&times;`
- 10^8・10⁸ などの累乗 → `10<SUP>8</SUP>`（⁴・⁸ は UMIN の特殊文字一覧にない）
- 段落の区切り → 次の段落の頭に `<BR>`。見出しは [Purpose] などの形で本文に残した（登録画面に見出しの欄が別にあるなら、見出しと <BR> を外して欄ごとに貼る）
- タグは半角。貼った後、登録画面のプレビューで、× と上付きが正しく出ることを確かめる

変換後の確認（タグと実体参照を戻して数えた値）: 英語 477 words（正典と同じ）、日本語 1166 字（見出しの語・空白を除く。正典は 1,166 字）。

## 英語の題名

```
A system for Monte Carlo radiation transport simulation through conversation with an AI agent
```

## 英語の抄録本文

```
[Purpose]<BR>Running Monte Carlo simulations requires users to learn specialized input formats and to secure enough computing power for adequate statistical precision. We aimed to lower both hurdles with a system in which users specify simulations through conversation with an AI agent and the calculations run on cloud GPUs.<BR>[Methods]<BR>Using Claude Code (Anthropic), we developed a Monte Carlo code in Python for photon, electron, and positron transport up to 10 MeV that runs on GPUs via CUDA. Photon cross sections are taken from evaluated data, including NIST data, and electron elastic-scattering cross sections from the publicly available code ELSEPA. The code is operated by conversational AI agents. Their behavior is defined in Agent Skills, a vendor-neutral instruction format, so the same system can be used with Claude Code, Codex (OpenAI), and similar agent tools. The agent first clarifies what the user wants to calculate, discusses feasible methods, and agrees on the output format; it then generates the input automatically. After the user approves the setup in a 3D view, the calculation runs on GPUs in Google Colaboratory (Google).<BR>Evaluation 1 benchmarked the code against the well-established Monte Carlo code EGS5. We compared percentage depth dose (PDD) and off-center ratio (OCR) profiles for a 60 keV, 10 &times; 10 cm beam in water, and PDD for a 10 MV beam in a multilayer phantom of water, lung, bone, and lead. Local percentage differences were calculated relative to the EGS5 value at each point and summarized by their mean and maximum absolute values. PDDs covered the full depth, including the 10 MV buildup region, and OCRs covered the field and extended to 3 cm beyond the geometric field edge. Each code simulated 1.0 &times; 10<SUP>8</SUP> incident photons per case.<BR>Evaluation 2 tested whether the system could carry out two dissimilar requests through dialogue: simulated X-ray images at five incident photon counts from 10<SUP>4</SUP> to 10<SUP>8</SUP>, and the dose perturbation caused by 58 intraprostatic I-125 seeds in 10 MV external-beam irradiation, computed with 3.0 &times; 10<SUP>8</SUP> incident photons each with and without the seeds.<BR>[Results]<BR>In Evaluation 1, for the 60 keV water phantom, the mean and maximum absolute differences were 0.16% and 0.6% for PDD and 0.39% and 1.3% for OCR; the OCR maximum occurred just outside the field edge. For the 10 MV multilayer phantom, they were 0.17% and 1.3% for PDD, with the maximum immediately upstream of the lead layer.<BR>In Evaluation 2, both requests were executed as intended. Image noise decreased visibly as the photon count increased. The dose increase averaged over all seeds was 3.3% about 0.2 mm from the seed surface and about 0.3% approximately 2 mm from the surface; the mean prostate dose increased by 0.5%.<BR>[Conclusion]<BR>Under the tested conditions, the code agreed with EGS5 to within 0.4% on average and 1.3% at most. The system carried out the intended Monte Carlo calculations through dialogue.
```

## 日本語の抄録本文

```
[Purpose]<BR>モンテカルロ計算の実施には、専門的な入力形式の習得と、十分な統計精度を得るための計算資源という 2 つの障壁がある。本研究では、条件設定を AI との対話で行い、計算をクラウド上で実行することで、モンテカルロ計算を容易に実行できるシステムを構築することを目的とした。<BR>[Methods]<BR>Claude Code（Anthropic）を用いて、最大 10 MeV の光子・電子・陽電子輸送を扱うモンテカルロコードを Python で記述し、GPU（CUDA）で実行できるようにした。光子の断面積には NIST の公開データなどの評価済みデータ、電子の弾性散乱には公開コード ELSEPA を用いた。さらに、このコードを対話型の AI エージェントで操作するシステムを構築した。AI エージェントの動作は、特定の AI に依存しない手順書形式 Agent Skills で定義し、Claude Code や Codex（OpenAI）等で利用可能とした。AI は計算目的の聞き取り、計算方法の検討、出力形式の確認を対話で順に行い、確定した条件から計算の入力を自動生成する。使用者が 3D 表示で条件を承認した後、Google Colaboratory（Google）の GPU で計算が実行される。<BR>評価 1 として、水ファントム（照射野 10&times;10 cm）の 60 keV での PDD と OCR、および水・肺・骨・鉛の積層ファントムの 10 MV での PDD を実績のあるモンテカルロコード EGS5 と比較した。差は各点の EGS5 の値に対する局所の割合とし、その絶対値の平均と最大で示した。評価範囲は、PDD はビルドアップを含む全域、OCR は照射野内と照射野端から外側 3 cm までとした。入射光子数は、いずれも両コードで 1.0&times;10<SUP>8</SUP> とした。<BR>評価 2 として、性質の異なる 2 件の計算依頼（入射光子数を変えた X 線画像、前立腺内の 58 本の I-125 シードが 10 MV の外照射の線量分布に与える影響）を対話で実施した。入射光子数は、X 線画像では 10<SUP>4</SUP> から 10<SUP>8</SUP> まで 5 段階で変え、シードの計算ではシードの有無の各 3.0&times;10<SUP>8</SUP> とした。<BR>[Results]<BR>評価 1: 60 keV の水ファントムでは、PDD の差は平均 0.16%、最大 0.6%、OCR では平均 0.39%、最大 1.3%（照射野端のすぐ外側）だった。10 MV の積層ファントムの PDD は平均 0.17%、最大 1.3%（鉛の直前）だった。<BR>評価 2: いずれの依頼でも、意図した計算が実行された。X 線画像では、画像ノイズが低下していく過程が視認できた。シードの計算では、全シードの平均で、線量の増加はシード表面から約 0.2 mm で 3.3%、約 2 mm で約 0.3% であった。前立腺全体の平均では +0.5% であった。<BR>[Conclusion]<BR>構築したシステムは、検証した条件で EGS5 との差は平均 0.4% 未満、最大 1.3% であり、性質の異なる 2 件の計算依頼を対話で実施できた。
```
