# JSMP133 英語抄録（v5 の英訳、Claude 初稿 2026-10-28）

英訳元: [abstract_ja_draft.md](abstract_ja_draft.md) の「②⁺ 正式採用の抄録本文（登録用 v5）」。言い回しの根拠は [english_phrasing/](english_phrasing/)。
使用者の決定: OCR は off-axis ratio (OAR) と書く。題名に "radiation transport" を入れる。旧英語案（`abstract_en_draft.md`、v3 の訳）は 2026-10-28 に使用者の指示でゴミ箱へ移した。
状態: **第 4 稿・最終候補**（2026-10-28、使用者採用。Codex の確認済み）。旧: 第 3 稿（2026-10-28）。使用者の指示で、意味を保ったまま直訳調を避け、英語として自然な流れに全体を書き直した。A1・A2 は日本語 v5 のまま（使用者決定）。第 2 稿は下の「第 2 稿」節に残す。

## Title

A system for Monte Carlo radiation transport simulation through conversation with an AI agent

## Abstract

[Purpose]
Running Monte Carlo simulations requires users to learn specialized input formats and to secure enough computing power for adequate statistical precision. We aimed to lower both hurdles with a system in which users specify simulations through conversation with an AI agent and the calculations run on cloud GPUs.

[Methods]
Using Claude Code (Anthropic), we developed a Monte Carlo code in Python for photon, electron, and positron transport up to 10 MeV that runs on GPUs via CUDA. Photon cross sections are taken from evaluated data, including NIST data, and electron elastic-scattering cross sections from the publicly available code ELSEPA. The code is operated by conversational AI agents. Their behavior is defined in Agent Skills, a vendor-neutral instruction format, so the same system can be used with Claude Code, Codex (OpenAI), and similar agent tools. The agent first clarifies what the user wants to calculate, discusses feasible methods, and agrees on the output format; it then generates the input automatically. After the user approves the setup in a 3D view, the calculation runs on GPUs in Google Colaboratory (Google).
Evaluation 1 benchmarked the code against the well-established Monte Carlo code EGS5. We compared percentage depth dose (PDD) and off-center ratio (OCR) profiles for a 60 keV, 10 × 10 cm beam in water, and PDD for a 10 MV beam in a multilayer phantom of water, lung, bone, and lead. Local percentage differences were calculated relative to the EGS5 value at each point and summarized by their mean and maximum absolute values. PDDs covered the full depth, including the 10 MV buildup region, and OCRs covered the field and extended to 3 cm beyond the geometric field edge. Each code simulated 1.0 × 10^8 incident photons per case.
Evaluation 2 tested whether the system could carry out two dissimilar requests through dialogue: simulated X-ray images at five incident photon counts from 10^4 to 10^8, and the dose perturbation caused by 58 intraprostatic I-125 seeds in 10 MV external-beam irradiation, computed with 3.0 × 10^8 incident photons each with and without the seeds.

[Results]
In Evaluation 1, for the 60 keV water phantom, the mean and maximum absolute differences were 0.16% and 0.6% for PDD and 0.39% and 1.3% for OCR; the OCR maximum occurred just outside the field edge. For the 10 MV multilayer phantom, they were 0.17% and 1.3% for PDD, with the maximum immediately upstream of the lead layer.
In Evaluation 2, both requests were executed as intended. Image noise decreased visibly as the photon count increased. The dose increase averaged over all seeds was 3.3% about 0.2 mm from the seed surface and about 0.3% approximately 2 mm from the surface; the mean prostate dose increased by 0.5%.

[Conclusion]
Under the tested conditions, the code agreed with EGS5 to within 0.4% on average and 1.3% at most. The system carried out the intended Monte Carlo calculations through dialogue.

## 語数

本文 **477 words**（見出しを除き空白区切り。「×」も 1 語）。題名 14 words（"A system for …"。2026-10-29 に 13 と書いたのは数え間違い）。

## 第 3 稿で直したこと（直訳調 → 英語の流れ）

- Purpose: 「〜という 2 つの障壁がある」→ "is often hindered by two barriers"。目的は "lowers both barriers by ..." として、2 つの障壁と手段（対話・クラウド）を対応させた。
- Methods: 日本語の「〜した。さらに〜した。」の積み上げをやめ、コード → データ → エージェント → 対話の流れ → 実行の順に、主語をそろえて 1 本の流れにした。対話の手順は first / then / finally。
- 評価 1: 「〜と〜を EGS5 と比較した」の長い主語をやめ、"we compared ... with EGS5:" の後にコロンで比較対象を並べた。差の定義は "At each point, ... relative to the local EGS5 value"。
- Results: 数値を "showed mean and maximum absolute differences of ..." の 1 文にまとめ、最大の位置は "with ..." / "the latter ..." で添えた。シードは主語を "The seeds" にして因果を明確にした。
- Conclusion: "the results differed from EGS5 by less than 0.4% on average and by at most 1.3%"。"agreed" や "reproduced" は使わない（数値だけを述べる）。
- 語数: 最初の書き直しは 494 words で上限に近すぎたため、断面積・対話の手順・評価 1 の導入・差の定義の 4 文を短くした（意味は同じ）。
- 意味の確認: 数値・評価範囲・光子数・最大の位置は第 2 稿・日本語 v5 と同じ。"only 0.5%" のような評価語は足していない。

## AI と書くときの注意を当てはめた点検（2026-10-28、english_phrasing/ai_writing_guidelines.md）

- AI に特有とされる語（delve, underscore, meticulous, showcase, pivotal, comprehensive, robust, leverage, notably, crucial など 40 語）は 0 件。
- 試していないことを主張しない: "works with ... other agent tools" は、確かめたのが Claude Code と Codex だけなので、v5 の「等で利用可能とした」に合わせて "can be used with ... similar agent tools" に弱めた。
- 数値は v5 と出典で照合済み（AI 同士の合意を根拠にしていない）。
- 抄録本文に AI 使用の開示文は入れない（JSMP の規定が見つからない。登録画面で確かめる）。論文にするときは、原稿の作成と英訳に Claude・Codex を使ったことを開示する。

- EGS5 の形容（使用者決定 2026-10-28）: "general-purpose"（多目的の意味で、信頼の含みがない）を "well-established" に変えた。日本語も「汎用コード」→「実績のある」に変更。

- 評価 2 の目的に答える文（使用者決定 2026-10-28）: Results の評価 2 の冒頭に "both requests were executed as intended." を追加。日本語は「いずれの依頼でも、意図した計算が実行された。」

- 通しの再点検（2026-10-28）: Purpose の 2 つの障壁を動名詞でそろえた（mastering / securing）。"taken relative to" → "calculated relative to"。評価 2 の列挙を "X-ray imaging ... , and the effect of ..." の対にそろえた。Results の主語を "the 60 keV water phantom showed"（ファントムが示す、という擬人的な言い方）から差を主語にした。"about 2 mm" に "from the surface" を足した。Conclusion を "our results" にした。

- Codex の最終確認（[english_phrasing/reviews/codex_en_review2.md](english_phrasing/reviews/codex_en_review2.md)）: 要修正なし。推奨 1（入射光子数を incident photon counts / incident photons と明示）と任意 2（output format）を反映。

- Agent Skills の形容（使用者決定 2026-10-28）: "general-purpose"（多目的の意味）を "vendor-neutral" に変更。意図は「メーカーを問わずどのエージェントでも使える手順書」。日本語も「汎用の」→「特定の AI に依存しない」。

## 第 4 稿（2026-10-28、使用者決定）

日本語の言葉選びと文の順番から離れ、意図から英語で組み直した。Purpose は使う人の障壁として書いた。評価 2 は "tested whether" で目的を示した。Conclusion の 2 文目は研究の目的に答える文。"agreed with EGS5 to within ..." を採用した（使用者決定。数値つきの医学物理の一般的な言い方）。"enabled users" は、開発者しか試していないため使わない。

Codex の確認（[english_phrasing/reviews/codex_en_review3.md](english_phrasing/reviews/codex_en_review3.md)）: 要修正なし。推奨 1（差の定義を local percentage differences ... relative to the EGS5 value と明示）と任意 2（OAR の範囲に照射野内を明記）を反映。

シードの距離（使用者決定 2026-10-28）: 3.3% は 0.25 mm ボクセルの最初の層（中心は表面から 0.14〜0.23 mm、平均 0.20 mm。シードに触れるボクセルは除外）の値なので「within 1 mm」→「about 0.2 mm from their surface」。「nearly vanished」は、2 mm で +0.32%（600 バッチ。最初の +0.31% は 200 バッチだけで集計した値）・3 mm 以上でも +0.2% 残るため不正確なので、数値「about 0.3% at 2 mm」にした。再計算は保存データ（`~/.cache/chatcarlo/p9c2/prod_prostate/`）で、`make_radial.py` と同じ区間（軸から 0.25 mm 幅）。

シードの文の Codex 確認（[english_phrasing/reviews/codex_en_review4.md](english_phrasing/reviews/codex_en_review4.md)、2026-10-28）: 文法は問題なし。推奨・任意（58 本の平均と明示、増加率が減ること、表面からの距離と明示）を反映（使用者決定）。

シードの本数（使用者決定 2026-10-28）: 58 本は Methods（評価 2）に移し、Results は "averaged over all seeds"。

最終チェック（2026-10-28、使用者決定）: Results の書き出しを "In Evaluation 1," にそろえた。

用語（使用者決定 2026-10-29）: 日本の学会で発表するため、OAR（off-axis ratio）を日本で一般的な **off-center ratio (OCR)** に戻した。2026-10-28 の「OAR を使う」決定を置き換える。

題名（使用者決定 2026-10-29）: "Development of a Conversational AI Agent System for Monte Carlo Radiation Transport Simulation" を "Monte Carlo Radiation Transport Simulation through Conversation with an AI Agent"（11 words）に変更。"Development of" で始める直訳調と、名詞が 4 つ続く形をやめ、Purpose と同じ "through conversation with an AI agent" で対話による計算を前に出した。→ 同日、"system"（作ったもの）が消えるため "A System for Monte Carlo Radiation Transport Simulation through Conversation with an AI Agent"（14 words）に変更（使用者決定）。登録の書式に合わせてセンテンスケース "A system for Monte Carlo radiation transport simulation through conversation with an AI agent" にした（Monte Carlo は固有名詞、AI は略語なので大文字）。

## 第 3 稿（第 4 稿の前の版。記録）

[Purpose]
Monte Carlo simulation is often hindered by two barriers: mastering code-specific input formats and securing the computational resources required for adequate statistical precision. We aimed to develop a system that lowers both barriers by letting users set up simulations through conversation with an AI agent and by running the calculations in the cloud.

[Methods]
Using Claude Code (Anthropic), we developed a Monte Carlo code in Python for photon, electron, and positron transport up to 10 MeV, with GPU acceleration via CUDA. Photon cross sections come from evaluated data, including NIST data, and electron elastic-scattering cross sections from the publicly available code ELSEPA. The code is operated by conversational AI agents whose behavior is defined in Agent Skills, a vendor-neutral instruction format, so that the system can be used with Claude Code, Codex (OpenAI), and similar agent tools. In the dialogue, the agent asks about the purpose of the simulation, discusses suitable methods, confirms the desired output format, and then generates the input automatically. Once the user has approved the setup in a 3D view, the calculation runs on GPUs in Google Colaboratory (Google).
Evaluation 1 compared our results with those of the well-established Monte Carlo code EGS5: percentage depth dose (PDD) and off-axis ratio (OAR) profiles for a 60 keV beam with a 10 × 10 cm field in water, and PDD for a 10 MV beam in a multilayer phantom of water, lung, bone, and lead. Differences were calculated relative to the local EGS5 value at each point and summarized as mean and maximum absolute values. PDDs were evaluated over the full depth, including the 10 MV buildup region, and OARs over the field and up to 3 cm beyond its geometric edge. Each code simulated 1.0 × 10^8 incident photons per case.
In Evaluation 2, two different types of simulation requests were carried out through dialogue: X-ray imaging at five incident photon counts from 10^4 to 10^8, and the effect of intraprostatic I-125 seeds on the dose distribution of 10 MV external-beam irradiation, with 3.0 × 10^8 incident photons each with and without the seeds.

[Results]
In Evaluation 1, the mean and maximum absolute differences for the 60 keV water phantom were 0.16% and 0.6% for PDD and 0.39% and 1.3% for OAR, with the OAR maximum just outside the field edge. For the 10 MV multilayer phantom, the PDD differences were 0.17% (mean) and 1.3% (maximum), the latter immediately upstream of the lead layer.
In Evaluation 2, both requests were executed as intended. The X-ray images showed a visible reduction in noise as the photon count increased. The seeds raised the dose by up to 3.3% within 1 mm of their surface; this increase had nearly vanished at about 2 mm from the surface, and the mean prostate dose rose by 0.5%.

[Conclusion]
Under the tested conditions, our results differed from EGS5 by less than 0.4% on average and by at most 1.3%, and the system completed two different simulation requests through dialogue.


## A1・A2（使用者決定 2026-10-28: そのまま）

A1（評価 2・Conclusion の「対話で実施」）と A2（「1 本のコード」）は、日本語 v5 も英語も変えない。理由は abstract_plan.md の 2026-10-28 追記。

## 第 2 稿（451 words、Codex のレビューを反映した版。記録として残す）


[Purpose]
Performing Monte Carlo simulations involves two barriers: learning specialized input formats and securing the computational resources needed for adequate statistical precision. This study aimed to develop a system that makes Monte Carlo simulations easy to perform by specifying conditions through conversation with an AI agent and running calculations in the cloud.

[Methods]
Using Claude Code (Anthropic), we wrote a Monte Carlo code for photon, electron, and positron transport up to 10 MeV in Python and enabled it to run on GPUs (CUDA). Photon cross sections were based on evaluated data, including publicly available NIST data; electron elastic-scattering cross sections were calculated using the publicly available code ELSEPA. We built a system in which conversational AI agents operate this code. Agent behavior was defined in Agent Skills, a general-purpose format for agent instructions, so that the system can be used with Claude Code, Codex (OpenAI), and other agents. The AI sequentially elicits simulation objectives, discusses calculation methods, confirms output formats, and automatically generates input from agreed conditions. Following user approval of conditions in a 3D view, calculations run on Google Colaboratory (Google) GPUs.
Evaluation 1 compared percentage depth dose (PDD) and off-axis ratio (OAR) profiles in a water phantom at 60 keV (10 × 10 cm field), and PDD in a multilayer phantom containing water, lung, bone, and lead at 10 MV, against the general-purpose Monte Carlo code EGS5. Local percentage differences relative to EGS5 at each point were summarized using their mean and maximum absolute values. PDDs covered the entire depth range, including the 10 MV buildup region; OARs covered the field and extended 3 cm beyond the geometric field edge. Each code simulated 1.0 × 10^8 incident photons per case.
Evaluation 2 performed two different simulation requests through dialogue: X-ray imaging at five incident photon counts from 10^4 to 10^8, and dose perturbations from intraprostatic I-125 seeds during 10 MV external-beam irradiation, using 3.0 × 10^8 incident photons each with and without seeds.

[Results]
In Evaluation 1, for the 60 keV water phantom, mean and maximum absolute differences were 0.16% and 0.6% for PDD, and 0.39% and 1.3% for OAR. The OAR maximum occurred just outside the field edge. For the 10 MV multilayer phantom, mean and maximum absolute PDD differences were 0.17% and 1.3%, respectively, with the maximum immediately upstream of the lead layer.
In Evaluation 2, image noise visibly decreased as photon counts increased. Dose increased by up to 3.3% within 1 mm of the seed surface, returning nearly to the seed-free level approximately 2 mm from the surface. Mean prostate dose increased by 0.5%.

[Conclusion]
Under the tested conditions, mean absolute differences from EGS5 were below 0.4%, with a maximum of 1.3%. Two different simulation requests were completed through dialogue.


## 訳の注（日本語 v5 との対応）

- 「水ファントム（10×10 cm）」の 10×10 cm は照射野の大きさ（10×10 cm の平行ビーム。ファントムは 30×30×20 cm）。
- 上付き文字は登録画面で使えない可能性があるため 10^8 と書いた。使えるなら 10⁸ に戻す。
- Codex のレビュー（第 1 稿に対するもの）は [english_phrasing/reviews/codex_en_review.md](english_phrasing/reviews/codex_en_review.md)。
