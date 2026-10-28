# 関連論文の一覧（2026-10-28 調査）

本文は保存していない。URL から読む。「参考にする点」は Claude の要約。

## A. AI エージェントでモンテカルロ計算を動かす研究（本研究と同じ分野）

1. **DosimeTron** — Tzanis E, Klontzas ME, Tzortzakakis A. *DosimeTron: Automating Personalized Monte Carlo Radiation Dosimetry in PET/CT with Agentic AI.* arXiv:2604.06280 (2026). <https://arxiv.org/abs/2604.06280>
   - 参考にする点: 構造化抄録（Purpose / Materials and Methods / Results / Conclusion）。"agentic AI system"、"natural-language interaction" という呼び方。参照コード（OpenDose3D）との一致を数値で示し、結論は "feasibility" と処理時間でまとめている。
2. **Sato ら（PHITS）** — Sato T, Hashimoto S, Ogawa T, et al. *Toward AI-Agent-Driven Particle Transport Simulations: Implementation of AI-Assisted Workflows for PHITS.* arXiv:2607.11309 (2026). <https://arxiv.org/abs/2607.11309>
   - 参考にする点: "AI assistants" と "AI agents" を区別する。エージェントが入力の編集・実行・エラー確認・結果解析を行う、という書き方。人による確認（human verification）を重視し、結論を条件つきで述べる。
3. **PHITSBench** — Ji X, Boriskina SV. *PHITSBench: an execution-scored benchmark for AI-assisted PHITS radiation-transport input generation using natural language.* arXiv:2607.09789 (2026). <https://arxiv.org/abs/2607.09789>
   - 参考にする点: 「自然言語で入力を作る」ことを "input generation using natural language" と呼ぶ。実際に計算を走らせて参照結果と比べる評価を "execution-scored" と呼ぶ。
4. **AutoFLUKA** — Ndum Ndum Z, Tao J, Ford J, Liu Y. *AutoFLUKA: A Large Language Model Based Framework for Automating Monte Carlo Simulations in FLUKA.* arXiv:2410.15222 (2024). <https://arxiv.org/abs/2410.15222>
   - 参考にする点: 使用者の障壁を "steep learning curve" や「手作業で時間がかかり誤りやすい」と書く。本研究の Purpose（2 つの障壁）の言い方の参考になる。

## B. GPU モンテカルロコードを参照コードと比べる研究（評価 1 の書き方）

5. **gDPM** — Jia X, Gu X, Sempau J, et al. *Development of a GPU-based Monte Carlo dose calculation code for coupled electron-photon transport.* Phys Med Biol (2010). arXiv:0910.0329. <https://arxiv.org/abs/0910.0329>
   - 参考にする点: "coupled electron-photon transport"。水–骨–水、水–肺–水の slab phantom で "depth-dose and lateral profiles" を比べる。
6. **gDPM v2.0** — Jia X, Gu X, Graves YJ, et al. *GPU-based fast Monte Carlo simulation for radiotherapy dose calculation.* Phys Med Biol (2011). arXiv:1107.3355. <https://arxiv.org/abs/1107.3355>
   - 参考にする点: 差を「統計的に有意でない領域の割合」で示す書き方。本研究は局所の差の平均と最大で示すので、形は採らない。
7. **oclMC** — Tian Z, Shi F, Folkerts M, et al. *An OpenCL-based Monte Carlo dose calculation engine (oclMC) for coupled photon-electron transport.* arXiv:1503.01722 (2015). <https://arxiv.org/abs/1503.01722>
   - 参考にする点: water–bone–lung–water slab の平均差を % で示す。差は最大線量に対する割合なので、本研究の「各点の EGS5 の値に対する局所の割合」と定義が違う。英語では定義を明記する。
8. **GPUMCD** — Hissoiny S, Ozell B, Bouchard H, Després P. *GPUMCD: a new GPU-oriented Monte Carlo dose calculation platform.* Med Phys (2011). arXiv:1101.1245. <https://arxiv.org/abs/1101.1245>
   - 参考にする点: 扱うエネルギー範囲を "photon and electron transport from 0.01 to 20 MeV" のように範囲で書く（A2 の書き方の参考）。
9. **Venselaar ら** — Venselaar J, Welleweerd H, Mijnheer B. *Tolerances for the accuracy of photon beam dose calculations of treatment planning systems.* Radiother Oncol 60:191–201 (2001).（③ の 2026-10-24 調査で既出）
   - 参考にする点: 線量分布を領域（ビルドアップ、照射野内、ペナンブラ、照射野外）に分けて許容を変える慣例。"buildup region"、"penumbra"、"outside the field" の語。

## C. 前立腺シードと外照射（評価 2 の書き方）

10. **Kanda ら** — Kanda D, Hanada T, Yoshida K, et al. *Evaluation of dose perturbations around iodine-125 seed sources in supplemental external beam prostate radiotherapy.* J Radiat Res (2023). <https://pmc.ncbi.nlm.nih.gov/articles/PMC10215002/>
    - 参考にする点: "dose perturbation(s) around I-125 seeds"、"supplemental external beam radiotherapy"、ビームの上流・下流（upstream / downstream）、"dose enhancement"、影響が "spatially limited" という言い方。
11. **Roswell Park の研究（6・18 MV、フィルムと DOSXYZnrc）** — UroToday の抄録紹介: <https://urotoday.com/recent-abstracts/urologic-oncology/prostate-cancer/45015-experimental-measurements-and-monte-carlo-simulations-of-dose-perturbation-around-a-nonradioactive-brachytherapy-seed-due-to-6-and-18-mv-photons-abstract.html>
    - 参考にする点: "dose perturbation around a nonradioactive brachytherapy seed"、"localized dose enhancement"、シード表面からの距離（mm）で影響の広がりを示す。

## D. 用語の確認

12. **Agent Skills** — Anthropic が公開した、エージェントへの手順書の形式（SKILL.md）。2025 年 12 月に open standard として公開され、Codex などほかのツールも読み込める、と二次資料（ブログ）にある。<https://codex.danielvaughan.com/2026/05/05/agent-skills-open-standard-portable-skills-codex-cli-cross-agent/>。一次資料（agentskills.io、anthropics/skills）は未確認。"open standard" と書くなら、一次資料で確かめてから書く。

## 見つからなかったもの

- Google Colaboratory で放射線のモンテカルロ計算を回した論文の抄録は見つからなかった。"cloud GPU (Google Colaboratory)" は本研究独自の書き方になる。
- EGS5 と OCR を局所の差で比べた英語の抄録は見つからなかった。
- 日本医学物理学会の英語抄録の投稿規定は、今回も Web で見つけられなかった（登録画面で確認）。
