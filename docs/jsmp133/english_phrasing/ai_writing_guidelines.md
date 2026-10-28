# AI と一緒に抄録・論文・スライドを作るときの注意（2026-10-28 調査、Claude）

Claude と Codex の共有資料。公式文書の本文は保存していない。二次資料からの情報は「二次」と書いた（一次資料で確かめてから引用する）。

## 1. 投稿規定・倫理（論文に進むときに必須）

| 項目 | 内容 | 出典 |
|---|---|---|
| 開示 | AI 支援技術を使ったかどうかと、その使い方を、カバーレターと原稿の中で開示する | ICMJE Recommendations（2025-04 更新、AI–Assisted Technology の節）<https://www.icmje.org/icmje-recommendations.pdf> |
| 著者にしない | AI は正確さ・誠実さ・独自性に責任を持てないので著者にしない（COPE の立場と同じ） | ICMJE、Wiley（二次） |
| 人の責任 | AI が出した主張・引用・解析は著者が確かめる。自信ありげに誤ることがある | ICMJE、Wiley（二次） |
| 開示の場所（Wiley。Medical Physics の出版社） | 原稿の一部を AI で作ったら Methods（または開示文・謝辞）に詳しく書く。綴り・文法・一般的な編集だけなら開示は不要。翻訳は開示が要るとする要約もある | Wiley AI Guidelines（二次: <https://www.enago.com/responsible-ai-movement/publisher-ai-guidelines/wiley-ai-guidelines/>） |
| 研究データ | 元の研究データや結果を AI で作ったり変えたりしない | Wiley（二次） |
| JSMP | 日本医学物理学会の生成 AI の方針と、第 133 回の抄録規定は Web で見つからなかった。登録画面と大会ページで確かめる | — |
| 他学会の例 | 学習に使われない設定で使う・個人情報を入れない・使った事実と役割を開示する（日本緩和医療学会の専門医症例報告書ガイドライン） | <https://jspm.ne.jp/files/specialistCertification/ai_guide.pdf> |

**本研究で特に注意する点**: 本研究は「AI を使って作ったシステム」が題材なので、抄録・論文の文章に AI を使ったことと、研究の中で AI を使ったこと（コードの開発・対話の入口）を分けて書く。論文にするときは、Methods の AI の役割とは別に、原稿の作成と英訳に Claude・Codex を使ったことを開示文に書く。

## 2. AI と一緒に良い文章を書く方法

| 方法 | 理由 | 出典 |
|---|---|---|
| 一度に全文を頼まず、節ごとに目的を決めて直させる（「明確に」「論理を強く」など） | 学術の文章は直しを重ねて良くなる。一発の指示では支えにくい | arXiv 2505.11336 <https://arxiv.org/abs/2505.11336> |
| AI の文は「読みやすく見える」ので、人が意識して直す | 読みやすさが自信になり、直しが減る傾向がある | arXiv 2606.15583 <https://arxiv.org/abs/2606.15583> |
| 中身（細部・考察）は人が足す | AI の科学文は細部と洞察が薄くなりやすい | 同上の調査で見た複数の論文 |
| 人が最後に判断する（human in the loop） | 著者と責任の問題 | 物理論文を AI と共著した例の論考（検索結果） |
| AI 特有の語を避ける: delve, underscore, meticulous, commendable, showcase, intricate, pivotal など | PubMed の抄録でこれらの語が ChatGPT 公開後に急増した。動詞が多い | Matsui（medRxiv 2024 → 2025 出版）<https://medrxiv.org/node/812931>、Juzek & Ward（COLING 2025）<https://arxiv.org/abs/2412.11385>、Science Advances の 1,500 万抄録の解析（Times Higher Education の記事 <https://www.timeshighereducation.com/node/737694>） |
| 数値・引用は一次資料と照合する。AI が書いた引用をそのまま使わない | 存在しない文献や数値を作ることがある | ICMJE、Wiley |

**この共同作業での運用（使用者の決めた流れ、2026-10-28）**: Claude が書く → Codex CLI が読み取り専用でレビューする → Claude が必要な指摘を反映する → 使用者が決める。数値は v5 と出典で照合し、AI 同士の合意を根拠にしない。

## 3. スライドを AI で作るときの注意

| 方法 | 内容 | 出典 |
|---|---|---|
| assertion–evidence 形式 | 見出しを「主張の 1 文」（2 行以内）にし、本体は図・写真・式などの証拠で示す。箇条書きを減らす。理解と記憶がよくなったという研究がある | Penn State（Alley）<https://writing.engr.psu.edu/research.html>、チェックリスト <https://mycourses.aalto.fi/pluginfile.php/1938299/mod_page/content/15/AE-checklist.pdf> |
| 文字の大きさ | 見出し 28 pt、本文 18〜24 pt、太字のサンセリフ 1 書体、箇条は 2〜4 項目 | 同上のチェックリスト |
| 図の忠実さを人が確かめる | AI のスライド生成は数値や引用を作ってしまうことがある（ベンダーの検証で、主張の正確さが 44% という報告。方法が不明なので参考程度）。自動の評価者は見た目の細かい問題を見逃し、人より甘い | SlideTailor arXiv 2512.20292 <https://arxiv.org/abs/2512.20292>、SlidesGen-Bench arXiv 2601.09487 <https://arxiv.org/abs/2601.09487>、Tosea のブログ（二次） |
| 図は計算結果から作る | スライドの図は、保存した計算結果からスクリプトで描き、AI に図の中身を描かせない（Wiley の「結果を AI で作らない」と同じ考え） | — |

**本研究のスライドに当てはめると**: 見出しは主張の 1 文（例: "Mean differences from EGS5 were below 0.4% under the tested conditions"）。図は既存の `ocr60_full_range/build_ocr.py` などで作ったもの。抄録と同じ数値・評価範囲だけを使い、「ペナンブラを高精度に再現」などの強めた言い方をしない（abstract_plan.md 2026-10-26）。

## 4. 運用で分かったこと（2026-10-28）

- **文章の点検は、数値が何を指すかの誤りを拾わない。** 英訳を Claude の自己点検と Codex の 3 回のレビューで見直しても、シードの「表面から 1 mm 以内で最大 3.3%」（実際は表面から約 0.2 mm の層の値）は見つからなかった。使用者が英文を読んでボクセルの大きさを疑い、集計のコードまでさかのぼって分かった（n=1）。数値を含む文は、レビューを頼む前に出典の集計の定義（区間・基準点・除外したセル・何本の平均か）を確かめ、その定義をレビューの依頼文にも書く。経緯は `docs/lessons_learned.md` の 2026-10-28「保存結果を集計し直すときは…」。
