# 公開ページの修正依頼（他の AI 向け・第 12 版、2026-10-29）

対象ページ: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site
前提: 公開版 33（`page_update_request_v11.md`、報告 `page_update_report_v11.md`）が反映済みのページ。
理由（使用者の決定、2026-10-29）: 日本の学会で発表するため、英語抄録でも日本で一般的な呼び方 **OCR（off-center ratio）** を使う。日本語抄録はもともと「OCR」なので、これで日英がそろう（`docs/jsmp133/abstract_plan.md` の 2026-10-29 追記、英語の正典は `docs/jsmp133/abstract_en_v5.md`）。

## 守ること（重要）
- 変更するのは、下の修正 1 の 4 か所だけです。それ以外の文章、数値、図、3D 表示、デザイン、構成、日本語抄録は変えないでください。
- 意見フォーム（設問、ボタン、送信の仕組み）、保存済みの回答、DB・マイグレーション、URL、公開範囲、用語解説のパネルの仕組みは変えないでください。フォームへのテスト送信や保存回答の削除もしないでください。
- **作業の前に、Sites の最新ソース（公開版 33 以降）を取得してください。** 手元の古い HTML を最新版として丸ごと再適用しないでください。
- 迷う箇所や、指示にない箇所に「OAR」「off-axis」が残っていると気づいたときは、変更せずに場所と内容を報告してください。
- 変更後に、「変更前→変更後」を `docs/jsmp133/page_update_report_v12.md` に報告してください。viveMonte リポジトリでは commit・push しないでください（Sites 側の保存・公開は前回と同じ手順）。

## 修正 1: 英語抄録（「5′. 抄録案（英語）」）の OAR を OCR にする
次の 4 か所だけを、それぞれ置き換えてください（2026-10-29 に公開ページで確認した位置）。

1. [Methods]（Evaluation 1）
   - 変更前: 「percentage depth dose (PDD) and off-axis ratio (OAR) profiles」
   - 変更後: 「percentage depth dose (PDD) and off-center ratio (OCR) profiles」
2. [Methods]（Evaluation 1）
   - 変更前: 「and OARs covered the field and extended to 3 cm beyond the geometric field edge.」
   - 変更後: 「and OCRs covered the field and extended to 3 cm beyond the geometric field edge.」
3. [Results]（Evaluation 1）
   - 変更前: 「0.39% and 1.3% for OAR;」
   - 変更後: 「0.39% and 1.3% for OCR;」
4. [Results]（Evaluation 1）
   - 変更前: 「the OAR maximum occurred just outside the field edge.」
   - 変更後: 「the OCR maximum occurred just outside the field edge.」

本文の語数は変わりません（477 words）。置き換えた後、英語抄録の本文が `docs/jsmp133/abstract_en_v5.md` の「## Abstract」節と文字単位で一致することを確かめてください。

## 報告に含めること
- 4 か所の変更前→変更後。
- 修正後のページ全体に「OAR」「off-axis」が 0 件であること（あれば場所と前後の文。変更しない）。
- 英語抄録の本文が 477 words、`abstract_en_v5.md` と一致すること。
- Sites の version、source commit、deployment ID、公開後の表示確認（英語抄録）。
