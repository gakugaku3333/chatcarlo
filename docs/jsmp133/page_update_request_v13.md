# 公開ページの修正依頼（他の AI 向け・第 13 版、2026-10-29）

対象ページ: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site
前提: 公開版 34（`page_update_request_v12.md`、報告 `page_update_report_v12.md`）が反映済みのページ。
理由（使用者の決定、2026-10-29）: 英語の題名を、直訳調の "Development of ..." をやめた形に変える（英語の正典は `docs/jsmp133/abstract_en_v5.md` の「## Title」、決定の経緯は `abstract_plan.md` の 2026-10-29 追記）。日本語の題名は変えない。

## 守ること（重要）
- 変更するのは、下の修正 1 の 1 か所だけです。それ以外の文章（英語抄録の本文、日本語の題名・抄録を含む）、数値、図、3D 表示、デザイン、構成は変えないでください。
- 意見フォーム（設問、ボタン、送信の仕組み）、保存済みの回答、DB・マイグレーション、URL、公開範囲、用語解説のパネルの仕組みは変えないでください。フォームへのテスト送信や保存回答の削除もしないでください。
- **作業の前に、Sites の最新ソース（公開版 34 以降）を取得してください。** 手元の古い HTML を最新版として丸ごと再適用しないでください。
- 旧い英語の題名がページの別の場所（ページの title 要素、OGP、見出し、説明文など）にもあると気づいたときは、変更せずに場所と内容を報告してください。
- 変更後に、「変更前→変更後」を `docs/jsmp133/page_update_report_v13.md` に報告してください。viveMonte リポジトリでは commit・push しないでください（Sites 側の保存・公開は前回と同じ手順）。

## 修正 1: 英語抄録（「5′. 抄録案（英語）」）の題名
- 変更前: 「Development of a Conversational AI Agent System for Monte Carlo Radiation Transport Simulation」
- 変更後: 「A system for Monte Carlo radiation transport simulation through conversation with an AI agent」

題名の行の見た目（「Title」の語や強調など）はそのままにし、題名の文字列だけを置き換えてください。大文字・小文字も上のとおりにします（登録の書式に合わせたセンテンスケース。先頭・Monte Carlo・AI だけが大文字）。

## 報告に含めること
- 修正 1 の変更前→変更後と、新しい題名が 13 words であること。
- 修正後のページ全体に「Development of a Conversational AI Agent System」が 0 件であること（あれば場所と前後の文。変更しない）。
- 英語抄録の本文（477 words）が変わっていないこと。
- Sites の version、source commit、deployment ID、公開後の表示確認（英語抄録の題名）。
