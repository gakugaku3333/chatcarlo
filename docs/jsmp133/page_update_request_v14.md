# 公開ページの修正依頼（他の AI 向け・第 14 版、2026-10-29）

対象ページ: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site
前提: 公開版 36（`page_update_request_v13.md`、報告 `page_update_report_v13.md`）が反映済みのページ。
理由（使用者の決定、2026-10-29）: 演題登録（UMIN）の書式がセンテンスケース（先頭のみ大文字、固有名詞と略語を除きそれ以外は小文字）なので、公開ページの英語の題名も同じ表記にそろえる。英語の正典は `docs/jsmp133/abstract_en_v5.md` の「## Title」、登録用は `docs/jsmp133/registration_text.md`。語句は変えず、大文字・小文字だけを変える。

## 守ること（重要）
- 変更するのは、下の修正 1 の 1 か所だけです。それ以外の文章（英語抄録の本文、日本語の題名・抄録を含む）、数値、図、3D 表示、デザイン、構成は変えないでください。
- 意見フォーム（設問、ボタン、送信の仕組み）、保存済みの回答、DB・マイグレーション、URL、公開範囲、用語解説のパネルの仕組みは変えないでください。フォームへのテスト送信や保存回答の削除もしないでください。
- **作業の前に、Sites の最新ソース（公開版 36 以降）を取得してください。** 手元の古い HTML を最新版として丸ごと再適用しないでください。
- 同じ題名がページの別の場所（title 要素、OGP、説明文など）にもあると気づいたときは、変更せずに場所と内容を報告してください。
- 変更後に、「変更前→変更後」を `docs/jsmp133/page_update_report_v14.md` に報告してください。viveMonte リポジトリでは commit・push しないでください（Sites 側の保存・公開は前回と同じ手順）。

## 修正 1: 英語抄録（「5′. 抄録案（英語）」）の題名の大文字・小文字
- 変更前: 「A System for Monte Carlo Radiation Transport Simulation through Conversation with an AI Agent」
- 変更後: 「A system for Monte Carlo radiation transport simulation through conversation with an AI agent」

大文字のまま残すのは、先頭の「A」、固有名詞の「Monte Carlo」、略語の「AI」だけです。題名の行の見た目（「Title:」の語や強調など）はそのままにしてください。

## 報告に含めること
- 修正 1 の変更前→変更後。題名が 14 words で、`abstract_en_v5.md` の「## Title」と文字単位で一致すること。
- 修正後のページ全体に「A System for Monte Carlo Radiation Transport」（タイトルケース）が 0 件であること（あれば場所と前後の文。変更しない）。
- 英語抄録の本文（477 words）が変わっていないこと。
- Sites の version、source commit、deployment ID、公開後の表示確認（英語抄録の題名）。
