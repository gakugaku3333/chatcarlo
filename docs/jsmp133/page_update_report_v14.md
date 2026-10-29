# 公開ページ修正報告（第14版、2026-10-29）

[第14版依頼](page_update_request_v14.md)に従い、Sitesの最新ソース（公開版36、`6000ae63d08b83ca3362b632c16d5e674d45e089`）を取得して編集した。

## 変更前 → 変更後

「5′. 抄録案（英語）」の題名1か所のみを、センテンスケースに変更した。

- 変更前：A System for Monte Carlo Radiation Transport Simulation through Conversation with an AI Agent
- 変更後：A system for Monte Carlo radiation transport simulation through conversation with an AI agent

語句は同じで、大文字・小文字だけの変更。先頭のA、固有名詞Monte Carlo、略語AIを大文字として保持。Title:の語と行の見た目は維持した。

## 検証

- 題名は **14 words**。`abstract_en_v5.md` の「## Title」と文字単位で一致。
- 題名1置換を逆適用したHTMLが変更前と完全一致。英語抄録本文は文字単位で変更なし、**477 words**を維持。
- ページ全体のHTMLに「A System for Monte Carlo Radiation Transport」は0件。app・content・workerのソースにも旧表記はなく、title・OGP・説明文などの別箇所で同じ題名は見つからなかった。
- 日本語の題名・本文、数値、図・3D、デザイン、構成、用語解説、意見フォーム、DB・マイグレーション等は変更なし。フォームへのテスト送信なし。
- `git diff --check` と本番ビルド成功。公開後に題名の大文字・小文字とレイアウトをブラウザで確認した。
- 確認画像：`runs/jsmp133_site_v37/english-title.png`。HTML控え：`runs/jsmp133_site_v37/review.html`。

## 公開情報

- URL: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site/
- Sites version: **37**
- source commit: `4bfa76ee8df46a118d127bab059e64bb56bb002a`
- saved version ID: `appgprj_6ada854f7c388191b7f4ae3247b0071e~appgver_157c32286948819193ca2519a7a8e3fe`
- deployment ID: `appgdep_6ae2dd0c45348191a6d88845f67b3589`
- status: `succeeded`
- 公開URL・公開範囲（public）は維持。

viveMonteリポジトリでのcommit・pushは実施していない。既存の変更・未追跡ファイルは保持した。
