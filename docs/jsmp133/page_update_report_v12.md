# 公開ページ修正報告（第12版、2026-10-29）

[第12版依頼](page_update_request_v12.md)に従い、Sitesの最新ソース（公開版33、`37ff1fdc6a5ff8e43ab5715d4197afb9169678aa`）を取得して編集した。

## 変更前 → 変更後

1. Methods（Evaluation 1）
   - 変更前：percentage depth dose (PDD) and off-axis ratio (OAR) profiles
   - 変更後：percentage depth dose (PDD) and off-center ratio (OCR) profiles

2. Methods（Evaluation 1）
   - 変更前：and OARs covered the field and extended to 3 cm beyond the geometric field edge.
   - 変更後：and OCRs covered the field and extended to 3 cm beyond the geometric field edge.

3. Results（Evaluation 1）
   - 変更前：0.39% and 1.3% for OAR;
   - 変更後：0.39% and 1.3% for OCR;

4. Results（Evaluation 1）
   - 変更前：the OAR maximum occurred just outside the field edge.
   - 変更後：the OCR maximum occurred just outside the field edge.

## 検証

- 指定の4置換を逆適用したソースが変更前と完全一致することを確認。変更は `content/review.html` の指定4か所のみ。
- 日本語抄録、数値、図・3D表示、デザイン、構成、用語解説、意見フォームと送信処理、DB・マイグレーションは変更なし。フォームへのテスト送信なし。
- ページ全体の可読テキスト（用語解説を含む）に `OAR`・`off-axis` は各0件。指定外の残存箇所なし。
- 英語抄録の本文は **477 words**。HTMLの文字参照を復元し、見出しと各段落の文字列を `abstract_en_v5.md` の「## Abstract」節と文字単位で比較し、一致を確認（HTMLとMarkdownによる段落区切りの違いのみ除く）。
- `git diff --check`、本番ビルド成功。
- 公開後に英語抄録のMethods・Resultsで、4か所のOCR表記とレイアウトを確認した。
- 確認画像：`runs/jsmp133_site_v34/english-ocr.png`。HTML控え：`runs/jsmp133_site_v34/review.html`。

## 公開情報

- URL: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site/
- Sites version: **34**
- source commit: `e9873579619d9c50ddcbb6eb93d19e32a6533c3d`
- saved version ID: `appgprj_6ada854f7c388191b7f4ae3247b0071e~appgver_e0a99dcda6088191a38dd5c90c6fb08e`
- deployment ID: `appgdep_6ae2c6f490908191b08fc784fb333b1e`
- status: `succeeded`
- 公開範囲：`public`を維持。

viveMonteではcommit・pushを実施していない。既存の変更・未追跡ファイルには手を加えていない。
