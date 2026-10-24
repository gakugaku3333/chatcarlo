# 公開ページ更新報告（第6版）

対象：<https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site>

## 変更前 → 変更後

補足資料「依頼2のねらい」の後の図1の画像だけを差し替えた。

- 変更前：均一領域を示す橙の矩形と、画像内の「橙の枠は…」の説明があるPNG。SHA-256：`e5a68502284f8e6c19da4611b7b74b5e7aa9406ecdf05d7244ed02e730a2364a`。
- 変更後：`docs/jsmp133/request_2_entry/frames_common_scale.png`。橙の矩形とその説明を除いたPNGを、再作図・再圧縮せず、そのままdata URLに埋め込んだ。206,927バイト、SHA-256：`0ced6bbe1cb07bfe480f6d0fae4ed33fde4e7f4a1eeedd8fa92706626d1fa800`。

## 検証

- Sites公式ソースヘルパーでリモート最新を取得。開始時HEADはversion 11の `db7ac356047f03bb7b6fb9713fee91a3a145b48d`。開始時・commit後の作業ツリーはクリーン。
- Sitesソースの変更ファイルは `content/review.html` だけ。対象imgのsrcを1か所だけ置換。新data URLを旧data URLに戻すと、変更前HTMLとバイト単位で完全一致することを確認。
- 図の説明文・alt・style・配置、ほかの文章・数値・図・3D表示・デザイン・構成・両感想フォーム・送信処理は変更なし。DB binding、API、認証、スキーマ、マイグレーションも変更なし。フォーム送信、保存済み回答の読み書き・削除・移行は実施していない。
- Sites公式 `build-site.mjs` によるビルド成功（/、/api/feedback、/responses）。公式ヘルパーでSitesソースだけcommit・pushし、同じcommitから公開用アーカイブを作成した。
- 公開成功後、公開URLからHTTP取得したHTMLの対象img（alt「入射光子数による画像の変化」）を抽出。PNG data URLの復号結果は、指定PNGと206,927バイトすべて一致。上記SHA-256も一致。
- ブラウザを再読み込みし、橙の矩形がない5段階の図1と、従来の説明文が表示されることを確認。
- 公開前後のURLとアクセス方針全体が一致。公開範囲 `public`、revision 2を維持。

## 公開結果

- **Sites version 12、公開成功**（`status: succeeded`、`type: publish`、`failure_message: null`）。
- 保存版ID：`appgprj_6ada854f7c388191b7f4ae3247b0071e~appgver_8343d49e942881919f661ce0fecbbe74`
- Sitesソースcommit：`9149fa5fc098aa953f8ba6eed874557a4e2638cc`
- 公開ID：`appgdep_6adcac18860081919ec0fcdfc010e3f6`
- 公開完了：2026-10-24 22:01:27 JST。
- URL：<https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site>（変更なし）。
- 第5版報告の「公開結果」は、version 11公開済み・公開ID・A〜DとPNGの確認結果を示す内容へ整理。第6版公開後の最新状態への参照も追記。
- viveMonteリポジトリでは報告書2件のみ更新し、commit・pushは実施していない。
