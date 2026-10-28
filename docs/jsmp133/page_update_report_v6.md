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

## 公開後の表示調整と締め（2026-10-24）

- version 13：IVR室3Dの線量凡例から長い統計・診断テキストを除き、カラーバーだけを残した。材料凡例は描画領域の上へ、カラーバーは右の専用領域へ配置した。線量分布に重ならないことを公開ページで確認。Sites commit `a312bcefefed82269834381fe62aecf4e52a91e1`、公開ID `appgdep_6adcb3cad2b881919d0705d8ac090a1d`。
- version 14（最終公開版）：材料とX線ビームを、色見本と名称を一組にした横並び・折り返し表示へ変更。Sites commit `08cc45c353d0c7666f31882db14bccc96408a4d2`、保存版ID `appgprj_6ada854f7c388191b7f4ae3247b0071e~appgver_19f1689f45dc819194a8325dc1b9b493`、公開ID `appgdep_6adcb4c896908191b3791faf9823f494`。公開成功時刻は2026-10-24 22:38:30 JST。
- version 13・14は公式ビルド成功後にSites側でcommit・pushし、保存・公開を実施。最終保存版の再照会でも同じcommitと公開IDを確認。Sites作業ツリーはクリーン。viveMonteではcommit・pushしていない。
- 各修正はIVR室のiframe内の凡例表示に限定。線量データ・描画処理と、iframe外の本文・図1・感想フォームは維持。最終公開ページのブラウザ表示で、材料の横並び・折り返し、カラーバーの保持、分布に重ならない配置を確認した。
- 使用者の表示方針：線量分布を優先し、常時表示の線量凡例はカラーバーだけ、材料は描画領域外で横並びにする。長い診断情報が分布の前に重なると閲覧を妨げるため、今後もこの配置を維持し、狭い画面では項目単位で折り返す。
- 利用方法：上記公開URLを閲覧者へ案内する。表示下限・上限、不透明度、視点切替はそのまま利用できる。次回編集はSitesの最新ソースを取得し、保存版IDとcommitを照合してから行う。
- 最終表示の控え：`runs/jsmp133_site_v6/legend_final.png`。一時的な配信用アーカイブと取得HTMLは削除し、Sitesソース作業ツリーとこの確認画像は保全した。
