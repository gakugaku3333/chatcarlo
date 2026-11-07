---
phase: 開発中
next: G5 A4u の H6 は **C2 以外の関門がすべて合格**（A1・A2・A3・B・B6。`judge` を初めて走らせて確認）。**H6 自体はまだ合格していない**。残るのは**ユーザーの判断 1 件**——本番の組は体積格子の経路で旧実装の **2.48 倍**遅く（g2b 500k: 279.24 → 691.32 ms）、計画が報告専用の D10(2) に登録した「1.3 を超えたら相談」の合図が出た。検出器の経路は 1.035 倍、治療は 1.080 倍。**移植そのものは悪化なし**（D10(1) は 1.0001・1.0085・0.9856・治療 0.9765、上限 1.0571）。相互作用の数は旧新で実質同じ（g2b 193,720 → 193,722）だが、**機構は切り分けられていない**——`G5_TEST_LEGACY_LOCAL` は履歴関数を丸ごと差し替えるため、A4u の診断側の変更の全部が候補。証拠の `~/chatcarlo_h6_2026-11-07k` は `judge` の入力なので消さない
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
