---
phase: 開発中
next: G5 A4u の H6 は**コード側の条件がすべて揃った**（`speed()` を計画の定めに戻し、`Owner.call` の再試行はセッション 10 で実測有効——VM が応答せず `colab exec` が 300 秒で 3 回切れたが3 回目で回復して A1 合格）。残るのは **`prepare` → `run` → `judge` の 1 周**だけ。**`prepare` の前に空きを 8 GiB 以上にする**——実測で `run` は開始時に 5.8 GiB 以上要り（完走した 5.83・5.97 対 中断した 4.45・4.71）、しかも空きはこの作業と無関係に 1 GiB ほどゆらぐ。`judge` では A1・A2・A3・B・B6 の合格を既に確認済みで、C2 の D10(1) も基準を満たしている
blocker: H6 の最後の 1 周に機械側の空き容量が足りない。APFS コンテナが 98.6% 使用で、ゴミ箱を空けても約 5.0 GiB（`run` は開始時に 5.8 GiB 以上要る）。さらに 3〜5 GiB 必要。`~/.cache/chatcarlo` の 25 GB はどれも文書から参照されているので、決まりに照らしてきれいな消去候補が無い。ELSEPAの表の配布方法とLICENSEは従来どおりユーザーの判断待ち（急がない）
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
