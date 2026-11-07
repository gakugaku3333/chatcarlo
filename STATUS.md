---
phase: 開発中
next: G5 A4u の H6 は**コード側の条件がすべて揃った**（`speed()` を計画の定めに戻し、`Owner.call` の再試行はセッション 10 で実測有効——VM が応答せず `colab exec` が 300 秒で 3 回切れたが3 回目で回復して A1 合格）。残るのは **`prepare` → `run` → `judge` の 1 周**だけ。**`prepare` の前に空きを 8 GiB 以上にする**——実測で `run` は開始時に 5.8 GiB 以上要り（完走した 5.83・5.97 対 中断した 4.45・4.71）、しかも空きはこの作業と無関係に 1 GiB ほどゆらぐ。`judge` では A1・A2・A3・B・B6 の合格を既に確認済みで、C2 の D10(1) も基準を満たしている。**今日の T4 の記録の正典は `docs/validation/gpu/g5/A4U.md`「H6 の T4 実施（2026-11-07）」に移した**（計画ファイルは計画に戻した）。生データは `a4u_h6_run/`（22 ファイル）
blocker: なし（**当日の「機械側で 3〜5 GiB 必要」は誤りだった**——自分の使い捨て 1.3 GB を数えていなかった。退避後は空き 6.02 GiB で、ゴミ箱を空ければ約 9.2 GiB になり要件 8 GiB を満たす）。ELSEPAの表の配布方法とLICENSEは従来どおりユーザーの判断待ち（急がない）
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
