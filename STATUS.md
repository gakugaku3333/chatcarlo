---
phase: 開発中
next: G5 A4u の H6（T4 の 1 セッション）。**実測で約 2.5 時間**（10 MV 21.0 分・4 MV 12.1 分＝A3 の残存データから実測、`seed_ratio` 116.2 分、＋小項目）。以前の「2.3 時間」も「4.5 時間」も誤りで、A4U.md に訂正済み。**bin を粗くしても履歴数を削っても効かない**（判定は既に 5 cm ブロック・臓器平均、封印済み参照の SEM が予算の 16〜32%、`seed_ratio` の N は既に最小）。削る余地は `seed_ratio`（全体の 78%）を回すか否かのみ——ユーザーの判断待ち。A3 の本番 2 本（1.8 GB、`~/.cache/chatcarlo/g5/a3/runs/`）は対の相手なので消さない
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-05
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
