---
phase: 開発中
next: G5 A4u の H6 は**関門がすべて合格**（A1・A2・A3・B・C2、C1 は報告専用）。残るのは**ユーザーの判断 1 件**——本番の組は診断の格子の作業で旧実装の **2.48 倍**遅く、計画が報告専用の D10(2) に登録した「1.3 を超えたら相談」の合図が出た。原因は測定済み: **相互作用の数は旧新で完全に同じ**（g2b 3.8744/履歴）で、増えたのは相互作用 1 回あたりの費用（解析的 Klein-Nishina → 表引きの束縛非干渉散乱）。g2b は g4 より相互作用が 5.0 倍多いので増分を 5 倍払う（g4 は 1.03 倍）。**移植そのものは悪化なし**（D10(1) は 1.0001・1.0085・0.9856・治療 0.9765、上限 1.0571）。起動役が報告専用の C3 の閾値を C2 の合否に混ぜているのも直す必要がある（閾値は動かさない）
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
