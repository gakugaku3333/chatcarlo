---
phase: 開発中
next: G5 A4u の H6 の**T4 のセッション 4**（B4〜B6・C1・C2）。**先にゴミ箱を空にする**——空きが 3.24 GiB で `prepare` の 3 GiB の床を越えられない（ゴミ箱の今日の使い捨て約 3.5 GiB を空ければ約 6.7 GiB）。ここまでの実機の結果: **A1 合格**（61 量、CPU の双子と T4）、**A2 合格**（格子 4 組）、**A3 完全合格**（`per_history_exact` 1.0・整数の相違 0）、**B1〜B3 合格**（到達数は封印済みの CPU と全桁一致）。止まった 2 件はどちらも「VM に xraylib が無い」で、A3 はホストで作った表の差し替え（編集なしの pack が束と完全一致することを根拠に）、B4/B5 は `freeze_inputs` での凍結で直した。xraylib を import 不能にして B1〜B5 を手元で全部通す検査を入れた。証拠の `~/chatcarlo_h6_2026-11-07c`・`d`・`e` は消さない
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
