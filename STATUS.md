---
phase: 開発中
next: G5 A4u の H6 の**T4 のセッション 7**（残りは C2 だけ）。**先にゴミ箱を空にして空きを 6 GiB 以上にする**——`prepare` は 3.5 GiB で通るが `run` は payload 1.27 GiB と回収ごとの reserve で 6 GiB 要る（セッション 6 は 4.45 GiB で始めて A2 の回収中に床に当たった。VM は停止済み）。セッション 5 で **A1・A2・A3・B（B1〜B6 全部）・C1 が合格**し、残るのは C2 のみ。C2 の検出器なし構成（g2b）の KeyError は直し、`prepare` が VM の前に CPU で測定経路を検査するようにした。証拠は全部 `docs/validation/gpu/g5/a4u_h6_run/` に入っているのでゴミ箱の中身は捨ててよい
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
