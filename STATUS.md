---
phase: 開発中
next: G5 A4u の H6 の**T4 のセッション 8**（残りは C2 だけ）。**先にゴミ箱を空にして空きを 6 GiB 以上にする**。セッション 5・7 で **A1・A2・A3・B（B1〜B6 全部）・C1 が合格**。C2 の不具合 2 件はどちらも「手元で一度も走っていない経路」で（検出器なしの g2b 構成、`fg` が None でない ledger の枝）、両方直して **`prepare` が VM の前に CPU で通す**ようにした。次に C2 で止まるなら GPU 固有の理由に限られる。`.DS_Store` で `prepared()` が拒否する問題も `ours()` で解決（VM 作成前なので損失なし）。証拠は全部 `docs/validation/gpu/g5/a4u_h6_run/` にあるのでゴミ箱の中身は捨ててよい
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
