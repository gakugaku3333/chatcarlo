---
phase: 開発中
next: G5 A4u の H6（T4 の 1 セッション）。**着手前の前提確認で、H6 の時間の内訳が 10 倍間違っていたことが判明**（581.8 秒は 460 万履歴の値で 126.4 µs/履歴。本番は 10 MV 4,600 万・4 MV 2,600 万なので実測分は約 2.3 h ではなく約 4.5 h が上限側。訂正は A4U.md の「訂正（2026-11-05…）」）。束は検証済みで既定の場所にある（`bundle_manifest_sha256` 9f54db5a…、4 ファイルのハッシュもディスクと一致）。既存の T4 道具に物理名・tag の焼き込みは無く、束から来る。次は固定費を `thin` 50 万で測り、H6 の起動役を codex-kyodo で作る
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-05
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
