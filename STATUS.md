---
phase: 開発中
next: G5 A4u の H6 の**T4 の 1 セッション**（`a4u_h6.py prepare` → `run` → `judge`）。手元は全部終わった——起動役と fixture の実装・独立検証、CPU の B1〜B5 全合格、**全テスト通し合格**（`tests/` 1100 passed、MV 1804 passed・17 skipped。不合格 4 件は空き容量切れで、再実行して 8.48 秒で合格）。**実行の前に空きを 7 GiB 以上にする**——両スイートの通しは一時出力に約 6 GiB 使い3 GiB の床を自力で突き抜ける。pytest の一時出力は `~/.Trash/pytest-temp-2026-11-06b/` に退避済み（空にするのはユーザー）
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-06
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
