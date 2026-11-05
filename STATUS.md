---
phase: 開発中
next: G5 A4u の H6。**起動役は実装・手元検証まで完了**（`a4u_h6.py`・`a4u_h6_fixtures.py`、群 A/B/C。CPU の B1〜B5 は全部合格し到達数を封印）。**T4 の 1 セッションは未実施**。先に要るのは**ゴミ箱を空にすること**——pytest の一時出力 2.6 GiB を `~/.Trash/pytest-temp-2026-11-06/` へ退避したが同じ volume なので空きが戻らず（今 3 GiB＝床と同じ）、全テストの通し実行ができない（477＋301 passed・不合格 0 まで進んで停止条件で中断）。範囲外の修正 1 件: `seed_cuda.py` の `GPU.observe()` が必ず `NameError` になる欠陥（CUDA 専用で CPU の試験が到達しないため A2b から残存、B3 が止まる位置）を直した
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-06
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
