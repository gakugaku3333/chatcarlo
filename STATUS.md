---
phase: 開発中
next: G5 A4u の H6 の**T4 のセッション 5**（B4〜B6・C1・C2）。**先にゴミ箱を空にする**——空きが 3.04 GiB で `prepare` の床（3.5 GiB 必要）を越えられない。ゴミ箱に約 5.5 GiB（セッション 1〜4 の作業ディレクトリ 4 本と再生成できる payload 2 本）が入っており、空ければ約 8.5 GiB になって残りの周回に十分。**証拠は全部リポジトリの `docs/validation/gpu/g5/a4u_h6_run/`（288 KB、push 済み）に移したので、ゴミ箱の中身は捨ててよい**。ここまでの実機の結果（2 セッションで再現）: A1 合格（61 量）・A2 合格（格子 4 組）・A3 完全合格（`per_history_exact` 1.0・整数の相違 0）・B1〜B3 合格（到達数は CPU と全桁一致）。捕まえた実機だけの不具合 3 件はすべて修正し、うち 2 件をテストに昇格（`tests/test_gpu_nvrtc_macros.py` 15 件）
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
