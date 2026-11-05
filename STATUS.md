---
phase: 開発中
next: G5 A4u の H6（T4 の 1 セッション）を**小項目だけに縮小**（ユーザー決定 2026-11-06）。約 10 分: G1 の 51 量・G4 の画像・G2 のボクセル・D8 (3)・低エネルギーの CPU 対 T4・故障注入・NVRTC の経路・S1 の T4 分・D10 の速さ。**MV は外す**——変更分の上限が 500 keV で 0.26%（許容 1%）、効きが最大の 20〜150 keV は H3a・H4・H5・中心軸 PDD で EGS5 判定済み、よって A3 の合格を引き継ぐ。**`seed_ratio` も外す**——封印済み参照は EGS5 でなく**旧 P9c 試作**で、同一物理の移植試験だったため、物理を変えた後に当てても正しさの情報が出ない（参照の取り直し＝シード近傍の EGS5 照合が課題として残る）。根拠は A4U.md「H6 を縮小した」
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-06
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
