---
phase: 開発中
next: G5 A4u の H6 の**T4 のセッション 2**。セッション 1（2026-11-07、11 分 47 秒）は A1 のコンパイルで不合格——**実機の NVRTC だけで出る不具合を 1 件捕まえた**（A4u が診断側に入れた `photon_physics.h` の `NAN` に対し、診断の入口 `g1_gpu.cu` に守りが無い。治療と seed の入口には元からある）。`g5_base.h` に守りを足し、教訓は 4 回目なので [tests/test_gpu_nvrtc_macros.py](tests/test_gpu_nvrtc_macros.py)（7 件・0.27 秒・GPU 不要）に昇格させた。次の順序は「コードの変更を終える → **コンパイルだけの下見 1 回**（断片 5 個・約 1 分。計画に無い追加でユーザーの判断待ち）→ `prepare` やり直し（約 4 分）→ `run`（poll 60 秒に変更済み、待ちの下限 6 分）→ `judge`」。証拠の `~/chatcarlo_h6_2026-11-07c` は消さない。実行の前に空きを 7 GiB 以上にする
blocker: ELSEPAで作った表の配布の方法（表を置くか、生成スクリプトだけか）とリポジトリのLICENSEはユーザーの判断待ち。急がない
updated: 2026-11-07
---

治療領域拡張の段階の状況は [docs/validation/mv/STATUS.md](docs/validation/mv/STATUS.md)、常設ルールは
[.claude/rules/mv.md](.claude/rules/mv.md)、発表は [docs/presentation_jsmp133.md](docs/presentation_jsmp133.md)、
全体の計画は [親計画](docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) を参照。
記録の置き場所の決まりは [CLAUDE.md](CLAUDE.md) の「記録の置き場所」。

水60 keVの追加100バッチは、[結果記録](docs/validation/gpu/water60_100M/RESULTS.md)の保全表に従い、約27 MBの水中バッチ表とreceiptを残して約3 GBの原本を整理済み。全格子の統合NPZは保持。
