# 計画: CLAUDE.md を常設ルールと段階の記録に分ける

- 状態: approved（2026-10-19、ユーザー承認。「何も失われていない」の確認は Codex が独立にレビューする）
- 作成者: claude
- 実装担当: claude
- 日付: 2026-10-19

## 目的

CLAUDE.md（632 行・64 KB）は毎回のセッションで全文が読まれる。公式の目安は 1 ファイル 200 行未満
（https://code.claude.com/docs/en/memory）。大半は治療領域拡張の段階ごとの記録で、各段階の RESULTS と重なる。
毎回必要な決まりだけを残し、場面ごとのルールは `paths:` 付きの `.claude/rules/` に、段階の記録は
`docs/validation/mv/STATUS.md` と各 RESULTS に移す。今後、段階の記録は CLAUDE.md に書かない。

## 前提の事実（2026-10-19 に確認）

1. `docs/validation/mv/p1c/RESULTS.md`・`p1d/RESULTS.md` は P1e/P1f の凍結対象（`prototypes/mv_electron/p1e_common.targets()`
   が `docs/validation/mv/p1,p1b,p1c,p1d/` の全ファイルを数える）。中身を変えたりファイルを足したりしない。
2. CLAUDE.md はどの凍結リスト・実行中の検査にも入っていない（`test_p1f.py` は入っていないことを確かめている）。
3. AGENTS.md は CLAUDE.md の見出し「Commands」「Cross-section/dose-coefficient data provenance」を名前で参照する。
4. 古い文書が「Physics」「Dose/H\*(10)タリー」「Known sharp edges」「Therapy-energy extension」の節を参照する。
5. `@import` は読み込み量を減らさない。`paths:` 付きの rules とサブフォルダの CLAUDE.md だけが必要時に読まれる。
6. 自動メモリーは PC ごと（MacBook と Mac mini で共有されない）。

## 照合で見つかった、CLAUDE.md にしかない記録（S1 で移す）

| # | 内容 | 移す先 |
|---|---|---|
| 1 | P1c の K9(i) のずれ（約 7σ、P1b からの系統的なずれ。P1b の「仕組みは健全」はこの但し書きつき） | STATUS.md（P1 の注記） |
| 2 | P1c の mf と dcslib の差が 81 bin 中 29 bin で 2σ 超 | STATUS.md |
| 3 | タリー: 0 でないボクセル 442,111 → 477,913（+8.1%） | docs/plan_tally_speedup.md |
| 4 | P3b-1: 毎ステップの確保を外す前は ≈700 µs/履歴 | p3b/RESULTS_P3B1.md |
| 5 | P7: 履歴数の上限を 2.4 億に上げたのはユーザーの決定 | p7/RESULTS.md |
| 6 | 圧縮済みキャッシュの一覧・大きさ、p2/egs5 展開の必要、`np.savez` を変えない | STATUS.md と rules/mv.md |
| 部分 | P1d の残差の変化が 0 と区別できない、P2 Q7 の余裕なし、P2b 最大 0.38%、P2c は Heitler、P3r の速度 310 と 301 | STATUS.md |

## 対象範囲

- 変更してよい: `CLAUDE.md`、`AGENTS.md`、`.claude/rules/mv.md`（新設）、`.claude/rules/chatcarlo.md`（新設）、
  `docs/validation/mv/STATUS.md`（新設）、`docs/plan_tally_speedup.md`、`docs/validation/mv/p3b/RESULTS_P3B1.md`、
  `docs/validation/mv/p7/RESULTS.md`、この計画書、自動メモリー（`~/.claude/projects/.../memory/`）
- 変更禁止: `docs/validation/mv/p1*/` の中身、コード、テスト、manifest・ハッシュの JSON、`data/`
- この計画の外（別承認）: `~/.claude/skills/shime` への追記（S6）

## 手順

- S0: 実行中の計算がない・CLAUDE.md に未コミットの変更がないことを確認し、変更前の行数と SHA256 を記録。
- S1: 上表の記録を移す。追記には「追記（2026-10-19、CLAUDE.md から移した。判定は変えない）」と書く。
- S2: `.claude/rules/mv.md` と `.claude/rules/chatcarlo.md` を作る（`paths:` 付き）。
- S3: CLAUDE.md を書き直す（150 行前後）。見出し「Commands」「Cross-section/dose-coefficient data provenance」
  「Known sharp edges」は残す。移した節の行き先の表と、記録の置き場所の決まりを置く。
- S4: AGENTS.md に rules と STATUS.md への案内を足す。
- S5: 自動メモリーの進捗の重複を STATUS.md への案内に置き換える。

## 受入条件

- [ ] CLAUDE.md が 200 行未満。rules の 2 ファイルもそれぞれ 200 行未満で、`paths:` を持つ。
- [ ] 旧 CLAUDE.md のすべての数値・リンクが、新しい置き場所（CLAUDE.md、rules、STATUS.md、各段階の文書、計画書、
      lessons_learned、その他 docs）のどこかにある。スクリプトで確認し、見つからないものは 0 件か、理由つきで列挙する。
- [ ] 新しいファイルのリンクがすべて実在するファイルを指す。
- [ ] `p1c/RESULTS.md`・`p1d/RESULTS.md` の SHA256 が `p1f/freeze_hashes.json` の値のまま。
- [ ] `git diff --stat` に変更してよいファイルだけが出る。
- [ ] Codex が「何も失われていない」を独立にレビューし、指摘への対応を記録する。
- [ ] 新しいセッションで MV のファイルを読むと `mv.md` が読み込まれる（ユーザーが確認）。

## テストコマンド

コードは変えないため pytest は対象外。確認は受入条件のスクリプトと `git diff --stat`。

## 追補1（2026-10-19、実装での変更・判明したこと）

- 照合表の #4（P3b-1 の約 700 µs）は `p3b/RESULTS_P3B1.md` に「697 µs/粒子」として、#5（P7 の上限 2.4 億）は P7 計画書の
  85・96 行目にあった（照合の見落とし）。どちらも書き足していない。
- STATUS.md には、一覧と注記のほかに、旧 CLAUDE.md の「Therapy-energy extension」節の本文（38〜318 行）を末尾に
  そのまま移した（リンクだけ STATUS.md からの相対に直した）。要約し直して注意書きを落とす危険を避けるため。
  新しい段階は本文に足さず、一覧に 1 行足す。
- P2 の Q7 の統計不足は 2 件（`dcslib` 10 MeV fe 0.3345%、`elsepa` 10 MeV k 0.3354%）。CLAUDE.md の記述は 1 件分だった。
- 自動メモリー: `project_mv_extension.md` を STATUS.md への案内とメモリーにしかない未決事項だけにした。
  `project_status.md`（1,224 行、2026-07〜08 の経緯）は消さず、先頭に「古い。リポジトリが正」の注記だけを入れた。
  書き換え前の版は作業用の一時フォルダに控えた。
- 結果: CLAUDE.md 632 行・64 KB → 186 行・13.8 KB。`mv.md` 111 行、`chatcarlo.md` 105 行。旧 CLAUDE.md の数値・リンク・
  コード名 1,035 個のうち、新しく書いた部分に 743、STATUS.md の本文に 242、その他の docs に 40 個。10 個は表記の違い
  （`../../` に書き換えたリンク 9 個、`200,000` の切り出し 1 個）で、実際の欠落はない。P1c・P1d の RESULTS の
  SHA256 は凍結値のまま。

## 追補2（2026-10-19、Codex の独立レビュー）

原文: [reviews/2026-10-19-claude-md-split-review-1.md](reviews/2026-10-19-claude-md-split-review-1.md)。61 項目を照合し、
所在が見つからない項目は 0 件。凍結対象の変更なし。ただし、作業のときに読まれる場所への移し忘れが実質 5 件あり、すべて直した。
1. CLAUDE.md の再検証の要約に「42 合格・1 不合格・4 統計不足、補助判定は不合格を置き換えない、Rayleigh fit 警告は未定量」を戻した。
2. mv.md に「合格構成を変える高速化は別計画で鎖 2 に戻る」を足した（P5d RESULTS 59 行）。
3. mv.md に「表の定数の log は端末上で同じ log で作り、ホストで作って渡さない」を足した（P5i RESULTS 32 行）。
4. mv.md に「結果を引くときの留保」の節を足した（P1c 未判定と P1b の但し書き、P3s の S4 統計不足、P3b-1 の B8 不合格の維持、
   P4a の Fano の適用範囲、判定の範囲とビルドアップの過剰）。
5. 本体の計画・レビューでも先に chatcarlo.md を読むよう、CLAUDE.md と AGENTS.md に書いた（`paths:` は広げていない）。
