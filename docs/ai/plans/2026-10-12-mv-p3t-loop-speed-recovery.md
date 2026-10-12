# 計画: P3t — P3rで増えた計算時間の回復（結果を1ビットも変えない整理）

- 状態: implemented（2026-10-12。隔離された試作の中の2行の変更で、合否は完全一致で機械的に決まるため、Claudeが実装した。Codexには実装後の差分を独立にレビューさせた。結果は末尾の「実施の記録」）
- 作成者: claude
- 実装担当: claude
- 日付: 2026-10-12
- Codexレビュー: 実装後の差分レビュー1巡（原文: `reviews/2026-10-12-mv-p3t-review1-codex.md`）
- 規模: 軽〜中（物理も乱数の使い方も変えない。本体`chatcarlo/`には触れない）

## 目的

P3r（制動放射の光子の輸送）で、試作は約4割遅くなった。離散事象を切って結果がP3sと完全一致する状態でも約3割遅く、これは物理ではなく輸送ループの書き方による無駄である（P3rのDISCUSSION 6節。原因は当時切り分けていなかった）。まず離散事象を切った状態をP3sと同じ速さに戻す（ユーザー指示、2026-10-12）。

## 0. 前提事実（検証済み。2026-10-12に実行）

Mohan 10 MVスペクトル、`elsepa`、own刻み、50,000履歴、seed 777、1本だけで実行、各条件を交互に3回（最小値、µs/履歴）。

| 条件 | P3rのまま | (a)を外す | (a)(b)を外す |
|---|---|---|---|
| P3s | 208 | 209 | 205 |
| 離散事象を切る | 267 | 243 | 204 |
| 離散・その場 | 307 | 284 | 224 |
| 離散・輸送 | 315 | 292 | 231 |

- (a) `photon_transport_p3r.py` 277行: `priority=event_order(bb,eb,mb,hb,rb,ab)`。**戻り値はどこでも使われていない**（`grep priority` は定義の1行だけ）。呼ぶたびに長さ6の配列を新しく作る。ステップごと（約650回/履歴）に実行される。
- (b) 同233〜234行: `restricted_stopping(...)` が内部で `quantities(...)` を呼び、直後にもう一度 `quantities(...)` を呼んでいる。同じ引数で表を2回引いている。エネルギーの係数を更新するたびに実行される。
- どちらを外しても、返り値の全配列のハッシュは変わらなかった（3条件とも）。この表は作業用の試験版で測ったもので、生データは残していない。振り返りの後に同じ切り分けを測り直し、`docs/validation/mv/p3t/breakdown.json` に保存した（(a) 23〜28 µs、(b) 36〜57 µs）。
- `photon_transport_p3r.py` は `run_p3r.py` の `SOURCES`（ハッシュ固定の対象）に入っている。書き換えると、P3rの検査記録・解析の再実行が「ソースが違う」で止まる。

## 方針

- これまでの段階と同じく、**新しいモジュール `photon_transport_p3t.py` を作り、`photon_transport_p3r.py` は書き換えない**（P3rの記録の再現性を保つ）。次の段階（CPUの並列実行、P3b）はP3tを使う。
- 変更は (a) の行を削ることと、(b) を「`quantities` を1回呼び、その戻り値の軟らかい放射阻止能を使って `S=collision(ev)-(hard(ev)[1] if hard_on else 0.)+bsoft` と書く」ことだけ。演算の順序は `restricted_stopping` と同じに保つ。
- `bremsstrahlung_p3r.py` は書き換えない（P3tからそのまま使う）。

## 対象範囲

- 変更してよい: `prototypes/mv_electron/photon_transport_p3t.py`（新規）、`prototypes/mv_electron/run_p3t.py`（新規）、`prototypes/mv_electron/run_p3t_breakdown.py`（新規、振り返りで追加）、`prototypes/mv_electron/tests/test_p3t.py`（新規）、`docs/validation/mv/p3t/`（新規）、P3rのDISCUSSION 6節への追記、`CLAUDE.md`・`STATUS.md`・親計画の進捗の更新。
- 変更禁止: `chatcarlo/`、`tests/`、`scripts/`、`examples/`、既存の `prototypes/mv_electron/*.py`（特にハッシュ固定の `photon_transport_p3r.py`、`bremsstrahlung_p3r.py`、`run_p3r.py`、`tests/test_p3r.py`）、過去の段階の結果ファイル。
- スコープ外: 表の引き方そのものの高速化（結果のビットが変わりうる）、CPUの並列実行、P3sから残る約100 µsの原因調査。

## 受入条件

**注意（事前登録ではない）**: この計画ファイルは、0節の測定と変更の試作を終えた後に書いた。受入条件と、T4のしきい値「+3%以内」は、離散事象を切った状態がP3sとほぼ同じ速さになる結果を見た後に置いたものである。T1〜T3（完全一致）は合否が機械的に決まるので影響は小さいが、T4は事前に決めた基準としては読まないこと。

- [x] T1: P3tとP3rの返り値（15個すべて）が、離散・その場、離散・輸送、離散事象を切る、陽電子の線源、電子の線源、無限媒質、カーマと対生成の採点なし、横方向の採点あり、一次光子の記録あり、刻みの規則 `egs`・`const`、の11条件で完全一致（型・形・バイト列。レビュー後に `np.array_equal` から強めた）。
- [x] T2: 離散事象を切ったP3tの返り値のうちP3sと共通の13個が、P3sと完全一致。
- [x] T3: Mohanスペクトル5万履歴でも T1（3条件）と T2 が成り立つ。
- [x] T4: 離散事象を切ったP3tの時間が、交互3回の最小値でP3sの+3%以内。
- [x] T5: `pytest prototypes/mv_electron/tests -q` が全部通る。

## 停止条件

T1〜T3のどれかが不一致なら、速度の数字を報告せずに止めて原因を調べる。

## テストコマンド

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m prototypes.mv_electron.run_p3t 3 50000
```

## 実施の記録（2026-10-12）

結果は [docs/validation/mv/p3t/RESULTS.md](../../validation/mv/p3t/RESULTS.md)。

Codexの実装後レビュー（致命0、実質2、軽微3）を反映した: 一致の比較をバイト列に強める、一致を確かめてから時間を保存する、T4を自動判定する、ソースのハッシュを記録する、テスト条件を入れ替える。反映後の再レビューはしていない。
