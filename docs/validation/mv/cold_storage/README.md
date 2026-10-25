# 閉じた段階のキャッシュの退避（2026-10-26）

計画: [2026-10-26-cache-batch-tables-to-drive.md](../../../ai/plans/2026-10-26-cache-batch-tables-to-drive.md)。
決まり: [mv.md](../../../../.claude/rules/mv.md) の「Stage-close data review」。

## 何をどこへ移したか

| 分類 | 対象（`~/.cache/chatcarlo` からの相対パス） | 行き先 | 一覧 |
|---|---|---|---|
| 表＋Drive（K） | `p8g3/colab`、`statsstack/p8f`、`mv_entry/96782f068a81`（`transfer_parts` を除く）、`p8g1/p8g_ref_gate2.npz`、`p8g2/p8g2_ref_gate2.npz`。計 1,340 ファイル、2.29 GB | gmail 側 Drive の `マイドライブ/viveMonte/cache_cold/<同じ相対パス>` | `p8g3_colab.sha256`、`statsstack_p8f.sha256`、`mv_entry_96782f068a81.sha256`、`ref_gate.sha256`。Colab で Drive から取った一覧は `colab_all.sha256`（1,340 件すべて一致） |
| 消す（D） | EGS5 の作業フォルダの `gsdist.dat`・`pgs5job.GSlst`、`p8g1/cli`・`cli_attempt1`、`p8g2/cli`、`mv_entry/96782f068a81/transfer_parts`。計 75 ファイル、5.23 GB | ゴミ箱（Drive には送っていない） | `deleted_20261026.sha256` |

- 元は `~/.Trash/cache_cold_*`（K）と `~/.Trash/cache_deleted_*`（D）に移した。ゴミ箱を空にするのは使用者。
- 2026-10-26 にクローンにした組があるので、空く量は見かけの合計より小さい（K と D で 3.61 GB の計算）。
- 関門の記録:
  - 移す前の必須の検査（`gate_T.txt`、31 件合格・skip 0）
  - 凍結の検査の前後の一覧（`protected_before.json`、`protected_after.json`。どちらも既存の 2 件だけで完全一致）

## 手元に残したもの

- `statsstack/` と `stats4mv/` の `batch_tables.npz`・`.json`（全 10 計算の外側のバッチの表、平均・SEM、N・seed・元のハッシュ）
- `pool_tables.py`（表だけから RESULTS の出力を再現する。通常・`orig`・`detail` の 5 つのモードで、元の `pool.py` の出力とバイト一致）
- キャッシュの小さなファイル（`complete.json`・`identity.json`・`egs5job.out`・入力・`statsstack/out`・`stats4mv/out` など）

## 戻し方

1. Drive の `cache_cold/<相対パス>` を、`~/.cache/chatcarlo/<相対パス>` へ写す。
2. `~/.cache/chatcarlo` で `shasum -a 256 -c <一覧>` を実行して照合する。一覧のパスは `~/.cache/chatcarlo` からの相対パス。
3. D は Drive にない。要るときは作り直す（EGS5 は実行し直すと `gsdist.dat` を作る。束は各段階の `export` で作る）。

| 戻す対象 | 戻すと動くもの |
|---|---|
| `p8g3/colab`、`statsstack/p8f` | `statsstack/pool.py`（表を使わない元の集計）、P8g-3 の再実行・照合 |
| `mv_entry/96782f068a81` | `stats4mv/pool.py` |
| `*_ref_gate2.npz` | P8g-1・P8g-2 の関門 2 |
| （D を作り直したうえで）`p8e/egs`、`statsstack/work`、`stats4mv/work` | P8e・P8f の再実行、EGS5 の追加の計算、P9a の `egs-replay`・`egs-pilot`・`egs` |

## Colab から Drive を照合するときの注意

- `colab drivemount -s <セッション>` は、セッションごとに Google の許可画面の承認が要る（使用者が行う）。
  標準入力に Enter を送るまで待つので、FIFO をつないで背景で動かし、URL を使用者に渡す。
- 承認した直後の 1 回目は `mount failed` になった。同じセッションでもう一度 `drivemount` すると通った（n=1）。
- Drive for desktop のアップロードには時間がかかる（2.29 GB で約 25 分）。Colab から見えるファイル数がそろってから SHA256 を取る。
