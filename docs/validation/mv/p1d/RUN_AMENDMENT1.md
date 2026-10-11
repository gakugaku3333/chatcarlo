# P1d 追補1の実行と再開

追補1（approved）のA0〜A6を実行。元の停止記録は `stop1/` にSHA256一覧つきで保全した。元の `progress.json` の `restart_allowed: false` は保持する。凍結1のR0不合格は継続して記録し、再開後のR0は凍結2で判定する。

再開コマンドは次のとおり。

```bash
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py
```

入口は追補approved、凍結2の一致、メモリ上再現成功の3条件を確認する。保存済みR1標本と完了済み輸送runはSHA256と条件を検証して読み込み、未完了runだけを固定seed・固定履歴数で開始する。各輸送runは一時NPZ作成、完了レシート保存、NPZ確定の順で保存する。未完了runは履歴途中からの継続ではなく、同じ条件でそのrunを最初から実行する。履歴数やseedの追加はしない。

凍結2対象は `amendment1_hashes.json` の明示一覧。新しいPythonファイルや旧成果物の追加・削除も不一致として検出する。凍結2から外した対象、凍結1との既知の不一致、許可されたソース変更を `amendment1_changes.json` に分けて記録した。今回のコード差分全体は `amendment1_diff.txt`。

実行コマンド:

```bash
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --seal
.venv/bin/python prototypes/mv_electron/run_tests_p1d.py
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --reproduce
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched
```

A0のコピー・SHA256取得、A1の一覧作成、変更前ソースの一時保全はインラインPythonで行った。A3の `--seal` が変更前ソースとの差分を保存し、凍結1の共通ファイルの許可されていない変更がないことを検証した。凍結2の追加・削除・変更の検出は一時ディレクトリの模擬対象で検証した（凍結済みの実ファイルは編集していない）。受入テストの具体的なコマンド・終了コードは `test_commands.json`。輸送子プロセスの全コマンドは `amendment1_commands.json`。

再開後の進捗は `amendment1_progress.json`、計算ログは `amendment1_execution.log`。各backendの `runs/` にNPZとJSONレシートを保存する。時間計測は輸送子プロセス終了後に単一プロセスで行い、統計に加えない。結果に合わせる調整、EGS5の再実行、計画ファイルの編集、commitは行わない。
