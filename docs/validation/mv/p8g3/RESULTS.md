# P8g-3 — Codex 実装と手元の関門

計画: [2026-10-24-mv-p8g3-stack-production.md](../../../ai/plans/2026-10-24-mv-p8g3-stack-production.md)（approved、§10 優先）。

Codex 担当の実装と手元の関門は完了。T4 の関門、本計算、書き出し、Colab/CLI の実操作、回収済み本計算への verify は未実行で、Claude に引き継ぐ。
合算スクリプトは作成していない。既存の核・記録・計画・規約は編集せず、git add / commit はしていない。

## 実装

- `gpu/p8g3_core.py`: 呼び出し ID `j=0..399`、`n=200000`、`seed=820301`、`start=j*200000`。保存・再開・最終照合で同じ条件検査を使う。
- `gpu/p8g3_colab.py`: 封印済み P8g-2 書き出しをそのまま検証・読み込み。40×256、容量 18。VM の最初に P8g-1/P8g-2 の 20,000 履歴（SEED2）の全履歴出力をバイト比較する。核は変更しない。atomicAdd の数え上げ先は、既存 P8g-2 GPU ラッパーが `cp.zeros` で確保するグローバルメモリをそのまま使う。
- `session_<uuid>.json`: 合格した初回照合・指紋・GPU 名・起動設定・再開記録を不変ファイルとして保存。各受領が ID とセッション記録の SHA256 を参照する。本計算のセッションには、小規模 T4 再開関門の記録も必要。
- 各呼び出しの受領に実際に渡した引数、出力回数が全履歴で 1 という検査結果、核の秒数を保存。GPU の `_groups_d`・`_groups_i`・`_absolute`・`additions` も保存する。
- 再開前に既存の全組を検証し、最小の保存済み ID を再計算して凍結済み P8f の `check_resume` で比較する。CPU 関門では数値再開比較を行わず、`cpu-schedule-only` と明示する。
- 公開対象は npz/受領が両方ある組だけ。pending、npz だけ、受領だけは未完了として再計算する。既存の検証済み組の二重保存や、取得済み組との競合は拒否する。
- `complete.json` は期待する ID 全集合、各 npz・受領・セッションの SHA256、種・総履歴数・指紋・GPU 名・起動設定を記録する。verify は全組を読み直し、その完了記録も照合する。
- `gpu/p8g3_cli/`: T4 起動、50 MiB 分割転送、再送前の受領・セッション検査、10 分ごとの確認と増分回収。セッションを先に回収し、結果は一時領域でハッシュ・条件・健全性を検証してから公開する。ログの成功/失敗印に加えて、控えた PID の消失・ゾンビ化も失敗として扱う。Colab の各呼び出しには上限時間がある。取得失敗・確認の最終失敗・割り込み時にも、再回収を試みて `colab stop` を呼ぶ。

## 実行結果

| コマンド | 結果 | 記録 |
|---|---|---|
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k p8g3` | 25 passed、1474 deselected | [pytest.log](pytest.log) |
| `.venv/bin/python -m prototypes.mv_electron.gpu.p8g3 local` | 終了 0、内部の上記 25 試験も合格 | [local.log](local.log)、[gate0.json](gate0.json) |
| `…p8g2 local`（一時コピー） | 16 passed、終了 0 | [p8g2_local.log](p8g2_local.log) |
| `…p8g local`（一時コピー） | 8 passed、終了 0 | [p8g_local.log](p8g_local.log) |
| `…p8f local`（一時コピー） | 32 passed、終了 0 | [p8f_local.log](p8f_local.log) |

CPU 実輸送の小規模駆動は、4 外側バッチ×2 呼び出しを平坦化した 8 呼び出し×2,000 履歴、種 820301。
連続実行と、3 組保存後に止めて残り 5 組を再開した実行を比較した。両方の完了記録を検査し、時間以外の全数値配列がバイト一致した。CPU 再開記録は数値再開照合の代用ではない。

400 呼び出しのスケジュール試験では、実 GPU ラッパーの呼び出し先を記録用オブジェクトに置き換え、実引数の全 400 組と保存側に渡る受領条件を独立に突合した。区間は `[0, 80000000)` を重複・欠落なく覆う。GPU 実輸送の代用ではない。

回帰は [run_regressions.py](run_regressions.py) でソースを一時ディレクトリにコピーして実行し、旧段階の出力先もコピー内にした。その他の入力は既存の読み取り専用データを参照した。元の GPU ソース、旧テスト、p8g1/p8g2/p8f/p8f_prof 記録の **253 ファイルの SHA256 は不変**。前後の記録は [protected_before.json](protected_before.json)、[verification.json](verification.json)、コマンド・終了値は [regression.json](regression.json)。

新規駆動/CLI の 9 Python ファイルは AST 構文検査も通過した。CLI・書き出し・本計算結果への verify は実行していない。

## 故障の注入と拒否

[tests/test_p8g3_contracts.py](../../../../prototypes/mv_electron/tests/test_p8g3_contracts.py) の試験で、以下を注入した。
具体的な拒否理由 **30 件**は [fault_observations.json](fault_observations.json)。これは CPU/NumPy の試験記録であり、T4 の合格記録ではない。

| 故障 | 拒否・復旧の確認 |
|---|---|
| p1: 保存済み npz の 1 バイト変更 | 再開前の SHA256 照合で拒否 |
| p2: 同じ ID の二重保存／ID の欠落 | 二重保存を拒否、完了集合の欠落を拒否 |
| p3: `violations` を壊す | `run_health` で拒否し、npz を保存しない |
| p4: 実呼び出しは全て start=0、受領は予定の start | 独立した実引数の記録との比較が拒否。保存アダプターに異なる実引数を渡す試験も拒否 |
| visits=0／visits=2 | 出力回数検査で拒否し、保存しない |
| 再開の整数不一致／浮動小数点の許容幅超過 | P8f の既存の規則で拒否。1 ULP の許容内の例は通過 |
| セッションが欠落／不合格／封印後変更 | 受領を読む段階で拒否。再送前の検査でも欠落を拒否 |
| 受領の種・実引数・出力回数・ID の改変 | 条件検査で拒否 |
| セッション上書き／完了封印変更／想定外 ID | 不変記録・完了照合・ID 集合検査で拒否 |
| npz のみ／受領のみ／pending のみ | 完了済みと数えず、再計算して完了まで復旧 |
| ダウンロード途中で停止 | 公開 npz を作らず、セッションは先に保全。再取得で復旧 |
| 取得済み組と異なるリモートの封印 | 上書きせず拒否 |
| 取得失敗／確認を 3 回失敗 | finally で再回収を試み、stop を呼ぶことを記録用 CLI で確認 |
| VM 初回の履歴不一致／必須の小規模再開関門なし | 初回照合／本計算用セッションの検査で拒否 |

## Claude への引き継ぎ（以下は未実行）

先に §10.2 の抄録数値再現を Claude の合算スクリプトで行う。
T4 の小規模な保存→回収→停止→別 VM への再送→再開照合は必須。本計算を先に開始しない。
各 watch は完了時に回収して VM を停止する。途中の VM 消失時は、手元の検証済み組だけを同じ `--output` から新 VM に再送する。

```bash
# 封印済み P8g-2 export を検証し、別の P8g-3 export にコピーする
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3 export

# 小規模関門: 最初の VM で 4 個中 3 個を保存して終了
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.start --session p8g3-trial-a --mode trial-save
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.watch --session p8g3-trial-a

# watch の回収・stop 後、別 VM に再送して数値再開照合、残り 1 個を計算
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.start --session p8g3-trial-b --mode trial-resume
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.watch --session p8g3-trial-b

# 小規模関門の合格を検証してから本計算へ
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.start --session p8g3-production --mode production
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3_cli.watch --session p8g3-production
.venv/bin/python -m prototypes.mv_electron.gpu.p8g3 verify
```

既定の手元の保存先は `~/.cache/chatcarlo/p8g3/colab/`、中に `trial/` と `results/`。
本計算の最終 verify は results の 400 組を検査する。小規模関門の qualification とその封印済み入力は trial に保持する。
VM 上の小規模関門コードには、実 T4 出力を使う整数・浮動小数点・出力回数・健全性・破損・重複・欠落・不合格/欠落セッションの注入と拒否、保存途中切れの復旧も含む。これらの実 T4 観測は Claude の実行後に記録する。

本計算の合算は Claude が §5・§10.2 に従って行う。この段階のコードに合算・χ² 判定・抄録更新は含めない。
