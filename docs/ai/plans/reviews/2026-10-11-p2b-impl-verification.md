# P2b（B: 事後解析）の実装と検証の記録（2026-10-11）

- **実装**: Codex（`codex exec -s workspace-write`）。`prototypes/mv_electron/posthoc_v2_p2.py`と`docs/validation/mv/p2/posthoc_v2/`の成果物を書き、1回目の実行で合成データの検査36項目、骨子レビューの表との照合（最大差0.000454ポイント）、既存の試算との照合（最大差0）を通した。そのあと利用上限に達して止まり、最終報告は残っていない（ログの末尾に`usage limit`）。
- **Claudeが実行したテストコマンド**: 解析の実行2回で`posthoc_v2.json`がバイト単位で一致。`--check`は合成36項目・照合2件・入力ハッシュ不変・保存済み成果物の照合がすべて通過。`run_checks_p2.py --verify-untouched`は保護968・EGS5入力952・本計算凍結39ファイルが一致し、`q0_regression.json`、`regression/code_diff.patch`、`RESULTS.md`、`q6.json`、`pilot_lock.json`のsha256は実行の前後で同じ。`git status --short -- chatcarlo tests scripts examples`は空。`pytest prototypes/mv_electron/tests`は同日朝に101件合格・1件スキップ（新しいファイルはテスト対象の既存コードを変えていない）。
- **Claudeの独立した再計算**: 解析スクリプトを使わず、`result.npz`と`pilot_lock.json`から28比較・130ブロックの`Δ_k`、`SEM_k`、状態を別のコードで計算し、JSONの値と最大3.7×10⁻¹⁴（% `Dref`）で一致。ブロックの境界もA-2の表と一致。
- **確かめていないこと**: A-8の形の差とA-9のブートストラップの値は、独立に再計算していない（合成データの検査と決定性の確認だけ）。図は目視で確認した。
