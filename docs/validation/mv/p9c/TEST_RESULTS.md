# P9c CPU検査記録（2026-10-24）

| 検査 | 結果 |
|---|---|
| `.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p9c.py -q`（最終版） | **23 passed** |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | 1626 passed, 3 skipped, 1 failed（1340.04秒） |
| 上の全体実行に含まれる既存MVテスト（P9c以外） | **1611 passed, 3 skipped** |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q --lf`（最終版で失敗箇所を再実行） | **1 passed**, 109 deselected |
| `.venv/bin/python -m pytest tests/ -q` | **718 passed**, 2 warnings（244.75秒） |
| `python -m prototypes.mv_electron.gpu.p9c local --work runs/p9c_cpu_check` | **V0・V1a・V2a・V3（決定論的）・V5・V6・V6b 合格**。詳細は [CPU_GATES.json](CPU_GATES.json) |
| Python構文検査、`git diff --check` | 合格 |

全体実行はP9c専用テスト16件を収集した時点のもの。実行中に断面出力を水セル限定に
揃えたため、収集済みの旧テスト `test_sections_water_and_centers` は、除外セルのNaNと
旧期待値0.5の不一致で失敗した。最終版では有効セルだけ比0.5、除外セルはNaNとして
検査し、専用23件と失敗箇所の再実行で合格した。既存MVテストに失敗はない。

本体の警告2件は既存のSpekPyのゼロ放射警告と、SpekPy不在を模した近似スペクトルの警告。

専用テストは、指定のCPU関門に加え、40本のランダム線分、小規模な2照射野の輸送、
7門・両側が完了する前の中断、再開とログマーカー、関門不合格と矩形の入射失敗での停止、
半開区間のセル選択と小数表記の格子面、粗格子の端面接触、CUDAエクスポートのホスト構文検査を含む。
CUDAの構文検査はCPUのスタブによるもので、NVRTCコンパイルやGPU実行の合格を意味しない。

Colab、NVMLの実測、GPU関門、T0、V1c・V2b、本計算は未実行。
`~/.cache`への書き込みを伴う確認は行わず、[README](README.md)に実行担当者の手順を残した。
P9b以前の変更禁止のコード・封印結果・計画§8は変更していない。
`p9c_cli.py`は作成していない。git commitは行っていない。
