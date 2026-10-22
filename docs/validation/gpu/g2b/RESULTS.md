# G2b 手元受入記録

2026-10-23。正典: [承認済み計画](../../../ai/plans/2026-10-22-chatcarlo-gpu-g2b-cli-dose-grid.md)。
計画の関門 A・B を CPU の双子で実施。Colab は実行していない。

## 関門 A: 合格

S2 (`docs/validation/gpu/g2/scene_lead_slab.yaml`)、解像度 2 cm、
200,000 履歴、バッチ 50,000、seed 2401、source_seed は既定の導出。

CLI の保存結果と、同条件の `run_transport(engine='gpu', dose_grid=True)` から
既存の保存規約で作った期待値について、13 キーすべての dtype・形・バイト列が一致。
時間・条件識別の行を除き、CLI の表示も API の結果を同じ表示経路に渡した場合と一致した。
最大値、R、寄与バッチ数、警告を含む。

- 原点: `[-70.0, -70.0, -50.05]` cm
- 形: `(70, 70, 61)`、298,900 ボクセル
- バッチ数: 4
- 総カーマ: 6428.95 MeV（32.1447 keV/history、CLI 表示桁）
- 最大吸収線量: 6.522112890301559e-14 Gy/history、R=0.019（表示桁）、寄与 4/4
- 最大 H*(10): 0.01973655260030853 pSv/history、R=0.003（表示桁）、寄与 4/4

最大値について既存の空気中・線源近傍の警告が出る。この最大値を実在する評価位置の
被ばく値として扱わない。ここでの合格は CLI の配線と API との一致の確認であり、
線量の物理精度の新たな保証ではない。

生成物: `/tmp/g2b_a.npz`。
[CLI 全出力](gate_a_cli.txt)、[配列別 dtype・形・SHA256](gate_a_arrays.json)。

## 関門 B: 合格

`tests/test_gpu_g2b.py` で以下を検査した。

- 承認後の原点・解像度・形・総数が同じ別形状・layout_version・grid 削除・
  g2_tables.npz 改変を拒否。
- 最終引数の cfg 差し替え、cfg の同一アドレスでの書き換え、g2_table の
  1 ulp 差し替えを observation mismatch で拒否。カーマと H*(10) の
  出力ポインタ交換を output mapping mismatch で拒否。
- 正常な観測後に同じ cfg・g2_table バッファを書き換えても、次の観測で検出。
- CPU の 19・20、CUDA の 25・26 の観測位置と、6 出力の対応。
  CUDA の引数組み立ては CuPy なしで検査。
- 正しい identity・SHA256 を付け直した結果でも、kerma/h10 の欠落、
  float32、長さ・次元違い、NaN、Inf、負値を集計前に拒否。
- 照合済み実バッチから順序つきの和、累積値の差分による二乗和、
  寄与バッチ数、履歴数・バッチ数、R、SEM、Gy/history、pSv/history を独立再構成し、
  保存規約による全配列とバイト一致。
  250,000 履歴 / バッチ 100,000（端数 50,000）、100,000 履歴の1バッチ、
  統計無効を実施。寄与ゼロも検査。
  二乗和をバッチ値から直接求めた値との比較は相対 1e-12。
  0.1・0.2・0.3 の人工バッチでも丸めを検査。
- 格子付き worker の再利用・集計と API が一致。
- 物体からはみ出し、既存 tally_bbox 内に収まる検出器を置き、
  検出器と格子の同時保存を API とバイト比較。job.grid が検出器の四隅を含む。
  格子の範囲は計画どおり Geometry.tally_bbox を使用。
- preview の格子の文字表示と条件の一致、格子なし job のバイト一致、
  格子なし条件・HTML の既存規約を維持。
- モックで colab 呼び出しの timeout=300、状態取得の3回までの再試行、
  起動済み worker の TimeoutExpired 後の二重起動防止、
  ゾンビ・pid なしの G1_FAILED、失敗前の結果回収、stop、
  確認ごとの heartbeat 更新を検査。実際の Colab は使っていない。

## 第8節のコマンド

```bash
.venv/bin/python -m pytest tests/test_gpu_g2b.py -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m chatcarlo run docs/validation/gpu/g2/scene_lead_slab.yaml --engine gpu --gpu-device cpu --seed 2401 -n 2e5 --batch-size 50000 --dose-grid --resolution 2 --dose-out /tmp/g2b_a.npz
```

- G2b: 55 passed。[全出力](local_tests.txt)。
- 全 tests/: 717 passed、2 warnings、228.87 秒、終了コード 0。[全出力](all_tests.txt)。
  警告は既存の anode cutoff / SpekPy 不在時の fallback テストからのもの。
- CLI: 終了コード 0。[全出力](gate_a_cli.txt)。
- 補助の G4 / Colab 転送回帰: 80 passed。[全出力](g4_regression_tests.txt)。

## 差分の範囲

今回の実装は計画第3節の許可ファイル、新規テスト、新規の本ディレクトリに限定。
既存テストの変更は指定された G4 の dose-grid 拒否ケースの除去だけ。
transport.py、tally.py、materials.py、dose_coefficients.py、kernel.py、
G1/G2a の C++・CUDA 核には差分がない。ファイルの削除、commit は行っていない。

作業開始時から docs/jsmp133/ と未追跡 data/ に別作業の差分があった。
作業中に以下の別作業の差分も現れた。いずれも今回の操作で編集しておらず、保持した。
したがって、作業ツリー全体では禁止対象 prototypes/ の差分が存在する。

- prototypes/mv_electron/entry/preview.py
- prototypes/mv_electron/tests/test_entry_scene.py
