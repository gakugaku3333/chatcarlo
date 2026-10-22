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

## Colab（T4）の関門 C・C2・D（Claude、2026-10-23）

承認: ユーザーが C・C2・D の承認用プレビューを承認し、Claude が `approve` を実行（1 回目・2 回目とも）。
判定スクリプト: [judge_colab.py](judge_colab.py)（C・C2）、[run_fault.py](run_fault.py)（D）。実行一式は `data/gpu_g2b/`（公開しない）。

### 関門 C 1 回目: 停止（記録に使わない）

条件ハッシュの旧い一式（`data/gpu_g2b/run1_gate_c`）。VM の worker が 0 バッチで終了し、`launch` が `G1_FAILED`（ゾンビ検出）で止まって `colab stop` した。
VM のログは回収していなかった。原因の有力な説明: CUDA の出力の照合 `check_outputs` が、渡さない出力（`np.uint64(0)`）で `.data.ptr` を読んで例外になる
（NumPy のスカラーも `.data` を持つ）。手元のテストは偽の CUDA バッファしか使っておらず、この形を通っていなかった。
直す前のコードで同じ例外を手元で再現し、修正とテスト（`test_t6_cuda_absent_outputs`）、worker 失敗時に VM のログ末尾を `worker_tail.log` に回収する変更を 59835e2 で入れた。
コードが変わり条件ハッシュも変わったので、C・C2・D は新しい一式を作り直して承認を取り直した。

### 関門 C: 合格

S2・2 cm・4×10⁶ 履歴・バッチ 2×10⁵・seed 2410・source_seed 2411。条件 SHA256 `ac583d75…`。回収済みなしから `new_batches=20`・`reused_batches=0`、VM の計算 2.77 s。

- (i) 20 バッチすべての受領票の観測（G1 と G2 の引数）が照合を通過。
- (ii) 保存した `.npz` のカーマ・H*(10) が照合済みバッチの和（累積の差分による独立の作り直し）とバイト一致。寄与バッチ数・バッチ数・原点・形・解像度も一致。R は相対 1e-12 で一致（保存側は線量から作るので最後の桁は比べない）。
- (iii) G2a の参照（sha256 `9e6d3ab5…`、シーン・格子・履歴数を判定前に照合）と G2a 関門 1 の判定ですべて合格:

| 量 | 適格ボクセル | 最大 \|z\| | 許容 | \|z\|>3 の割合 | z² の平均 | 領域の総和の z（水・空気・骨・鉛層・全体） |
|---|---|---|---|---|---|---|
| カーマ | 69,247 | 5.39 | 8.08 | 0.59% | 1.12 | 0.17・0.32・0.97・−2.50・−0.70 |
| H*(10) | 81,052 | 5.70 | 8.17 | 0.57% | 1.10 | −0.12・0.40・1.07・−1.39・0.29 |

この標本の精度で不一致を検出しなかった、という意味であり、数 % 以内の一致の保証ではない。記録: [gate_c.json](gate_c.json)。

### 関門 C2: 不合格（バイト一致の条件）。再開の配線は通過

計画からの変更: 途中で止める代わりに、条件が同じ（`job.json` がバイト一致）別の一式に、関門 C の最初の 8 バッチ（結果と受領票）を置いてから `run` した（10 分ごとの確認では途中で止める時点を作れないため）。

- `reused_batches=8`・`new_batches=12`、再利用分を含む 20 バッチの観測が照合を通過、(ii)・(iii) も合格（[gate_c2.json](gate_c2.json)）。
- **不合格**: 新しく計算した 12 バッチの `deposits`・`kerma`・`h10` が関門 C と相対 最大 2.8e-15 違い、保存した `.npz` が C とバイト一致しなかった（kerma で 10,448 ボクセル、相対 最大 5.2e-16）。
  同じバッチの `endings`・`scatter_sum`・`n_fluorescence`・`box_sum`・`box_sum2` はバイト一致（同じ履歴を同じ経路で追っている）。
- 有力な説明（未確認）: 核の atomicAdd による浮動小数点の足し込みの順が実行ごとに変わる。計画は「同じ種なら GPU でもビット再現」を前提にしていたが、その前提が誤り。
- 判定基準は結果を見て変えていない。基準を変えるなら追補を書いて承認を取り、新しい一式で回す（計画 4 節）。
- 完了した一式を再実行して `reused_batches=20` になる追加の検査はまだしていない。

### 関門 D: 合格

S2・2 cm・10⁵ 履歴・1 バッチ・seed 2420・source_seed 2421。`fault.json` で VM の最終の引数に注入:

| 故障 | 結果 |
|---|---|
| `g2_cfg`（`cfg` の解像度を 2 倍にした配列に差し替え） | `observation mismatch: batch_000000.npz 26:cfg` で停止、`.npz` なし |
| `g2_table`（`g2_table` の 1 要素を 1 ulp） | `observation mismatch: batch_000000.npz 25:g2_table` で停止、`.npz` なし |

記録: [gate_d_g2_cfg.json](gate_d_g2_cfg.json)、[gate_d_g2_table.json](gate_d_g2_table.json)。終了後 `colab ls` でセッションが残っていないことを確認。
