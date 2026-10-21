# G4 CLI — 手元での実装・受入確認

日付: 2026-10-21。対象は承認済み計画
`docs/ai/plans/2026-10-21-chatcarlo-gpu-g4-cli.md` の手元で実行可能な範囲。
ユーザーの指定により Colab は実行していない。git commit も行っていない。
G1 の関門 1・2 の記録はともに `pass` を確認した。

## 実装

- scene の `detector:` を検証し、`numpy`／`gpu` の CLI に渡す。
  宣言のない既存 example の検証結果・警告と preview HTML は変更前と一致。
- `run --engine gpu --gpu-device cpu|colab`、Colab 用の条件固定 preview、
  `approve RUN_DIR`、送信前の承認照合を接続した。
  Colab の実行には、承認済みの束とジョブをそのまま使う。
- 生成記録は条件・目録・ジョブ・スナップショット・HTML をハッシュで結び、
  承認後の上書きと改変を拒む。
- 核に実際に渡す引数列を観測し、引数の位置別にスカラー値と入力配列の SHA256 を照合する。
  CPU でも束から線源を再生成して照合する。静的配列のハッシュは Core 作成時にキャッシュ。
  `seed`／`start` の故障注入と、CUDA の引数位置の照合は手元でも検査した。
- バッチ集計を共通化し、材料別の付与・SEM・R・転帰・検出器像・ROI 共分散を扱う。
  新規／再利用のバッチ数と計算時間を分けて記録する。
- 検出器の NPZ・JSON・PNG を保存する。Colab は実行フォルダに保存し、
  `--detector-out` が指定された場合はそこにも写す。
  PNG は u が横、v が縦、画素 (0,0) が左下。
- 依頼 D との回帰判定は計画の事前固定式を実装した。
  核・入力・実効ジョブの不一致は `inapplicable`、runtime／settings の差は別項目で記録する。

C++／CUDA の核、束の形式、`source_host.py`、`transport.py`、物理・幾何・タリー本体、
`prototypes/`、`docs/jsmp133/`、G1 の関門記録は変更していない。

## 手元の検査

`tests/test_gpu_g4.py` に、宣言の異常系、直接 API と CLI のバイト一致、
改変の拒否、故障注入、承認を通す CPU の統計無効ケース、検出器なし・ROI・1 バッチ・
端数・検出ゼロの集計、独立の足し算と統計式、PNG の向き・軸・単位、
移動・傾斜した検出器の描画範囲、MV の引数転送、回帰判定の検査を追加した。
CUDA の引数位置の手元検査はデバイス配列の代役を使い、CUDA 実行そのものではない。

`local_baseline.json` は変更前の HEAD
`ff64414c9710c36389f8270ef5d199ecf33079ea` のコードで先に保存した固定値。
G1 S1・S2（各 1,000 履歴、seed 42、batch_size 400、30×30 cm／60×60 画素の検出器）と、
kernel の `water_phantom_pdd_ocr.yaml`（10,000 履歴、seed 42、既定チャンク設定）について、
配列のバイト列の SHA256、スカラー、材料別付与、統計、転帰を比較して一致した。
NumPy は G1 の既存の変更前固定値検査が通過している。

### 計画 §8 のコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -q -k "gpu or mv_dispatch or scene or detector"
.venv/bin/python -m pytest prototypes/mv_electron/tests -q -k entry
.venv/bin/python -m chatcarlo validate docs/validation/gpu/g4/scene_portrait_detector.yaml
.venv/bin/python -m chatcarlo run docs/validation/gpu/g4/scene_portrait_detector.yaml --engine gpu -n 1e4 --seed 2040 --detector-out /tmp/g4_cpu.npz
```

| コマンド | 最終コードでの結果 |
|---|---|
| 全体 pytest | 557 passed、2 warnings（147.38 秒） |
| 絞り込み pytest | 201 passed、356 deselected（73.81 秒） |
| MV entry pytest | 71 passed、1373 deselected（33.33 秒） |
| validate | OK、463 物体 |
| CPU twin CLI run | 成功、10,000 履歴、検出 4,425、3 ファイル保存 |

警告 2 件は既存の SpekPy のゼロ放出検査と、SpekPy 不在を模擬するフォールバック検査に伴うもの。
CLI の matplotlib は、書き込み可能な一時フォルダにフォントキャッシュを作成して正常に PNG を保存した。
`git diff --check` も通過。

## 指定 CLI の計算結果

- シーン: [scene_portrait_detector.yaml](scene_portrait_detector.yaml)。依頼 D の肖像シーンをそのまま写し、
  依頼 D と同じ DET（中心 `[0,0,0]`、法線 `[0,0,1]`、u 軸 `[1,0,0]`、15×18 cm、100×120 画素）を足した。
- engine/device: gpu/cpu、seed: 2040、source_seed: 3712336832425968660、batch_size: 200000。
- 条件 SHA256: `dcee25f75719954126ec464784a2f1573409240e3c18973ea1f0f4e87f758545`。
- 検出: 4,425。吸収割合 0.4989、脱出割合 0.0586、平均相互作用回数 0.7608。
- 材料別付与（合計 MeV）: aluminum `169.26472277178752`、air `4.724912941809377`。
- 新規 1 バッチ、再利用 0 バッチ、Core 呼び出し時間 `0.09224958391860127` 秒。
- 既定 batch_size では 1 バッチなので R は全画素で未定義。CLI は既存のバッチ不足の案内を表示した。
  統計の数値検査は別の複数バッチのテストで行った。
- 保存物: `/tmp/g4_cpu.npz`、`/tmp/g4_cpu.json`、`/tmp/g4_cpu.png`。
  NPZ の全画像配列と材料別付与は直接 `run_transport(engine="gpu", gpu_device="cpu", ...)` とバイト一致。
  PNG は実画像も確認した。

## 未実行（ユーザー指定）

Colab の新規 GPU バッチによる通し、保存済み依頼 D の 10⁶ 履歴 A との実データ回帰比較、
VM 上でのジョブ改変 (f) と故障注入 (g) は実行していない。
これらは G4 全体の受入済みとは扱わない。回帰の判定式・前提照合・実行と回収の配線は実装し、
手元で検査可能な部分だけを確認した。
