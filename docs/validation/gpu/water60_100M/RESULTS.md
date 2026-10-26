# 水60 keV: 5,000万光子の追加と1億光子への統合

2026-10-26。使用者の同条件追加の依頼に基づく。

## 結果

元50,000,000光子＋追加50,000,000光子＝100,000,000光子。
1バッチ500,000光子、各100バッチ、統合200バッチ。
60 keV単色、10×10 cm平行ビーム、水30×30×20 cm、1 cm格子。
物理入力・係数表・実装束は元と同一。transport/source seedのみ変更。

- 照射野内の線量R中央値: 0.2323488% → 0.1630567%。
- ボクセルごとのR比の中央値: 0.7072091。
- 照射野内2,000ボクセルの最少寄与バッチ数: 200/200。
- 表面層の中心近傍ボクセル（中心x=y=z=0.5 cm、各辺1 cm）の体積平均線量:
  4.7634317023×10⁻¹⁵ Gy/history、R=0.0800753%、寄与200/200。
- 総トラックカーマ: 37.7082182572 keV/history。

線量・H*(10)は1入射光子あたり（Gy/history、pSv/history）。mAs/CTDIによる絶対値校正はない。
図の誤差棒は±1 SEM。100Mには元50Mが含まれるため両推定値は相関する。
研究・教育用の計算結果である。

## 実行・統合

追加実行コマンド（当日の出力ログ: `run.log`。承認済みrunへの再投入はColabを起動するため、既存結果の確認には下のローカル再集計を使う）:

```sh
.venv/bin/python -m chatcarlo run runs/diag_water_60kev_gpu/scene.yaml \
  --engine gpu --gpu-device colab -n 5e7 --seed 20261026 \
  --batch-size 500000 --dose-grid --resolution 1 \
  --run-dir runs/diag_water_60kev_gpu/add50M_20261026/run \
  --dose-out runs/diag_water_60kev_gpu/add50M_20261026/dose.npz
```

追加のGPU native実行時間は計41.8863秒（転送・圧縮・検証時間は含まない）。
Colabセッションの終了を確認済み。`prefetch.py`は標準ドライバーと同じreceipt
照合関数で完了バッチを先行回収し、標準ドライバーが全native入力・線源再現を検証した。

`merge.py`は両runのjob、物理束、実行シーン、グリッド、配列、寄与バッチ数を確認し、
等サイズバッチのpooled分散（群間平均差項を含む）から200バッチのSEMとRを再計算する。
200バッチの直接標本統計を別に生成して統合式を検証した。
入力・出力のSHA256と種は`merge_provenance.json`に記録。

```sh
.venv/bin/python docs/validation/gpu/water60_100M/merge.py
.venv/bin/python docs/validation/gpu/water60_100M/analyze.py
```

旧個別バッチはローカルに残っていない。元NPZの由来は旧job・承認・完了ログで確認し、
統合には保存された平均・SEM・寄与バッチ数・raw sumを使用した。
元のヒアリング要件記録はなく、追加依頼は`requirements.md`に記録した。
`scene_100M.yaml`は合計光子数を反映した物理仕様であり、実際の二つの種を使った
実行条件は`merge_provenance.json`および両runのjobが正典である。

## 成果物・保全

| ファイル | 内容 | 保全 |
|---|---|---|
| `runs/diag_water_60kev_gpu/dose.npz` | 元50M、変更なし | 手元に保持 |
| `dose.npz` | 追加50M | 手元に保持 |
| `dose_100M.npz` | 統合100M、200バッチ | 手元に保持 |
| `run/results/` | 追加100バッチとreceipt、線源再現、約3 GB | 手元に保持 |
| `merge_provenance.json` | 統合の由来とSHA256 | 手元に保持 |
| `summary.json` / `comparison_50M_100M.png` | 集計表・比較図 | 手元に保持 |
| `audit_A.md` / `audit_BC.md` | 独立監査報告 | 手元に保持 |

生データの表中の相対パスは、元50Mを除いて`runs/diag_water_60kev_gpu/add50M_20261026/`を基準とする。
この約3 GBの生データと各132 MBの線量NPZは手元に保持し、削除・Driveへの移動は実施していない。
Gitに保存するのはこの結果文書、小さな記録、CSV、比較図、再集計スクリプトだけである。

## 後日の利用

- `field_statistics.csv`: 水中照射野内2,000ボクセルの元/統合Rと寄与バッチ数。R中央値・比をNPZなしで再計算できる。
- `depth_dose.csv`: 中心近傍x=y=0.5 cm列の20深さの線量・SEM・R（Gy/history）。比較図の数値表。
- `summary.json`: 報告した集計値。`merge_provenance.json`は元/追加/統合NPZのSHA256と実行jobを保持。
- `original_*` / `additional_*`: 両実行のjob・approval・generationの当日スナップショット。
- `audit_A.md` / `audit_BC.md`: 原文の監査報告。報告中の相対パスは当日のrunフォルダを指す。
- スクリプトはこの結果専用。`merge.py`は既存NPZ2個から再集計し、`analyze.py`は元50Mと統合100MのNPZから集計・図を再生成する。GPUを再実行しない。
- 別の場所へデータを置く場合は`CHATCARLO_WATER60_RUN_DIR`で追加runフォルダを指定できる。元50Mはその親の`dose.npz`と`run/`を読む。
- 検証時は`CHATCARLO_WATER60_OUTPUT_DIR=/private/tmp/<専用フォルダ>`で出力先を変え、既存成果物を上書きせず比較する。`analyze.py`は入力runフォルダ内の統合NPZを読む。

今回の監査は条件付き合格。旧個別バッチの再検証不能、非衝突透過の直接観測なし、escaped energy未保存の制約を維持する。
今回新たなユーザーの常設の好み・方針変更はなかった。`furikaeri`はログ探索の途中で中断され、完了・対処・学びの記録はしていない。
