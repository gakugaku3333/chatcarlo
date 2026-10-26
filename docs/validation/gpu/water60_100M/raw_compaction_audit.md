# 監査報告書 — 水60 keV追加50Mバッチデータの整理 / ステージB・C
監査人: 氷室 律（独立監査）

## 判定: 条件付き合格

追加50Mの水中全18,000ボクセルのバッチ別値とスカラー集計は、元100ファイルから正確に抽出されている。保持する三つの全格子NPZと記録を残し、下記の情報喪失を了承する条件で、`raw_compaction.json`の`source_files`に列挙された100個の元バッチNPZを削除する整理は妥当である。監査では削除を実施していない。

## 所見

| # | 重大度 | 所見 | 根拠（ファイル・コマンド・数値） | 推奨対応 |
|---|---|---|---|---|
| 1 | 情報 | 水全域のバッチ別値は損失なく抽出されている。 | `additional_water_batches.npz`のkerma/h10は各float64、shape=(100,18000)。元バッチ全100個から`raw[k][water_flat_indices]`を別スクリプトで読み、保存値と配列完全一致。 | 表を保持し、追加50Mの水内再集計に利用できる。 |
| 2 | 情報 | 水の全18,000 voxelが重複・欠落なく含まれる。 | 保存flat indicesを`np.unravel_index`で復元。中心座標x,y=-14.5〜14.5 cm、z=0.5〜19.5 cmの全直積と集合完全一致。C順flat番号と元shape=(130,130,120)も保存。 | 表の座標・形状・格子幅とindicesを一緒に保持。 |
| 3 | 情報 | スカラー量・全格子合計・履歴範囲も正確に保存されている。 | 全100個でdeposits/endings/scatter_sum/n_fluorescenceを直接照合し完全一致。grid_kerma_total/grid_h10_totalは元全格子sumと一致。starts=0〜49,500,000、histories各500,000は各receipt identityと一致。元box_sum/box_sum2は全100件空配列。 | 追加50Mの材料別付与量、終端数、全格子合計のバッチ統計も再計算できる。 |
| 4 | 情報 | 元ファイルの由来と保持NPZはハッシュで照合できる。 | 元NPZ全100個のSHA256・bytes、元receipt全100個のSHA256、集約receiptのJSON内容を元実物と独立照合し全一致。compactと集約receiptのSHA256、保持NPZ3件のSHA256もraw_compaction.jsonと一致。 | manifest、job、承認、生成記録、既存監査、source_referenceと記録を保持。元NPZの完全再生成が可能という意味ではない。 |
| 5 | 情報 | 既存結果の再利用に元全格子バッチNPZは必要ない。 | `docs/jsmp133/60kev_100M/build_figures.py`はplot定義を読込み、入力を保持対象dose_100M.npzへ置換して使用。`docs/jsmp133/ocr60_out_of_field_100M/analyze.py`もdose_100M.npzと既存EGS5比較CSVを使用。rawバッチを読まない。 | この三つの全格子NPZと既存比較CSV・解析コード・merge_provenanceを削除対象に含めない。 |
| 6 | 要注意 | 空気の個別voxelのバッチ別値は失われる。 | compactは水18,000 voxelのみ。全格子2,028,000 voxelのうち残り2,010,000 voxelの個別バッチ値は保存しない。全格子平均・SEM・R・hitは3 NPZに残る。 | 空気内の任意領域統計・voxel間共分散、水と空気の共分散を後から厳密に求められるとは説明しない。全空気合計のバッチ値は全格子合計−水合計で丸め差を伴い再構成できる。 |
| 7 | 要注意 | 水内でも100M全体の共分散は復元できない。 | compactに残すのは追加50Mの100バッチ。旧50Mの個別バッチは整理前から存在しない。 | 水内の任意領域の平均・規格化について、追加50Mの共分散は再計算可能。統合100Mの厳密な共分散・規格化SEMを復元できるという主張はしない。既存OCR解析のunknown covariance/SEM boundsの限定を維持。 |
| 8 | 情報 | 保存容量は大きく減る。 | 元100 NPZ合計2,978,634,350 bytes、compact表27,229,252 bytes。表単体で99.08585%減。receipt・記録などの小ファイルは別途保持。 | 実際の空き容量は削除後のファイルシステムで確認する。 |

## 実施した独立検証

`compact_batches.py`を読取り、対象ファイルを書き換えない独立Pythonスクリプトを`.venv/bin/python`で実行した。`np.load`で元100バッチとcompact表を読み、以下を検証した。

- 元ファイル名・順番・bytes・SHA256、元receipt SHA256、集約receipt全件の内容一致。
- 保存水格子の全座標集合、indicesの一意性、水領域18,000 voxelの完全被覆。
- kerma/h10、材料別付与、終端数、散乱和、蛍光数、全格子合計の完全一致。保持値と元値の比は非ゼロ値で1、ゼロ値も完全一致。
- 履歴範囲・バッチ光子数のreceiptとの一致。旧・追加・統合dose NPZのハッシュ一致。
- 実装の検査経路では全格子raw合計とhitを追加dose.npzへ照合し、水の平均・SEMも照合していることを確認。今回の独立検証は元からの抽出値そのものを全件照合したため、その保存精度を直接確認できた。
- 既存二つの図・OCR解析の入力参照を読み、保持全格子NPZ・既存EGS5比較データに依存し、削除予定raw NPZに依存しないことを確認。

結果: `ALL 100 raw hashes, receipt hashes/contents, exact water/scalar extraction, unique indices/coordinates, NPZ preservation hashes passed`。

## 未確認事項

- 削除は未実施。削除範囲の実行結果、保持ファイルの削除後再検証、実際の空き容量は未確認。
- 空気個別voxelのバッチデータから可能な将来の未指定解析は保証しない。既存指定解析の再利用には影響なし。
- 全100Mの共分散および旧50M個別バッチの再検証は、元データが従前からないため不可。
