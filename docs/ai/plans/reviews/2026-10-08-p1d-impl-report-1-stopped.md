**R0の検証で停止し、実装・計算は未完了です。** [停止報告](docs/validation/mv/p1d/RESULTS.md)と[manifest](docs/validation/mv/p1d/manifest.json)を保存しました。

実装には、凍結対象を計画の必須範囲より広く取った問題と、描画処理の`dref` import漏れがあります。追加で凍結したP2計画のハッシュが変わり、固定済みの検証コードが停止しました。このファイルを編集する操作はしていません。

| 項目 | dcslib | mf |
|---|---|---|
| R0 | 不合格：ハッシュ1件不一致 | 同左 |
| R1 | 未評価：0.2 cm試作が未完了 | 合格 |
| R2 | 未評価。K5bのみ合格 | 同左 |
| R3 | 未評価 | 未評価 |
| R4 | 未評価 | 未評価 |

完了したR1の値です。「差 ± SEM」、単位は%です。

| backend／道のり | 横二乗平均 | 深さ平均 |
|---|---:|---:|
| dcslib／0.05 cm | −0.308 ± 0.291 | −0.00222 ± 0.00495 |
| mf／0.05 cm | −0.530 ± 0.288 | +0.00083 ± 0.00498 |
| mf／0.2 cm | −0.345 ± 0.167 | +0.02097 ± 0.01137 |

確認した事実は、必須の物理コード・旧成果物・事前登録・当初のP1dコードのハッシュ一致、追加前後4条件の完全一致、既存403件・試作75件のテスト合格です。有限の刻みが広がりのずれを生むという説明は、未検証の仮説です。

未完了は、dcslibのR1残り1標本、輸送14run、R2〜R4、補助比較、性能計測、図、および描画import漏れへの対処です。保存済み結果のやり直し・増統計・seed変更・EGS5再実行・commitは行っていません。

主要な実行コマンドは以下です。保存結果再生と凍結検証は、R0不一致で終了コード1でした。

```bash
.venv/bin/python prototypes/mv_electron/run_tests_p1d.py
.venv/bin/python prototypes/mv_electron/run_p1d.py --freeze
.venv/bin/python prototypes/mv_electron/run_p1d.py
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched
.venv/bin/python prototypes/mv_electron/stop_report_p1d.py
```

成果物のハッシュ整合と、利用者の絶対パスが含まれないことを確認しました。