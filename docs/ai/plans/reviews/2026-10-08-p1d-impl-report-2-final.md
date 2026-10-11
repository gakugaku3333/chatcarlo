A0〜A6を完了しました。**輸送14本を保存し、実行項目に未完了はありません。**

凍結1のR0不合格は[停止記録](docs/validation/mv/p1d/stop1/RESULTS.md)に保全しました。再開後の凍結2は **2,106ファイルすべて一致**、保存済み回帰4条件も完全一致です。R1の元の7標本は保持し、A4のメモリ上再現も完全一致しました。

| 項目 | dcslib | mf |
|---|---|---|
| R0・凍結2 | 合格 | 合格 |
| R1・横二乗平均の差：0.05 / 0.2 cm | 合格：−0.308 / −0.295% | 合格：−0.530 / −0.345% |
| R2 | 全項目合格 | 全項目合格 |
| R3・2 MeV線量最大差/Dref | 合格：0.445% | 合格：0.821% |
| R4・10 MeV線量最大差/Dref | 未評価・参考計算済み：0.578% | 未評価・参考計算済み：0.792% |

両backendで、**2 MeV・水・指定条件（K1_max=0.025）で電子輸送の試作が成立**しました。dcslibは研究専用の対照、mfは解析式で低Z限定です。刻みはP1cの結果を見たあとに変えました。選定の根拠はEGS5に依存しない検査です。

補助指標は次のとおりです。全値・SEM・範囲差・後方散乱比は[結果報告](docs/validation/mv/p1d/RESULTS.md)に記載しています。

| 補助指標 | dcslib | mf |
|---|---:|---:|
| P1に対する残差低減率 | 85.29% | 72.83% |
| 同backend旧刻みに対する低減率 | 18.20% | 12.94% |
| 上記の95%区間 | −26.97〜44.30% | −19.62〜30.35% |
| 保存済みms_halfとの最大差/Dref | 0.616% | 0.373% |
| 最大｜差/合成SEM｜ | 3.15 | 2.48 |
| 2 MeV・採点あり単価 | 273.60 μs | 267.20 μs |

同backend旧刻みに対する低減率の区間は両方とも0を含み、**刻み変更によるEGS5残差の改善は統計的に断定できません**。mf−dcslibの最大差は0.554% Dref、深さ0.245 cmでした。時間測定の一部に短い保存R4再解析が重なった点も報告に明記しました。

R1の新旧差はdcslibで合成SEMの2.31倍、mfで1.89倍でした。両backendは同じseedを使うため相関しています。有限刻みによる偏りや、旧K=0.05の合否差へのseedゆらぎの関与は**仮説で、原因は未確認**です。旧判定は変更していません。

実行した主要コマンド：

```bash
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --seal
.venv/bin/python prototypes/mv_electron/run_tests_p1d.py
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py --reproduce
.venv/bin/python prototypes/mv_electron/run_p1d_amendment1.py
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1d.py --verify-untouched
```

本体403件・試作75件のテストが成功し、保護対象のGit差分は空でした。固定条件・判定ロジックの変更、合わせる調整、EGS5再実行、計画編集、commitは行っていません。

成果物：[線量図](docs/validation/mv/p1d/depth_2.png)、[全コード差分](docs/validation/mv/p1d/amendment1_diff.txt)、[実行記録](docs/validation/mv/p1d/amendment1_commands.json)、[manifest（156ファイルのSHA256確認済み）](docs/validation/mv/p1d/manifest.json)。