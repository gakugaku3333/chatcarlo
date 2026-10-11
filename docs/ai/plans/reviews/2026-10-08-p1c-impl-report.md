`mf`を実装しました。**N5が不合格のため、登録規則どおりN6・N7は未判定です。** 調整・EGS5再実行・commitはしていません。

|項目|値・判定|
|---|---|
|N1|71配列が完全一致。P1・P1bの264ファイル不変：合格|
|N2|積分倍増の最大相対変化2.31×10⁻¹⁴、Born閉形式との差4.44×10⁻¹⁶：合格|
|N3|事前値の照合15/15件が相対10⁻³以内：記録のみ|
|N4|GS規格化誤差最大5.42×10⁻⁷：合格|
|N5|K9(i)、経路0.05 cm・K=0.05で横方向変位二乗の差−2.02048%、SEM 0.29235%。許容幅2%を超過：不合格。他の検査は合格|
|N6|N5未達により未判定|
|N7|N5未達により未判定|

pytestは本番403件、試作71件が通過しました。16条件の計測も完了し、基準条件は2 MeVで213.45、10 MeVで731.96 µs/一次履歴でした。

**確かめた事実**は上の検査失敗と保存値です。有限刻み・標本変動の寄与、低エネルギー散乱係数差と深いbinの差との因果は、原因未特定の**仮説**として記録しました。未完了はN6・N7と、それに依存する残差低減率・R50/Rp比較です。

計算・検証の主要コマンド：

```bash
.venv/bin/python -m prototypes.mv_electron.regression_p1c --before
.venv/bin/python -m prototypes.mv_electron.regression_p1c
.venv/bin/python prototypes/mv_electron/run_p1c.py
.venv/bin/python prototypes/mv_electron/run_p1c.py --benchmark-only
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python prototypes/mv_electron/run_checks_p1c.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1c.py --verify-untouched
```

詳細・全実行記録は [RESULTS.md](docs/validation/mv/p1c/RESULTS.md)、図は [depth_2.png](docs/validation/mv/p1c/depth_2.png) に保存しました。再解析・manifest照合・絶対パス検査は通過しています。