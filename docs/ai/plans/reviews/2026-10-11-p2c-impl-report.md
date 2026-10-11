**C4のJSON保存エラーで停止しました。受入条件は未達です。** 私の実装で `numpy.bool_` をJSON用の型へ変換していなかったことが原因です。停止指示に従い、修正・再実行・許容変更はしていません。

| 検査 | 結果・値 |
|---|---|
| C1 | 合格。6/10 MeVで全対象配列が完全一致 |
| C2 | 合格。積分の最大偏差 `1.7764e-15`、独立値との最大相対差 `1.8801e-12`。率は `0.05747282522707444 /cm` |
| C3 | 合格。最小p値 `0.0308666`。最大運動量残差 `5.1979e-15 MeV`、最大方向長偏差 `4.4409e-16` |
| C4 | 4条件の数値判定は合格でしたが、保存失敗。回収できた1 MeV条件は消滅7,094回、`P=0.03547`、`P_th=0.03544687953`、差 `2.31205e-5`、許容幅 `0.00230379662`。残り3条件の個別値は未保存 |
| C5〜C8 | 未実行、値なし |

指定コマンドの結果は次のとおりです。

| コマンド | 結果 |
|---|---|
| `run_p2c --checks` | 終了コード1 |
| `pytest prototypes/mv_electron/tests -q` | 停止により未実行 |
| `pytest tests/ -q` | 停止により未実行 |
| `run_checks_p2 --verify-untouched` | 停止により未実行 |
| `git status --short -- chatcarlo tests scripts examples` | 終了コード0、出力なし |

4つの新規コード・テストファイルを作成し、`--analyse` のコードも書きました。**`--run` と `--analyse` は実行していません。**

全数値・エラー・残した証拠は[停止報告](/Users/oishifamily/Projects/viveMonte/docs/validation/mv/p2c/IMPLEMENTATION_STATUS.md)に記録しました。[checks.json](/Users/oishifamily/Projects/viveMonte/docs/validation/mv/p2c/checks.json)はC1〜C3までの記録で、全検査合格の証拠には使えません。