**計画全体は未完了です。** T0〜T4とT5の自己点検を保存しました。Q1は不合格で、条件を調整せずQ6を未評価にしています。本計算は0回です。

| 検査 | 状態・主な値 |
|---|---|
| Q0 | 未評価。現時点の保護ハッシュと電子回帰は一致。第三凍結は未実施 |
| Q1 | **不合格**。対生成補間誤差：核場0.230052%、電子場0.235614%（許容0.2%） |
| Q2 | 合格。平均エネルギー差最大2.348σ、χ²検査の最小p=0.07665 |
| Q3 | 合格。収支最大相対残差1.066×10⁻¹⁴、違反・上限到達0 |
| Q4 | 合格。解析値との差最大1.788σ、相対SEM最大0.0547% |
| Q5 | 未評価。線量・カーマ比未取得 |
| Q6 | 全3 backend ×4エネルギーで未評価 |
| Q7 | 未評価。刻み半減計算未実施 |

EGS5自己点検は8ケースで合格しました。ただし、実効電子下限が10 keVより約1 eV高く、T5の厳密な基準設定は未完了です。T6の実行ドライバーにも未実装部分があります。

確認した事実は閾値近傍の補間基準超過です。「XCOMの丸めが原因」という説明は未検証の仮説です。初期の実装・検査不具合も保存しています。

テストは本番403件合格、試作全体95件合格・1件skip、P2単独8件合格でした。主な実行コマンドは次のとおりです。

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_photon_p2.py -q
.venv/bin/python prototypes/mv_electron/run_checks_p2.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p2.py --verify-untouched
```

詳細・全コマンド・未完了項目は `docs/validation/mv/p2/RESULTS.md`、`commands.json`、`progress.json` に保存しました。保存済み検査を再利用する再開コマンドは以下です。未実装のT6を完了するコマンドではありません。

```bash
.venv/bin/python prototypes/mv_electron/run_checks_p2.py --resume
```

計画ファイルの編集・commitはしていません。