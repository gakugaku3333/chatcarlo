B0〜B5を完了しました。**正式K5bは1回で合格し、T5・T6も合格**です。[結果全文](docs/validation/mv/p1f/RESULTS.md)に記録しました。

凍結1は旧2332ファイルを保全。凍結2は、許可されたコード差分と固定データ・`stop1/`を含む2400ファイルです。凍結2、再計算禁止の32ファイル、最終manifest7615件はすべて一致しました。

|項目|状態|主な値|
|---|---|---|
|T0|合格|保存済み回帰一致、凍結照合一致|
|T1|合格|同梱再現差0、4条件×51点×2巡一致|
|T2|合格|補間差最大0.128010%、CDF差0.000166341、ERROR最大6.1×10⁻⁶。受入警告58、不受入0|
|T3|未評価・記録済み|水2 MeV G1/G2＝0.617379/1.757546 cm⁻¹。合否規則なし|
|T4|合格|K3差最大0.059674%、CDF最小p＝0.274596|
|T5|合格|K5/K5b/K7/K8/K9すべて合格。K9(i)横変位差−0.412348%／−0.106889%|
|T6|合格|最大線量差0.750223% Dref（許容3%）、R50/Rp差−0.072783%／−0.207620%|
|T7|未評価・参考計算済み|最大線量差0.550619% Dref。合否規則なし|

補助指標：elsepaのR50/Rp＝0.695093/0.887756 cm、後方エネルギー比0.626989%。elsepa−dcslib最大差は0.405526% Dref（−2.050 SEM）。dcslib/mfの対EGS5最大差は0.444706%／0.821470%です。

実行した主要コマンドは以下です。pytest各検査を含む全文・終了コードは[実行記録](docs/validation/mv/p1f/amendment1_execution_commands.json)と[検査記録](docs/validation/mv/p1f/amendment1_commands.json)に保存しました。

```text
.venv/bin/python /tmp/p1f_amendment1_b0.py
.venv/bin/python prototypes/mv_electron/run_p1f_amendment1.py --precheck
.venv/bin/python prototypes/mv_electron/run_p1f_amendment1.py --freeze
.venv/bin/python prototypes/mv_electron/run_p1f_amendment1.py --execute
.venv/bin/python prototypes/mv_electron/run_p1f_amendment1.py --production-tests
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --verify-untouched
.venv/bin/python prototypes/mv_electron/run_p1f_amendment1.py --verify-untouched
.venv/bin/python /tmp/p1f_amendment1_report.py
```

B1の単独・全体実行の一致と禁止経路の検出はB3前に確認済み。本体403件合格、試作87件合格・既存1件skipでした。

確認した事実は、旧K5bの実行エラー、条件・ハッシュの一致、今回の検査結果です。**T6の値は追補前から既知**でした。Claudeによる再開は運用記録に基づき、実行者を成果物だけでは証明できません。水素差やK9偏りの原因は仮説・未解明です。

B0〜B5に未完了はありません。T3/T7の合否、既存skip、水素差の原因、表の配布・ライセンス適用は未評価または対象外です。保存run・標本・表の再計算、EGS5再実行、計画編集、commitは行っていません。成果物の利用者絶対パスも0件です。