# P1f 結果

問いG': ELSEPAによる水素・酸素の表の自前生成が、再現可能な手順で成立。
問いH': 成立の判定に必要な検査が不合格または未評価。未評価を物理的不合格と扱わない。

P1e（`IHEF=1`）は表の検査で不合格だった。P1fは、その結果を見たあとに作り方を変えたものである。変更の根拠はEGS5に依存しない検査である。

事前登録: [PREREGISTRATION.md](PREREGISTRATION.md)。2つのseedは20261013/20261014。やり直し・増統計・seed選び直し・合わせるための調整なし。警告は事前登録の分類で全件を保存。

|項目|状態|値・根拠|
|---|---|---|
|T0|合格|既存4 backend各10000履歴の全配列完全一致。凍結 2332 ファイル。追加・削除・変更0。|
|T1|合格|同梱DCS最大相対差 0、断面積差 [0.0, 0.0, 0.0]。4条件×51点×独立2巡、全配列一致。|
|T2|合格|{"table_total_relative": 0.0024870168246844404, "table_transport_relative": 0.00043111969558307095, "doubling_relative": 4.3087755585702325e-13, "ERROR_max": 6.1e-06, "interpolation_relative": 0.0012801037101111135, "CDF_difference": 0.00016634088880806974}。受入整合警告 58、不受入警告 0、無効置換 0。|
|T3|未評価|記録項目。合否規則なし。計算済み。|
|T4|合格|詳細は summary.json|
|T5|不合格|詳細は summary.json|
|T6|未評価|T0,T1,T2,T4,T5合格時のみ判定|
|T7|未評価|最大線量差/Dref=0.005506186、R50/Rp相対差=[-0.0014708667151760912, -0.0008049015282676597]、合成SEM/Dref=0.0026377239。参考のみ、合否規則なし。|

T3/T7は合否規則のない記録項目。計算済みかどうかを状態とは別に示す。

警告の受入上限（全断面積0.5%、第1輸送断面積0.1%）は、既知の全断面積差約0.25%を見たうえで事前に置いた。全断面積の上限はELSEPA内部閾値の2.5倍である。結果を見たあとで除外していない。

T2不合格の全内訳（警告の全件は [warnings.json](warnings.json)、数値の全件は [t2.json](t2.json)）:

|Z|電荷|MeV|種類|不合格の量|差 [全,G1,G2] / CDF|
|---|---|---|---|---|---|

不合格 0 件。格子点 204、直接中間点 200、旧切り替え前後の記録点 8。±1%の2点比較で連続性は主張しない。

K9(i)（差は[横変位二乗平均, 深さ変位平均, 補助量]、後ろの量は判定対象外）:

|道のり cm|差|合成SEM|状態|
|---|---|---|---|
|0.05|[-0.004123475350008134, -3.195704512513117e-05, -0.00012565819941338322, -0.00020829204799949238]|[0.002903419538635964, 4.986014433644906e-05, 9.008770169749142e-05, 0.00020105592641632387]|合格|
|0.2|[-0.0010688930674433742, 6.342789624639208e-05, 0.00020043905841963736, 0.00025394502742629083]|[0.001678650113878373, 0.00011317295707288834, 0.00023361483540425816, 0.0005795487822632415]|合格|

K5/K5b/K7/K8/K9(ii): {"K5": {"status": "pass", "global_relative": 3.4850982343070936e-17, "max_history_relative": 6.217248937900877e-15, "unprocessed_particles": 0, "unprocessed_MeV": 0.0, "stack_failures": 0, "step_limit_failures": 0, "back_energy_fraction": 0.006269889116447767}, "K5b": {"status": "fail", "command": ".venv/bin/python -m pytest prototypes/mv_electron/tests/test_p1f.py::test_actual_p1f_transport_and_k5b -q", "exit_code": 1}, "K7": {"status": "pass", "rows": {"split_aligned": {"Dref": 3.2159458582928044, "reference_ranges_cm": [0.6947986526763167, 0.8877556380864275], "other_ranges_cm": [0.6947986526763166, 0.8877556380864274], "range_difference_over_reference_R50": [-1.597906127694194e-16, -1.597906127694194e-16], "range_difference_SEM_over_R50": [0.0007748852656177737, 0.0009283580561119304], "max_dose_difference_over_Dref": 8.285385937793285e-16, "max_dose_difference_SEM_over_Dref": 0.0023538527827999915, "evaluation_bins": 120, "statistics_ok": true, "pass": true, "boundary_events": 6448354, "boundary_queries": 1287604265, "status": "pass"}, "split_shifted": {"Dref": 3.2159458582928044, "reference_ranges_cm": [0.6947986526763167, 0.8877556380864275], "other_ranges_cm": [0.6947986526763167, 0.8877556380864272], "range_difference_over_reference_R50": [0.0, -4.793718383082582e-16], "range_difference_SEM_over_R50": [0.0007748852656177757, 0.0009283580561119196], "max_dose_difference_over_Dref": 8.285385937793285e-16, "max_dose_difference_SEM_over_Dref": 0.0023538527827999855, "evaluation_bins": 120, "statistics_ok": true, "pass": true, "boundary_events": 7106035, "boundary_queries": 1288261946, "status": "pass"}}}, "K8": {"status": "pass", "rows": {"ms_half": {"Dref": 3.2159458582928044, "reference_ranges_cm": [0.6947986526763167, 0.8877556380864275], "other_ranges_cm": [0.6947102826641476, 0.8875369275196402], "range_difference_over_reference_R50": [-0.00012718794405934457, -0.0003147826581771453], "range_difference_SEM_over_R50": [0.0007807685068155411, 0.0009029368584916592], "max_dose_difference_over_Dref": 0.005490058745618873, "max_dose_difference_SEM_over_Dref": 0.0020910529185292144, "evaluation_bins": 120, "statistics_ok": true, "pass": true, "boundary_events": 20786, "boundary_queries": 2274856510, "status": "pass"}, "energy_half": {"Dref": 3.2159458582928044, "reference_ranges_cm": [0.6947986526763167, 0.8877556380864275], "other_ranges_cm": [0.6937569325541781, 0.8884246112150708], "range_difference_over_reference_R50": [-0.0014993122368990078, 0.0009628302042129555], "range_difference_SEM_over_R50": [0.0007779003441730367, 0.0008903495875532033], "max_dose_difference_over_Dref": 0.004213063500388399, "max_dose_difference_SEM_over_Dref": 0.0022213251305241554, "evaluation_bins": 120, "statistics_ok": true, "pass": true, "boundary_events": 20664, "boundary_queries": 1553160878, "status": "pass"}}}, "K9ii": {"status": "pass", "prototype_cm2": [0.0066826321711395235, 0.06460265432946534], "reference_cm2": [0.006690003055515246, 0.06476063874744237], "relative": [-0.0011017759356097545, -0.0024395129670222297], "difference_SEM_relative": [0.0073342153606901364, 0.0029207561149361314], "prototype_cross_N": [986544, 721730], "reference_cross_N": [296036, 216598]}, "status": "fail"}

dcslib照合（計算済み、判定対象外）:

|Z|電荷|全断面積比|G1比|G2比|DCS角度比|
|---|---|---|---|---|---|
|1|-1|0.99997274–1.0077636|0.99999199–1.0135278|0.99999075–1.0151045|0.99791935–1.0433016|
|1|1|0.99999049–1.0067942|0.99999639–1.0107644|0.99999589–1.0123284|0.9976285–1.0420495|
|8|-1|0.99976824–1.0016264|0.99991855–1.0014048|0.99990637–1.0012506|0.99820296–1.0039782|
|8|1|0.99789681–1.0015844|0.99675774–1.0001303|0.99673645–1.0001344|0.99585358–1.0037842|

水の合成G1/G2/後方半球積分、IHEF=0/1の比、旧切り替え前後の2点は t3.json / t2.json に全件保存。

補助指標（合否に使わない）:

- elsepa: {"max_dose_difference_over_Dref": 0.007502233691723832, "R50_cm": 0.6950934474634677, "Rp_cm": 0.8877556380864275, "range_difference_cm": [-0.0005062806817944798, -0.0014442009475119644], "range_difference_over_reference_R50": [-0.0007278333520118242, -0.0020761953880614106], "back_energy_fraction": 0.006269889116447748, "status": "not-evaluated", "reason": "補助記録のみ。旧判定とR3の停止規則を変更しない。"}
- dcslib: {"max_dose_difference_over_Dref": 0.004447062222119866, "R50_cm": 0.694671695653847, "Rp_cm": 0.888065955566623, "range_difference_cm": [-0.0009280324914151539, -0.0011338834673164655], "range_difference_over_reference_R50": [-0.0013341472888289466, -0.0016300803773167613], "back_energy_fraction": 0.006189691646554222, "status": "not-evaluated", "reason": "補助記録のみ。旧判定とR3の停止規則を変更しない。"}
- mf: {"max_dose_difference_over_Dref": 0.008214696905288836, "R50_cm": 0.6931996468779574, "Rp_cm": 0.8860936123530929, "range_difference_cm": [-0.0024000812673047323, -0.003106226680846591], "range_difference_over_reference_R50": [-0.0034503769483985812, -0.004465537514123234], "back_energy_fraction": 0.006214256055948225, "status": "not-evaluated", "reason": "補助記録のみ。旧判定とR3の停止規則を変更しない。"}
- elsepa_minus_dcslib: {"max_difference_over_Dref": 0.004055260309659622, "depth_cm": 0.5750000000000001, "difference_over_combined_SEM": -2.0502016851193576, "comparison": "elsepa seed20261013 vs P1d saved dcslib seed20261009"}

輸送単価: {"rows": [{"T": 2.0, "scoring": true, "N": 20000, "K": 0.025, "fe": 0.05, "seed": 20261013, "wall_seconds": [5.385322332964279, 5.395827708998695, 5.379800875089131], "us_per_primary": 269.26611664821394}, {"T": 2.0, "scoring": false, "N": 20000, "K": 0.025, "fe": 0.05, "seed": 20261013, "wall_seconds": [5.3089218330569565, 5.304093708051369, 5.317493750015274], "us_per_primary": 265.4460916528478}, {"T": 10.0, "scoring": true, "N": 20000, "K": 0.025, "fe": 0.05, "seed": 20261013, "wall_seconds": [20.469844916951843, 20.43308300001081, 20.502820166992024], "us_per_primary": 1023.492245847592}, {"T": 10.0, "scoring": false, "N": 20000, "K": 0.025, "fe": 0.05, "seed": 20261013, "wall_seconds": [20.32361820805818, 20.21010575001128, 20.16068541596178], "us_per_primary": 1010.505287500564}], "method": "single process, median 3 x 20000, fixed seed, JIT excluded, statistics not pooled", "table_bytes": 151113248}

生成時間・表サイズ:

|巡/Z/電荷|ケース数|秒|bytes|
|---|---|---|---|
|generation_pass1_z1_q-1|51|131.9173|606335|
|generation_pass1_z1_q1|51|136.6920|606335|
|generation_pass1_z8_q-1|51|114.7431|606335|
|generation_pass1_z8_q1|51|114.1608|606335|
|generation_pass2_z1_q-1|51|132.7042|606335|
|generation_pass2_z1_q1|51|133.0663|606335|
|generation_pass2_z8_q-1|51|113.0661|606335|
|generation_pass2_z8_q1|51|110.9379|606335|

生成前の代表条件による見込み: {"rows": [{"Z": 1, "charge": -1, "seconds": 2.992008417029865}, {"Z": 1, "charge": 1, "seconds": 3.09943275002297}, {"Z": 8, "charge": -1, "seconds": 2.6111739589832723}, {"Z": 8, "charge": 1, "seconds": 2.649327374994755}], "table_cases": 408, "estimated_table_seconds": 1157.898135105148, "method": "2 MeV representative per element/charge, before generation; midpoint/probe work additional; 10 MeV cost is higher than 2 MeV, so this is a rough estimate"}

確かめた事実:

- アーカイブSHA256と展開全ファイルの内容を照合。ELSEPAソースは変更していない。同梱再現はP1eの保存出力を読み取りだけで確認した。
- ELSEPAのstdoutの第2輸送断面積はDCS積分由来。継承フィールドseries_xsの第3欄は独立な級数値ではなく、警告受入判定には使わない。
- IHEF=0の格子点・直接計算点の警告、ERROR、級数/表断面積、収束の出力行、計算時間を保存した。生成表と生出力はP1f専用のリポジトリ外の保存先にある。
- 水素とdcslibの差は上の数値として確認した。電子密度・核分布・電子交換はEGS5説明と共通だが、EGS5のMCPOL/MABS/IHEF/陽電子交換の指定は確定できていない。
- T2で検査した表のハッシュをGS表と各runのレシートに結び付ける。陽電子の表は検査だけで、輸送には未使用。

仮説: 水素差の由来として旧版の原子データや実装版の違いが考えられるが、今回原因は調べていない。P1eの不合格が切り替えの跳びを連続スプラインでまたいだためという説明も、原因を分離した実証ではない。

利用条件: 計画/P1eの取得時に確認したMendeley掲載表記は「CC BY NC 3.0」。同梱ソース/readmeには別のライセンス記述がないことをP1eが確認した。今回はネットワークを使っていない。CC BY-NC 3.0の条文そのものと生成表への適用は未確認。

未完了の項目:
T6、T7。停止規則または未完了のrunによる。統計不足・未評価を合格と数えない。

実行コマンド・終了コード: [commands.json](commands.json)。データ出典・保存先・再生成手順: [DATA.md](DATA.md)。保存物のハッシュ: [manifest.json](manifest.json)。EGS5再実行・commit・計画編集なし。
