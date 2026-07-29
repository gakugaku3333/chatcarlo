# njitスカラーDDA前提検証

## 目的

本番コードを変更せず、njitスカラーDDAが現行numpy実装より速いか、さらに
end-to-end 3倍化に足るかを反証可能な形で確認する。

## 事前登録した判定基準

- 比較対象は同じ固定入力を受け取る現行`accumulate_track_length_multi`と
  njitスカラーDDA。JIT初回コンパイルと入力生成は計測外とする。
- 各targetを毎回ゼロクリアし、同一プロセスでAB/BAを交互に実行する。
- ゲートA: `median(njit DDA) < median(numpy DDA)`なら通過。
- DDA速度比を `S_DDA = median(numpy) / median(njit)` とする。
- ゲートB: 既報の全体0.538秒、DDA 0.402秒、残余0.136秒を固定し、
  `S_E2E = 0.538 / (0.136 + 0.402 / S_DDA)` を計算する。
  `S_E2E >= 3.0`なら通過。それ未満ならゲートA通過時も3倍化プロジェクトには進まない。
- 主要判定はwater60_free（N=200,000、resolution=2 cm）の直接A/Bだけで行う。
  chest_room由来区間の速度比は一般化確認の参考値とする。

## 入力契約

方向ベクトルは単位長でなければならない。`length_cm`は幾何学的な飛程上限であり、
非正規化方向を渡すと現行実装と同様に物理的な長さの意味が変わる。

## 将来統合時の制約

今回のシングルスレッド採択結果だけではカーネル内統合可能とは判断しない。
prange領域で共有グリッドを直接更新するとデータ競合する。候補は
チャンク/スレッド専用グリッド後reduce、atomic加算、区間バッファを残してDDAのみ
後段実行、の3つである。次段階には競合回避方式の選定と専用グリッド等のメモリ見積もりが
別途必要であり、本計画では実装しない。

## 結果

- 正しさ: PASS。列挙した単一レイ境界ケース、非立方体・1200 voxel長区間、
  同一voxelへの複数加算、6000-ray fuzz、chest_room実輸送由来103,336区間の
  kerma/H*(10)について、ボクセル配列・非ゼロvoxel集合・グリッド合計が一致した。
  報告された最大絶対誤差・最大相対誤差はいずれも全ケースで0だった。
- water60_free（kernel由来1,022,763区間、N=200,000、2 cm）:
  numpy中央値0.379355秒、njit中央値0.058222秒、DDA速度比6.516倍。
  ゲートAはPASS。
- 健全性確認の統合経路1回: 0.526749秒（既報0.538秒と同オーダー）。
- 固定式による予測:
  `0.538 / (0.136 + 0.402 / 6.516) = 2.721倍`。3倍未満なのでゲートBはFAIL。
- chest_room一般化参考値（実輸送由来103,336区間）:
  numpy中央値0.327111秒、njit中央値0.074777秒、4.374倍。
- 採否: **棄却（DO_NOT_PROCEED）**。スカラーDDA単体は速いが、
  事前登録した3倍化条件を満たさないため、3倍化プロジェクトには進まない。
- pytest: 326 passed、8 failed、2 warnings（73.43秒）。失敗8件は全て並列テストで、
  sandboxが`os.sysconf("SC_SEM_NSEMS_MAX")`を`PermissionError: [Errno 1]
  Operation not permitted`として拒否した同一の環境要因。本番コード・testsは
  変更禁止のため変更していない。

全テストコマンドの生出力は
`docs/speedup_baseline/dda_njit_premise_check_result.txt`に記録した。

## end-to-end実測（追記）

### 初回（境界セマンティクス差により停止）

前計画は「3倍」という一括目標に対する判定であり、本追記は
「着実な高速化として採用できるか」という別の問いに対する実測である。
したがって、前計画のゲートB FAILおよびDO_NOT_PROCEED判定は変更しない。

ただし、速度測定より先に行うべき正しさゲートのうち、複数バッチ・端数バッチ条件が
事前登録した許容値を満たさなかったため、計画の「いずれか一致しない場合は速度測定へ
進まず、原因を報告して停止する」に従ってend-to-end timingは実行しなかった。

- 1バッチ条件（N=200,000、`batch_size=200_000`）:
  `kerma_keV`、`h10_track_pSv_cm3`ともボクセル値・非ゼロボクセル集合が完全一致
  （最大絶対誤差0、最大相対誤差0）。`KernelBatchResult`の全6配列もビット一致。
- 複数バッチ・端数バッチ条件（60k×3 + 20k）:
  非ゼロボクセル集合は両targetとも一致したが、`kerma_keV`は最大絶対誤差
  8.0606259871274233e-06、最大相対誤差1.0180686290485363e-10、
  `h10_track_pSv_cm3`は最大絶対誤差2.1541636670008302e-06、最大相対誤差
  1.0509332453466174e-10で、許容`atol=2e-10, rtol=2e-12`を超えた。
- 当時の原因候補: 1バッチでは完全一致し、複数バッチでも非ゼロ集合は一致しているため、
  DDAの幾何学的な配線ミスではなく、既に値を持つtargetへ次バッチを加える際の
  numpy集約経路とnjitスカラー経路の浮動小数点加算順の差と判断した。
  呼出し単位の寄与を一時ゼログリッドへ集約してからtargetへ加える方式も確認したが、
  同程度の差（最大相対誤差約1.05e-10）が残り、許容には入らなかった。
- 後の区間単位調査で判明した真因は、加算順ではなく次の4つの境界セマンティクス差だった。
  1. 試作は区間端点から`_EPS_PLANE`以内の面も交差面として扱っていた。
  2. 試作は入口直後のprobeで初期セルを決め、参照の部分区間中点分類と異なっていた。
  3. 試作は`tol = 2e-12`以内の近接到着を同着扱いし、参照が残す正長薄片を落としていた。
  4. 試作は範囲外添字をclampし、参照の範囲外破棄と異なっていた。
- `S_E2E_measured`: 未測定。
- 採用判定: **高速化を確認できず**（速度の不合格ではなく、前提となる正しさゲート未通過）。
- 予測との乖離・残余変化・次のボトルネック候補: timingを実行していないため評価不能。
- 本計画の"end-to-end"はkernelのdose-grid経路（`run_dose_grid`）全体を指し、
  CLI起動・シーン読込・`bake_scene_materials`等を含まない。
- `chest_room`はcylinder / sphereを含み、box専用kernelではend-to-end測定不能。
- pytestは326 passed、8 failed、2 warnings。8件は計画に記載済みの並列テストで、
  すべてサンドボックスが`os.sysconf("SC_SEM_NSEMS_MAX")`を拒否する同一の
  `PermissionError: [Errno 1] Operation not permitted`だった。

アダプタ契約（2 target、shape、重み長）、非ゼロtargetへの積算、空入力、および
意図的な例外後のモンキーパッチ復元はすべてPASSした。全生出力は
`docs/speedup_baseline/dda_njit_e2e_result.txt`に記録した。

### 修正後

参照実装と同じく、AABBクリップ後に各軸の
`ceil(p_small + _EPS_PLANE)`から`floor(p_big - _EPS_PLANE)`までの面を列挙し、
全交点と両端点を区間内でソートした。隣接点間が正長の場合だけ中点でボクセル分類し、
範囲外はclampせず破棄する。`_EPS_PLANE`は`chatcarlo.tally`からimportしている。
近接時刻の許容同着判定は廃止し、厳密に同じ時刻のときだけ差がゼロとなって落ちる。

正しさの結果:

- 完全精度のwater60_free回帰ケースは参照とボクセル単位で一致し、
  `voxel(1,25,24)`へのkerma/H\*(10)寄与はいずれもゼロ。
- 境界近傍fuzzは区間単位で、端点3,000本、AABBクリップ点1,500本、
  2軸・3軸の厳密/近接同着500本がすべて一致。決定値
  `0, 0.5*EPS, 0.99*EPS, 1.01*EPS, 2*EPS`、面の両側、正負方向、
  斜め方向、短い初回境界を含む。
- 既存境界ケース、6000本random fuzz、chest_room由来区間も全PASS。
- kernel統合経路は1バッチ（200,000）と複数・端数バッチ（60k×3+20k）の双方で、
  kerma/H\*(10)の値と非ゼロ集合が完全一致（最大絶対誤差0、最大相対誤差0）。
  `KernelBatchResult`全6配列もビット一致。

end-to-end測定はwater60_free、N=200,000、`batch_size=200_000`、
`n_chunks=8`、AB 4回・BA 4回で実施した。

| 指標 | min (s) | median (s) | max (s) |
| --- | ---: | ---: | ---: |
| `E2E_A` | 0.501122583 | 0.513267500 | 0.554781458 |
| `DDA_in_A` | 0.372562125 | 0.380829541 | 0.423874125 |
| `residual_A` | 0.127399500 | 0.129714438 | 0.136502500 |
| `E2E_B` | 0.196290333 | 0.199722438 | 0.204999417 |
| `DDA_in_B` | 0.070123959 | 0.070733292 | 0.071057167 |
| `residual_B` | 0.125873458 | 0.128791854 | 0.134806792 |

反復ごとのend-to-end速度比は
`2.722486, 2.523152, 2.524552, 2.553029, 2.485482, 2.769186, 2.556147, 2.626401`。
8反復すべてで`E2E_B < E2E_A`、中央値比`S_E2E_measured = 2.569904 > 1.0`のため、
事前登録した採用判定は **高速化を達成**。

- 固定予測2.721倍に対する実測比は0.944470（実測は予測より約5.55%低い）。
  同時測定した分解による予測は2.560605倍で、実測と近い。
- `median(E2E_A)=0.513267500s`（既報0.538sに対して約4.60%短い）。
- 残余中央値は0.129714438sから0.128791854sへ−0.000922584s（−0.711%）。
  10%基準内であり、DDA置換以外の有意な相互作用は観測されない。
- 境界修正後のDDA単体速度は5.491倍で、修正前6.516倍から約15.7%低下した。
  それでもend-to-endの実高速化は上記のとおり達成した。
- 置換後の最大成分は`residual_B`である。内訳は今回プロファイルしていないため、
  transport、区間バッファ確保・連結、重み計算の集合が次のボトルネック候補。
- 計時ラッパーのオーバーヘッドはDDA呼出し1回につき`perf_counter`一組
  （この条件では各アーム1回）で、無視できる水準。
- `ru_maxrss=1125564416`はプロセス開始以来の累積ピーク参考値のみ。
  同一プロセス内のアーム別メモリ差は主張しない。
- pytestは326 passed、8 failed、2 warnings。8件は計画に記載済みの並列テストで、
  すべてサンドボックスが`os.sysconf("SC_SEM_NSEMS_MAX")`を拒否する同一の
  `PermissionError: [Errno 1] Operation not permitted`だった。

正しさ・速度測定の全生出力は
`docs/speedup_baseline/dda_njit_premise_check_result.txt`と
`docs/speedup_baseline/dda_njit_e2e_result.txt`に記録した。

### kernel.py恒久組み込み後

`chatcarlo/tally_njit.py`へ境界修正後のスカラーDDAを移設し、
`kernel.run_batch_with_tally` / `run_dose_grid`の
`use_njit_dda=True`既定経路として恒久組み込みした。`False`では従来のnumpy
参照実装へ切り戻せる。

water60_free、N=200,000、resolution=2cmの直接A/Bでは、1バッチ
（`batch_size=200_000`）と複数・端数バッチ（`batch_size=60_000`）の双方で、
kerma・H\*(10)グリッドと`KernelBatchResult`全6配列が`np.array_equal`で一致した。

同じ条件をAB/BA交互に8反復した本番引数経路のend-to-end再測定結果:

- numpy (`use_njit_dda=False`): min 0.507760625s、median 0.516252230s、
  max 0.551558125s
- njit (`use_njit_dda=True`): min 0.197448500s、median 0.202292374s、
  max 0.269662833s
- 反復ごとの速度比:
  `2.619678, 2.564341, 2.470462, 2.521338, 2.543524, 1.923508, 2.551472, 2.585708`
- 8反復すべてでnjit経路が高速、中央値比`2.552010 > 1.0`のため採用ゲートPASS。

モンキーパッチ経由の既報中央値2.569904倍との差は約0.70%であり、大きな乖離は
ない。今回のスクリプトは本番の`use_njit_dda`引数を直接使用し、DDA単体の
計時用モンキーパッチは行っていない。

pytest全体は342 passed、8 failed、2 warnings。8件は既知の並列テストと完全に
一致し、すべて`os.sysconf("SC_SEM_NSEMS_MAX")`に対する
`PermissionError: [Errno 1] Operation not permitted`だった。

### residual_Bの内訳プロファイル（Codex実装・Claude独立検証）

「置換後の最大成分はresidual_Bである。内訳は今回プロファイルしていない」
（上記「### 修正後」節）を受けた別タスクとして、`run_batch_with_tally`
（use_njit_dda=True経路）の内部処理を`docs/speedup_baseline/residual_b_breakdown_profile.py`
で直接呼び出し、ステップごとに`time.perf_counter()`計測した
（本番コードは変更していない。読み取り専用でimportして中身を再構成しただけ）。
water60_free、N=200,000、resolution=2cm、batch_size=200,000、n_chunks=8、
seed=20260728、8反復（Codex実行1回・Claude独立再実行1回、いずれも定性的に一致）。

Claude独立再実行の中央値（秒）:

| ステップ | 中央値 |
| --- | ---: |
| chunk_plan + buffer確保 | 0.002658 |
| transport（`_run_batch_scalar_tally`、njit並列） | 0.059517 |
| overflow検査 | 0.000025 |
| concatenate（5配列） | 0.002063 |
| **重み計算**（`mu_en_rho`×2材料＋`h_star_10_per_fluence`＋`mat_names`構築） | **0.069011** |
| （参考）DDA accumulator | 0.069175 |

`non_DDA_including_transport`（=E2E_B − DDA_in_B相当の再構成値）median 0.134957s
は既報residual_B median 0.128792sと+4.79%で近い（測定境界がモンキーパッチなし・
直接ステップ呼び出しに変わったことによる差として妥当な範囲）。

**次のボトルネック候補は重み計算であり、transportではない**（重み計算median
0.069s > transport median 0.060s、Codex実行・Claude再実行の両方で同じ順序）。

原因候補（Claude確認済み、未修正）: `kernel.py:1095`の
`mat_names = np.array(tables.material_names, dtype=object)[seg_mat_all]`が、
njit輸送本体から直接出てくる整数材料コード配列`seg_mat_all`（int64）を
わざわざオブジェクト配列の文字列へ変換してから`material_groups`に渡している。
`material_groups`（`chatcarlo/materials.py:541`）自身のdocstringには
「整数コード配列も受け付ける——object配列のtolist()＋Python文字列比較を
避けるため」という高速経路が既に実装済みだが、`kernel.py`側がこの高速経路を
使わず、わざわざ低速経路（object配列のtolist()+set()+文字列等価比較）を
踏んでいる。`seg_mat_all`を変換せずそのまま`material_groups`へ渡す
（ループ内で名前が必要なら`material_code_name(code)`を使う、既存の高速経路と
同じパターン）ことで、この変換・低速経路コストを削減できる可能性がある
——ただし本タスクはプロファイルの範囲であり、この修正は実装していない。

生ログ: `docs/speedup_baseline/residual_b_breakdown_result.txt`（Codex実行分）。
Claude独立再実行はターミナル出力のみで別途保存していないが、中央値は上表の
とおりCodex実行分（transport 0.057s、重み計算 0.069s）と一致した。
`.venv/bin/python -m pytest tests/test_kernel.py -q`は14 passed（Claude自身で再実行し確認）。

### material_groupsを経由しないシーンローカル材料コード高速経路

承認済み計画
`docs/ai/plans/2026-07-29-material-groups-fast-path-kernel.md`に従い、
`kernel._compute_tally_weights`を追加した。`seg_mat_all`をobject材料名配列へ
変換せず、シーンローカルコードを直接グループ化する。旧実装との浮動小数点
演算順（`mu_en_rho * density`を配列へ格納後、energyを乗算）は維持した。
実装開始時HEADは`73aa899afac2779af3ae6c911aeaa6c54549df86`。

別subprocess・別worktreeでの正しさ比較（water60_free、N=200,000、
resolution=2cm）は、1バッチ（200,000）と複数・端数バッチ（60k×3+20k）
の双方でkerma/H\*(10)グリッドがshape・dtype・値とも完全一致
（最大絶対差0）。water/lead/airの3材料条件も完全一致し、実区間数は
water 541,510、lead 141,815、air 266,753で全材料が実際に通過された。

固定された同一segment入力を使う重み計算AB/BA交互8反復では:

- 旧実装: min 0.070075625s、median 0.071086000s、max 0.072513500s
- 新実装: min 0.048440584s、median 0.048789730s、max 0.049838000s
- 反復ごとの旧/新比:
  `1.440597, 1.472879, 1.442041, 1.411566, 1.433409, 1.441808, 1.496958, 1.488239`
- 比の中央値1.441925で、事前登録した性能ゲート1.20以上をPASS。
- 探索的成分中央値はobject配列構築0.003868834s、旧set grouping
  0.007085834s、新`np.unique` grouping 0.004951041s、
  補間＋mask fill 0.059018604s（`np.unique`はgrouping側に含む）。

旧/new worktreeのend-to-end AB/BA交互8反復（同一arm内でDDAを個別計測）:

| 指標 | min (s) | median (s) | max (s) |
| --- | ---: | ---: | ---: |
| old E2E | 0.202969000 | 0.206909333 | 0.215715375 |
| old DDA_in | 0.068936250 | 0.069432959 | 0.070747792 |
| old residual_B | 0.133368166 | 0.137453062 | 0.146242750 |
| new E2E | 0.183021500 | 0.189902000 | 0.193815875 |
| new DDA_in | 0.068843000 | 0.069508146 | 0.072072250 |
| new residual_B | 0.113639083 | 0.120184792 | 0.124855084 |

反復ごとのresidual_B旧/新比は
`1.172142, 1.141848, 1.193820, 1.083775, 1.204509, 1.078530, 1.207583, 1.171300`、
E2E旧/新比は
`1.114813, 1.088613, 1.117906, 1.050371, 1.129518, 1.050951, 1.112386, 1.112991`。
`tests/test_kernel.py`は23 passed。

#### Claude独立検証

Codexは実装完了時、`pytest tests/ -q`がサンドボックス側で3回とも20%・95件通過
表示のまま終了サマリを出さず終了し、「既知8件以外にfailureがないか確認できて
いない」と正直に報告（機械的事実と原因説明を分けて自己申告する姿勢どおり）。
Claudeが同一コマンドを自分の環境で実行したところ、**359 passed, 0 failed**
（既知の8件のsandbox制約由来failureも含めて今回は1件も再現しなかった）で
完走し、この未確認事項を解消した。

そのほか以下をすべてClaude自身の環境で再実行し、Codexの報告と定性的に一致
することを確認した:
- `tests/test_kernel.py -v`: 23 passed（自分で再実行して確認）。
- `git diff --stat`: `chatcarlo/kernel.py`・`tests/test_kernel.py`・
  本ファイルの3件のみ変更、対象範囲外ファイルは無傷。
- `kernel.py`の差分をコードレビュー: `_compute_tally_weights`が
  `material_groups`を一切参照していないこと、ndim/dtype/長さ/範囲外コード
  の契約チェックが計画どおり実装されていることを確認。
- `material_weights_worktree_compare.py --mode correctness`: 自分で再実行し、
  1バッチ・端数バッチ・3材料の全条件でkerma/H\*(10)最大絶対差0、3材料の
  区間数もwater 541,510／lead 141,815／air 266,753とCodex報告値と完全一致。
- `material_weights_worktree_compare.py --mode timing`: 自分で再実行し、
  8反復全てでE2E比・residual_B比とも新実装が高速（E2E比中央値約1.146、
  residual_B比中央値約1.219）。個別の秒数はCodex実行分と多少異なるが
  （測定ノイズの範囲、両実行とも同じ結論）、8反復全勝という定性的結論は一致。
- `residual_b_breakdown_profile.py`（重み計算単体のAB/BA、8反復）: 自分で
  再実行し、旧median 0.069360020s・新median 0.047984042s・比の中央値
  1.445952でCodex報告値（1.441925）と近い。性能ゲート（>=1.20）PASSを
  独立に確認。
- `dda_njit_e2e_benchmark --mode correctness`: PASS（全差0）を確認。

受入条件は全項目Claude独立検証済みでPASS。計画の状態は`implemented`とする。
