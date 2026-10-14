# P5a CPU核の実装・検査（2026-10-14）

承認済み計画 `docs/ai/plans/2026-10-14-mv-p5a-cpu-kernel.md` に従う。
**P5aは合格。固定した検査はすべて合格し、P5bへ進める（`p5a_summary.json` の `p5b_ready: true`）。**
Codex担当のA1・B1・B2・B4〜B7はCodexが実行し、Claudeがリポジトリの外の作業場所で走らせ直して同じ結果を確かめた。
B3・C1〜C4とC1前のSEM確認はClaudeが実行した（2026-10-14）。

既存ファイルは読み取り専用、作業生成物は `.p5a0_work/p5a/` に保持する。
新規実装は共通の核 `p5_kernel.h`、差し替え可能な乱数 `p5_rng.h`、
CPU入口 `p5_cpu.cpp`、入力・検査・保存・解析の入口 `p5a.py`、
速い検査 `test_p5a.py`。GPU入口はP5bの対象。

## 機械的な結果

| 検査 | 状態 | 数値 |
|---|---|---|
| A1 | pass | 153,239,591個の出力値、不一致0。定義域外18例を両側で拒否（36回）。 |
| B1 | pass | `history_rng=True`、種20291950、20,000履歴。10出力・75,339要素、不一致0。 |
| B2 | pass | 入れ直しなし、種20291951、20,000履歴。10出力・75,339要素、不一致0。 |
| B4 | pass | 3種類ともコンパイル・実行でき、不一致を検出。最初の不一致履歴は(a)9、(b)0、(c)0（0起点）。 |
| B5 | pass | B1の診断を有効にしても10出力の不一致0。記録2,000,000ステップ、記録容量を超えたステップ11,176,915。 |
| B6 | pass | 2条件とも10出力が完全一致。容量2: `counts[5:7]=[0,161]`, 未解決350.18953094694643 MeV。上限2ステップ: `[161,0]`, 未解決341.59807523840266 MeV。 |
| B7 | pass | Philox、種20291960、2,000履歴・4バッチ。全実行と分割・中断・再開で、バッチごとの10出力の不一致0。 |

B1・B2の健全性: 数値出力はすべて有限、`violations=[0,0,0]`、
`counts[5:7]=[0,0]`、`brems[2:4]=[0,0]`、`totals[3]=0`。
最大のスタック深さは両方とも7（容量64）。B6は失敗を注入した検査なので、
上の失敗件数と未解決エネルギーが参照と一致することを判定した。

A1の内訳: collision 102,105点、hard 102,105点×2、coefficients
102,149点×4、table_rates 101,455点×2。粗い集計のscore_segment/localは
100,457/100,452例、細かい集計は101,807/101,802例で、全ビンを比較。
gs_mu_cachedとsample_Wはそれぞれ10種×1,000回の戻り値と、各種の次の乱数を比較した。
電子の元のESTAR節点、表の節点の直前・直後、閾値、ビン境界を含む。
対数で一様な点は比率を1回計算し、その後は乗算で生成した。

意図的な誤りの最初の差（単独履歴での出力名と添字）:

| 誤り | 履歴 | 出力・添字 | P4 | C |
|---|---:|---|---:|---:|
| (a) 制動放射の光子のエネルギー×(1+1e-12) | 9 | batches[0,138] | 0.24213149922601082 | 0.2421314992260359 |
| (b) 電子の付与を隣のビンへ | 0 | batches[0,9] | 0.005210279901761705 | 0 |
| (c) 光子の相互作用の選択で余分な乱数 | 0 | batches[0,9] | 0.005210279901761705 | 0 |

各単独履歴のC側の診断は `B4_fault1_first.npz`（1,086ステップ）、
`B4_fault2_first.npz`（451）、`B4_fault3_first.npz`（999）に保存。
粒子の種類、エネルギー、xyz、乱数の消費回数、ステップ番号を含み、
この3例の記録のあふれは0。B5の全履歴記録は容量を超える部分を数えるが、
診断の有効・無効は輸送出力を変えない。通常版で不一致が出た場合は、
最初の不一致履歴・出力名・添字・両側の値とC側の記録を保存して停止し、
最初の違うステップと原因はClaudeが調べる。

## 再現条件と成果物

- 手元: Darwin arm64、Python 3.11.13、NumPy 2.4.6、Apple clang 21.0.0。
- CPU指定: `-O2 -ffp-contract=off -std=c++11 -shared -fPIC`。
  乱数・故障・容量の指定はそれぞれ `P5_RNG_MT`/`P5_RNG_PHILOX`、
  `P5_TRANSPORT_FAULT`、`P5_STACK_CAP`。fast-mathなし。
- P5a0の共通ヘッダーは読み取り専用でincludeし、Philox、実数変換、interp、quantitiesを再利用。
- `.p5a0_work/p5a/gate.json` と `chain0.json` に、検査結果、入力配列の形とハッシュ、
  実際のソースのSHA-256、コンパイラとビルド指定、出力のハッシュを保存。
- B1/B2の両側の出力、故障版の単独履歴の診断、B7のバッチは同じ作業ディレクトリ。
  診断・入力・npzを追跡対象にはしない。削除処理は追加していない。
- C1は共通の種と全体の履歴番号を使い、part 0が0〜49、part 1が50〜99。
  保存は親のPython入口が行う。ソース・ビルド指定・入力配列・乱数・種・履歴範囲を
  照合し、保存済みバッチの欠け、重複、形や型の不整合を拒否する。
  途中のファイルは完成したバッチとして読まず、書き終わってから移す。
  メタデータの更新は短いロックで直列化する。計算用の子プロセスを自動では作らない。
- 新規テストには、B3未実行時の本計算の拒否、保存構成の変更、保存済みバッチの欠落、
  不正な出力の形、独立な両側のSEMとビン間の共分散の保持も含む。

## Claudeが実行した検査（2026-10-14）

作業場所 `~/.cache/chatcarlo/p5/p5a/`（`gate.json`、`chain0.json`、`sem_check.json`、`analysis.json`）。

### B3: 保存済みのP3rの本番のバッチの再現 — pass

P3rの本番 `p3r_proto` のバッチ0（種 4067114529）とバッチ2（種 3112960579）、各10万履歴を、Cの核（MT、入れ直しなし）で走らせ直した。
保存されている `coarse`・`fine`・`totals`・`maxima`・`violations`・`counts`・`maxdepth`・`nspec`・`steps`・`brems` が**すべてバイト単位で一致**（例: バッチ0の `coarse` のSHA-256は両側とも `662ec439…`）。
先に、表（電子・GS・光子・NIST）とスペクトルのハッシュが、本番の時と同じであることを確かめた。

### C1の前のSEMの確認 — pass

保存済みのP3rの側の深い5ブロックのSEMは 0.089 / 0.078 / 0.074 / 0.064 / 0.064 %（`Dref` 比）。差のSEMの見込みは 0.090〜0.126 %で、止める基準の0.23%を下回った。

### C1・C2: 新しい乱数（Philox）での1,000万履歴 — pass

種 20291960、履歴番号 0〜9,999,999、100バッチ。2プロセス（各50バッチ）で約40分。相手は保存済みのP3rの本番（MT、1,000万履歴）。`Dref` = 9.798535080485913×10⁻¹² Gy cm²。

| 深さ (cm) | 差（Philox − MT、`Dref` 比 %） | 差のSEM (%) | 判定 |
|---|---:|---:|---|
| 5〜10 | +0.080 | 0.133 | pass |
| 10〜15 | +0.139 | 0.114 | pass |
| 15〜20 | −0.003 | 0.105 | pass |
| 20〜25 | +0.131 | 0.101 | pass |
| 25〜30 | −0.015 | 0.094 | pass |

許容は差の絶対値が1%以内、かつ差のSEMが0.33%以下。どの差もSEMの1.3倍以内。
C2（健全性）: 出力はすべて有限、`violations` 0、ステップ上限・スタックのあふれ0、未解決のエネルギー0。

### C3: 報告（判定しない）

- 相互作用の数の差（Philox − MT、1履歴あたり、±SEM）: Compton +0.0045 ± 0.0025、光電 +0.00018 ± 0.00023、対生成 −0.000094 ± 0.000067、制動放射 −0.000013 ± 0.00014、Møller +0.0005 ± 0.0044、飛行中の消滅 −0.000007 ± 0.000016。いちばん大きいのはComptonの1.8σ。
- ビルドアップ（0〜2 cm）: −0.33 ± 0.15 %（2.2σ）。**判定の対象外**。深い5ブロックの差はすべて1.4σ以内なので、新しい乱数に系統的なずれがある兆しとは読まないが、ビルドアップは独立な量で1回だけの観測。乱数の違いのほかに、両側が別の乱数列であることによる偶然で説明できる大きさ（2.2σは、およそ3%の確率で起こる）。
- 乱数の方式による系統のずれは、鎖0（MTの経路での完全一致）と、この鎖1の結果の両方から見て、検出できる大きさでは見えない。

### C4: 時間（報告）

同じ機械（M3）、1プロセス、5万履歴、同じ設定: **Cの核（Philox）164 µs/履歴、P4の核（Numba、MT）284 µs/履歴**。CPUのままで約1.7倍速い。理由は調べていない（固定長のスタック、ループの中で配列を作らない、などが候補）。

## 最終テスト

指定のgateとchain0（`--skip-saved`）は、最終ソースで終了コード0。
`PYTHONPATH=. .venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5a.py -q` は
**16 passed in 17.53s**、終了コード0。
`PYTHONPATH=. .venv/bin/python -m pytest prototypes/mv_electron/tests -q` は
**670 passed, 2 skipped in 876.50s (0:14:36)**、終了コード0。
基準の654 passed / 2 skippedに新規16件が加わった件数。
`git diff --check` は終了コード0。開始時に存在した追跡対象・非ignoreファイルの
内容の変化は0件。新規ファイルは許可された7ファイルだけで、うち6ファイルが
`git status --short` に出る。`p5a_summary.json`（小さい数値要約）は既存の
`.gitignore` 71行の `docs/validation/mv/p5/*.json` に該当するため、
ファイルは存在するがstatusには出ない。既存のignore規則は編集していない。

途中の全体テストは、停止・再開の受入条件に関する追加検査を最終版にそろえるために
中断した（134 passed, 1 skipped、終了コード2）。これは最終回帰結果には数えない。
入力束のスカラー変換とPython入口の構文の実装中のエラーは、比較開始前に解消した。
通常版のA1・鎖0に不一致はなかった。

## 付録: 固定条件で到達する関数・枝の一覧（実装前の整理）

対象は計画の `run_coupled(10., N, seed, .05, .025, 30.,
*run_p3s.inputs('elsepa'), annihilate_in_flight=True,
spectrum_edges=edges, spectrum_cum=cum, brems_mode=2,
score_kerma=False, score_pair=False, **run_p3s.rule_kwargs('own'))`。
以下の「写す」は到達可能性による予定範囲であり、実装済みを意味しない。

| 部分・関数 | 写す枝と理由 |
|---|---|
| `run_coupled`, `mass_geometry`, `expand_regions` | Python側の入力構築。単層、密度1、厚さ30 cm、領域なし、粗0.2 cm・細0.05 cmに固定。表はPythonで読み検証し、C連続float64として渡す。 |
| `_run_coupled` の履歴開始 | Mohanの累積分布の選択、ビン内一様抽出、`nspec`、平行光子の状態作成。通常のMT種初期化と `history_rng` の履歴別入れ直し、Philoxの全体履歴番号に対応。 |
| `photon_state`, `boundary` | 光子の状態、正方向・負方向・ゼロ方向、表面の同着・脱出。単層内の境界で終了する。 |
| `coefficients`, `kn_total`, `log_interp`, `pair_interp` | Compton・光電・原子核対生成・電子場対生成。正の表の対数補間、対生成閾値以下のゼロ、閾値直上の外挿、端点。 |
| `compton`, `rotate` | 組成選択の両枝、棄却の反復、散乱・反跳、反跳運動量ゼロ、方向の極付近と通常枝。 |
| `pair`, `pair_small_total` | 2.1 MeV以下の一様抽出と上の棄却の全枝、3乱数の最大値、閾値・遮蔽の棄却、両符号の割当て、スタックへの順番。 |
| `initial`, `electron_initial`, 両方の `energy_init` | 23成分の状態。エネルギー・散乱・硬衝突・制動放射の初期残距離。カットオフ未満の二次粒子でも元の順番で乱数を消費する。 |
| `collision`, `moller_params`, `hard`, `primitive0`, `primitive1` | 衝突阻止能、20 keV以下の硬衝突ゼロ、直上の局所積分、通常の積分。 |
| `table_rates`, `bracket`, `gs_mu_cached` | ELSEPAの独立エネルギー格子を含むレート補間、GSのゼロ衝突と条件付き分布、全補間枝。 |
| `rule_value`, `scattering_hinge`, `hinge_transition` | 上端を含む20 keV・1 MeVの段階規則、旧ステップの散乱抽出後に新しいヒンジを作る順番。 |
| 荷電粒子ループ | カットオフと終端テール、評価エネルギーのキャッシュ、境界→エネルギー→散乱→硬衝突→制動放射→消滅の同着優先順位。実エネルギー終端、非有限・負のステップ、ステップ上限。 |
| `sample_W`, `moller_directions` | Møllerの棄却反復、親子の方向、親の再初期化、二次粒子の初期化、あふれ時の両娘の未解決エネルギー。 |
| `quantities` | P5a0のヘッダーを読み取り専用でincludeして再利用。カットオフ以下、最初の区間、通常区間。表の生成はC側で行わない。 |
| `shape`, `sample_k`, `direction`, `new_remaining`, `advance_remaining` | 20 MeV以下のH/Oの全遮蔽枝、制動放射の棄却と包絡超過、T超過、残距離、放出光子を運ぶモード2、親の中断・復帰とスタックあふれ。 |
| `annihilation_rate`, `sigma`, `sample`, `zeta_min`, `photons`, `annihilation` | 陽電子の飛行中の消滅、棄却と角度のクリップ、カットオフ後の静止消滅、陽電子の脱出と静止質量収支、スタックあふれ。 |
| `local`, `score_segment`, `tracked_local`, `tracked_segment`, `touched_sum`, `add_clear` | 粗細の付与、ゼロ損失・短い線分・順逆方向・ビン境界・端のクリップ、前から順の総和、100群への同じ順の加算とクリア。 |
| 履歴の終了 | `totals`, `maxima`, `violations`, `counts`, `maxdepth`, `step_batches`, `brems` の全更新。失敗時の現在粒子とスタックの未解決エネルギー。 |

| 写さない部分 | 固定条件で到達しない理由 |
|---|---|
| `source_state`, 陽電子・荷電粒子・Fano線源、`sample_field`、線源位置オフセットの変更 | 線源は平行光子スペクトル、表面の原点に固定。 |
| `mass_coordinate`, `mass_local`, `mass_segment`、層間の光子再飛行、Fano故障 | 層・質量ビンなし。密度は1。境界を跨ぐと板から脱出する。 |
| `elastic`, `elastic_rates`, `single_cached`, `table_single` | `backend_rates` はELSEPA、`mode=0` に固定。単一散乱や解析DCSの枝は選ばれない。 |
| `stopping`, `restricted_stopping` | 核は `collision-hard[1]+quantities[2]` を直接計算。同じ式は写すが未呼出し関数は追加しない。 |
| `transfer_coefficient`, `kn_transfer`, NIST μenの飛程集計 | `score_kerma=False`。 |
| 対生成由来線量、横方向集計、`score_box_bins`, `score_box_point`, `deposit_segment`, `deposit_point` の領域・記録枝 | `score_pair=False`, `score_lateral=False`、領域・付与記録なし。領域なしの呼出しは出力に作用しない。 |
| `brems_mode=1`, `disable_brems=True`, 無限媒質、散乱・硬衝突オフ | `brems_mode=2`, `disable_brems=False`, 有限板、`mode=0`, `hard_on=True` に固定。 |
| `photoelectric` の補助関数 | 光電分岐は直接電子を初期化し、この補助関数は呼ばない。 |
| `event_order`, 各 `samples`・積分・表生成・検査用関数、飛行中消滅の `differential`, `primitive`, `cdf` | 輸送から呼ばれない。事象優先順位は核内の比較の並びで保持する。 |

対象出力は計画の10個だけ。照射野・層・GPU入口・物理変更は含めない。
