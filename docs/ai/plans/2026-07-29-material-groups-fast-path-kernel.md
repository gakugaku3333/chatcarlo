# 計画: kernel.pyの重み計算からobject材料名配列の構築を除く

- 状態: implemented
- 作成者: claude
- 実装担当: codex
- 日付: 2026-07-29
- 改訂1: 2026-07-29（Codex read-onlyレビューで「対象範囲」節の当初案が
  安全でないと判明したため全面修正。旧版は「material_groupsの整数コード
  高速経路へ切り替える」だったが、以下の理由で撤回した）
- 改訂2: 2026-07-29（Codex read-onlyレビュー2回目。正しさ設計の懸念は
  解消と判定されたが、(1)正しさ比較の同一プロセスimport切替方式が
  `sys.modules`・Numba dispatcherキャッシュ混在で信頼できない、
  (2)修正前後のE2E AB/BA比較を実行する手段が計画内になかった、
  (3)材料コード入力契約が狭すぎる、の3点を反映した）

## 改訂の経緯（撤回した当初案とその理由）

当初案は`kernel.py`の`seg_mat_all`（int64の材料コード配列）を
`chatcarlo/materials.py`の`material_groups`へ直接渡し、その整数コード
高速経路（`names.dtype.kind in "iu"`分岐、`material_code_name(code)`で
名前へ戻す）を使う案だった。Codexのread-onlyレビューでこれは**正しさの
問題**（性能問題ではない）と指摘された:

- `kernel.py`の`seg_mat_all`は`SceneMaterialTables.material_names`への
  **シーンローカルな0起点index**である（`kernel.py`の`SceneMaterialTables`
  docstring「材料コードはmaterial_namesのindex（0起点）」参照）。
- 一方`materials.py`の整数コード高速経路が前提とする`material_code()`/
  `material_code_name()`は、**プロセス全体で使い回すグローバルな
  インターン表**（`_MATERIAL_CODES`、挿入順で採番、シーン間で共有）である。
- この2つのコード体系は無関係であり、`seg_mat_all`をそのまま
  `material_groups`へ渡すと、インターン表の状態（それまでにどの材料が
  どの順で登録されたか、他のシーン処理から漏れてきた状態を含む）次第で
  `IndexError`になるか、**別材料の名前へ誤変換され、静かに間違った
  μen/ρ・H\*(10)係数で線量を計算する**（silent data corruption）。
  spy（呼び出し引数のdtypeが`"iu"`であることの確認）ではこの不一致を
  検出できない——dtypeは正しく整数でも、値の意味が違うため。

本改訂では、`material_groups`の整数コード高速経路は使わない。代わりに、
`seg_mat_all`（シーンローカルコード）を**そのままキーにして直接グループ化**
する（`material_groups`を経由しない）。これによりコード体系の変換自体を
なくし、object配列構築のコストだけを除く。

## 目的（1〜3行）

`docs/plan_dda_njit_premise_check.md`「### residual_Bの内訳プロファイル」で、
`kernel.run_batch_with_tally`のresidual_B内で最大の成分が「重み計算」
（transport中央値0.060sを上回るmedian 0.069s）であり、原因候補として
`kernel.py:1095`の`mat_names = np.array(tables.material_names,
dtype=object)[seg_mat_all]`（1M要素規模のobject配列構築＋その後の
文字列ベースグルーピング）を特定した。本計画はこのobject配列構築を除き、
`seg_mat_all`（シーンローカルint64コード）を直接使ってグループ化する
ことで、同じ計算結果をより速く得ることを目指す。

## 対象範囲

- 変更してよい:
  - `chatcarlo/kernel.py`（`run_batch_with_tally`内、重み計算部分のみ）。
    以下の非公開関数を新設し、`run_batch_with_tally`から呼ぶ:

    ```python
    def _compute_tally_weights(tables: SceneMaterialTables,
                                seg_mat_all: np.ndarray,
                                seg_e_all: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray]:
        """seg_mat_all(シーンローカル材料コード, tables.material_namesへの
        0起点index)とseg_e_all(keV)から、(kerma_weights, h10_weights)を返す。

        契約:
        - seg_mat_all.ndim == 1、seg_e_all.ndim == 1（2次元以上はValueError）
        - seg_mat_all.dtype.kind in "iu"（整数dtypeのみ。bool・float・object
          はTypeErrorで拒否する——float配列は`np.unique`までは動いてしまい、
          名前参照の段階で不明瞭な例外になるか暗黙変換される余地があるため、
          事前に明示的に拒否する）
        - len(seg_mat_all) == len(seg_e_all)（不一致はValueError）
        - seg_mat_allの全要素は0 <= code < len(tables.material_names)
          （範囲外はValueError。範囲外を黙ってclampしたり無視したりしない。
          `len(tables.material_names) == 0`で非空のseg_mat_allが来た場合も
          同様にValueError）
        - 空配列（長さ0）はエラーにせず、入力と同じshapeの空配列ペアを返す
          （int32・int64いずれの空配列入力でも動作する）
        - 戻り値は入力と同じ長さの1次元float64配列のペア
          `(kerma_weights, h10_weights)`
        - 入力配列（seg_mat_all・seg_e_all）を変更しない
        - material_groupsは使わない（シーンローカルコードとグローバル
          インターンコードの体系が異なるため、経由すると誤変換の恐れがある
          ——上記「改訂の経緯」参照）。`np.unique(seg_mat_all)`で得た
          コード集合を`tables.material_names[code]`でソートし
          （旧実装`sorted(set(mat_names.tolist()))`と同じ「名前の辞書順」を
          維持する——旧実装の観察可能な処理順との一致を保つため）、
          コードごとの真偽マスクで`mu_en_rho`・`density`を呼ぶ。
        """
    ```

  - `docs/speedup_baseline/residual_b_breakdown_profile.py`
    （`kernel_mod._compute_tally_weights`を直接呼ぶ形に更新する。
    旧実装（object配列構築＋`material_groups`の文字列経路）を比較用に
    再現するローカル関数`_legacy_weight_calculation`をこのベンチマーク
    ファイル内に追加し、A/B比較する——このローカル関数は比較専用で
    本番コードとして扱わない）
  - 新規 `docs/speedup_baseline/material_weights_worktree_compare.py`
    （worktree間の正しさ比較・E2E AB/BA比較を行う親スクリプト。
    `--mode correctness`と`--mode timing`の2モードを持つ。詳細仕様は
    「受入条件」「実装方針」参照）
  - `tests/test_kernel.py`（`_compute_tally_weights`の単体テストと、
    `run_batch_with_tally`を通した多材料グリッド一致テストを追加）
  - `docs/plan_dda_njit_premise_check.md`（本計画の実装結果を追記。
    既存節は書き換えない）
- 変更禁止:
  - `chatcarlo/materials.py`（`material_groups`・`mu_en_rho`・`density`・
    `material_code`・`material_code_name`は一切変更しない。今回は
    `material_groups`を経由しない設計にするため、そもそも触る必要がない）
  - `chatcarlo/tally.py`・`chatcarlo/tally_njit.py`・`chatcarlo/transport.py`
    ・既存の`tests/`ファイル（`tests/test_kernel.py`以外）
  - `docs/speedup_baseline/dda_njit_e2e_benchmark.py`
    （既存のend-to-end A/B計測スクリプトは変更しない）
  - `run_batch_with_tally`の呼び出しシグネチャ・チャンク処理・DDA呼び出し
    部分（重み計算部分以外は一切変更しない）

## 受入条件（検証可能な形で列挙）

- [ ] `_compute_tally_weights`は`material_groups`を呼ばない
      （grepで`material_groups`の参照が`_compute_tally_weights`の
      実装内に存在しないことを確認する）。`np.array(...,
      dtype=object)[seg_mat_all]`のようなobject配列構築も行わない。
- [ ] 契約テスト（`tests/test_kernel.py`、`_compute_tally_weights`単体、
      輸送やnjitを介さない）:
      - 空配列（`seg_mat_all`・`seg_e_all`とも長さ0）: 例外にならず、
        長さ0の`(kerma_weights, h10_weights)`を返す。int32・int64の
        両方の空配列で確認する。
      - `len(seg_mat_all) != len(seg_e_all)`: `ValueError`。
      - `seg_mat_all.ndim == 2`（または`seg_e_all.ndim == 2`）: `ValueError`。
      - `seg_mat_all`がfloat配列（例: `[0.0, 1.0]`）: `TypeError`。
      - `seg_mat_all`がbool配列: `TypeError`。
      - 範囲外コード（負値、`len(tables.material_names)`以上）:
        `ValueError`（黙ってclampしない）。
      - `len(tables.material_names) == 0`で非空の`seg_mat_all`: `ValueError`。
      - 単一材料のみの配列・同一エネルギー: 全要素が同じweightになる。
      - 単一材料のみの配列・異なるエネルギー混在: 旧実装（下記の
        `_legacy_weight_calculation`と同一ロジックのテスト内ローカル
        参照実装）と`np.array_equal`で完全一致（エネルギー依存の
        μen/ρ・H\*(10)係数が単一材料内でも正しく効くことの確認）。
      - 2材料混在（water/air、既存の`make_water_case`相当）: 旧実装と
        `np.array_equal`で完全一致。
      - 3材料混在（water/air/lead等）: 同上、旧実装と完全一致。
      - 同じ材料コードが配列内で非連続に出現するケース（例:
        `[0, 1, 0, 1, 0]`）: 旧実装と完全一致（グループ化がインデックス
        順を壊さないことの確認）。
      - `tables.material_names`の順序を反転したシーン
        （例: `["air", "water"]` vs `["water", "air"]`、コード自体は
        同じでも指す材料が入れ替わるケース）で、コードが指す材料名に
        従って正しくweightが変わることを確認する（コード⇄名前の対応が
        シーンローカルであることの直接的な検証）。
      - `seg_mat_all`のdtypeがint32でも動作し、int64と同じ結果を返す。
      - 戻り値`(kerma_weights, h10_weights)`のshape・dtypeが入力と
        整合すること（`(len(seg_e_all),)`、`float64`）。
      - 呼び出し前後で`seg_mat_all`・`seg_e_all`が変更されていないこと
        （呼び出し前にコピーを取り、呼び出し後に比較する）。
- [ ] 統合レベルの正しさ（bit-exact）: 実装着手時点のHEADコミット
      （実装開始時に`git rev-parse HEAD`で確定し、比較スクリプトの
      出力にそのハッシュを明記する）を`git worktree add`で別ディレクトリへ
      チェックアウトする（ユーザーの作業ツリー・未追跡ファイルには
      一切触れない。`git stash`は使わない）。正しさ比較は**別subprocess**
      で実行する（同一プロセス内で`sys.path`を差し替えて新旧`kernel`
      モジュールを読み込む方式は、`sys.modules`キャッシュとNumba
      dispatcher/キャッシュが新旧で混在しうるため使わない）。
      手順:
      1. 親スクリプトが、旧worktree・新worktreeそれぞれをcwd（または
         `PYTHONPATH`先頭）にした子プロセスを`subprocess.run`で起動する。
      2. 各子プロセスは対象条件（下記）で`run_dose_grid`を実行し、
         結果を一時`.npz`へ保存し、実際に読み込んだ
         `chatcarlo.kernel.__file__`の絶対パスを標準出力に明記する
         （意図したworktreeを読んだことを親スクリプトが検証できるように
         する）。
      3. 親スクリプトが両方の`.npz`を読み込み、`kerma_keV`・
         `h10_track_pSv_cm3`グリッドのshape・dtype・値
         （`np.array_equal`）を比較する。
      条件:
      - water60_free（2材料）、N=200,000、resolution=2cm、1バッチ
        （batch_size=200,000）と複数・端数バッチ（batch_size=60,000、
        60k×3+20k）の両方
      - 材料3種以上を含むシーンを最低1つ追加する（例: water/air/leadの
        平板を挟んだ箱シーン。各材料に最低1本の区間が実際に生じている
        こと——`seg_mat_all`から材料コードごとのsegment countを数え、
        全材料でcount > 0であることをassertする。`n_scatter`は材料別の
        区間有無を証明しないため使わない。鉛を1本も通らない偶然のシーンに
        ならないようにする）
      - 一致しない場合は原因を報告し、この計画の「実装」を完了と
        みなさず停止する。比較完了後、`git worktree remove`で
        旧worktreeを片付ける。
- [ ] 性能改善の確認: `docs/speedup_baseline/residual_b_breakdown_profile.py`
      を更新し、同一プロセス内でAB/BA交互8反復により、旧実装
      （`_legacy_weight_calculation`、ベンチマーク内のローカル再現コード）
      と新実装（`kernel_mod._compute_tally_weights`、本番コード）の
      重み計算単体の時間を比較する。測定の再現性のため、全8反復で
      **同一の`seg_mat_all`・`seg_e_all`**（1回のtransportで生成した
      固定入力）を使い回す（反復ごとにtransportし直さない——重み計算
      だけを比較対象にするため）。各armとも計測前に同一回数のwarm-upを
      行い、戻り値をchecksum等で消費して測定対象が実際に実行されたことを
      保証する。判定基準（事前登録、後から動かさない）:
      - 正しさゲートと性能ゲートは独立: 正しさ（上記bit-exact条件）は
        必須PASSであり、性能が目標未達でも正しさが崩れていい理由には
        ならない。
      - 主判定: 各反復ペア（同一AB/BA試行内の旧実装・新実装）の速度比
        `旧実装時間 / 新実装時間`を8反復分求め、**その中央値が1.20以上**
        であれば性能受入条件PASSとする。
      - 中央値が1.20未満（1.0超も含む）の場合は性能受入条件はFAILと
        報告するが、正しさが確認できていれば「性能改善は限定的だが
        正しさは確認済み」として報告し、**この変更を採用するかどうかは
        ユーザー判断とする**（自動的に実装を差し戻さない）。
      - 個別反復での単発逆転（新実装が旧実装よりわずかに遅い回が1回だけ
        ある等）は、それ単体では性能受入条件のFAIL事由にしない
        （OSスケジューリング・キャッシュ揺らぎ由来のノイズを許容する）。
        ただし8反復全ての速度比・min/median/maxを両実装について報告する。
      - 参考情報として、重み計算をさらに「object配列構築」
        「`set(names.tolist())`によるグルーピング」「`mu_en_rho`・
        `h_star_10_per_fluence`の補間計算」に分けて計測し、20%という
        改善目標がobject配列構築の除去だけで現実的に達成可能かを
        見積もる一次データとして報告する（この分離計測自体は探索的で
        よく、閾値判定の対象にはしない）。`np.unique`の時間は
        「グルーピング」側に含めることを明記する。
- [ ] end-to-end再測定: water60_free条件、`use_njit_dda=True`固定で、
      修正前後（上記と同じ実装着手時HEAD vs 修正後）のend-to-end
      `run_dose_grid`実行時間を、正しさ比較と同じsubprocess方式で
      AB/BA交互8反復により比較する。各armのsubprocess内で、E2E全体の
      時間に加えて`residual_B`（=E2E − DDA_in、`docs/plan_dda_njit_premise_check.md`
      と同じ定義）を同一反復・同一arm内で個別に測定する（過去に報告済みの
      DDA中央値を両armから流用しない——ペア測定にならずノイズ評価が
      弱くなるため）。residual_Bの中央値を報告する（具体的な改善幅は
      事前登録せず観測値をそのまま報告する——重み計算の改善幅が全体の
      residual_Bにそのまま反映されるとは限らないため、ここは探索的な
      報告でよい）。測定境界を以下のとおり固定する:
      - 各subprocessは起動後、まず`N_HISTORIES=2_000`で1回フル経路
        （scene/material table/grid構築＋`run_dose_grid`）を実行して
        Numba JITコンパイルを済ませ、この結果は破棄する（
        `dda_njit_e2e_benchmark.py`の`timing()`と同じウォームアップ規約）。
      - ウォームアップは計測対象タイマーの外側で行う。本測定の
        `time.perf_counter()`はウォームアップ後、`N_HISTORIES=200,000`の
        本番条件でのみ開始する。
      - E2Eタイマーは`make_water_case()`相当のscene/material
        table/grid構築の**外側**から`run_dose_grid`呼び出しの終了まで
        （`dda_njit_e2e_benchmark.py`の`timed_arm`と同じ境界:
        「CLI起動・シーン読込・`bake_scene_materials`等を含まない」）。
      - `DDA_in`は、本測定の`run_dose_grid`呼び出し内部でDDA
        accumulator（`tally_njit.accumulate_track_length_multi_njit`）
        呼び出し1回ごとに`time.perf_counter()`で計測し、同一反復内で
        合算した値を使う（`residual_b_breakdown_profile.py`の
        ステップ計測と同じ手法をE2E測定に組み込む形でよい）。
      - 各反復についてE2E・`DDA_in`・`residual_B = E2E − DDA_in`を
        出力し、旧・新それぞれのmin/median/maxと、反復ごとの
        `residual_B`比またはE2E比を報告する。
- [ ] pytest全体を実行し、既知の8件（`os.sysconf("SC_SEM_NSEMS_MAX")`を
      `PermissionError: [Errno 1] Operation not permitted`として拒否する
      並列テスト、`docs/plan_dda_njit_premise_check.md`に記載済み）**以外の
      failureが1件でもあれば停止**する。

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_kernel.py -v
.venv/bin/python -m docs.speedup_baseline.dda_njit_e2e_benchmark --mode correctness
.venv/bin/python -m docs.speedup_baseline.residual_b_breakdown_profile
.venv/bin/python -m docs.speedup_baseline.material_weights_worktree_compare --mode correctness
.venv/bin/python -m docs.speedup_baseline.material_weights_worktree_compare --mode timing
```

## 実装方針

- `material_groups`は経由しない。理由は「改訂の経緯」節のとおり、
  シーンローカルコードとグローバルインターンコードの体系が異なるため。
- `_compute_tally_weights`は`np.unique(seg_mat_all)`で出現コード集合を求め、
  `tables.material_names[code]`で名前を引いてソートし（旧実装の
  `sorted(set(mat_names.tolist()))`と同じ「名前の辞書順」を維持——
  レイリー元素抽選等、材料処理順に依存する乱数消費がある箇所と処理順を
  揃えるため、CLAUDE.mdの`material_groups`に関する既存の注意書きと同じ
  理由）、コードごとの真偽マスクで`mu_en_rho(name, seg_e_all[mask]) *
  density(name)`を計算する。
- 統合レベルの正しさ・E2E比較は`git worktree add`で実装着手時HEADを
  別ディレクトリにチェックアウトし、旧・新それぞれ**別subprocess**で
  実行して結果を`.npz`へ保存、親スクリプトが比較する（`git stash`は
  ユーザーの未追跡ファイルを巻き込むため使わない。同一プロセス内での
  `sys.path`差し替えは`sys.modules`・Numbaキャッシュの混在リスクがあるため
  使わない）。各子プロセスは実際に読み込んだ`chatcarlo.kernel.__file__`を
  出力し、意図したworktreeを読んだことを親スクリプトが検証する。
  worktreeは比較完了後に`git worktree remove`で片付ける。
- ベンチマークスクリプトの「旧実装再現コード」（`_legacy_weight_calculation`）
  は比較目的のためだけの一時的なローカル関数とし、本番コードへは持ち込まない。

## 書かなかったこと（スコープ外を明示）

- `material_groups`自体のアルゴリズム変更（今回は経由しないため対象外）
- transport.py（逐次numpy輸送経路）側で同様の重み計算がある場合の対応
  （run_batch_with_tallyのみが対象）
- `mu_en_rho`・`h_star_10_per_fluence`のPCHIP/log-log補間自体の高速化
  （今回は分離計測で「補間計算がどの程度を占めるか」を参考報告するに
  とどめ、補間自体の最適化は別タスクとする）
- chest_room等cylinder/sphere込みシーンでのend-to-end測定
  （既存計画と同じくbox専用kernelでは対象外）
- `materials.py`のグローバルインターンコード体系とkernel.pyのシーンローカル
  コード体系を統一する設計変更（今回のバグ根本原因ではあるが、影響範囲が
  broadすぎるため本計画のスコープ外。将来必要なら別計画とする）
