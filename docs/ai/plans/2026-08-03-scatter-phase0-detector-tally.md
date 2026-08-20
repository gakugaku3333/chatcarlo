# 計画: 終端型検出器タリー（散乱線補正プロジェクト Phase 0）

- 状態: implemented（Codex実装→Claude独立検証→ギャップ発見→Codex追加実装→Claude再検証、完了）
- 作成者: claude
- 実装担当: codex
- 日付: 2026-08-03
- 親計画: `mcscatter:docs/plans/2026-08-03-roadmap.md`（状態: approved。2026-08-20に別プロジェクト `~/Projects/mcscatter` へ移設）
- 注記（2026-08-20）: 散乱線補正プロジェクトは本リポジトリから分離したが、**本計画の成果物 `chatcarlo/detector.py` はChatCarloの機能としてここに残る**。移設先ではMC-GPU画像タリーの独立照合手段として使う。CLAUDE.md「Scatter correction moved out」参照。

## 実施結果（Claudeの独立検証、2026-08-03）

**1回目実装**: `chatcarlo/detector.py`（新規）、`transport.py`/`geometry.py`/`tally.py`の
最小差分（計85行）で完了。差分を精読し、輸送ループの核心（`hit_det`を
`noninteract`/`interact`両マスクから除外する処理、`ds`短縮とgrid積算の順序、
`tally_bbox`/`bbox`の分離、`VoxelGrid.end_batch`の共通ヘルパー化）が計画どおり
正しいことを確認。全393テスト通過（既存382＋新規11、回帰なし）。

**1回目検証で発覚したギャップ**: 受入条件は約30項目あったが永続テストは11件のみで、
特に**D-1（検出器なし時のビット一致、設計全体の絶対制約）とD-3（kernel併用エラー）が
無テスト**だった。E-2の2σ判定ロジックがスクリプトに未実装（Codexの報告文の判定結論は
生データから独立再計算し数値的には正しいと確認したが、再現可能な形で残っていなかった）。
ミューテーション検証「7件検出」は自己申告のみで証跡なし。

**2回目実装（ギャップ埋め）**: 欠落していた9カテゴリの受入条件に対応するテスト10件を追加
（`test_detector_none_is_bit_identical_for_low_and_high_level_transport`ほか）。
STPRスクリプトに`e2_judgments()`（結合SEMの2σ判定、`_two_sigma_status`）を追加。
7件のミューテーションを実際に注入・復元した記録を本ファイル下部に追記。

**2回目検証**: 全403テスト通過（並列を含む。Codex環境のサンドボックス制約
`SC_SEM_NSEMS_MAX`はこちらの環境では発生せず、実際に`n_workers=2`の実マルチプロセス
（実pickle化込み）で検出器タリーが正しく動作することを別途確認した。
`_InlineProcessPool`によるworkers=2テストはロジック検証として妥当だが実pickle化は
検証していなかったため、この点は自分で追加検証した）。B-1のBeer-Lambert照合は
`chatcarlo.materials.linear_mu`から取得しており手打ちでない、E-2判定ロジックは
既存CSVに対して実行し独自の手計算と数値まで完全一致することを確認。
**E-2判定CSV（`stpr_water_slab_judgments.csv`）はロジック追加後に一度も生成されて
いなかった**ため、既存STPR CSVから軽量に（MC再実行なしで）自分で生成した。

**最終結果**: STPR(0cm)=0.00557–0.00719（<0.01合格）、STPR(20cm)=3.426–3.502
（2.0–8.0合格）、E-2a単調性は60/80kVの25→30cm遷移のみ判定保留・他は確認、
E-2b kV順序は20cm以上の**全比較**が2σ未満で判定保留（Codexの報告文「一部」は
過小表現だったが、悪い方向への誤りではない）。E-3散乱内訳の逆転を確認。
変更禁止範囲（`physics.py`/`materials.py`/`kernel.py`/`scene.py`/`__main__.py`）は
一貫して無傷。

## 目的

グリッドなし2次元X線画像の散乱線補正（Virtual Grid的処理のMC版）の**唯一の観測窓**として、
検出器平面に到達した光子を一次線/散乱線を弁別してピクセル毎に集計するタリーを実装する。
これがなければ後続Phase（厚さ逆算・2.5Dモデル・減算）の正しさを検証する手段が存在しない。

背景・Phase分割の根拠は親計画を参照。**本計画は補正アルゴリズム自体を一切実装しない。**

## 前提（実装者はここを読んでから着手すること）

1. **既存の輸送ループの構造**（`chatcarlo/transport.py` の `transport_photons`）:
   各反復で `idx = np.where(alive)[0]` → `t_boundary, escape = geometry.next_boundary(o, d)` →
   `will_interact = tau[idx] < tau_to_boundary` → `ds = np.where(will_interact, tau[idx]/mu_safe, t_boundary)` →
   `ends = o + d*ds[:,None]` → **gridへのtrack-length積算** → `pos[idx] = ends` →
   `noninteract = ~will_interact` 側の `tau` 減算とepsilon移動 → `interact = will_interact` 側の
   相互作用処理、の順。**`will_interact`/`noninteract`/`interact` は `idx` に対する
   ローカル配列であり、グローバル添字は `idx[...]` で取る**（下記の落とし穴を参照）。
2. **`n_scatter` は「散乱次数」ではなく「相互作用回数」**。光電吸収の分岐でも
   `n_scatter[photo_idx] += 1` が実行される。したがって散乱次数の分類には使えない。
   **新たに `n_compton_rayleigh`（int配列）と `had_fluorescence`（bool配列）を追跡する。**
3. **`escaped` は世界境界に達した場合だけ真**。`detected` を独立の終端事象として追加し、
   `absorbed` / `detected` / `escaped` を排反にする。
4. **世界境界は物体bbox＋`bbox_margin_cm`（既定50cm）で決まる**（`Geometry._compute_bbox`）。
   物体から50cm以上離れた検出器には光子が到達する前に `escaped` で終端される。
   **かつ `run_transport` は同じbboxから `VoxelGrid` を作る**ので、bboxを素朴に拡張すると
   線量グリッドが巨大な空気領域まで広がりメモリ・時間が跳ねる。両方を同時に解決する
   設計をD-2で固定した。
5. **統計機構はスナップショット差分方式**（`VoxelGrid.end_batch` のdocstring参照）。
   `total` を一切書き換えないことが「統計ON/OFFでtotalがビット一致」の根拠。
6. **`VoxelGrid.end_batch` は4つのことを同時に行っている**:
   (a) 初回だけ `_kerma_prev`/`_h10_prev` を**遅延確保**、(b) `np.copyto` でprevバッファを
   **再利用**、(c) kermaとH*(10)の寄与有無を**ORして** `n_batches_hit` を**一度だけ**増やす、
   (d) `n_batches`/`n_histories` を**一度だけ**増やす。
   **「共通ヘルパーを2回呼ぶ」だけでは (c)(d) が壊れる**（二重加算またはOR判定の喪失）。
   分割方法はD-6で固定した。
7. **既存のエネルギー保存テストの許容差は相対 `1e-9`**（`tests/test_fluorescence.py`）。
   本計画もこれに合わせる（初版の `1e-12` は非現実的だった。B-3参照）。
8. **scene validatorは `geometry` に1個以上の物体を要求する**（`chatcarlo/scene.py`）。
   「厚さ0cmの水スラブ」はそのままでは表現できない。D-7で扱いを固定した。

## 対象範囲

- 変更してよい:
  - `chatcarlo/detector.py`（**新規**。`DetectorPlane` と `DetectorTally`）
  - `chatcarlo/transport.py`（検出器判定の組み込み、`BatchResult` 拡張、
    `_run_batches`・`run_transport`・並列worker経路への受け渡しと集約、
    グリッドbboxの分離、kernel併用エラー）
  - `chatcarlo/tally.py`（**モーメント積算の共通小部品への括り出しのみ**。
    推定式 `_batch_variance`/`standard_error`/`relative_error` の**数式は変更しない**）
  - `chatcarlo/geometry.py`（世界bboxへの検出器の反映。D-2）
  - `tests/test_detector_tally.py`（新規）、`tests/test_detector_geometry.py`（新規）
  - `scripts/stpr_water_slab.py`（新規）
  - `docs/plan_scatter_correction_feasibility.md`（Phase 0の実施結果の追記のみ。このファイルは2026-08-20に `mcscatter:docs/feasibility.md` へ移設済み）

- 変更禁止:
  - `chatcarlo/physics.py`・`chatcarlo/materials.py`・`chatcarlo/spectrum.py`
  - `chatcarlo/data/` 配下のデータファイル
  - `chatcarlo/kernel.py`・`chatcarlo/tally_njit.py`（**併用エラーの判定は
    `transport.py` 側に置く**。kernel本体は触らない）
  - `chatcarlo/tally.py` の推定式そのもの、および `VoxelGrid`/`ScalarMoments` の
    **外部から見た振る舞い**（既存テストがビット一致で通ること）
  - `chatcarlo/scene.py`・`chatcarlo/__main__.py`（**Phase 0はPython API限定**。D-1）
  - `docs/egs5_crosscheck/` 配下の既存結果
  - 既存テストの期待値

## 設計の確定事項

### D-1. 検出器の宣言方法: Python APIのみ

`run_transport(..., detector=DetectorPlane(...))` として渡す。scene.yamlへの
`detector:` セクション追加とCLI配線は**スコープ外**。Phase 0の検証はスクリプトから
叩ければ十分で、スキーマ設計を前倒しすると検証本体より周辺が大きくなる。
scene化はPhase 1bで補正パイプラインの形が見えてから行う。

### D-2. 世界bboxと線量グリッドbboxを分離する

（Codexレビュー指摘への対応。**Codexの代替案とは異なる案を採る。理由も記す**）

- **世界bbox（輸送の脱出境界）には検出器を含める。** 検出器の角4点を
  `Geometry._compute_bbox` の対象に加える。
- **線量グリッド（`VoxelGrid`）のbboxには検出器を含めない。** `run_transport` が
  `VoxelGrid.from_bbox` に渡すbboxを、**geometry由来のbbox（従来と同一）**に固定する。

**Codexは「bboxを拡張せず、脱出予定の飛行だけを検出器まで延長する」案を提示したが採らない。**
その案は検出器までの空隙（エアギャップ、実機で100cm級）を**無相互作用として飛ばす**ことに
なり、(a) 空気の減弱（60 keVで100cm あたり約2%）、(b) 空気による散乱、の両方を落とす。
**エアギャップは散乱除去の物理そのもの**（エアギャップ法）であり、散乱補正プロジェクトの
Phase 0で最初から落とすのは筋が悪い。bboxを拡張して空気を正しく輸送し、線量グリッド側の
bboxだけ分離すれば、Codexが懸念した「巨大な空気ボクセル」は発生しない。

**受入条件A-7で、検出器を足しても線量グリッドの `shape`/`origin_cm` が
ビット一致することを固定する**（この分離が効いていることの証明）。

**実装契約（Codexレビュー2回目の指摘、採用）**: 現行は同一の `Geometry.bbox_min/max` が
世界脱出境界と `VoxelGrid.from_bbox` の**両方**に使われている。既存挙動を壊さずに
分離するため、以下を固定する:

- `Geometry` に **`tally_bbox_min` / `tally_bbox_max` を新設**する（物体だけから計算。
  値は**検出器なしの現行 `bbox_min`/`bbox_max` とビット一致**）。
- `bbox_min`/`bbox_max` は**輸送用world bbox**として、物体bboxと検出器4隅を
  合わせてから既存と同じ `bbox_margin_cm` を付けたものにする。
- **`run_transport` と並列worker（`_run_worker`）の両方**で、`VoxelGrid.from_bbox` には
  **`tally_bbox_*` を使う**（片方だけ直すとworkers>=2で不整合になる）。
- **`Geometry(..., detector_plane=None)` を新設**する。`transport_photons(..., detector_tally=...)`
  だけでは、遠方の検出器より先に世界境界へ達してしまうため、
  **低レベル経路でも検出器を世界bboxへ渡せる**必要がある。
- 既存呼び出しは既定引数で従来のbbox計算を通る（非検出器経路を壊さない）。

### D-2b. 空の geometry は扱わない（D-7の設計変更により不要になった）

初版は厚さ0cmのベースラインを「空の `Geometry`」で実現しようとしていたが、
**D-7を「同一寸法の空気スラブ」に変更したため、geometryが空になるケースは発生しない**
（Codexレビュー3回目で「空geometryの実行経路が未確定」と指摘されたのを、経路を作るのではなく
**そもそも空にしない**ことで解消した）。`Geometry([])` の許容は本計画では**実装しない**。

### D-3. 排反分類の真理値表

history毎に `had_fluorescence`（bool）と `n_cr` ≡ `n_compton_rayleigh`（int）を追跡し、
**検出された時点の値**で一意に分類する。上の行が優先。

| # | 条件 | カテゴリ | index |
|---|---|---|---|
| 1 | `had_fluorescence == True` | `fluorescence_secondary` | 3 |
| 2 | `n_cr == 0`（かつ蛍光なし） | `primary` | 0 |
| 3 | `n_cr == 1`（かつ蛍光なし） | `single_scatter` | 1 |
| 4 | `n_cr >= 2`（かつ蛍光なし） | `multiple_scatter` | 2 |

- 行1が最優先なのは、蛍光光子は元の一次線とエネルギーも方向も無関係な**別の光子**であり、
  その後の散乱回数で一次線側の議論に混ぜるべきでないため。
- 行2の `primary` は「一度もCompton/Rayleighを起こさず蛍光でもない」＝幾何学的な直進線。
- **各行に対応するテストを個別に置く**（B-5）。合計一致だけでは誤分類が通る。

### D-4. 粗ビンのRはPhase 0では出さない

粗ビンのRは細ビンのRから事後合成できない（親計画の判断5、Codexが実コードで検証済み）。
Phase 0では細ビン（=検出器ピクセル）についてのみバッチ統計を積む。再ビニングは
決定論的な後処理関数として提供するが、**粗ビンのRは計算せず返り値に含めない**
（誤った数値を返すくらいなら返さない）。粗ビンRの実装はPhase 2へ送る。

### D-5. MC統計誤差と量子ノイズ予測を別出力として分離する

- **MC統計誤差**: バッチ統計（`sum2`/`n_batches_hit` → `relative_error`）。
  history数を増やせば0に近づく。
- **量子ノイズ予測の材料**: 検出光子のエネルギーの**1次・2次モーメント**
  （ピクセル毎の `Σ E`、`Σ E²`、検出光子数 `Σ 1`）。実光子数へのスケーリング（mAs校正）
  自体はPhase 0では行わず、**材料を出すところまで**。
- 別の属性名で持ち、docstringで違いを明記する（下記API契約）。

### D-6. 統計ヘルパーの分割方法（Codexレビュー指摘、採用）

「ヘルパーを2回呼ぶ」案は破綻する（`n_batches_hit` の二重加算、OR判定の喪失、
Python の `prev=None` をヘルパー内で代入しても呼出側の参照は更新されない）。
**安全な分割**は、ヘルパーを「prevは呼出側が確保済み」「deltaを返す」
「`copyto` はヘルパー内で行う」小部品にすること:

```python
def accumulate_moment_and_snapshot(total, prev, sum2, n_histories_in_batch):
    """delta = total - prev を計算し、sum2 += delta²/n_b を積み、prevへcopytoしてdeltaを返す。
    n_batches_hit・n_batches・n_historiesは呼出側の責務（複数量でOR/一度だけ更新するため）。"""
```

`VoxelGrid.end_batch` は、遅延確保 → ヘルパーをkerma/H*(10)の2回呼んでdeltaを2つ得る →
**ORして `n_batches_hit` を一度だけ更新** → `n_batches`/`n_histories` を一度だけ更新、
という形に書き換える。`DetectorTally.end_batch` はカテゴリ数ぶん呼び、同様にORで一度だけ。

**既存 `VoxelGrid` の結果がビット一致すること**を既存テストで確認する（C-5）。

### D-7. 厚さ0cmのベースラインは「同一寸法の空気スラブ」で表現する

scene validatorは `geometry` に1個以上を要求し、`Geometry([])` はbbox計算で落ちる。
新しいAPIを作らずにこれを回避するため、**厚さ0cm条件は「水スラブと同一寸法・同一位置の
`air` スラブ」として表現する**（材料だけ差し替える）。geometryは常に非空になり、
既存の `run_transport(scene, ...)` 経路をそのまま使える。

**シーンはYAMLファイルを介さず、Pythonで組んだ raw dict を `validate_scene` に通して
`Scene` を得る**（`chatcarlo/scene.py` の既存API。ファイルI/Oもスキーマ拡張も不要）。

**したがって厚さ0cmのSTPRは厳密な0にはならない**（空気自体がわずかに散乱する）。
Codexは1回目のレビューで「厳密に0であるべき」としたが、それは「散乱体が全く無い」
場合の話であり、空気を輸送する本設計では成り立たない。合格基準は
`STPR(0cm) < 0.01` とし、**この理由をスクリプトのコメントに明記する**（E-2d）。

### D-8. 法線と受理方向の規約（Codexレビュー指摘、明文化）

- `d` は**光子の進行方向**。`normal` は**検出器の受光面が向いている方向**（線源側を向く）。
- **受理条件は `d · normal < 0`**（光子が受光面に正面から入射する）。
- 具体例で固定する: 線源が z=+100 cm、検出器が z=0 の xy平面にあり光子が −z 方向へ
  進むとき、`normal = (0, 0, +1)`（線源側を向く）、`d = (0, 0, -1)` で `d·normal = -1 < 0`
  → **受理**。反対向きに進む光子（背面からの入射）は非受理。
- **テストでこの座標例そのものを検証する**（A-1）。

### D-9. 始点が検出器面上にある場合の規約（Codexレビュー指摘、明文化）

`t_det == 0`（区間の始点がちょうど面上）は、**受理方向（`d·normal < 0`）なら検出とする**。
逆方向なら非受理（そのまま輸送を続ける）。これにより「検出器面から出発した光子が
永久に検出されない」ことも「背面から出た光子が即座に検出される」ことも防ぐ。

### D-10. STPRの定義（Codexレビュー指摘、明文化）

- **分子 = 非primaryの検出エネルギーフルエンスの合計**
  （`single_scatter + multiple_scatter + fluorescence_secondary`。蛍光も散乱側に含める
  ——「一次線でないもの」がグリッドで除去される対象という実務的定義に合わせる）
- **分母 = primaryの検出エネルギーフルエンス**
- **集計領域 = 検出器中心の ROI**（下記E-1で固定するサイズ）。全画素積分ではなく中心ROIに
  するのは、有限照射野の周辺部（半影・照射野外）の寄与で比が薄まるのを避けるため。
- 面積で割ったフルエンス像をROIで**面積積分**して比を取る（ピクセル面積は分子分母で
  相殺するが、定義として明示する）。

## API契約（Codexレビュー指摘、これがないと実装できない）

### `chatcarlo/detector.py`

```python
CATEGORY_NAMES = ("primary", "single_scatter", "multiple_scatter", "fluorescence_secondary")
CAT_PRIMARY, CAT_SINGLE, CAT_MULTIPLE, CAT_FLUOR = 0, 1, 2, 3

@dataclass
class DetectorPlane:
    center_cm: np.ndarray      # (3,) 検出器中心
    normal: np.ndarray         # (3,) 受光面の向き（線源側。D-8）
    u_axis: np.ndarray         # (3,) ピクセル行方向の単位ベクトル
    size_cm: tuple             # (u_len, v_len) 全長 [cm]
    shape: tuple               # (nu, nv) ピクセル数
    # v_axis は normal × u_axis から導出（__post_init__）。
    # normal と u_axis が直交していなければ ValueError（黙って正規化しない）。

    def corners_cm(self) -> np.ndarray:      # (4,3) 世界bbox計算用（D-2）
    def intersect_segments(self, o, d, ds):  # 純関数。(t, iu, iv, accept) を返す
        """区間 o→o+d*ds と面の交差。accept は D-8/D-9/A-4 の規約を満たすもののみ真。

        入力: o (N,3) float, d (N,3) float（単位ベクトル）, ds (N,) float
        出力: t (N,) float, iu (N,) int64, iv (N,) int64, accept (N,) bool

        座標規約（Codexレビュー2回目の指摘により明文化）:
          - `normal`・`u_axis` は**単位長を要求する**（__post_init__で検証、
            違えばValueError。黙って正規化しない）。`v_axis = normal × u_axis`。
          - t = ((center_cm - o) · normal) / (d · normal)
          - 交点 p = o + d*t。面内座標 (uu, vv) = ((p - center_cm)·u_axis,
            (p - center_cm)·v_axis)
          - 画素番号 iu = floor((uu + u_len/2) / (u_len/nu))、iv も同様（半開区間）
          - accept = (d·normal < 0) & (0 <= t) & (t <= ds)
                     & (0 <= iu < nu) & (0 <= iv < nv)
          - **非受理の要素では t=inf, iu=-1, iv=-1 を返す**（呼出側が誤って
            使った場合に静かに0番画素へ入らないようにするため）
        """

@dataclass
class DetectorTally:
    plane: DetectorPlane
    track_uncertainty: bool = False
    roi: tuple | None = None            # ((iu_lo, iu_hi), (iv_lo, iv_hi)) 半開区間。STPR用（下記）
    # --- 以下はすべて field(init=False) で __post_init__ が確保する（利用者は渡さない） ---
    # --- 主出力（エネルギーフルエンス [keV/cm²]、**累積値**。history正規化しない） ---
    category_fluence: np.ndarray        # (4, nu, nv) float
    # --- 量子ノイズ材料（D-5、MC統計誤差とは別物） ---
    photon_count: np.ndarray            # (nu, nv) float — 検出光子数 Σ1
    energy_sum_keV: np.ndarray          # (nu, nv) float — ΣE
    energy_sum2_keV2: np.ndarray        # (nu, nv) float — ΣE²
    # --- MC統計誤差（track_uncertainty=False なら **すべて None**） ---
    category_sum2: np.ndarray | None    # (4, nu, nv) float — Σ(バッチ寄与²/n_b)
    total_sum2: np.ndarray | None       # (nu, nv) float — totalの独立Q
    n_batches_hit: np.ndarray | None    # (nu, nv) int32
    # --- ROIスカラー統計（STPRのSEM用。下記「ROI統計」を参照） ---
    roi_P: float; roi_S: float          # 累積のROI primary / non-primary 和
    roi_QP: float; roi_QS: float; roi_CPS: float
    n_batches: int
    n_histories: int

    def total_fluence(self) -> np.ndarray            # (nu, nv) = category_fluence.sum(axis=0)
    def category_relative_error(self) -> np.ndarray  # (4, nu, nv)。OFF時は ValueError
    def total_relative_error(self) -> np.ndarray     # (nu, nv) — total_sum2 を使う
    def stpr(self) -> float                          # roi_S / roi_P
    def stpr_sem(self) -> float                      # デルタ法（下記）
    def end_batch(self, n_histories_in_batch: int) -> None
    def merge_from(self, other: "DetectorTally") -> None   # 並列集約（親側で加算）
```

**確保と `None` の規約**（Codexレビュー2・3回目の指摘、明文化）: 配列は
`field(init=False)` として `__post_init__` が確保する。`track_uncertainty=False` のとき:

- **`None` になるもの**: `category_sum2`・`total_sum2`・`n_batches_hit`・
  `roi_QP`・`roi_QS`・`roi_CPS`・スナップショット（`prev`）
- **常に保持するもの**: `category_fluence`・量子ノイズ3配列・**`roi_P`・`roi_S`**
  （累積和なので統計機構と無関係。`stpr()` は統計OFFでも計算できる）
- 統計OFFで `category_relative_error()`/`total_relative_error()`/`stpr_sem()` を
  呼んだら**`ValueError` を送出する**（NaNや0を黙って返さない）。`stpr()` は動く。

**`merge_from` の契約**（Codexレビュー2回目の指摘、明文化）: 相手の `plane` と
`track_uncertainty`・`roi` が**自分と一致することを検証**し（違えば `ValueError`）、
`category_fluence`・量子ノイズ配列・`category_sum2`・`total_sum2`・`n_batches_hit`・
ROIスカラー・`n_batches`・`n_histories` を**加算する**。
**worker内のスナップショット `prev` は決してmergeしない**（バッチ境界はworker内で閉じている）。

**`total` のRはカテゴリ別のQからは合成できない**（粗ビンRと同じ理由——カテゴリ間の
共分散項が要る）。したがって `total_sum2` を独立に積む。

### ROI統計（STPRのSEMを出すために必要。Codexレビュー2回目の最重要指摘）

**`category_sum2`/`total_sum2` だけからは中心ROIのSTPRのSEMは算出不能**である
（ROI内の画素間共分散と、primary和/非primary和のバッチ間共分散の両方が欠ける）。
受入条件E-1「各STPRのSEM」・E-2の2σ判定はこのままでは満たせない。
**したがってROIをスクリプト側の後処理にせず、`DetectorTally` に `roi` を持たせ、
バッチごとに以下のスカラーを積む**:

- `P_b` = そのバッチのROI内 primary 寄与和、`S_b` = 同 non-primary 寄与和
- `roi_QP += P_b²/n_b`、`roi_QS += S_b²/n_b`、`roi_CPS += P_b·S_b/n_b`

`stpr_sem()` は比のデルタ法で返す。**history当たり平均で統一する**
（Codexレビュー3回目の指摘: 分母に累積和 `S²`/`P²`/`SP` を使いながら分散だけ
平均の分散にすると **SEMが 1/N 倍ずれる**。初版の式はこの誤りがあった）:

```
p = P / N,  s = S / N                      # N = 総history数、P/S は累積和
v_P = (Q_P − P²/N) / (M − 1)               # 既存 _batch_variance と同形
v_S = (Q_S − S²/N) / (M − 1)
c   = (C_PS − P·S/N) / (M − 1)
Var(p) = v_P / N,  Var(s) = v_S / N,  Cov(p, s) = c / N
Var(s/p) ≈ (s/p)² · [ Var(s)/s² + Var(p)/p² − 2·Cov(s,p)/(s·p) ]
stpr_sem = sqrt(Var(s/p))
```

比 `s/p` は `S/P` に等しい（Nが相殺する）ので `stpr()` の値は変わらない。
**変わるのは分散のスケールだけ**であり、そこが誤りの入りやすい箇所なので
上式のとおり実装すること。**この導出を実装のdocstringに書き、受入条件C-6で
ブルートフォース照合する**。`roi=None` なら `stpr()`/`stpr_sem()` は `ValueError`。

### `chatcarlo/transport.py`

- `BatchResult` に追加: `detected`（(N,) bool）、`n_compton_rayleigh`（(N,) int）、
  `had_fluorescence`（(N,) bool）。**検出器を渡さない場合もこれらは常に返す**
  （`detected` は全False。追加は既存フィールドの値に影響しない）。
- `transport_photons(..., detector_tally: DetectorTally | None = None)`
- `_run_batches(..., detector_tally=None)`: バッチ末尾で `detector_tally.end_batch(n)` を
  **`grid.end_batch(n)` と同じ位置**で呼ぶ。
- **`run_transport(..., detector: DetectorPlane | None = None, detector_roi: tuple | None = None)`**:
  内部で `DetectorTally(plane=detector, track_uncertainty=track_uncertainty, roi=detector_roi)` を
  作り、`TransportResult.detector`（新フィールド、既定 `None`）で返す。
  **ROIの受け渡しはこの `detector_roi` 引数が唯一の方法**（Codexレビュー3回目の指摘:
  「`run_transport` が内部で `DetectorTally` を作る」規約と「スクリプトが
  `DetectorTally(roi=...)` を渡す」要求が矛盾していた）。
  `detector is None` かつ `detector_roi is not None` は `ValueError`。
- 並列時: 各workerが自分の `DetectorTally` を持ち、親が**worker番号順に** `merge_from` する
  （既存の `ScalarMoments.merge_from` と同じ規律）。
- `engine="kernel"` かつ `detector is not None` は `ValueError`（D-3の受入条件）。

## 受入条件（検証可能な形で列挙）

### A. 幾何・ビン規約

- [ ] **A-1 受理方向の規約**: D-8の座標例（線源 z=+100、検出器 z=0、`normal=(0,0,1)`）
      そのものをテストにし、`d·normal < 0` の光子が検出され、背面から入射する光子が
      検出されないことを確認する。
- [ ] **A-2 終端型の規約**: (a) 各historyの**最初の受理交差だけ**が記録される
      （散乱して面を再度横切っても二重計上されない）、(b) 記録後は `detected=True` /
      `alive=False` で輸送が止まる、(c) 検出器の矩形外を通る光子は記録されない。
- [ ] **A-3 区間短縮の順序**: 飛行区間の途中で交差する条件で、(a) gridへの総track lengthが
      **検出器面までで止まる**こと（検出器背後のボクセルのカーマが0）、
      (b) 検出により `tau` 残量を使った次ステップや相互作用が**起きない**こと。
- [ ] **A-4 タイの規約**: `t_det` が材料境界距離または相互作用距離と数値的に一致する条件で、
      **受理検出器を優先（`t_det <= ds` で検出）**という規約どおり一意に処理されること。
- [ ] **A-5 ピクセル境界規約**: 半開区間 `[lo, hi)` を採用。ピクセル境界・検出器外周・角を
      ちょうど通る光子の帰属をテストで固定する。**上限側の外周にちょうど当たる光子は
      範囲外**（半開区間の帰結）——これを明示的にテストする。
- [ ] **A-6 縮退ケース**: 面と平行に走る光子（`d·normal == 0`）、始点が面上の光子（D-9）、
      相互作用点が面上の光子で、例外を出さず規約どおりに処理されること。
- [ ] **A-7 世界bboxと線量グリッドの分離**（D-2）: (a) 検出器を渡すと世界bboxが
      検出器を含むよう拡張され、**物体から50cm以上離れた検出器にも光子が到達する**こと、
      (b) 検出器を渡さない場合の `Geometry.bbox_min`/`bbox_max` が既存とビット一致すること、
      (c) **検出器の有無で `VoxelGrid` の `shape`/`origin_cm`/`voxel_size_cm` が
      ビット一致すること**（グリッドが空気領域へ広がっていないことの証明）。
      **(c)は workers=1 と workers=2 の両方で実行する**——`run_transport` 側だけ
      `tally_bbox_*` に直して並列worker側を直し忘れる誤りは、workers=1では検出できない。
- [ ] **A-9 ROI引数の受け渡し**: `run_transport(detector=..., detector_roi=...)` で
      ROIが `DetectorTally` へ正しく渡ること、`detector is None` かつ
      `detector_roi is not None` が `ValueError` になること。
- [ ] **A-8 幾何学的投影**: (a) 無物体・単色・平行ビームで、照射野に対応するピクセルだけが
      一様に加算されること、(b) 発散ビーム（`cone` または `rect`）で、既知の幾何学的投影位置と
      ピクセル位置が一致すること（拡大率 = SID/SOD で検証）。

### B. 物理的整合性

- [ ] **B-1 一次線のBeer-Lambert照合**: 単一材料スラブ・平行ビーム・単色で、
      `primary` 像の各ピクセル値が `exp(-μt)` と一致すること。
      **比較量の定義**（Codexレビュー2回目の指摘）: `category_fluence` は**累積値**
      （history正規化しない）なので、`exp(-μt)` と直接比べられるのは
      **「スラブありのprimary像 ÷ スラブなしのprimary像」というピクセル毎の比**である。
      同一seed・同一history数の2run（スラブあり／なし）で比を取って照合する。
      **合格閾値（事前登録）: 相対差が統計誤差の3σ以内、かつ全ピクセルの平均相対差 < 0.5%。**
      μは `chatcarlo/materials.py` から取得し、テスト内で手打ちしない。
- [ ] **B-2 終端の排反性**: 各historyで `absorbed`/`detected`/`escaped` の
      **ちょうど1つだけ**が真であることを全history分、要素単位で検査する。
- [ ] **B-3 エネルギー収支の恒等式**:
      `入射エネルギー = 衝突沈着 + 検出器到達エネルギー + 未検出脱出エネルギー`。
      **合格閾値: 相対差 < 1e-9**（既存 `tests/test_fluorescence.py` の
      エネルギー保存テストと同じ許容差。初版の `1e-12` は段階的加算・`np.sum` の
      丸め蓄積を考えると非現実的だったので改めた）。
      - **収支に使うのは衝突沈着**（`energy_deposited`）であって、gridの
        track-lengthカーマではない（別の推定量。混ぜると合わない）。
      - 検証には**テスト専用に**history別の「全衝突沈着」「検出終端エネルギー」
        「未検出脱出終端エネルギー」を保持してよい（本番経路のメモリを増やさないこと）。
      - **蛍光あり・なしの両方**で確認する。
- [ ] **B-4 カテゴリの完全性**: 4カテゴリの和が `total_fluence()` と**ビット一致**すること。
- [ ] **B-5 真理値表の各行**: D-3の4行それぞれについて、該当するhistoryが実際にその
      カテゴリへ入ることを個別テストで確認する。行1（蛍光）は鉛など高Z材料＋K吸収端以上の
      エネルギーで蛍光を強制的に起こす条件を作る。

### C. 統計

- [ ] **C-1 統計ON/OFFのビット一致（絶対制約）**: `track_uncertainty=True/False` で
      `category_fluence`・`photon_count`・材料別沈着・`grid.kerma_keV` が
      **ビット一致**すること（`np.array_equal`）。
- [ ] **C-2 Rのブルートフォース照合**: n=6,000・batch_size=1,000（M=6）で、各バッチの
      検出器像を独立に記録して手計算したSEMと、実装が返す値が `rtol=1e-9` で一致すること。
      **カテゴリ別と `total` の両方**について行う（`total` が独立Qを使っていることの検証）。
- [ ] **C-3 端数バッチ**: `n_histories=2,500`・`batch_size=1,000` で最終バッチの `n_b=500` が
      正しく渡り、`n_batches=3`・`n_histories=2,500` になること。
- [ ] **C-4 量子ノイズ材料の分離**: `photon_count`/`energy_sum_keV`/`energy_sum2_keV2` が
      MC統計誤差の配列とは別属性として得られ、単色・無物体・既知入射光子数の条件で
      解析的期待値（単色なら `Σ E² = N·E²`）と厳密に一致すること。
- [ ] **C-6 STPRのSEMのブルートフォース照合**（上記「ROI統計」の検証）:
      n=6,000・batch_size=1,000（M=6）で各バッチの `P_b`/`S_b` を独立に記録し、
      デルタ法の式を外部で手計算した値と `stpr_sem()` が `rtol=1e-9` で一致すること。
      **共分散項を0に落とした誤実装が検出できること**もミューテーションで確認する（F-1に追加）。
- [ ] **C-7 統計OFF時の挙動**: `track_uncertainty=False` で
      `category_relative_error()`/`total_relative_error()`/`stpr_sem()` が
      **`ValueError` を送出する**こと（NaN・0を黙って返さない）。
- [ ] **C-5 既存 `VoxelGrid` の不変性**（D-6の共通化の副作用がないこと）:
      ヘルパー括り出し後も既存の統計テスト（`tests/test_uncertainty.py`・
      `tests/test_uncertainty_transport.py`・`tests/test_tally.py`）が全通過し、
      同一seedで `kerma_sum2`/`h10_sum2`/`n_batches_hit` が括り出し前と**ビット一致**すること
      （括り出し前の値をgit worktreeで取得して比較する）。

### D. API・並列・エンジン境界

- [ ] **D-1 既存輸送への非侵襲性（ビット一致）**: 検出器を渡さない場合、同一seedで
      以下がビット一致すること。**低レベルと高レベルを分けて列挙する**:
      - `transport_photons` の戻り値: `n_scatter`・`absorbed`・`escaped`・`final_energy`・
        `energy_deposited`・`n_fluorescence`
      - `run_transport` の戻り値: `energy_deposited_MeV`・`fraction_absorbed`・
        `fraction_escaped`・`mean_scatter_events`・`grid.kerma_keV`・`grid.h10_track_pSv_cm3`・
        `energy_deposited_sem_MeV`
      - 検出器判定は**乱数を一切消費しない**こと（これがビット一致の根拠）。
        崩れたら設計の読み違いなので**手を広げず停止して報告する**。
- [ ] **D-2 並列**: `_run_batches`・並列worker・親側集約まで検出器タリーが通り、
      同一 `(seed, workers)` で再現すること、workers数を変えた場合にカテゴリ別総和が
      統計的に同等（結合3σ以内）であること。
- [ ] **D-3 kernelエンジンとの併用禁止**: `engine="kernel"` と `detector` の同時指定が
      **明示的に `ValueError`** になること。既存 `kernel_engine_compatible` と同じスタイルの
      日本語メッセージにする。

### E. STPR再現スタディ

- [ ] **E-1 実験条件を定数として固定し、スクリプト冒頭で出力する**（Codexレビュー指摘）。
      以下を `scripts/stpr_water_slab.py` の定数として宣言する:
      - スラブ: 厚さ 0/5/10/15/20/25/30 cm、横断面 40×40 cm、材料 `water`。
        **厚さ0cm条件は同一寸法・同一位置の `air` スラブで表現する**（D-7）。
        シーンはPythonのraw dictを `validate_scene` に通して構築する（YAMLファイル不要）
      - 検出器: 43×43 cm、ピクセル 256×256、スラブ出射面から **air gap 5 cm**
      - 線源: 点線源、SID 180 cm（スラブ入射面まで SOD = 180 − 5 − 厚さ）、
        照射野は**検出器全面を覆う `cone`**（技報の「照射野=検出器サイズ」条件）
      - スペクトル: SpekPy、管電圧 60/80/100/120 kV、**濾過 Al 2.5 mm**、ヒール効果なし
      - 集計ROI: **検出器中心の 60×60 ピクセル**（Codexレビュー3回目の指摘:
        43cm/256 = 0.16796875 cm なので「10cm」は59.535画素になり整数にならない。
        **画素数で定義し、対応する実寸 10.078 cm をスクリプトが出力する**。
        端画素の面積加重はしない）。`run_transport(detector_roi=((98,158),(98,158)))`
        として渡す（後処理でROIを切ると共分散が復元できずSEMが出せない。上記「ROI統計」）
      - `n_histories = 2e6`、`batch_size = 1e5`（M=20）、`seed = 42`
      - 出力: 条件・STPR・**各STPRのSEM**をCSVで保存
- [ ] **E-2 合格判定（事前登録。後から動かさない）**:
      - (a) **単調増加（統計的判定）**: 隣接する厚さ間で `STPR(t+Δ) > STPR(t)` を
        **結合SEMの2σ以上の差**で確認する。**差が2σ未満の区間は「判定保留」として記録**し、
        不合格にはしないが報告に明記する（MCノイズ下で厳密な単調性を要求しない。
        Codexレビュー指摘）。
      - (b) **kV依存の順序（統計的判定）**: 厚さ20cm以上で
        `STPR(120kV) > STPR(100kV) > STPR(80kV)` を同じく2σ基準で判定する。
        2σ未満は判定保留として記録する。
      - (c) **オーダーゲート**: 厚さ20cmでのSTPRが **2.0〜8.0** に入ること。
        これは**粗い探索ゲート**であり精密な再現試験ではない
        （技報Fig.3は目視読み取り、かつ照射野・エアギャップ・線質に強く依存する。
        ユーザー承認済みの「オーダーでの判定」方針に従う）。
      - (d) `STPR(0cm) < 0.01` であること（D-7のとおり空気の散乱があるので厳密な0ではない）。
      - **(a)(b)以外が不合格なら、閾値を動かさず不合格として記録し停止して報告する。**
- [ ] **E-3 散乱内訳の定性挙動**: 5cmでは `single_scatter > multiple_scatter`、
      25cm以上では逆転すること（同じく2σ基準、2σ未満は判定保留）。

### F. テストの検出力

- [ ] **F-1 ミューテーション検証**: 以下5つの変異を順に注入し、追加したテストが
      **実際に失敗する**ことを確認して報告に含める。検出できない変異があれば、テスト条件を
      調整するか、**検出できない旨をコメントに正直に書く**
      （`2026-08-01-kernel-cli-wiring-phase1.md` の `_CROSSCHECK_HISTORIES` のコメントが手本）。
      1. primary判定を `n_cr == 0` から `n_cr <= 1` に変える
      2. ピクセル割り当てを1ピクセルずらす
      3. 受理方向の判定（`d·normal < 0`）の符号を反転する
      4. 区間短縮（`ds = min(ds, t_det)`）を外し、短縮前の `ds` をgridへ積算する
      5. `hit_det` を `noninteract`/`interact` マスクから除外する処理を削除する
         （下記の落とし穴。これが検出されないなら A-3(b) のテストが弱い）
      6. `stpr_sem()` のデルタ法から共分散項 `−2·Cov(S,P)/(S·P)` を削除する
         （C-6が検出するはず）
      7. `run_transport` 側だけ `tally_bbox_*` に直し、並列worker側を `bbox_*` のままにする
         （A-7(c) をworkers>=2でも実行しないと検出できない。**A-7(c)は
         workers=1と2の両方で実行すること**）

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/test_detector_tally.py tests/test_detector_geometry.py -q
.venv/bin/python scripts/stpr_water_slab.py --out docs/speedup_baseline/stpr_water_slab.csv
```

## 実装方針

### 輸送ループへの組み込み（`transport_photons`）

`ds`/`ends` を計算した**直後、grid積算の前**に:

1. `t_det, iu, iv, hit_det = detector_tally.plane.intersect_segments(o, d, ds)` を呼ぶ。
   `hit_det` は **`idx` に対応するローカル配列**（長さ `len(idx)`）。
   受理条件は D-8（`d·normal < 0`）、D-9（`t_det == 0` の扱い）、
   `0 <= t_det <= ds`、交点が矩形内（A-5の半開区間）。
2. `ds = np.where(hit_det, t_det, ds)` で**短縮**する。
3. `ends = o + d*ds[:,None]` を**短縮後の `ds` で再計算**する。
4. grid積算（`accumulate_track_length_multi`）は**短縮後の `ds`** で行う。
5. `pos[idx] = ends`。
6. **【最重要の落とし穴。Codexレビュー指摘】検出したhistoryを、相互作用側・
   非相互作用側の両方のマスクから除外する**:
   ```python
   noninteract = (~will_interact) & (~hit_det)
   interact    =  will_interact  & (~hit_det)
   ```
   これを怠ると、検出済みhistoryが同じ反復で相互作用したり、`tau` 減算と
   epsilon移動を受けたりする。
7. `alive[idx[hit_det]] = False`、`detected[idx[hit_det]] = True`。
   **`hit_det` はローカル配列なのでグローバル添字は `idx[hit_det]`**。
   `pos[idx]` と `pos[idx[hit_det]]` を混同しないこと。
8. 検出器タリーへ加算: 加算値は `e`（**相互作用前のエネルギー**）をピクセル面積で
   割ったエネルギーフルエンス。カテゴリはD-3の真理値表で決める。
   **同一ピクセルへの複数historyの加算は `np.add.at` を使う**（単純な fancy indexing は
   重複添字で最後の1件しか反映されない）。具体形:
   ```python
   gidx_hit = idx[hit_det]
   cat = classify(n_compton_rayleigh[gidx_hit], had_fluorescence[gidx_hit])
   np.add.at(tally.category_fluence,
             (cat, iu[hit_det], iv[hit_det]),
             e[hit_det] / pixel_area_cm2)
   ```
   `photon_count`/`energy_sum_keV`/`energy_sum2_keV2` も**同じ `iu[hit_det], iv[hit_det]`**
   で別々に `np.add.at` する（D-5）。`will_interact` は短縮後に再計算しない
   ——手順6のマスク除外で正しく終端する。

**検出器判定は乱数を一切消費しない。** これが受入条件D-1の根拠。

### `n_compton_rayleigh` / `had_fluorescence` の更新箇所

- Compton分岐（`compt_idx`）・Rayleigh分岐（`rayl_idx`）で `n_compton_rayleigh += 1`
- 蛍光放出時（`emit_idx`）で `had_fluorescence = True`
- **既存の `n_scatter` の更新は一切変更しない**（D-1のビット一致のため）

### 再ビニング（後処理、D-4）

```python
def rebin_area_preserving(fluence_image, factor):
    """エネルギーフルエンス像[keV/cm²]を factor×factor で粗化する（面積加重平均）。

    入力がフルエンス（面積で割り済み）なので粗画素の値は構成細画素の**平均**
    （総和ではない）。factor が各辺を割り切らない場合は ValueError。
    カテゴリ軸がある場合は最後の2軸に対して作用する。
    **粗ビンの統計不確かさRは返さない**——細ビンのRから合成できないため
    （細ビン間の共分散項が必要。親計画の判断5参照）。粗ビンRが必要なら
    粗解像度のモーメント積算を別途行う（Phase 2）。
    """
```

`photon_count`（面積で割っていない計数）は**和**で粗化する必要があるので、
フルエンス用とは**別の関数**（または `mode` 引数）にし、取り違えを防ぐ。

## 書かなかったこと（スコープ外を明示）

- scene.yamlへの `detector:` セクション追加とCLI配線（D-1。Phase 1bで設計）
- 厚さ逆算・2.5Dモデル構築・散乱減算（Phase 1a/1b）
- heightfieldプリミティブ（Phase 1a）
- 検出器応答モデル（吸収効率・DQE・エネルギー応答関数）。Phase 0の検出器は
  **理想的に100%吸収する面**
- 検出器背後からの後方散乱（終端型の近似。**ユーザー承認済み**。親計画参照）
- 粗ビンの統計不確かさR（D-4でPhase 2へ）
- 量子ノイズの実光子数スケーリング（mAs校正）そのもの（D-5、材料までがPhase 0）
- kernel経路の検出器対応（併用はエラーにするだけ）
- EGS5相互検証（親計画でユーザーが受入条件から除外）
- 分散低減（next-event estimation / forced detection）
- `chatcarlo/tally.py` の推定式の改良

## 失敗時の停止条件

- **E-2(c)(d) の閾値を満たさない場合、閾値を動かさない。** 不合格として記録し、原因
  （タリーの誤り／シーン設定／統計不足）を切り分けてから報告する。
- **D-1（検出器なしでのビット一致）が崩れた場合、原因が判明するまで先に進まない。**
  乱数消費が変わっている可能性が高く、放置すると以降の全比較が無効になる。
- **C-5（既存 `VoxelGrid` のビット一致）が崩れた場合も同様に停止する。**
  共通化は「振る舞いを変えない」ことが前提の変更であり、崩れたら分割方法が誤っている。
- `chatcarlo/kernel.py` や `chatcarlo/physics.py` に変更が必要になった場合、
  それは設計の読み違いなので**手を広げず停止して報告する**。
