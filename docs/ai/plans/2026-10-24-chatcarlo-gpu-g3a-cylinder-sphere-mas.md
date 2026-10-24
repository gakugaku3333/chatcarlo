# 計画: GPU エンジン G3a — cylinder・sphere と `mas` の校正（IVR 室の依頼 3 を Colab の GPU で計算する）

- 状態: draft（保留、2026-10-24: 依頼 3 は直方体版で GPU 計算済みのため実装しない。円柱・球を GPU に載せる需要が出たら再開）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-24
- Codexレビュー: 2巡
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md) の段 G3 の前半（G3 のうち cylinder・sphere・`mas` だけ。`ctdi_vol_mGy`・`heel_effect`・`rotation` は G3b に残す）
- 規模: 重（輸送の幾何と線量の絶対値。誤ると静かに違う線量が出る）

## 目的

抄録（`docs/jsmp133/abstract_ja_draft.md`）は「承認した後に Google Colab 上で計算が実行される」「計算をクラウド上の GPU で実行する」と書くが、依頼 3（IVR 室、`runs/angio_room_3d/scene.yaml`）は GPU エンジンが cylinder・sphere・`mas` を拒むため手元の CPU（`transport.py`）で計算した。ユーザー決定（2026-10-24）: GPU エンジンを拡張し、依頼 3 を Colab の GPU で統計数を上げて計算し直す。

この計画が答える問い: **cylinder・sphere を含み `mas` を持つ診断シーンを、`run --engine gpu --gpu-device colab --dose-grid` で（preview → approve → run の承認を通して）計算でき、その結果が `transport.py` と統計的に一致し、mAs で校正した絶対値（Gy・pSv）が本番と同じ式で出るか。**

## 0. 前提事実（検証済み、2026-10-24 に読解）

**F7 既存コードとの契約**
- `gpu/engine.py:gpu_engine_compatible` は `shape != 'box'` と `mas`・`ctdi_vol_mGy`・`rotation`・`heel_effect` を理由つきで拒む。`bundle.create` も同じ関数を呼ぶ。
- 幾何は `geometry.npz` の `centers`・`halves`（box の半幅）・`materials`・`background`・`world`・`detector`。核の構造体 `G1Geometry {centers, halves, materials, nb, bg, world, det}`（`g1_geometry.h`、`core.py:19` の ctypes 定義、`core.py:145` の CUDA 引数の並び、`CUDA_INPUTS`・`GEOMETRY_INPUTS` の観測）。
- 核の幾何は 3 関数: `g1_material`（リスト後方優先、`<=` の閉区間）、`g1_boundary`（全物体の入口・出口のうち `t > 1e-6` の最小、と世界の箱の出口。`exit <= obj` なら脱出）、`g1_box`（`|d|<1e-12` を平行とする slab 法）。履歴の核 `g1_history.h` は境界を越えたあと `1e-6` だけ進める。
- 参照の `geometry.py`: `_intersect_sphere`（`b=oc·d`、`disc=b²−c`、`disc>=0` で命中）、`_intersect_cylinder`（軸 `x|y|z`、側面は `a<1e-12` を軸平行、軸平行で `cc<=0` なら側面は無限区間、端面は slab、`t_enter>t_exit` で外れ）、`_contains_sphere`（`<= r²`）、`_contains_cylinder`（`r² <= R²` かつ `|z−c| <= h/2`）。`next_boundary` の `eps=1e-6`。
- 線源の抽選はホスト（D7）。`rect` の発散照射野と `kvp` の多色は G1 で対応済み。
- 校正: `transport.py:595` は `mas` があると `n_photons_real = photon_count_through_field(src)`（`source.py:218`、SpekPy 必須）。`__main__` は `result.n_photons_real` があれば校正済みの値を表示し `.npz` に `dose_Gy`・`h10_pSv` を入れる。GPU の `aggregate.py` は `TransportResult` を作るが `n_photons_real` を入れない（未確認なら実装で確認）。VM には SpekPy が無いが、校正はホスト（Mac）で 1 回だけ計算すればよい。
- 線量格子は G2b で CLI・Colab の承認に通っている。格子の仕様は `job.grid` だけ。

**F10 対象シーンの現物**（`runs/angio_room_3d/scene.yaml`）
- 9 物体: cylinder 2（`patient_torso` axis y、`operator` axis z）、sphere 1（`patient_head`）、box 6。`mas: 100`、`kvp: 80`、`rect` 25×25 @ SID 110。
- 2026-10-22 の CPU 計算: 2×10⁶ 履歴、5 cm 格子 (120,120,79)、29 秒。H*(10) の R<0.10 は 6.9%、未到達 27.5%。
- `examples/` で cylinder・sphere を持つのは abdomen_table・angio_room・ct_room・chest_room・portable_ward。

**F3 破壊的操作**: なし。既存の `runs/angio_room_3d/` は上書きしない（新しい結果は `runs/angio_room_3d_gpu/`）。既存の受領票は核の原典が変わるので束のハッシュが変わり再利用されなくなるが、ファイルは消さない。

## 1. 対象範囲

**変更してよい**
- `chatcarlo/gpu/` の `g1_geometry.h`・`g1_history.h`（必要なら）・`g1_cpu.cpp`・`g1_gpu.cu`・`core.py`・`g2_core.py`・`bundle.py`・`engine.py`・`aggregate.py`・`cli_run.py`・`approval.py`・`colab_driver.py`（幾何の引数の受け渡しと観測だけ）・`condition.py`（必要なら）。
- `tests/test_gpu_g3a.py`（新規）、`docs/validation/gpu/g3a/`（新規。関門の実行スクリプトと記録）。
- 既存の GPU テスト（`tests/test_gpu_g1_*`・`test_gpu_g2*`・`test_gpu_g4.py`、`docs/validation/gpu/g2/run_gates.py`）のうち、**引数の並び・構造体の定義・引数の位置の辞書だけ**を新しい ABI に合わせる機械的な変更（Codex レビュー 1 の指摘 2）。**期待値・固定ケース・許容・判定の式は変えない。** 変えた行は実装報告に一覧で出す。
- `.claude/rules/chatcarlo.md` の GPU の段落（対応範囲の 1 文）。

**変更禁止**
- `transport.py`・`geometry.py`・`source.py`・`kernel.py`・`materials.py`（参照実装）。既存テストの期待値。`runs/angio_room_3d/`。

**スコープ外**: `ctdi_vol_mGy`・`heel_effect`・`rotation`（G3b）、治療領域、速度の最適化。

## 2. 設計

- **幾何の表現**: `geometry.npz` に `kinds`（int64、物体数、0 box・1 cylinder・2 sphere）と `params`（float64、形 (物体数, 4)）を足す。box は `halves` だけを正典にし、`params` の行は 0。cylinder は `params=[半径, 半高さ, 軸 0/1/2, 0]`・`halves` の行は 0、sphere は `params=[半径, 0, 0, 0]`・`halves` の行は 0。`axis` の省略は `geometry.py` と同じ既定（実装で `scene.py`/`geometry.py` を読んで確認し、報告する）。
- **核**: `G1Geometry` に `kinds`・`params` を足し、`g1_material` と `g1_boundary` を型で分岐させる。cylinder・sphere は `geometry.py` の式を項の順に C++ へ写す（`eps=1e-12` の平行判定、`disc>=0`、軸平行で `cc<=0` の無限区間、`t_enter>t_exit` で外れ、`<=` の包含）。内積は `geometry.py` の `einsum` と同じ `x,y,z` の順の和で書く（einsum の内部順序までは契約にしない。関門 B の許容で吸収する）。**box の分岐の演算と乱数の消費は今のコードのまま**（関門 A）。
- **引数の観測**: `kinds`・`params` を `GEOMETRY_INPUTS` に加え、CPU の構造体・CUDA の引数の両経路で実引数のハッシュを観測し、承認済みの束から作った期待値と照合する。引数の位置が動く場合は `g2_core.py` の固定オフセット（`19+i` など）と `approval.py` の線源の位置（15・16・17）も合わせる。dtype・形・値域（`kinds∈{0,1,2}`、軸 ∈{0,1,2}、半径・半高さ > 0、有限）は束を作るときと読むとき（`bundle.load` の後）に検査する。
- **承認の照合**（`approval.py`）: 今の `centers`・`halves` の照合を型ごとに広げ、scene の `shape`・`size_cm`/`radius_cm`/`height_cm`/`axis` と束の `kinds`・`halves`・`params` が一致しなければ `preview geometry mismatch` で止める。
- **校正**: `mas` は核に入れない。共有の関数 1 つ（例: `gpu/calibration.py` は作らず `aggregate.py` か `engine.py` に置く）で `photon_count_through_field(source)` を 1 回計算し `result.n_photons_real` に入れる。手元（`engine.run`・CPU の CLI）は scene の `source`、Colab の集計（`cli_run.run`）は照合済みの束の `source.json` を渡す。`mas` が無ければ `None`（今の出力のまま）。条件への `mas` の重複追加はしない（束の `source.json` と `settings.json` の scene の SHA が manifest 経由で条件と受領票に入っている。レビュー 1 の参考 13）。算出した `n_photons_real` と SpekPy の版を結果のメタデータに残す。
- `gpu_engine_compatible` は cylinder・sphere・`mas` を通し、`ctdi_vol_mGy`・`heel_effect`・`rotation` は今までどおり拒む。

## 3. 受入条件（事前登録）

- **関門 A（box の不変）**: 実装の前に、基準のコミット（実装開始時の HEAD）で CPU の双子を作り、次の条件の履歴ごとの記録 `d`・`i` と格子を保存する（`docs/validation/gpu/g3a/baseline/`、スクリプトも同じ場所）: G1 の既存の box シーン（G1 の関門のシーン）と G2 の格子つきのシーン、種 2 組、各 2×10⁴ 履歴・バッチ 10⁴。変更後に同じ条件で再計算し、`d`・`i`・格子が**バイト一致**。コンパイラとフラグは記録する。既存の GPU テスト（ABI の機械的な追従後）がすべて合格。
- **関門 B（幾何の関数）**: 核の幾何の関数を CPU の双子の probe で呼ぶ。固定の world の中で、点 10⁵ 個・単位ベクトルに正規化した方向で、cylinder（軸 x・y・z）・sphere・box が重なるシーンを使う。
  - 材料: GPU の添字を `material_names` で名前に戻し、`Geometry.material_at` と**全件一致**。
  - 境界: 脱出フラグが全件一致。距離は両方有限なら `|Δ| ≤ 1e-12·max(1,|t|)`、片方だけ無限なら不合格。
  - 固定ケース: 内外、接線の両側、側面と端面の同着、`t = 1e-6` の前後、軸平行で中・外、球の中心を通る方向。
- **関門 C（統計の一致）**: `runs/angio_room_3d/scene.yaml` と `examples/chest_room.yaml` で、CPU の双子（`run_transport(engine='gpu', gpu_device='cpu')`）と `transport.py` を各 10⁶ 履歴・バッチ 10⁵（10 バッチ）、5 cm 格子で比べる。種は GPU 101・`transport.py` 202（固定）。
  - 材料別の吸収エネルギー: 1 履歴あたりの平均に直して z = Δ/√(SEM₁²+SEM₂²)。全材料で |z| ≤ 4。
  - ボクセル: 両方で R<0.05 のボクセルのカーマと H*(10) で |z| ≤ 3 が 99% 以上。適格ボクセル数を報告し、**0 個なら不合格**。
  - 領域の積算: IVR は患者（体幹＋頭）・術者・防護衣の後ろ・天吊り板の後ろ、胸部は患者・操作室の鉛遮蔽の後ろの固定の箱（ボクセルの集合を実装前にスクリプトで固定）。H*(10) とカーマの和で |z| ≤ 4。領域和の SEM は、両エンジンとも**バッチごとに領域内の和を先に取り、その 10 個のバッチ値から**求める（ボクセル間の共分散があるため、ボクセルの SEM の二乗和は使わない）。バッチごとの格子は検証スクリプト側で 1 バッチずつ実行して取得し、参照実装は変えない。
  - z の分母が正なら比較する。両側とも SEM が 0 のときは、平均が一致すれば一致、異なれば不合格。件数を報告する。
- **関門 D（校正）**: `mas` のシーンで、(1) API の `engine.run`、(2) CPU の CLI（`run --engine gpu --gpu-device cpu --dose-grid --dose-out`）、(3) Colab の集計（`cli_run.run` の Colab 経路の集計部分を、受領済みの結果のフォルダで呼ぶテスト）のそれぞれで `n_photons_real` が `transport.py` の値とビット一致し、`.npz` の `dose_Gy`・`h10_pSv` が `per_history × n_photons_real` とビット一致。`mas` なしのシーンでは今までどおり校正のキーが無い。
- **関門 E（T4）**: 種は（transport 種, source 種）=(301, 302)・(303, 304)。同じ VM で作った同じ線源の配列を CPU の双子と GPU に渡す（G2 の関門 2 と同じ形）。依頼 3 のシーン、種 2 組、各 10 バッチ×10⁴ 履歴。
  - 整数の記録はバイト一致。格子は `docs/validation/gpu/g2/run_gates.py` の `grid_paired_judge`（加算回数の記録つき）で合格。
  - cylinder 3 軸・sphere の関数の probe（関門 B の固定ケース）を T4 でも走らせ、CPU の双子と材料・脱出フラグが一致、距離は相対 1e-12 以内。
  - 故障の注入を 3 種: (a) 承認後に束の `params` を書き換える（ファイルの SHA で停止）、(b) 別の条件の受領票を流用する（識別情報で停止）、(c) 束を変えず、核に渡す直前に `kinds` または `params` を差し替える（`CHATCARLO_G4_FAULT` の新しい値、観測の不一致で停止）。(c) は同じアドレスの中での書き換えも含め、ハッシュのキャッシュで見逃さないこと。
- **本計算**: 関門 A〜E の合格後、依頼 3 のシーンを preview → approve → run で Colab の GPU で計算する。回収量は 1 バッチあたり格子 2 量で約 18.2 MB（未圧縮、1,137,600 ボクセル）、ホストの RAM も全バッチを保持するので、**バッチ数を 20 以下**にして batch_size を決める（例: 2×10⁷ 履歴なら 10⁶ × 20 バッチ、展開後 約 0.36 GB）。履歴数はユーザーと決める。結果は `runs/angio_room_3d_gpu/`、3D 表示は `preview --dose`。

## 4. テストコマンド

```bash
.venv/bin/python -m pytest tests/test_gpu_g3a.py tests/test_gpu_g1_geometry.py tests/test_gpu_g1_regression.py tests/test_gpu_g2.py tests/test_gpu_g2b.py tests/test_gpu_g4.py -q
.venv/bin/python -m pytest tests/ -q
```

## 5. 停止条件

- 関門 A が通らなければ（box の結果が 1 ビットでも変わる）、先に進まず報告する。
- 関門 C で基準を満たさなければ、基準を動かさず、原因（幾何か校正か）を調べて報告する。
- T4 での失敗は Claude が調べる。核の変更が要るならこの計画の範囲で直し、関門 A から取り直す。

## 6. レビューの反映

- レビュー 1（`data/logs/codex_g3a_review1.md`、致命 2・実質 9・軽微 1・参考 2）: 致命 1（`approval.py` の照合）と致命 2（既存テストの ABI）は対象範囲の不足で、目的・前提は崩れていないため骨子は作り直さず範囲を直した。実質 3〜11・軽微 12 はすべて §2・§3 に反映。
- レビュー 2（`data/logs/codex_g3a_review2.md`、致命 0・実質 2・軽微 1）: 前回 14 件は解消・一部解消。SEM 0 の扱い、領域和の SEM をバッチ値から求めること、胸部の領域、関門 E の種を反映。致命が無く前巡と同種の精密化なので 3 巡目は回さない。
