# 計画 G1: chatcarlo/gpu の診断の核（box・多色・発散照射野・平面検出器、CPU の双子と GPU）

- 状態: approved（ユーザー承認 2026-10-21。全体の進め方も承認: 依頼 A+B のコミットの後に実装を始める）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-21
- Codexレビュー: 2巡（骨子: 致命 0・実質 3・軽微 2・参考 3。詳細: 致命 0・実質 8・軽微 4・参考 3。すべて反映。2 巡目の実質はどれも同じ設計の精密化なので、手順どおり Claude が反映し 3 巡目は行わない。原文 `data/logs/codex_gpu_review{1,2}.md`）
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md)（前提事実と構成の決定 D1〜D8。D1〜D4・D7 はユーザー承認済み 2026-10-21）

## 目的

診断領域の光子輸送を、P8f と同じ形（共通の C++ 履歴関数、CPU の双子、CuPy RawKernel、Philox）で `chatcarlo/gpu/` に作る。
Python の API（`run_transport(..., engine='gpu')`）と依頼 D 用の GPU ランナーから、box だけのシーン・多色スペクトル・
`rect`/`cone`/`parallel` 照射野・平面検出器のタリーを計算できるようにする。答える問いは 3 つ。

1. C++ の物理の関数と 1 履歴の輸送は、`kernel.py` の Numba の実装と、同じ MT19937 の乱数列でバイト一致するか（写し間違いがないか）。
2. CPU の双子（Philox）の結果は、本番の参照 `transport.py`（NumPy、PCG64）と統計的に一致するか（乱数の置き換えを含む移植全体の検査）。
3. GPU（T4）の結果は、同じ線源の入力と同じ Philox の種で、CPU の双子と一致するか。

実用の目標: 抄録依頼 D の肖像のシーンを、依頼 D と同じ検出器・同じ保存形式の GPU ランナーで計算できること。
CLI（`__main__.py` の `cmd_run`）は今、検出器を渡さず画像も保存しないので、CLI での画像化は G4 で扱う。

## 0. 前提事実（検証済み、2026-10-21。親計画 §0 に加えて）

- **本番の輸送ループ**（`transport.py` `transport_photons`、104〜280 行）: 光子 1 個が履歴の全部。蛍光は**同じ光子を蛍光線のエネルギー・等方の方向に置き換えて続ける**
  （スタックなし）。1 歩の順番: 材料の判定 → μ と内訳 → 次の境界（`Geometry.next_boundary`、世界の箱の脱出を含む）→ 相互作用か境界か
  （`tau < mu*t_boundary`）→ 検出面の交差（`DetectorPlane.intersect_segments(o,d,ds)`。当たれば `ds=t_det`、光子は止まる）→
  境界なら `tau -= mu*t_boundary` と 1e-6 cm 押し出し → 相互作用なら `r_type` で光電／コンプトン／Rayleigh。
  検出は `classify(散乱の回数, 蛍光の有無)` の 4 区分（`detector.py` 69〜72 行: 蛍光あり=3、散乱 0=0、1=1、2 以上=2）に
  `E/画素の面積` を足し、`photon_count`・`energy_sum_keV`・`energy_sum2_keV2` も足す。
- **`kernel.py` の 1 履歴**（`_transport_one`、624〜722 行）: 同じアルゴリズムのスカラー版（検出器なし）。乱数は Numba の `np.random.random()`
  （MT19937）。`run_batch(..., n_chunks=1)`（1275 行）は `_chunk_plan` の種で `np.random.seed` し、履歴ごとの配列
  `n_scatter・absorbed・escaped・final_energy・energy_deposited・n_fluorescence` を返す。P5 の MT（`p5_rng.h`、`P5_RNG_MT`）は
  P8f の関門 1a で Numba の `np.random` の列をバイト再現した実績がある。
- **世界の箱**: `Geometry` は物体の外接箱 + 50 cm、検出器があればその四隅 + 50 cm まで広げる（`geometry.py` 139〜145 行）。
- **線源**（`source.sample_source_photons`、67〜212 行）: `rect` は SID の面の矩形に一様な目標点、`cone` は立体角一様、`parallel` は面上の一様な位置。
  エネルギーは `spectrum.sample_spectrum`（73〜85 行）が `rng.choice(len(e), p=w)`。`kvp` の表は SpekPy、**SpekPy が無ければ警告を出して Kramers に切り替わる**
  （`spectrum.py` 18 行 `_HAS_SPEKPY`、68 行の警告）。だから VM 上でスペクトルを作らせてはいけない（§1-3）。
- **import の依存**: `transport.py` を import すると `materials`（xraylib・SciPy）・`physics`・`source` が読み込まれる。`detector.py` と `tally.py` は NumPy だけ。
- **`kernel.py` の蛍光の表**: `_k_shell_table_for_element`（65 行）は、蛍光線の最大エネルギーが 5 keV 未満の元素の K 殻分率を最初から 0 にする
  （O・Al で非ゼロの要素 0 個、Codex が確認）。だから 434 行の 5 keV 未満の局所吸収の分岐には、実在の材料の表では到達しない。
- **`beam_basis`**（`source.py` 28〜35 行）: `|d_z| < 0.999` かどうかで u の作り方が分かれる。
- **依頼 D の肖像のシーン**: box 463 個（アルミ）、70 kVp、2.5 mmAl、`rect`、検出器 z=0・15×18 cm・100×120 画素（`run_history_noise.py` 24〜27 行）。
- **試し計算**（2026-10-21、`transport.py`、各シーン 10 本×10⁵ 履歴、workers 4、M1 Mac。シーンの YAML は §1-5 に固定）:

  | シーン | 10⁶ 履歴の時間 | 主な量の相対 SEM（合計 10⁶ 履歴） |
  |---|---|---|
  | S1 水 60 keV 平行 | 8.7 s | 水の付与 0.077%、脱出 0.11%、一次の全体 0.75%、単散乱の全体 0.53%、ROI 1.6〜8.8% |
  | S2 多材料 120 kVp cone | 10.8 s | 軟部 0.14%、骨 0.14%、銅 0.60%、鉛 1.0%、アルミ 0.73%、蛍光区分の全体 6.1%、鉛の下の ROI の蛍光 6.9% |
  | S3 肖像 70 kVp rect | 29.2 s | アルミの付与 0.064%、一次の全体 0.080%、ROI の一次 0.6〜0.9%、多重散乱の ROI 7.9〜11% |

  初めの S2（70 kVp、厚さ 1 mm の銅を検出器の上）は蛍光区分が 10⁶ 履歴で 1.7×10⁻⁴（相対 SEM 44%）しか届かなかったので、120 kVp と厚さ 0.2 mm の鉛に変えた。

## 1. 設計

### 1-1. ファイル（すべて新規）

- `chatcarlo/gpu/g1_common.h`: `P5_FN` 相当の修飾子、Philox（`p5a0_common.h` の `philox`・`uniform`・`draw` を写す。出典の行を注記）、MT19937（`p5_rng.h` の MT を写す。CPU だけ）。
- `chatcarlo/gpu/g1_physics.h`: `kernel.py` の `_lerp_lookup`・`_element_index_frac`・`_select_element`・`_mu_and_parts_scalar`・`_sample_fluorescence_scalar`・
  `_sample_compton_bound_scalar`・`_sample_rayleigh_cos_theta_scalar`・`_scatter_direction_scalar`・`_isotropic_direction_scalar` を、**同じ演算の順で**写す。
- `chatcarlo/gpu/g1_geometry.h`: `_intersect_box_scalar`・`_material_at_scalar`・`_next_boundary_scalar`、検出面の交差（`DetectorPlane.intersect_segments` の 1 本版、同じ式と同じ受け入れ条件）。
- `chatcarlo/gpu/g1_history.h`: 1 履歴の関数 `g1_history`。順番は `transport_photons` の 1 歩と同じ（§0）。乱数の呼び出しの順は `_transport_one` と同じ。
  検出器が無ければ `_transport_one` と同じ動き。
- `chatcarlo/gpu/g1_cpu.cpp`（`clang++`／VM では `g++`、ctypes）、`chatcarlo/gpu/g1_gpu.cu`（`cp.RawKernel(..., backend='nvrtc')`、オプションは P8f と同じ `--fmad=false` 系）。
- `chatcarlo/gpu/bundle.py`: 束づくり（表、幾何、検出器、スペクトルの表、核のソース。目録に各 SHA256）。
- `chatcarlo/gpu/source_host.py`: `sample_source_photons` の `rect`/`cone`/`parallel`（heel・rotation なし）を、**NumPy だけで動く形に写したもの（VM 用）**。
  手元の API（`run_transport(..., engine='gpu')`）は既存の `sample_source_photons` をそのまま使い（親計画 D7）、VM 用の写しは関門 0d で既存の関数とのバイト一致に縛る。
  エネルギーは束の (energy, weight) の表から同じ `rng.choice`。
- `chatcarlo/gpu/core.py`: **NumPy と CuPy と ctypes だけを import する**駆動（CPU の双子と GPU、バッチごとの生の配列を返す）。VM はこれだけを使う。
  `chatcarlo/gpu/__init__.py` は何も import しない。
- `chatcarlo/gpu/engine.py`: 手元の API。`core` の生の配列から `DetectorTally`・`ScalarMoments`・`TransportResult` を組み立てる（`transport` などは関数の中で遅延 import）。
- `chatcarlo/gpu/colab_driver.py`: Colab CLI の小さな Python の駆動役（P8f の `p8f_cli/` は参考に読むだけで写さない）。束の目録から分割・送付・VM での組み立てと照合、
  `nohup` で起動、**10 分ごとの確認**（ユーザー決定 2026-10-21、20 分から変更）、結果は本体と受領票（SHA256 と識別情報）がそろって照合できてから確定、終了処理で `colab stop`。
  `.pending` などの途中のファイルを回収の対象にしない。存在だけで回収済みとしない。
- `chatcarlo/gpu/run_detector_image.py`: 依頼 D 用の GPU ランナー（`run_history_noise.py` と同じ引数・同じ `.npz`/`.json` の形、`engine` を記録する）。

### 1-2. 乱数と履歴番号

- 輸送の乱数: Philox(seed, history)。history は 0 から始まる通し番号（バッチ b の i 番目 = `start_b + i`）。ワーカー数・バッチの大きさ・GPU のスロットの割り当てに依らない。
- 線源の乱数: ホストの `np.random.default_rng(source_seed)`。バッチごとに `source_host` で `n_b` 個を順に抽選する（本番の `_run_batches` と同じく、
  1 本の Generator からバッチ順に引く）。輸送の seed と線源の seed は別の引数で、どちらも記録する。
- 検査用の MT モード（CPU だけ）: `_chunk_plan(n, 1, seed)` の種で初期化し、履歴を順に回す。`kernel.run_batch(..., n_chunks=1)` との比較だけに使う。
- **再開**: 回収済みのバッチは輸送を省くが、**線源は同じ順番・同じ個数で抽選して捨てる**（Generator の状態を連続実行と同じにする）。
  再利用する結果の識別情報: 輸送 seed・線源 seed・履歴の範囲・バッチの分け方・線源の辞書の SHA256・`source_host.py` と核のソースと表の SHA256。
- **API**: `run_transport` の末尾にキーワード専用の引数 `gpu_device: str = "cpu"`（`"cpu"` か `"cuda"`）と `source_seed: int | None = None` を足す。
  `engine='gpu'` では `seed` を必須とし（無ければエラー）、`source_seed` が無ければ `np.random.SeedSequence(seed).generate_state(1, np.uint64)[0]` とする。
  `n_workers` は 1 だけ（それ以外はエラー）。`engine == 'gpu'` の分岐は `run_transport` の先頭（既存の処理より前）に遅延 import と早期 return で置き、既存の行は変えない。
  実際に使った輸送 seed・線源 seed・実効の worker 数は結果に記録する（ランナーも）。

### 1-3. 線源の入力と VM

- 線源の配列（位置 3・方向 3・エネルギー 1、float64）はバッチごとに作って装置へ送る。VM 上でも同じ `source_host` で作る（VM に送るのは
  スペクトルの表と `source_host.py` で、SpekPy・xraylib は要らない）。スペクトルの表は Mac で `spectrum._default_spectrum` から作り、束の目録に SHA256 を書く。
- 関門 2 の CPU と GPU の比較は、**同じ VM 上で、同じ線源の配列**を両方に渡して行う（プラットフォームの違いを持ち込まない）。

### 1-4. 履歴ごとの出力と集計

- 履歴ごとの出力（関門 0b・2 の比較用、記録する場合だけ）: 散乱の回数、Compton+Rayleigh の回数、蛍光の回数、終わり方（吸収／脱出／検出）、
  最後のエネルギー、付与の合計、材料ごとの付与、検出の画素 (iu, iv) と区分、消費した乱数の数。
- 集計: 材料ごとの付与（keV）、脱出・吸収・検出の数、散乱の回数の和、蛍光の数、検出器の `category_fluence`・`photon_count`・`energy_sum_keV`・`energy_sum2_keV2`。
  GPU は atomic の加算。バッチごとに配列をホストへ戻し、既存のクラスの決まりどおりに積む（`detector.py`・`tally.py` は変えない）:
  - `DetectorTally`: 累積の画像に**今回のバッチを加えてから** `end_batch(n_b)`（上書きしない。スナップショットの差が壊れる）。
  - `ScalarMoments.add_batch`: **今回のバッチ単独の**材料ごとの付与を渡す（`end_batch` は無い）。
  - `TransportResult`: `run_transport` の `numpy` 経路と同じ正規化（`energy_deposited_MeV` は全履歴の和、`energy_deposited_sem_MeV` は 1 履歴あたりの平均の SEM、
    `fraction_*`・`mean_scatter_events`・`n_fluorescence` も同じ定義）。
- 健全性（毎バッチ）: 非有限の値・負の付与・範囲外の画素を拒否し、吸収 + 脱出 + 検出 = 履歴数を確かめる。

### 1-5. 関門 1 のシーン（固定）

- **S1**: 単色 60 keV、`parallel` 10×10 cm、焦点 (0,0,100) から −z。水の box 30×30×20 cm、中心 (0,0,15)。
- **S2**: 120 kVp、2.5 mmAl、`cone`（直径 20 cm、SID 100 cm）。軟部組織の box 30×30×10（中心 (0,0,10)）、その中に重なる骨 6×20×4（中心 (−3,0,10)）、
  銅 5×5×0.1（中心 (4,0,10)）、空気の中の鉛 6×6×0.02（中心 (4,−6,2)、蛍光が検出器に届く）、**下面が検出面に接する**アルミ 8×8×1（中心 (0,6,0.5)）。蛍光あり。
- **S3**: 依頼 D の `scene_portrait.yaml`（コミット済みのものをそのまま。SHA256 を記録）。
- 検出器: S1・S2 は z=0・30×30 cm・60×60 画素、法線 +z、u=+x。S3 は依頼 D と同じ 15×18 cm・100×120 画素。
- シーンの YAML は `docs/validation/gpu/g1/scenes/` に置く（S3 は依頼 D のファイルを参照し、SHA256 で照合）。

## 2. 判定（事前登録）

### 関門 0（手元、Codex が実装しテストにする）

- **0a 関数単位**: §1-1 の各関数を、`kernel.py` の対応する njit 関数と、同じ入力・同じ MT の種で比べる。各関数 10⁵ 入力（乱数の入力と、端:
  吸収端の直上・直下、格子の端、box の面上・角・面に平行な方向、`w=0`）。**戻り値と、書き換える作業配列（`_mu_and_parts_scalar` の `photo_e`・`compt_e`・
  `rayl_e`・`idx_e`・`frac_e` の有効範囲）と、MT の状態（624 語・index・続く 16 個の乱数。P8f の `test_p8f_gates.py` と同じ確かめ方）がバイト一致**。
  libm の差でバイト一致しない関数が出たら止まって報告する（許容を後から作らない）。CPU のビルドは P8f と同じ `-O2 -ffp-contract=off` 系とし、
  NumPy・Numba・LLVM・clang・CPU の版を記録する。
  **人工の蛍光の表**（テスト専用の入力。K 殻分率と放出確率が正、線が 5 keV 未満）を 1 つ足し、正常版では局所吸収・故障版では放出することを固定する。
- **0b 1 履歴の鎖（MT）**: 検出器なしで、次の固定の場合（全履歴で同じ位置・方向・エネルギー。`kernel.run_batch` の API に合わせる）について、
  `kernel.run_batch(..., n_chunks=1, seed=2026102100+i)` と CPU の双子の MT モードの、履歴ごとの 6 出力が 10⁴ 履歴すべてでバイト一致:
  | i | 幾何 | 位置 → 方向 | エネルギー | 蛍光 |
  |---|---|---|---|---|
  | 1 | S1 | (0,0,100) → −z | 60 keV | あり |
  | 2 | S2 | (4,0,100) → −z（銅を通る） | 100 keV | あり |
  | 3 | S2 | (4,−6,100) → −z（軟部組織のあと鉛を通る） | 100 keV（鉛の K 吸収端 88 keV より上） | あり |
  | 4 | S2 | 3 と同じ | 100 keV | なし |
  | 5 | S2 | (−3,0,100) → −z（重なった骨） | 60 keV | あり |
  | 6 | S2 | (1,2,100) → 方向 (0.3,0.2,−1) を正規化（斜め） | 80 keV | あり |
- **0c 検出面**: 交差の関数を `DetectorPlane.intersect_segments` と 10⁵ 本の線分（端: 面に平行、裏から、`t=ds` ちょうど、画素の境界上、検出器の外周上）でバイト一致。
  検出面は z=0・法線 +z のものに加えて、**傾けて動かした面**（法線 (0.2,−0.1,1) を正規化、u をそれに直交させた斜めの向き、中心 (1,−2,3)）も使う。
  **履歴関数でも強制の場合を 1 件ずつ**: 検出面が box の面と重なる（S2 のアルミ）、相互作用の点がちょうど検出面の上（MT の列を選んで作る）。
  どちらも `transport_photons` と同じく検出が先になることを、手で書いた期待値と比べる。
- **0c' 集計**: エネルギー・区分・画素の違う人工の履歴（10 件ほど）から、4 種類の画像（`category_fluence`・`photon_count`・`energy_sum_keV`・`energy_sum2_keV2`）と
  材料ごとの付与・脱出・吸収・検出の数を、手計算の値と比べる。人工のバッチで、不揃いの最後のバッチ・寄与 0 のバッチ・バッチが 1 つだけ（R が NaN）・
  `track_uncertainty=False` を確かめる。
- **0d 線源**: `source_host` と `sample_source_photons` が、S1・S2・S3 の線源の辞書（S2・S3 の `kvp` はスペクトルの表に置き換えた辞書と元の辞書の両方）と、
  **斜めの方向**（(0.3,0.2,−1) を正規化）・**`beam_basis` の分岐の境目**（`|d_z|` = 0.9989 と 0.9991）の `rect`/`cone`/`parallel` について、
  同じ `default_rng(seed)` から、**10⁴ 個ずつ 5 バッチ続けて**抽選した位置・方向・エネルギーと、抽選後の Generator の状態がバイト一致。
  再開（最初の 2 バッチを抽選して捨てる）と連続実行で、3 バッチ目の配列と最後の状態が一致する。
- **0e 世界の箱**: 束に入る世界の箱が、`Geometry(geoms, detector_plane=plane)` の `bbox_min/max` とバイト一致。
- **0f 故障の検出**: 次の 4 つの故障ビルド（コンパイル時の定数）で、0a〜0c のどれかが必ず落ちる: Rayleigh の (1+cos²θ)/2 の棄却を外す、
  束縛コンプトンの S/Z の棄却を外す、蛍光の 5 keV 未満の局所吸収を外す（人工の蛍光の表で落ちる）、後ろ優先を前優先にする。
- **0g VM の import**: xraylib・SciPy・SpekPy を import できなくした環境で、`chatcarlo.gpu.core` から小さな束（S1、10³ 履歴、CPU の双子）を実行できる。
- **0h numpy 経路の回帰**: 実装の前に基準のコミットのハッシュを記録し、S1・S2 を `transport.py` で `n=2×10⁵`・seed 42・workers 1・batch_size 10⁵・
  §1-5 の検出器・`track_uncertainty=True` で回した全項目（材料ごとの付与・SEM・R、割合、`mean_scatter_events`、`n_fluorescence`、検出器の 4 画像・
  `category_sum2`・`total_sum2`・`n_batches_hit`）を固定値として保存し、実装の後に完全一致を確かめる。

### 関門 1（手元、CPU の双子 対 `transport.py`、Claude が回す）

- 各シーン、両側とも **10 本×10⁶ 履歴**（独立な種。`transport.py` は seed 5000+i・workers 4、CPU の双子は輸送 seed 6000+i・線源 seed 7000+i。
  どちらも batch_size 2×10⁵）。
  試し計算の時間から、`transport.py` 側は 3 シーンで約 8 分。
- 比べる量（1 本ごとに 1 履歴あたりの値にし、10 本の平均と SEM = 標準偏差（ddof=1）/√10。ROI は**各本で先に合計してから**）:
  シーンの全材料（空気を含む。材料名はシーンから列挙）の付与、脱出の割合、`category_fluence[c].sum()/N` と 3 つの ROI の `category_fluence[c][ROI].sum()/N`（ROI は試し計算と同じ: S1 `[25:35,25:35]`・`[0:10,0:10]`・`[25:35,45:55]`、S2 `[25:35,25:35]`・`[32:44,12:24]`・`[20:30,40:50]`、
  S3 `[40:60,50:70]`・`[5:20,5:20]`・`[80:95,100:115]`、添字は `[iu, iv]`）。
  両側とも 10 本すべてで 0 の量は、比べずに「両側とも 0」と記録し K に数えない（片側だけ 0 なら不合格）。非ゼロで両側の SEM が 0 の量は、平均が完全一致なら合格、
  違えば不合格。NaN・inf が出たら不合格。
- 判定: 各量 z = (平均の差) / √(SEM₁² + SEM₂²)。**すべて |z| ≤ 3 なら合格。** |z| ≥ 4 が 1 つでもあれば不合格。3 < |z| < 4 が 1 つだけなら、
  両側を新しい固定の種（5100+i・6100+i・7100+i）で 1 回だけやり直し、そのときすべて |z| ≤ 3 なら合格。それ以外は不合格。
  比べる量の数 K（試し計算から 3 シーンで約 55）と、Σz²/K を報告する。
- 報告の書き方: 「この精度では差を検出しなかった」とし、各量の SEM と、検出できる差の目安（3√(SEM₁²+SEM₂²)）を並べる。同等性の証明とは書かない。

### 関門 2（T4、GPU 対 CPU の双子、同じ VM、Claude が回す）

- 2a: 関門 0a・0c の固定の入力（Philox の種を固定）を GPU の関数でも計算し、戻り値と消費数を CPU と比べる。整数の出力（画素番号、乱数の消費数、
  元素の選択）は**完全一致**。浮動小数の出力は一致率を報告する。
- 2b: S2 と S3 の 2,048 履歴（同じ線源の配列、同じ輸送の種）で、履歴ごとの出力の一致率と、整数の出力（散乱の回数、終わり方、画素）が分かれた
  履歴の数を報告する（報告だけ）。分かれた履歴があれば、**診断用の再実行**（その履歴だけを、歩ごとの位置・材料・乱数の消費を記録して CPU と GPU で回す経路）で、
  最初の 5 件の分かれた歩（何回目の歩で、どの量が）を記録する。
- 2c（合否）: S2 と S3 で、10 バッチ×10⁶ 履歴（同じ線源の配列、輸送 seed 8000、線源 seed 8001、バッチは対応させる）。関門 1 と同じ量について、各量を次の順で判定する:
  (i) 差（GPU − CPU）が、事前に決めた**丸めの上限**（その量への加算の回数 × 2⁻⁵² × 寄与の絶対値の和。P8f の `p8f_core.py` 57 行と同じ形）以内なら合格、
  (ii) そうでなければ、バッチごとの差の平均が、その差の SEM の 3 倍以内なら合格（履歴が分かれたことによる差）、(iii) SEM が 0 で (i) を超えるなら不合格。
  あわせて、1 バッチ（10⁵ 履歴）は履歴ごとの出力を記録し、そこから集め直した値と装置の集計を (i) の上限で照合する。健全性（§1-4）も毎バッチ確かめる。
- 速さ（報告だけ）: CPU の双子（1 スレッド）と GPU の履歴/秒、線源の抽選・転送・初期化を含む時間。依頼 D の 9 段階×2 本の GPU での所要時間の見積もり。

## 3. 対象範囲

- 変更してよい: 新規 `chatcarlo/gpu/`（§1-1）、`chatcarlo/transport.py` の `run_transport` の `engine` に `"gpu"` を足し、末尾にキーワード専用の
  `gpu_device`・`source_seed` を足すこと（§1-2。先頭の早期 return だけで、`numpy`・`kernel` の経路と既定値は変えない。`gpu` は box だけ・`kvp` か `spectrum`・`rect`/`cone`/`parallel`・`mas` などなし、を
  `gpu_engine_compatible` で検査し、外れたら理由つきで止める）、新規 `tests/test_gpu_*.py`、新規 `docs/validation/gpu/g1/`。
- 変更禁止: `chatcarlo/__main__.py`（依頼 A+B の未コミットの変更があるため。CLI は G4）、`kernel.py`・`physics.py`・`geometry.py`・`detector.py`・
  `source.py`・`spectrum.py`・`materials.py`・`tally*.py`（読んで写すだけ）、`prototypes/mv_electron/`（P5・P8f は読んで写すだけ）、`data/`、
  `docs/jsmp133/`（依頼 D のシーンとランナーは読むだけ）、依頼 A+B の未コミットのファイル、`examples/`。
- スコープ外: cylinder・sphere、ボクセルのタリー、`mas`・`ctdi_vol_mGy`・`heel_effect`・`rotation`、電子輸送、CLI、速さの工夫（box の加速構造など）。

## 4. 受入条件

- [ ] 関門 0a〜0f がすべて合格（`tests/test_gpu_g1_*.py`）。
- [ ] `transport.py` の `numpy` 経路が変わっていない: 既存の `tests/` 全件が合格し、S1・S2 の `transport.py` の固定の種の結果が変更前のコミットと同じ（新規テストで固定値と比べる）。
- [ ] `run_transport(scene, engine='gpu', device='cpu')`（CPU の双子）が、`TransportResult` の既存の項目（材料ごとの付与と SEM、脱出・吸収の割合、
  `detector` の `DetectorTally`）を返し、`gpu_engine_compatible` が cylinder・`heel_effect`・`mas`・`rotation`・ボクセルのタリーを理由つきで断る。
- [ ] 関門 1 が合格（Claude が回し、`docs/validation/gpu/g1/gate1.json` に、シーンの SHA256・種・履歴数・各量の平均と SEM・z・K・Σz²/K）。
- [ ] 束の目録: 表・幾何・検出器・スペクトルの表・核のソースの SHA256。VM 上の照合で 1 バイト変えた束を拒否する（テストは偽の束で）。
- [ ] Colab の起動役は偽の `colab` 実行ファイルで: 分割と組み立て、照合の失敗で止まる、セッションの消失で止まり回収済みを残す、再実行で回収済みを計算し直さない。
- [ ] 関門 2 が合格（Claude が T4 で回し、`gate2.json`）。
- [ ] `run_detector_image.py` が、依頼 D と同じ形の `.npz`/`.json` を書く（CPU の双子で 10⁴ 履歴の小さな実行をテストにする）。

## 5. テストコマンド

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -q -k gpu
.venv/bin/python -m chatcarlo validate examples/chest_room.yaml
```

Codex のテストは一時フォルダで完結させる（`~/.cache` に書けない）。関門 1・2 と Colab の実行は Claude が行う。

## 6. 停止条件

- 関門 0 が通らなければ関門 1 に、関門 1 が通らなければ Colab に進まない。関門 2 が落ちたら、依頼 D を GPU で回さない。
- `transport.py` の `numpy` 経路の出力が変わったら止まる。
- 判定の基準・履歴数・種・ROI は結果を見てから変えない。

## 書かなかったこと

- Numba の `cuda.jit` 案: CUDA Simulator で手元検査はできるが、将来 P8f の C++ の電子輸送をつなぐ負担が残る（親計画 D1、Codex の比較表）。
- 線源を装置側で抽選する案: 写す量が増え、heel・rotation のときにまた写し直しになる。転送が律速だと分かってから考える（親計画 D7）。
- `kernel.py` をそのまま検出器つきに拡張する案: `kernel.py` は参照実装として変えない。
- 463 個の box を速く調べる仕組み（格子など）: 速さは報告だけ。遅すぎれば G1 の結果を見て別に計画する。
