# 親計画: chatcarlo/ に、診断領域（低エネルギー）から治療領域までを 1 つの GPU エンジンで計算できるようにする

- 状態: approved（構成の決定 D1〜D4・D7 と段の分け方をユーザー承認 2026-10-21。D5・D8 は G5 の前に決める）
- 作成者: claude
- 実装担当: codex（Colab の実行・`~/.cache` を使う検査は Claude）
- 日付: 2026-10-21
- Codexレビュー: 1巡（骨子: 致命 0・実質 3・軽微 2・参考 3。原文 `data/logs/codex_gpu_review1.md`。すべて反映）
- 子計画: 第 1 段 [2026-10-21-chatcarlo-gpu-g1-diagnostic-core.md](2026-10-21-chatcarlo-gpu-g1-diagnostic-core.md)

## 目的

ユーザーの決定（2026-10-21）: 「chatcarlo/ で、GPU でも低エネルギーでも計算できるようにする計画を最初に行う」。
今は、診断領域（`chatcarlo/`、150 keV まで）は CPU（NumPy・Numba）だけ、治療領域（`prototypes/mv_electron/`、10 MV）は
GPU の核があるが平らな板の積み重ねだけで、Rayleigh と束縛コンプトンを持たない。抄録（JSMP133）の
「対話で確定した条件が Google Colab の GPU で計算される」を、診断・治療の両方で成り立たせるのが最終の目標。

この親計画が答える問い: **どの順番で、何を合格の条件にして、1 つの GPU エンジンへまとめるか。** 各段は子計画で詳細を決める。

## 0. 前提事実（検証済み、2026-10-21）

- **診断領域の per-history の物理は、すでにスカラーの形で `chatcarlo/kernel.py` にある**（Numba njit）。
  Rayleigh（`_sample_rayleigh_cos_theta_scalar`、475 行）、束縛コンプトン（`_sample_compton_bound_scalar`、440 行）、
  K 殻蛍光（`_sample_fluorescence_scalar`、400 行）、box の交差（`_intersect_box_scalar`、539 行）、
  表の焼き込み（`bake_scene_materials`、165 行。xraylib と NIST の表を固定長の配列にする）。`transport.py` と統計的に照合済み
  （`docs/speedup_baseline/kernel_crosscheck.py`）。乱数は MT19937（np.random）。
- `kernel.py` の制限（`transport.kernel_engine_compatible`、294〜310 行）: box だけ、単色（`source.spectrum` 1 要素）、
  `parallel` 照射野だけ、`mas`/`ctdi_vol_mGy`/`heel_effect`/`rotation` なし、workers=1。平面検出器のタリーは `transport.py` だけ
  （`transport.py` 154〜248 行、`detector.py` の `DetectorTally`）。
- 治療領域の GPU の核（P8f、`prototypes/mv_electron/gpu/p8f_*`）の形: 1 つの C++ の履歴関数 `p8f_history` を、
  CPU（`clang++` で共有ライブラリ、ctypes。`p8f.py` 88〜97 行）と GPU（`cp.RawKernel(..., backend='nvrtc')`、`p8f_colab.py` 39 行）の
  両方でコンパイルする。乱数は Philox（`p5_rng.h`、`rng_init(&s, seed, history)` で履歴ごとに独立）。表はハッシュつきの束で送る（D4 の常設ルール）。
  Colab CLI での自動実行は `prototypes/mv_electron/gpu/p8f_cli/` に実績（2026-10-21、T4）。
  P8f の光子の物理: 自由電子の Klein–Nishina（`p5_compton`）、光電効果と殻の選択・緩和（`p8f_select_shell`・`p8f_relax`）、対生成。
  **Rayleigh なし、束縛コンプトンなし**（比べる相手の EGS5 の入力で両方 0。`run_p8e.py` 155〜158 行）。幾何は z 方向の層だけ。
- 150 keV の上限は 3 か所: `scene.py` 80〜85 行（`source.spectrum`）・111〜113 行（`kvp` 20〜150）、`materials.py` 96 行
  （`_XS_GRID_MAX_KEV = 150.0`、断面積の格子の上限）。
- 過去の見積もり（路線 2、`docs/gpu_speedup/route2_scoping/RESULTS.md`、2026-08-20）は「`kernel.py` の CUDA 移植は、形・スペクトル・照射野・
  校正の 4 つの拡張が先に要り、Colab での開発の往復が遅い」で着手を見送った。**今回はこの 2 つの障害の前提が変わった**:
  **開発と検証の手段が改善した**（障害が消えたのではない。形・スペクトル・照射野・校正の拡張は残り、G1〜G3 に分ける）:
  (1) P8f の形（CPU の双子で手元で開発・検査し、GPU は関門だけ）で、Colab に依存する往復は関門の数回に減る、
  (2) Philox で CPU と GPU が同じ乱数列を使うので、対応する履歴・バッチどうしを直接比べられる（ただし `log`・`exp` などの数学関数の
  CPU/GPU の差は残るので、全面のバイト一致は求めない。P8f の `compare_histories` も一致率は報告だけで、判定は対応バッチの差と SEM）。
  乱数の置き換えを含む移植全体の統計的な検査は、G1 の関門 1（CPU の双子 対 `transport.py`）が担う。
- 抄録依頼 D の肖像のシーン（`docs/jsmp133/request_d_history_noise/scene_portrait.yaml`）: box 463 個（アルミ 463、土台 1 枚を含む）、
  `kvp: 70`、ろ過 2.5 mmAl、`rect` 照射野（`sid_cm: 100`）、z=0 の平面検出器。cylinder・sphere なし。
- 共有チェックアウトの状態（2026-10-21、全計算停止後）: 依頼 A+B（承認済み計画 `2026-10-21-mv-entry-colab.md`）の実装途中の
  変更が未コミットで残る（`chatcarlo/__main__.py` +55 行、`prototypes/mv_electron/entry/`、`tests/test_mv_dispatch.py` ほか）。
  この計画の実装は、それらのファイルに触れない（段 G1 は新規ファイルと `transport.py` の分岐だけ。`__main__.py` は G4 まで触らない）。

## 1. 構成の決定（ユーザーの承認が要る）

| # | 決定すること | 推奨 | 理由 |
|---|---|---|---|
| D1 | エンジンの作り方 | **P8f の形**（共通の C++ 履歴関数、CPU の双子＋CuPy RawKernel、Philox、ハッシュつきの束、Colab CLI）。`kernel.py` の Numba の cuda.jit への書き換えは採らない | 手元で CPU の双子を作れば GPU なしで開発・検査でき、GPU は CPU とのバイト一致で確かめられる。P8f で実績がある |
| D2 | 光子の物理の出どころ | **診断の物理は `kernel.py` のスカラー関数を C++ に写す**（表は `bake_scene_materials` が作るものをそのまま使う）。治療の電子輸送・対生成・殻の緩和は、後の段で P8f の関数を同じエンジンに入れる | 診断は `transport.py`・EGS5 60 keV で照合済みの物理と同じにして、過去の結果と比べられるようにする |
| D3 | 物理のスイッチ | Rayleigh・束縛コンプトン・蛍光（K 殻だけ／全殻の緩和）・電子輸送（カーマ近似／輸送）を、シーンの設定で切り替える。**診断の既定 = 今の `transport.py` と同じ**（Rayleigh・束縛あり、K 殻蛍光、5 keV 未満の線は局所吸収、カーマ近似）。**治療の既定 = 今の P8f と同じ**（Rayleigh・束縛なし、全殻の緩和、電子輸送） | 両方の既存の合格結果を壊さない。後の段で、治療の既定のまま P8f の鎖 0 がバイト再現することを回帰の関門にする |
| D4 | 置き場所 | 新規 `chatcarlo/gpu/`（C++ ヘッダー、CPU のビルド、CuPy の駆動、束づくり）。G1 は Python の API（`run_transport(..., engine='gpu')`）と専用のランナーから使い、CLI の `--engine gpu` は G4 で足す。既定の `numpy` 経路は変えない | 本番の参照実装 `transport.py` を残し、新しいエンジンと統計的に比べ続ける |
| D5 | 治療領域を `chatcarlo/` に入れること・150 keV の上限を外すこと | **方針の変更なので、段 G5 の前にユーザーが明示的に決める**。決めたら CLAUDE.md と `.claude/rules/mv.md` の「治療領域は `chatcarlo/` の外」の文を書き換える | 今の決まりと逆になる |
| D6 | 表の出どころ | 診断のモードは CLAUDE.md の表（μ の内訳は xraylib、カーマの μen は NIST XAAMDI、H*(10) は ICRP 74）を変えない。治療のモードは P8 の束（ESTAR・ELSEPA など）。どのモードがどの表を使うかを束の目録に書き、表はスクリプトだけで作る（手で打たない） | CLAUDE.md の「データの来歴」 |
| D7 | 線源の抽選 | **ホストで抽選し**、位置・方向・エネルギーの配列を CPU の双子と GPU に同じものとして渡す。輸送の乱数だけを Philox にする。手元は既存の `source.sample_source_photons` をそのまま使い、VM（xraylib・SpekPy が無い）は NumPy だけの写しを使って、既存の関数とのバイト一致に縛る（G1 関門 0d） | 線源を写し直さずに済み、多色・`rect`・`cone`（G3 の heel・rotation も）を既存の定義で使える。CPU と GPU を同じ入力から比べられる。転送は 100 万履歴あたり約 56 MB。抽選と転送の時間は G1 で測り、律速なら後の段で装置側に移す |
| D8 | G5 で治療の物理が扱う幾何 | G5 の前に決める（層だけに限るか、box などにも広げるか）。広げる場合は電子の境界の扱いが新しい研究になる | Codex の指摘 |

## 2. 段の分け方（各段は子計画で詳細を決める）

| 段 | 内容 | 合格の条件（要約） |
|---|---|---|
| **G1** | 診断の核: box・`kvp` スペクトル・`rect`/`cone`/`parallel` 照射野・平面検出器のタリー、Rayleigh・束縛コンプトン・K 殻蛍光（カーマ近似）。CPU の双子と GPU | 関数単位で `kernel.py` と一致、CPU の双子が `transport.py` と統計的に一致、GPU が CPU の双子と履歴ごとにバイト一致。**依頼 D の肖像のシーンを GPU で回せる** |
| G2 | ボクセルのカーマ・H*(10) のタリー（飛跡長、バッチの R/SEM） | `transport.py` の `accumulate_track_length` とボクセルごとに統計的に一致。2026-10-07 の EGS5 60 keV の組を GPU で再実行し、同じ判定基準で比べる（既存の 42 合格・1 不合格・4 統計不足と、EGS5 の Rayleigh の警告は、そのまま引き継いで報告する） |
| G3 | cylinder・sphere（後ろ優先の規則）、`mas`・`ctdi_vol_mGy`・`heel_effect`・`rotation`（校正はできるだけホスト側） | `examples/` の 6 シーンが `--engine gpu` で動き、`transport.py` と統計的に一致 |
| G4 | 対話の入口: CLI の `run --engine gpu`（検出器の設定と画像の保存を含む）を Colab CLI で自動実行し、承認した条件と実際の引数の一致を照合（依頼 A+B の承認・ハッシュの層を使う） | 入口から人の手を介さずに 1 回通る。わざと条件を変えると止まる |
| G5 | 治療の物理を同じエンジンへ（電子輸送・対生成・殻の緩和を P8f から）。150 keV の上限を外す（D5 の決定の後） | 治療の既定で P8f の鎖 0（保存バッチ）がバイト再現。P6・P8f の EGS5 判定を新しいエンジンで再現（既存構成の再現であり、新しく対応する構成の検証とは分けて報告する） |

## 3. 止まっている作業の扱い（ユーザーが決める）

| 作業 | 推奨 |
|---|---|
| 依頼 A+B（治療の入口、P7 の核、承認済み） | 承認・ハッシュの層はどの核にも使えるので、**再開して完成させる**（G4 で、この層を新しいエンジンの入口に流用する）。ただし再開はユーザーの指示の後 |
| 依頼 D（肖像、CPU で 9 段階×2 本） | **済（2026-10-21）**: G1 の GPU（T4）で 18 本、傾き −0.501±0.004（`docs/jsmp133/request_d_history_noise/RESULTS.md`）。以下は当初の方針: G1 の GPU で回し直す。事前記録の改訂（エンジンの変更）を本計算の前に書く。CPU で終わった分は消さずに残し、G1 の統計的な照合の参照に使える |
| 3D 線量表示の計画 | エンジンに依存しない（.npz を描く）。再開してよい |

抄録の文は、G1〜G4 が 11/1 までに通らなかった場合だけ、その時点で通った範囲に合わせて直す（この 1 行以外では扱わない）。

## 4. 対象範囲（親計画として）

- 変更してよい（各段の子計画で絞る）: 新規 `chatcarlo/gpu/`、`chatcarlo/transport.py`（`--engine gpu` の分岐だけ）、新規 `tests/test_gpu_*.py`、
  新規 `docs/validation/gpu/`、各段の子計画が名指しする文書。
- 変更禁止: `transport.py` の `numpy` 経路と `kernel.py` の振る舞い（参照実装）、`prototypes/mv_electron/` の既存ファイル（P5〜P8f、G5 では
  読んで写すだけ）、凍結対象（`.claude/rules/mv.md`「Data, freezing and deletion」）、`data/`、依頼 A+B の未コミットのファイル、`docs/jsmp133/`。
- スコープ外: 速さの工夫（状態機械・ワープの整列など。各段で速さは報告だけ）、heightfield などの新しい形、診断での電子輸送の既定化（D3 で別に決める）。

## 5. 停止条件

- 各段の関門が落ちたら、次の段に進まずに報告する。判定の基準は結果を見てから変えない。
- `transport.py` の `numpy` 経路の出力が 1 ビットでも変わったら止まる（既存テストと固定の出力で確かめる）。
