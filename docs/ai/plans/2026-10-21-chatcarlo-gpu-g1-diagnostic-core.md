# 計画 G1: chatcarlo/gpu の診断の核（box・多色・発散照射野・平面検出器、CPU の双子と GPU）

- 状態: draft（骨子。実装の詳細はまだ書いていない）
- 作成者: claude
- 実装担当: codex（Colab の実行と `~/.cache` を使う検査は Claude）
- 日付: 2026-10-21
- Codexレビュー: 1巡（骨子、親計画と同時。致命 0・実質 3・軽微 2・参考 3。すべて反映）
- 親計画: [2026-10-21-chatcarlo-gpu-engine.md](2026-10-21-chatcarlo-gpu-engine.md)（前提事実と構成の決定 D1〜D6 はそちら。ここでは繰り返さない）

## 目的

診断領域の光子輸送を、P8f と同じ形（共通の C++ 履歴関数、CPU の双子、CuPy RawKernel、Philox）で `chatcarlo/gpu/` に作り、
Python の API（`run_transport(..., engine='gpu')`）と依頼 D 用の専用ランナーから、box だけのシーン・`kvp` の多色スペクトル・`rect`/`cone`/`parallel` 照射野・平面検出器のタリーを計算できるようにする。
答える問いは 3 つ。

1. C++ の物理の関数は、`kernel.py` の Numba の関数と、同じ一様乱数を与えたとき同じ値を返すか（写し間違いがないか）。
2. CPU の双子の結果は、本番の参照 `transport.py`（NumPy、PCG64）と統計的に一致するか。
3. GPU（T4）の結果は、同じ線源の入力と同じ Philox の種で、CPU の双子と対応する履歴・バッチの範囲で一致するか（全面のバイト一致は求めない。親計画 §0）。

最初の実用の目標は、抄録依頼 D の肖像のシーン（box 463 個、70 kVp、`rect`、平面検出器）を、**依頼 D と同じ検出器・同じ保存形式の GPU 用ランナー**で回せること。CLI（`__main__.py` の `cmd_run`）は今、検出器を渡さず画像も保存しないので、CLI での画像化は G4 で扱う。線源はホストの `source.sample_source_photons` で抽選し、同じ配列を CPU の双子と GPU に渡す（親計画 D7）。

## 0. 前提事実（検証済み、2026-10-21。親計画 §0 に加えて）

- 平面検出器: `transport.py` 154〜248 行で、各飛行区間と検出面の交差（`DetectorPlane.intersect_segments`）を調べ、
  当たった光子を止めて `category_fluence[cat, iu, iv] += E / 画素の面積`、`photon_count`、`energy_sum_keV`、`energy_sum2_keV2` に加える。
  区分 `cat` は `detector.classify(散乱の回数, 蛍光の有無)` の 4 区分。バッチの R は `DetectorTally.end_batch`（スナップショットの差）。
- 線源（`source.sample_source_photons`、67〜212 行）: `rect` は SID の面の矩形に目標点を一様に抽選して焦点から結ぶ。
  `cone` は立体角一様。`parallel` は面上の一様な位置と同じ方向。エネルギーは `spectrum.sample_spectrum` が離散のビンを重みで選ぶ
  （ホストで SpekPy から作った (energy, weight) の表）。`heel_effect`・`rotation` は G1 の対象外（G3）。
- `kernel.py` の表（`bake_scene_materials`）: 材料×元素×格子の固定長の配列（`photo`・`compt`・`rayl`・`incoh_q/s`・`rayl_x/a`・
  `k_edge`・`k_omega`・`k_line_e/p`・`k_frac`・`density`）。G1 はこれをそのまま束にする（表を新しく作らない）。
- 手元に `clang++`（/usr/bin/clang++）あり。P8f の CPU の双子と同じ作り方ができる。

## 1. 判定（事前登録。数値は詳細を書く段で、試し計算で誤差を測ってから確定する）

- **関門 0（手元、関数単位）**: `kernel.py` の各スカラー関数（μ の引き、元素の選択、束縛コンプトン、Rayleigh、蛍光、方向の回転、box の交差・
  材料の判定・次の境界）と C++ の関数に、同じ入力と同じ一様乱数の列を与え、戻り値と消費した乱数の数が一致すること（許容は詳細の段で決める。
  乱数と数学関数に依存しない固定の場合はバイト一致、`log`・`exp` などが入る所は事前に決めた相対の許容）。
- **関門 1（手元、CPU の双子 対 `transport.py`）**: 検証の対象を先に固定する: `cone` と `rect` と `parallel`、**輸送される蛍光線を持つ材料**（K 線が 5 keV を超え、管電圧が K 吸収端より上。肖像のアルミの K 線は約 1.5 keV で局所吸収されるので、これでは試せない）、蛍光のあとの検出の区分、重なった box の後ろ優先、検出面と物体の境界の競合。`kernel.py` の K 殻の分率は表の補間、`transport.py` は xraylib の照会なので、関数を正しく写しても両者の一致は保証されない。だから比べる相手は `transport.py`。シーンは 60 keV の水の平行ビーム、蛍光を試す多材料の `rect`/`cone`、依頼 D の肖像。それらで、
  透過・材料ごとの付与・検出器の画像（区分ごと、ROI 平均）を比べ、各量の差 ÷ 合成した SEM の絶対値が 3 以下。比べる量の数 K と、
  偶然の不合格の扱い（新しい固定の種で 1 回だけやり直す）、平均も SEM も 0 の量の扱いを詳細の段で書く。報告は「この精度では差を検出しなかった」とし、SEM と検出できる差の大きさを並べる（同等性の証明とは書かない）。
- **関門 2（T4、GPU 対 CPU の双子）**: 同じ線源の入力と同じ Philox の種で、(a) 乱数と数学関数に依存しない固定の場合は厳密に一致、(b) 履歴ごとの出力の一致率と、経路・画素が分かれた履歴の数を報告する（分かれた履歴の扱いは詳細の段で決める）、(c) 合否は対応するバッチの差 ÷ SEM（P8f の関門 3 と同じ形）。代表の場合（単色の小さなシーン）を先に通し、次に肖像のシーン。
- 速さは報告だけ（合否なし）。線源の抽選・転送・初期化を含む時間と、依頼 D の 9 段階×2 本の所要時間の見積もりを出す。

## 2. 対象範囲

- 変更してよい: 新規 `chatcarlo/gpu/`（`*.h`・`*.cpp`・`*.cu`・Python の駆動・束づくり・Colab CLI の起動役・依頼 D 用の GPU ランナー）、`chatcarlo/transport.py` の
  `run_transport` の `engine` の分岐に `gpu` を足すこと（`numpy`・`kernel` の経路は変えない）、新規 `tests/test_gpu_*.py`、新規 `docs/validation/gpu/g1/`。
  `chatcarlo/__main__.py` は G1 では触らない（依頼 A+B の未コミットの変更があるため。CLI は G4）。
- 変更禁止: `kernel.py`・`physics.py`・`geometry.py`・`detector.py`・`source.py`・`spectrum.py`・`materials.py` の振る舞い（読んで写すだけ）、
  `transport.py` の既存の経路、`prototypes/mv_electron/`（P8f の `p5_rng.h` などは読んで写す。共通化は G5）、`data/`、`docs/jsmp133/`。
- スコープ外: cylinder・sphere、ボクセルのタリー、`mas`・`ctdi_vol_mGy`・`heel_effect`・`rotation`、電子輸送、速さの工夫、多数の box の高速化
  （463 個の box を毎回すべて調べる素朴な方法で始め、速さは報告だけ。遅すぎる場合の対策は G1 の結果を見て別に計画する）。

## 3. 停止条件

- 関門 0 が通らなければ関門 1 に、関門 1 が通らなければ Colab に進まない。関門 2 が落ちたら、依頼 D を GPU で回さない。
- `transport.py` の `numpy` 経路の出力が変わったら止まる（既存の `tests/` 全件と固定の出力）。

## 4. 詳細の段で決めること（骨子のレビューの後）

- 乱数の消費の順番（関門 0 で一様乱数の列を注入できる形）と、Philox の (seed, history) の割り当て（バッチ・ワーカーとの関係）。
- 履歴ごとの出力（関門 2 のバイト一致に使う量）と、検出器の集計の形（GPU では atomic で足す。バッチの R の取り方）。
- 束の目録とハッシュ（表・スペクトル・幾何・検出器・核のソース）。
- 関門 1 のシーン・履歴数・比べる量・K。試し計算で SEM を測ってから書く。
