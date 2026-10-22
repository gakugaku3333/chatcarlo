# G2a local acceptance — 関門 0・1 合格（2026-10-22）

**更新（Claude、2026-10-22）**: 下の「Stop record」の後、Claude が落ちた既存テスト 1 項目を置き換え（計画 §11）、関門 1 を実行した。runner の JSON 書き出しが NumPy の bool で落ちたので書き出しだけ直し（`run_gates.py` の `write`）、最初から流し直した。全テスト 618 passed（161.8 s、Claude が再実行）。**関門 2・3（Colab）は未実行。**

## 関門 1（CPU の双子 対 本番 NumPy、各 4×10⁶ 履歴＝20 バッチ × 2×10⁵、2 cm 格子）

参照は種を固定して作り直し、2 回の実行でハッシュが一致した（water 2680dc28…、lead 9e6d3ab5…）。

| シーン | 量 | 判定対象 N | max\|z\|（上限） | \|z\|>3 の割合 | z² の平均 | 領域合計の z | 判定 |
|---|---|---|---|---|---|---|---|
| S1 水 60 keV | カーマ | 129,727 | 4.98（8.42） | 0.58% | 1.08 | 水 +0.73、空気 −0.27 | 合格 |
| S1 水 60 keV | H*(10) | 134,445 | 5.00（8.44） | 0.56% | 1.08 | 水 +0.94、空気 −0.31 | 合格 |
| S2 鉛・骨 80 kV | カーマ | 69,280 | 4.83（8.08） | 0.55% | 1.10 | 水 −0.86、骨 +1.21、鉛の層 −0.29、空気 −0.63 | 合格 |
| S2 鉛・骨 80 kV | H*(10) | 81,039 | 5.12（8.17） | 0.53% | 1.09 | 水 −0.86、骨 +0.70、鉛の層 −0.45、空気 −0.21 | 合格 |

- 脱落（参照だけ適格／双子だけ適格）は両向きでほぼ同数（例: S2 カーマの鉛の層 523／499、空気 13,479／13,793）。
- \|z\|>3 の割合 0.53〜0.58% と z² の平均 1.08〜1.10 は、正規（0.27%、1.0）と自由度 19 の t（0.74%、1.12）の間で、各 SEM が 20 バッチから推定されていることと合う。
- 読み方: この標本の精度で不一致を検出しなかった。数 % 以内の一致の保証ではない。
- 時間（手元）: 本番の参照 25 s・30 s、CPU の双子 22 s・27 s（各 4×10⁶ 履歴）。


G2 tally and validation code were added within the approved scope. Acceptance is
incomplete. No Colab gate was executed. No commit or staging was performed.

| Command (from plan section 7) | Result |
|---|---|
| `.venv/bin/python -m pytest tests/test_gpu_g2.py -q` | 20 passed in 20.98 s |
| `.venv/bin/python -m pytest tests/ -q` | 1 failed, 617 passed, 2 warnings in 175.62 s; exit 1 |
| `PYTHONPATH=. .venv/bin/python docs/validation/gpu/g2/run_gates.py --gate 0` | 0a, 0b, 0c pass; exit 0 |
| `PYTHONPATH=. .venv/bin/python docs/validation/gpu/g2/run_gates.py --gate 1` | Interrupted, exit 130; no statistical verdict |

The new tests cover the production coefficient comparison (200,000 energies per
material, knots and both sides), 1,000,000 box overlaps, deterministic grid
boundary cases, artificial grid/history moments, terminal detector ties, the
frozen G1 native outputs, API grid/statistics, ABI positions, CUDA source assembly,
optional tables/default margin, runner hash rejection, historical world bounds,
and numerical reproduction by the imported historical comparison code.

## Stop record

The full test run and local gates were run concurrently. When an existing-test
failure marker was observed, gate 1 was interrupted under plan section 4.
`make_reference.generate` had completed 19 of 20 water reference batches. It had
not saved a new reference batch NPZ, started the lead reference, or run the twin.
`gate1.json` records this interruption, not a physical comparison failure.
Reference hashes and a gate-1 comparison are consequently unavailable.

The existing failing test is
`tests/test_gpu_g1_history.py::test_gpu_g1_compatible[dose_grid]`, line 42:
`assert not ok and reason`; the observed value is `ok=True`.
The approved plan requires `gpu_engine_compatible` to accept `dose_grid` and
permits only the new `tests/test_gpu_g2.py` among test-file edits. The existing
test was left unchanged. Full output: [tests_all.log](tests_all.log).

## Scope check

`git diff --stat` shows no changes in `tally.py`, `materials.py`,
`dose_coefficients.py`, `kernel.py`, the old 60 keV validation directory,
`approval.py`, or `cli_run.py`. The only `transport.py` difference passes
`grid_resolution_cm` in the GPU branch. Existing `colab_driver.py` workers,
source reference, launch and comparison functions are unchanged; only the new
`g2-gate` action and its dispatch were added. `git diff --check` passed.

`data/` remains untracked and was not staged. Gates 2 and 3 were not run.

## 関門 2（Colab T4、鉛のシーン S2、10 バッチ × 10⁵ 履歴、同じ線源と種）— 不合格（2026-10-22）

- 判定（`paired_judge`、計画どおり）: **カーマ 298,900 ボクセル中 61 個、H*(10) 64 個が不合格**（丸めの判定値にも、対応バッチの差の 3 SEM にも入らない）。箱の判定は合格と出たが、鉛のシーンには箱が無く（形 (0,)）、空の配列に対する判定で、箱を確かめた結果ではない。要約: [gate2_lead.json](gate2_lead.json)。
- 整数: 比較した履歴の記録（相互作用の回数・終わり方・吸収量など。途中の座標は比べていない）の不一致 0、ボクセルの加算回数の不一致 0、寄与の有無の不一致 0（10 バッチすべて）。
- 落ちた 61 個の差は相対 1.0e-15〜1.2e-14（中央値 3.6e-15）。7〜10 バッチで同じ向きの差が出ている。
- 推定（未検証）: 判定値の C=16 がボクセルに 1 回だけ足され、寄与の回数に比例する数学関数（`log`・`exp`）の CPU/GPU の差を覆っていない（計画の詳細の初版では C は「1 つの寄与あたり」と書いていた。2 巡目の書き直しでこの区別が落ちた。Codex 2 巡目の参考 11 の指摘）。物理や経路の食い違いではないと見ている。
- 計画の停止条件どおり、PDD の幾何（関門 2 の残り）と関門 3 には進まず、ユーザーの判断を待つ。判定の基準は変えていない。
- 時間（VM）: CPU の双子 6.36 s、GPU 1.18 s（10⁶ 履歴、線量を数える）。
- 経緯: 1 回目は Colab CLI が組み立ての前に止まり、2 回目は `g2-gate` が `chatcarlo` を import できずに VM の上で落ちた（Claude が `colab_driver.g2_gate` に `PYTHONPATH` を渡す 1 行を足した）。起動スクリプト `colab_g2.py` は Claude が書き、`colab` の呼び出しの制限時間、10 分ごとの確認、毎回の回収、生存記録を足した（教訓: docs/lessons_learned.md 2026-10-22）。

### 関門 2 の不合格の原因の確認（Claude、手元、2026-10-22。判定は不合格のまま）

**有力な説明: 係数の `log`・`exp` ではなく、線分の長さの評価が違う。座標の最後の 1 ビットの違いが、ボクセル境界での長さの割り振りを隣へ移していると考えると、観測とすべて合う。** 同じ姿の再現は GPU の差の原因の特定ではなく、DDA の内部の交差位置や短い区間の差の計算も候補に残る（Codex の指摘）。

1. 寄与の値そのものが違う: 加算が 1 回だけのボクセル（足し算の丸めが無い）でも 27% で値が違い、相対差は最大 3.0e-9。足し算の並べ替えの丸めだけでは説明できない（並べ替えの最悪の幅を超えるボクセルが約 64 万個）。
2. 長さの違い: 相対差 1e-13 超の加算 1 回のボクセル 16,832 個で、H*(10) の相対差 ÷ カーマの相対差 = 1.000（5〜95% で 0.999〜1.001）。係数の差なら 2 量で違うはずなので、両者に共通の「長さ」が違う。
3. 保存: バッチの全格子の和は CPU と GPU で一致（相対差 0）。符号つきの差の和は、差の絶対値の和の 2〜6%。
4. 再現: CPU の双子を写して、相互作用の点を 1 ulp ずらす版（`nextafter`、リポジトリ外の写し）を作り、同じ 1 バッチ（10⁵ 履歴）を比べた。記録の違う履歴 0、加算回数は一致、値の違うボクセル 48%（GPU 57%）、相対差の p99 5.5e-14（GPU 1.4e-13）、最大 2.0e-10（GPU 1.3e-10）、全体の差 0、2 量の差の比 1.000。GPU の差と同じ姿になった。
   - 仕組み: 散乱の向きの計算（`sin`・`cos`・`sqrt` など）は CPU と GPU で最後のビットが違うことがある（`.claude/rules/mv.md` の既知の性質）。経路は同じでも座標が ulp でずれ、境界をかすめる短い区間で相対差が大きくなる。
   - 未解明: 落ちたボクセルでは 10 バッチの差の向きが 9 割以上そろう。偶然の ±1 ulp なら向きはばらつくはずなので、GPU の数学関数に向きの偏りがある可能性がある（未確認）。
5. 判定の式が合わなかった理由: 丸めの判定値は「足し算の丸め＋固定の余裕」で、長さの割り振りの移動を想定していなかった。対応バッチの差の 3 SEM も、向きのそろった微小な差を救えない。

## 関門 2 の再実行（追補 2 の判定、新しい種: 輸送 3300・線源 3301）

### 鉛のシーン — 合格（2026-10-22）

- 格子（カーマ・H*(10)）: 10 バッチすべてで、幅（相対 1e-6 ＋ A）を外れたボクセル 0、全体の保存は合格、片側だけ 0 のボクセル 0。
- 整数: 加算回数・寄与の有無・履歴の記録の不一致はすべて 0。箱は無し（期待 0 個、一致）。要約: [gate2_lead_v2.json](gate2_lead_v2.json)。
- 時間（VM、10⁶ 履歴）: CPU の双子 6.57 s、GPU 1.22 s。
- 起動スクリプトの不具合（結果には影響なし）: VM の上の計算のプロセスは、終わった後も Jupyter のカーネルに回収されずゾンビ（状態 Z）で残り、`os.kill(pid, 0)` では生きて見えた。そのため起動スクリプトが終了に気づかず、Claude が止めた（結果は確認のたびの回収で手元に届いていた）。`colab_g2.py` を、`/proc/<pid>/stat` の状態 Z を終了と扱うよう直した。G1 の起動処理は完了の印のファイル（`done.json`）で判断している。

### 60 keV の PDD の幾何（47 箱）— 合格（2026-10-22）。関門 2 は完了

- 格子: 10 バッチすべてで幅を外れたボクセル 0、全体の保存は合格、片側だけ 0 のボクセル 0。
- 箱: 47 個（期待 47）。和・2 乗和とも 47 個すべてが丸めの判定値の内側（対応バッチの差の SEM による救済は 0）。
- 整数: すべて 0。要約: [gate2_pdd_v2.json](gate2_pdd_v2.json)。時間（VM、10⁶ 履歴）: CPU の双子 12.28 s、GPU 2.15 s。
- 起動スクリプトは終了を自分で検出し、結果を回収してセッションを止めた。
