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
