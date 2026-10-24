# 治療領域拡張（MV）の状況一覧

10 MV 級の光子・電子・陽電子の連成輸送の試作（`prototypes/mv_electron/`、本体 `chatcarlo/` の外）の段階ごとの状況。
各段階の正典はリンク先の RESULTS / DISCUSSION。ここは索引で、判定や数値の正は常にリンク先にある。

**書き方の決まり（2026-10-19）**: 段階が終わったら、(1) その段階の RESULTS（と必要なら DISCUSSION）を書き、
(2) 下の表に 1 行足す。CLAUDE.md には段階の記録を書かない。常設のルールが変わったときだけ
[`.claude/rules/mv.md`](../../../.claude/rules/mv.md) を直す。

## 段階の一覧

| 段階 | 日付 | 状態・判定 | 正典 | 一言 |
|---|---|---|---|---|
| 全体計画 | 2026-10-07 | draft | [plan](../../ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) | 発表の筋は [presentation_jsmp133](../../presentation_jsmp133.md) |
| P0 | — | 完了 | [RESULTS](p0/RESULTS.md) | xraylib は約 800 keV まで。MV の光子データは NIST XCOM |
| P1 | — | 完了（EGS5 比較は 3.02% で許容 3% を外れた） | [RESULTS](p1/RESULTS.md) | 電子だけの試作。弾性 DCS は遮蔽 Rutherford |
| P1b | — | 完了 | [RESULTS](p1b/RESULTS.md) | EGS5 と同じ DCS（`dcslib`、研究用の対照で配布しない）で 0.54% |
| P1c | — | 完了・**判定は未決** | [RESULTS](p1c/RESULTS.md) | `mf`（解析式、低 Z のみ）。K9(i) が −2.02% で止まった。下の注記を参照 |
| P1d | — | 完了 | [RESULTS](p1d/RESULTS.md) | `K1_max` = 0.025 を採用（のちに P3s の規則に置き換え） |
| 弾性 DCS の出典 | 2026-10-08 | ユーザー決定 | [全体計画](../../ai/plans/2026-10-07-native-10mv-photon-electron-transport.md) | G4EMLOW は商用不可で不採用、ELSEPA で自前生成 |
| P1e | 2026-10-09 | **不合格** | [RESULTS](p1e/RESULTS.md) | `IHEF=1` で表の検査 52/228 件が不合格 |
| P1f | 2026-10-09 | 合格 | [RESULTS](p1f/RESULTS.md) | `IHEF=0`、2 MeV で EGS5 と 0.75% |
| P2 | 2026-10-10 | Q0–Q4 合格、Q6 未判定、Q7 統計不足 | [RESULTS](p2/RESULTS.md) | 単色 1.25–10 MeV、30 cm 水平板 |
| P2b | 2026-10-11 | ユーザー決定（基準 v2） | [RESULTS](p2b/RESULTS.md)・[plan](../../ai/plans/2026-10-11-mv-p2b-criteria-redesign.md) | 5 cm ブロック平均が `Dref` の 1% 以内かつ SEM 0.33% 以内 |
| P2c | — | 完了 | [DISCUSSION](p2c/DISCUSSION.md) | 陽電子の飛行中消滅（常に有効） |
| P3a | — | 完了 | [DISCUSSION](p3a/DISCUSSION.md) | Mohan 10 MV の平板。照射野の PDD/OCR ではない |
| P3s | 2026-10-12 | ユーザー採用 | [DISCUSSION](p3s/DISCUSSION.md) | ステップ規則 K1 = 0.1 / 0.05 / 0.025 |
| P3r | 2026-10-12 | 完了 | [DISCUSSION](p3r/DISCUSSION.md) | 制動放射の光子を輸送 |
| P3t | 2026-10-12 | 完了 | [RESULTS](p3t/RESULTS.md) | ループの無駄を除去。新しい作業はこちら |
| P3u | 2026-10-12 | 完了 | [RESULTS](p3u/RESULTS.md) | 並列バッチのランナー |
| P3b（骨子） | 2026-10-12 | — | [plan](../../ai/plans/2026-10-12-mv-p3b-field-pdd-ocr.md) | 10×10 cm²、発散、SSD 100 cm |
| P3b-1 | 2026-10-12 | 完了（**B8 は不合格のまま記録**） | [RESULTS](p3b/RESULTS_P3B1.md) | 照射野の試作の核 |
| P3b-2 | 2026-10-12 | 完了 | [RESULTS](p3b/RESULTS_P3B2.md) | EGS5 の照射野ユーザーコード |
| P3b-3 | 2026-10-13 | 11 量すべて合格 | [RESULTS](p3b/RESULTS_P3B3.md)・[DISCUSSION](p3b/DISCUSSION_P3B3.md) | 照射野の PDD/OCR を EGS5 と比較 |
| P4 | 2026-10-14 | すべて合格 | [P4a](p4/RESULTS_P4A.md)・[P4b/c](p4/RESULTS_P4BC.md)・[plan](../../ai/plans/2026-10-13-mv-p4-independent-checks.md) | Fano・カーマ・飛程（EGS5 に頼らない検査） |
| P5（方針） | 2026-10-14 | — | [plan](../../ai/plans/2026-10-14-mv-p5-gpu-rewrite.md) | CuPy RawKernel、検証の鎖 0/1/2 |
| P5a0 | — | 完了 | [RESULTS](p5/RESULTS_P5A0.md) | ツールの試行 |
| P5a | — | 完了 | [RESULTS](p5/RESULTS_P5A.md) | C++ の核、保存バッチをバイト単位で再現 |
| P5b | — | 完了 | [RESULTS](p5/RESULTS_P5B.md) | GPU の外枠 |
| P5c | — | 合格（T4） | [RESULTS](p5/RESULTS_P5C.md) | Numba → C++ → GPU がつながった |
| P5d | — | 報告のみ | [RESULTS](p5/RESULTS_P5D.md) | 速度の基準値（P5a の値は再現せず） |
| P5e | — | 探索のみ | [RESULTS](p5/RESULTS_P5E.md) | 変種はすべて ±2% 以内 |
| P5f-0 | — | 完了 | [RESULTS](p5/RESULTS_P5F0.md) | ncu で分岐の分散を確認 |
| P5f-1（P5g） | 2026-10-16 | 採用 → のちに置き換え | [RESULTS](p5/RESULTS_P5F1.md) | 状態機械に書き直し |
| P5h | 2026-10-16 | 不採用 | [RESULTS](p5/RESULTS_P5H.md) | 約 1.5% 遅い |
| P5i | 2026-10-16 | 採用 → のちに置き換え | [RESULTS](p5/RESULTS_P5I.md) | 定数の log を事前計算 |
| P5j | 2026-10-17 | 採用 → のちに置き換え | [RESULTS](p5/RESULTS_P5J.md) | 定数項の除去 |
| P5k | 2026-10-17 | **変種 F が現在の合格構成** | [RESULTS](p5/RESULTS_P5K.md) | 終わった履歴の後に次を取る |
| P5l | 2026-10-18 | 不採用（F を超えず） | [RESULTS](p5/RESULTS_P5L.md) | GPU の高速化はここで一時停止（ユーザー決定） |
| P6 | 2026-10-18 | 11 量すべて合格 | [RESULTS](p6/RESULTS.md)・[plan](../../ai/plans/2026-10-18-mv-p6-gpu-field.md) | 照射野の PDD/OCR を GPU で |
| P7 | 2026-10-18 | 3 条件とも 11 量すべて合格 | [RESULTS](p7/RESULTS.md)・[plan](../../ai/plans/2026-10-18-mv-p7-field-energy-variants.md) | 5×5・20×20、4 MV |
| P8a | 2026-10-20 | 材料の検査は合格（未実施 2 点を記録して許可）。蛍光の感度は未決 → ユーザー決定で試作に蛍光を足す | [RESULTS](p8a/RESULTS.md)・[plan](../../ai/plans/2026-10-19-mv-p8a-materials.md) | 肺 0.26・骨・鉛。蛍光で鉛の線量 −0.26 ± 0.06% |
| P8e | 2026-10-21 | **合格**（水–肺–水–骨–水–鉛の積層、19 ブロックすべて \|差\| ≤ 1%・SEM ≤ 0.33%）。P8b〜P8d はユーザー決定で省略 | [RESULTS](p8e/RESULTS.md)・[plan](../../ai/plans/2026-10-20-mv-p8e-stack.md) | 最大 \|差\| 0.38%（水 18–19 cm）、鉛 +0.07 ± 0.10%。蛍光あり。CPU（Numba）の試作 |
| P8f | 2026-10-21 | **合格**（GPU、T4。判定 A: 保存済み EGS5、B: 保存済み P8e の CPU 試作。どちらも 19 ブロックすべて合格）。鎖 0 で保存バッチ 3 つを 12 出力ともバイト再現 | [RESULTS](p8f/RESULTS.md)・[plan](../../ai/plans/2026-10-21-mv-p8f-gpu-materials.md) | 2.0×10⁷ 履歴、276 µs/履歴。A の最大 \|差\| 0.39%。**A の χ² 50.3/19 で小さな系統的な差を示唆**（報告だけ）。Colab CLI で実行 |
| 入口（依頼 A・B） | 2026-10-21 | **合格**（使用者が 3D 表示で承認した条件 = GPU の核に渡った引数から作り直した条件、ハッシュ一致。わざと変える検査 5 つすべて検出。EGS5 と P3b-3 の 11 量すべて許容内） | [RESULTS](entry/RESULTS.md)・[plan](../../ai/plans/2026-10-21-mv-entry-colab.md) | `chatcarlo run` から Colab T4 を人の手なしで。4,600 万履歴、GPU 20.0 分。深部 8 比較の最大 \|差\| 0.42%・RMS 0.24%・SEM 0.15〜0.26%。VM が 96/100 バッチで消え、再開で残り 4 だけ計算 |
| 入口（4 MV、依頼 1） | 2026-10-24 | **通した**（Mohan 4 MV、10×10 cm²、2.6×10⁷ 個。承認条件 = 実計算条件、関門 2a/2c/2d 合格、止まらず。EGS5 との比較は入口に含めず、外で PDD 平均 1.6%・最大 6.4%、OCR 平均 2.0%・最大 5.6%） | [RESULTS 追補](entry/RESULTS.md) | 許可リストに `mohan4` を追加。Colab 呼び出しに制限時間、`uniform` の丸めの違い（Mac と Colab）を修正 |
| 10 MV の統計の追加 | 2026-10-23 | **完了**（両コード 1.0×10⁸ 個。EGS5 は種違い 2 本を合算、GPU は入口で 5,400 万を追加。合算前の χ² はすべて p ≥ 0.001） | [RESULTS](stats10mv/RESULTS.md)・[plan](../../ai/plans/2026-10-23-mv-egs5-10mv-statistics.md) | 0.5 cm 刻みで PDD 差の平均 0.53%・最大 1.46%（誤差平均 0.76%）、OCR 0.42%・1.36%（0.71%）、2σ 超 0。抄録 (1) の 10 MV の行。入口の回収順を修正 |
| 4 MV（依頼 1）の統計の追加 | 2026-10-24 | **完了**（両コード 1.0×10⁸ 個、0.5 cm 刻み。EGS5 は P7 条件 C に 3,700 万 × 2 本、GPU は入口で 7,400 万を追加。合算前の χ² はすべて p ≥ 0.001。PDD 差の平均 0.77%・最大 1.94%、OCR 0.60%・1.56%。深さ 10 cm の OCR の +3.4% は −0.47% に） | [RESULTS](stats4mv/RESULTS.md) | 抄録の依頼 1 を差し替え |
| P8g-1（積層の核の表の定数 log の事前計算） | 2026-10-24 | **合格**（関門 0〜2。検査した履歴で P8f と履歴ごとにバイト一致し、P8f の判定を引き継ぐ） | [RESULTS](p8g1/RESULTS.md)・[plan](../../ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md) | T4 で ×1.25〜1.31（40×256・20 万履歴で 198 µs/履歴）。ncu の測定は [p8f_prof](p8f_prof/)、上位計画 [P8g](../../ai/plans/2026-10-24-mv-p8g-fast-stack-kernel.md) |
| P8g-2（積層の核の状態機械と補充） | 2026-10-24 | **合格**（関門 0〜2。検査した履歴で P8g-1・P8f と履歴ごとにバイト一致し、P8f の判定を引き継ぐ） | [RESULTS](p8g2/RESULTS.md)・[plan](../../ai/plans/2026-10-24-mv-p8g2-state-machine.md) | T4・20 万履歴で 42.5 µs/履歴（40×256）。P8g-1 の ×4.68、P8f 本番の起動の ×6.53。1×10⁸ で約 1.2 時間 |
| 積層の統計の追加（P8g-3） | 2026-10-24 | **完了**（両コード 1.0×10⁸ 個。EGS5 は P8e 本番に 2,000 万 × 2 本、GPU は P8f 本番に P8g-2 の核で 8,000 万を追加。合算前の χ² はすべて p ≥ 0.16） | [RESULTS](statsstack/RESULTS.md)・[plan](../../ai/plans/2026-10-24-mv-p8g3-stack-production.md) | 220 点で差の平均 0.168%・最大 1.26%（誤差平均 0.175%）、2σ 超 21（見込み約 10）。骨 +0.29%・鉛直前の水で系統差（P8f で報告済みと同じ場所） |

**現在（2026-10-19）**: P7 まで完了。GPU の合格構成は P5k の変種 F。GPU の高速化は一時停止中で、次の段階はユーザーと相談して決める。

常設の決定と既知の近似: [KNOWN_APPROXIMATIONS.md](KNOWN_APPROXIMATIONS.md)。

## 注記（2026-10-19、CLAUDE.md から移した。判定は変えない）

CLAUDE.md の段階の要約にだけ書かれていて、リンク先の文書に文として書かれていなかった内容。P1c・P1d の
RESULTS は P1e/P1f の凍結対象（`prototypes/mv_electron/p1e_common.targets()`）なので、そちらには書き足さずここに置く。

- **P1c の K9(i)**（Claude の読み。仕組みは仮説のまま）: 道のり 0.5 mm・K = 0.05 の 1 ステップ後の横の広がりは、
  `mf` が −2.020%（SEM 0.292%）、P1b の `dcslib` が −1.956%（SEM 0.294%）。値は
  [P1d の事前登録](p1d/PREREGISTRATION.md) の 40 行目にある。両者の差は 0.06 ポイントで 1σ 以内、どちらも 0 から約 7σ
  離れている。したがって、その刻みでのヒンジ近似の 1 ステップの横の広がりに実在する系統的なずれで、P1b のときから
  あり、P1c では許容のすぐ外に出ただけと読める。**P1b の「輸送の仕組みは健全」はこの但し書きつきで読む。**
  K = 0.025 ではどちらも合格。P1d で新しい種で繰り返すと値が約 2σ 動いたので、P1c の「0.02 ポイントの不合格」は
  種によるばらつきの中にあった（[P1d RESULTS](p1d/RESULTS.md) の 25 行目）。
- **P1c の `mf` と `dcslib` の差**: 最大 0.71%（R50 付近、P1c RESULTS にある）。CLAUDE.md には「81 bin 中 29 bin で
  2σ を超える。本当のモデルの差」とあった。この 29/81 の数え方は文書に残っておらず、2026-10-19 の時点で再計算していない。
- **P1d の残差の変化**: 刻みを半分にしたことによる EGS5 との残差の変化は、0 と区別できない
  （[P1d RESULTS](p1d/RESULTS.md) の再標本化で、r_K の 95% 区間が 0 をまたぐ）。
- **P1d の停止**: 計算の途中で Claude が無関係な計画ファイル（P2 の計画）を編集し、ハッシュの不一致で止まった。
  追補で再開した（そのとき線量の結果はまだなかった）。教訓は [lessons_learned](../../lessons_learned.md) の 1580 行付近。
- **P2 の Q7**: 統計不足は 2 件で、差の SEM が `Dref` の 0.3345%（`dcslib` 10 MeV fe）と 0.3354%（`elsepa` 10 MeV k）。
  条件の 1/3%（0.3333%）をわずかに超えた。事前に決めた精度に余裕がなかった（[P2 RESULTS](p2/RESULTS.md) の 40・49 行目）。
- **P2b**: 基準 v2 を P2 に事後に当てはめると、12 比較すべてが 1% 以内で、最大 0.38%。
- **P2c**: 飛行中の陽電子消滅の断面積は Heitler の式（解析式、データファイルなし）。
- **P3r の速度**: CLAUDE.md には「約 310 µs/履歴（約 40% 遅くなった）」とあった。P3t の RESULTS では、P3r の同じ条件を
  301.3 µs/履歴と測っている。比べるときは P3t の値を使う。

## 終わった段階のキャッシュの圧縮（2026-10-19）

`scripts/archive_run.sh <dir>` で `~/.cache/chatcarlo/archive/` に `tar` + `zstd -19` で保存している。全ファイルの SHA256 を
記録し、展開して照合してから元をゴミ箱に移す。展開先は `<label>.origin` に記録されている。展開は
`zstd -dc <archive>.tar.zst | tar -xf - -C "$(cat <archive>.origin)"`。

| アーカイブ | 元の場所 | ファイル数 | 圧縮前 → 圧縮後 | 展開が必要になる作業 |
|---|---|---|---|---|
| `p2_egs5` | `~/.cache/chatcarlo/p2/egs5` | 1,741 | 1.4 GB → 276 MB | P2 の入力の読み込み（`docs/validation/mv/p2/manifest.json` が検査するので、展開するまで止まる） |
| `p2b` | `~/.cache/chatcarlo/p2b` | 158 | 422 → 82 MB | P2b の再解析 |
| `p3a` | `~/.cache/chatcarlo/p3a` | 720 | 328 → 64 MB | P3a の再解析 |
| `p6` | `~/.cache/chatcarlo/p6` | 447 | 1.9 GB → 728 MB | `p6 analyse` |
| `p7_results` | `~/.cache/chatcarlo/p7/results` | 619 | 2.04 GB → 733 MB | `run_p7 analyse` |
| `p7_work_pilot` | `~/.cache/chatcarlo/p7/work/pilot` | 315 | 2.9 GB → 287 MB | 試し計算の固定（pilot lock）の検査。`histories.npz` は大半が 0 で、682 MB → 9 MB ずつ |
| `p8f` | `~/.cache/chatcarlo/p8f` | 857 | 1.3 GB → 641 MB | `gpu/p8f.py` の全コマンド（`gate1`〜`analyse`）、`p8f_cli` での再実行（束 `export` と結果 `results` もここ）。P8e の材料の束は `p8e/` にあり圧縮していない |

`~/.cache/chatcarlo/p7/spectra` は P7 のコードが読むので圧縮しない。

## 段階ごとの要約（原文、2026-10-19 に CLAUDE.md から移した）

以下は CLAUDE.md の「Therapy-energy extension」節の 2026-10-19 時点の原文で、手を加えていない（リンクだけ、
このファイルからの相対パスに直した）。新しい段階はここに足さず、上の表に 1 行足す。

Work toward coupled photon/electron/positron transport for nominal 10 MV is under way, **entirely outside
`chatcarlo/`**. Nothing in the production package or the 150 keV limits has changed.

- Roadmap (draft): [docs/ai/plans/2026-10-07-native-10mv-photon-electron-transport.md](../../ai/plans/2026-10-07-native-10mv-photon-electron-transport.md).
  Presentation story it serves: [docs/presentation_jsmp133.md](../../presentation_jsmp133.md).
- P0 feasibility (done): [docs/validation/mv/p0/RESULTS.md](../../validation/mv/p0/RESULTS.md) — xraylib stops at
  ~800 keV, NIST XCOM chosen as the MV photon data path, EGS5 MV reference with electron transport established.
- P1 electron-only prototype (done): `prototypes/mv_electron/` (class-II condensed history, dual random hinge,
  Numba). [docs/validation/mv/p1/RESULTS.md](../../validation/mv/p1/RESULTS.md): all EGS5-independent checks pass;
  the 2 MeV EGS5 comparison missed the 3% tolerance (3.02%) with a screened-Rutherford elastic DCS.
- P1b elastic-DCS experiment (done): [docs/validation/mv/p1b/RESULTS.md](../../validation/mv/p1b/RESULTS.md).
  Feeding the prototype the same DCS EGS5 uses (research-only control, never to be shipped) brings the difference
  to 0.54%, so the transport mechanics are sound and the DCS approximation was the cause. EEDL fails badly at
  1–2 MeV (its angular tables jump from 256 keV to 10 MeV). The production elastic-DCS source is still undecided.
- P1c analytic spin correction (done, verdict open): [docs/validation/mv/p1c/RESULTS.md](../../validation/mv/p1c/RESULTS.md).
  A new `mf` backend multiplies the screened-Rutherford DCS by the McKinley–Feshbach Mott/Rutherford ratio — no
  external data file; valid for low Z only (tested: H and O). **The EGS5 comparison is formally undetermined, not
  passed**: the pre-registered sanity check K9(i) (lateral spread after one 0.5 mm step at K=0.05, against
  single-scattering reference) gave −2.02% against a 2% tolerance, and the stop rule forbids judging the dose
  comparison after that. Facts: the same condition gave −1.96% with the `dcslib` control in P1b (each SEM ≈ 0.29%),
  so `mf` did not make it worse (0.06 points apart, inside 1σ), and both are ~7σ from zero — a real systematic in
  the hinge's single-step lateral spread at that step size, already present in P1b, which landed just inside the
  threshold. K9(i) at K=0.025 passes for both. The mechanism is a hypothesis, not established. So read P1b's
  "transport mechanics are sound" with this caveat. Measured values outside the formal verdict: the saved 2 MeV `mf`
  run differs from EGS5 by 0.94% of Dref at most (tolerance 3%); `mf` vs `dcslib` differ by up to 0.71% near R50
  (29 of 81 bins beyond 2σ — a real model difference).
- P1d half-step retest (done): [docs/validation/mv/p1d/RESULTS.md](../../validation/mv/p1d/RESULTS.md). User decision
  after P1c: rerun with the multiple-scattering step limit halved (`K1_max` = 0.025; one halving only, chosen on the
  EGS5-independent check, new fixed seeds, P1–P1c verdicts untouched). All pre-registered checks pass for both
  backends; 2 MeV max dose difference vs EGS5 is 0.45% (`dcslib` control) and 0.82% (`mf`, analytic, low Z only),
  tolerance 3%. **`K1_max` = 0.025 is now the adopted step.** Caveats to keep: (1) the run stopped once on a hash
  mismatch caused by Claude editing an unrelated plan file mid-run and was resumed under a reviewed amendment —
  no dose result existed at that point; (2) the change in EGS5 residual from halving the step is not statistically
  distinguishable from zero; (3) repeated K9(i) values with new seeds differ from the earlier ones by ~2σ, so the
  P1c "fail by 0.02 points" was within seed-to-seed scatter — the cause of the offset is still a hypothesis.
- Elastic-DCS source for production (user decisions, 2026-10-08): Geant4's G4EMLOW `dpwa` tables were chosen first,
  then **dropped** — the dataset's top-level README says "DATA ARE *NOT* FOR COMMERCIAL USE AND MUST BE USED WITHIN
  GEANT4". The user then chose **self-generating tables with ELSEPA** (elsepa-2020, Mendeley Data
  doi:10.17632/w4hm5vymym.1, CC BY-NC 3.0 per the repository metadata; sha256 `1066d4d4…a7e5`). A trial build with
  gfortran outside the repo reproduces EGS5's `dcslib` transport cross sections for oxygen to ~0.01% and hydrogen to
  ~1% (options not yet matched). Neither archive nor generated tables are in this repository. `mf` stays as a
  data-free independent check.
- ELSEPA tables (P1e failed, P1f passed — 2026-10-09): [P1e](../../validation/mv/p1e/RESULTS.md) generated H/O
  electron and positron tables with ELSEPA's default high-energy factorization (`IHEF=1`) and **failed** its own
  table check (interpolation vs direct calculation, 52 of 228 cases; the factorization switch at 20.1 keV × Z makes
  the DCS jump by up to ~1.5%, and part of the check compared against the wrong reference).
  [P1f](../../validation/mv/p1f/RESULTS.md) regenerates them with pure partial waves (`IHEF=0`, 10 keV–10 MeV only)
  and passes every pre-registered check: interpolation error ≤ 0.13%, and with the new `elsepa` backend the 2 MeV
  max dose difference vs EGS5 is 0.75% (tolerance 3%; `dcslib` control 0.44%, `mf` 0.82%, same step). Caveats:
  P1f is a redesign made after seeing P1e fail (basis: EGS5-independent checks only); Codex ran out of quota mid-run
  and Claude finished the remaining runs with the frozen resume command; one check (K5b) first recorded a "fail" that
  was a test-harness JIT error and was redone under a reviewed amendment, at a point when the dose result was already
  visible; ELSEPA's "dcs table is not consistent" warning (58 cases, total cross section series-vs-table ≈0.25%) is
  accepted under a limit set in advance with that value known; hydrogen differs from EGS5's table by ~1% for an
  unknown reason (≈0.06% on water's G1). Positron tables are generated and checked but not yet used in transport.
  Tables and the ELSEPA archive live outside the repo (`~/.cache/chatcarlo/elsepa/`); regenerate with
  `prototypes/mv_electron/generate_elsepa_p1f.py`. Above 10 MeV the `elsepa` backend refuses to run.
  How to distribute the tables (CC BY-NC 3.0 program; licence text itself not yet read) is an open user decision.
- The prototype's tests are run separately from the main suite (`pytest prototypes/mv_electron/tests -q`);
  collecting both directories in one pytest call fails because both contain `test_transport.py`.
- P2 coupled photon/electron/positron slab (done, 2026-10-10): [docs/validation/mv/p2/RESULTS.md](../../validation/mv/p2/RESULTS.md).
  Monoenergetic 1.25/2/6/10 MeV photons on a 30 cm water slab vs EGS5. **Formal verdict stays as recorded: Q0–Q4 pass,
  Q6 undetermined, Q7 insufficient statistics** (the pre-registered precision had no margin: 0.3345% vs 0.3333%).
  Run notes incl. a disk-full stop and a 2-process helper: [EXECUTION_NOTES.md](../../validation/mv/p2/EXECUTION_NOTES.md).
- P2b criteria v2 (2026-10-11, user decision): literature-based redesign (Rogers & Mohan ICCR 2000, AAPM TG-105). Judge
  fixed 5 cm depth blocks beyond the peak window: block-mean difference ≤ 1% of a fixed `Dref` **and** SEM ≤ 0.33%;
  build-up and peak are reported, not judged. Applied post hoc to P2 (all 12 comparisons within 1%, max 0.38%) and to
  later stages. Plan: `docs/ai/plans/2026-10-11-mv-p2b-criteria-redesign.md`. A 6 MeV "+1.05%" difference between two
  EGS5 settings, found post hoc, did **not** reproduce at 5× statistics (+0.23 ± 0.13%).
- P2c positron annihilation in flight (done): [docs/validation/mv/p2c/DISCUSSION.md](../../validation/mv/p2c/DISCUSSION.md).
  Heitler cross section, analytic, no data file. In-flight probability matches theory within 0.3% (1–9 MeV). It removed
  a systematic offset vs EGS5 (whole-slab deposit +0.11% → −0.04% at 6 MeV, +0.26% → −0.05% at 10 MeV). Now always on.
- P3a Mohan 10 MV spectrum on the slab (done): [docs/validation/mv/p3a/DISCUSSION.md](../../validation/mv/p3a/DISCUSSION.md).
  All five deep blocks within 1% of EGS5 (max 0.29 ± 0.11%). **This is a broad parallel beam on a slab, not a field
  PDD/OCR** (that is P3b, not started). The spectrum file is untracked (redistribution terms unconfirmed); never
  copy its values into tracked files. Pair-production counts differ by −1.2% because EGS5's piecewise-linear fit smears
  the pair threshold (total attenuation still agrees within 0.06%). A small build-up excess of the prototype
  (≈ +0.3 ± 0.1% of `Dref`) persists across energies and settings; cause unknown.
- P3s step-size rule (done; **adopted by user 2026-10-12**): [docs/validation/mv/p3s/DISCUSSION.md](../../validation/mv/p3s/DISCUSSION.md).
  Multiple-scattering step K1 = 0.1 for T ≤ 0.02 MeV, 0.05 for T ≤ 1 MeV, 0.025 above (was 0.025 everywhere). ~1.35×
  faster, dose unchanged within statistics. Caveat: the electron-beam before/after check (S4) is formally "insufficient
  statistics" (no difference seen; the precision condition was set without sizing it). Prototype is still ~3× slower
  than EGS5; ~100 µs/history of non-step overhead is unexplained.
- P3r bremsstrahlung photon transport (done, 2026-10-12): [docs/validation/mv/p3r/DISCUSSION.md](../../validation/mv/p3r/DISCUSSION.md).
  Same Bethe–Heitler formula as EGS5, normalised to ESTAR's radiative stopping power by our own factor (no EGS5 correction
  table; user decision "B"). Photons above 10 keV are discrete events and are transported; the EGS5 reference was rerun
  with them transported too. Mohan 10 MV slab: all five deep blocks within 1% (−0.04 to +0.04%, SEM ≈ 0.1%); the dose
  change from transporting them (≈ −0.85% near the peak, ≈ 0 beyond 20 cm) is the same in both codes. Making radiative
  loss discrete did not move the build-up excess (still ≈ +0.3%). Caveats: Codex hit its usage limit and Claude wrote the
  check/production/analysis part of `run_p3r.py`; three check amendments (one chance failure, two harness/plan mistakes);
  bremsstrahlung count/energy created are 0.7 ± 0.2% / 1.2 ± 0.4% lower than EGS5, unexplained (not the formula: rates agree within 0.07%; the
  −1.4% pair count covers only ≈ 0.1%; EGS5's 1%-tolerance piecewise-linear electron tables remain the candidate, not confirmed);
  the prototype got ~40% slower (≈ 310 µs/history, 4× EGS5) — about half of that remains with discrete events switched
  off, so it is overhead in the copied loop (cause found and removed in P3t, below). Slab thickness must be a multiple of the 0.2 cm scoring
  bin (otherwise deposits fall outside the dose array; the scoring-violation counters flag it).
- P3t loop speed recovery (done, 2026-10-12): [docs/validation/mv/p3t/RESULTS.md](../../validation/mv/p3t/RESULTS.md).
  `photon_transport_p3t.py` is the P3r loop minus two redundant operations (an unused per-step `event_order` call that
  allocated an array, ≈ 23–28 µs; a duplicated bremsstrahlung table lookup per energy update, ≈ 36–57 µs). Outputs are
  byte-identical to P3r (and to P3s with discrete events off). Mohan slab, 50k histories, single process: P3s 201,
  events off 201 (was 254), photons transported 228 µs/history (was 301) — ≈ 3× EGS5 again. **Use P3t for new work**;
  `photon_transport_p3r.py` is left untouched because it is hash-frozen by `run_p3r.py`. The remaining ≈ 23 µs with
  discrete events on is probably the table lookup itself (not separated; changing it could change bits).
- P3u parallel batches (done, 2026-10-12): [docs/validation/mv/p3u/RESULTS.md](../../validation/mv/p3u/RESULTS.md).
  `batch_runner_p3u.run_batches` runs independent-seed batches of a named kernel (`batch_kernels_p3u.KERNELS`; currently
  the P3t slab) in up to 2 spawn processes, saves each batch from the parent only, resumes, and stacks in batch-index
  order. For a fixed config the arrays are byte-identical for 1 or 2 workers and across stop/resume, and each batch equals
  a direct P3r call with the same seed. Mohan slab, 2M histories: 492 s → 258 s (×1.90). The worker cap of 2 is the
  standing operating rule, not a technical limit (more is unmeasured). **New production runs (P3b onward) should go
  through this runner**; add a kernel rather than a new batch loop.
- Standing decisions and known approximations live in
  [docs/validation/mv/KNOWN_APPROXIMATIONS.md](../../validation/mv/KNOWN_APPROXIMATIONS.md): no positron-specific
  stopping/Bhabha/elastic data (estimated effect ≤ 0.1%); electron cutoff stays at EGS5's 10 keV (do not propose raising
  it); bremsstrahlung photons are transported since P3r; GPU target
  is Google Colab free tier.
- P3b-1 prototype field kernel (done, 2026-10-12; **no EGS5 comparison yet**):
  [docs/validation/mv/p3b/RESULTS_P3B1.md](../../validation/mv/p3b/RESULTS_P3B1.md). `photon_transport_p3b.py` carries
  full xyz for every particle, has the divergent 10×10 cm² source and scores an axis box, the whole slab and lateral
  profiles by straight-segment apportioning (legacy depth outputs untouched and byte-identical to P3t without a field).
  Registered in the P3u runner as `'p3b_field'`. All deterministic checks pass. **The pre-registered symmetry check B8
  failed** (1 of 9 contrasts, z = −3.59, 2M histories; the test ran at its designed ≈5% family-wise level, so a coarse
  SEM (≈ 2% per contrast) does not explain a false alarm — it only means a pass would have shown little); a diagnostic
  with a rule fixed beforehand (≈60× the volume, new seed) found no bias (max |z| 1.61, asymmetries ≤ 0.75 ± 0.5%), so
  the failure is consistent with a ≈1% chance event — B8 stays recorded as failed and symmetry is to be rechecked in the P3b-3 pilot.
  50%-dose positions match the geometric field edge (5.25/5.51/6.05 cm at 5/10/20 cm depth). Speed with regions:
  308 µs/history (1.32× the P3t slab) after removing a per-step allocation (was ≈ 700).
- P3b-2 EGS5 field user code (done, 2026-10-12; **still no prototype-vs-EGS5 dose comparison**):
  [docs/validation/mv/p3b/RESULTS_P3B2.md](../../validation/mv/p3b/RESULTS_P3B2.md). `p3b_egs.f` = `p3r_egs.f` plus the same
  divergent source and the same region scoring (boxes are written by Python from `expand_regions`; the apportioning code
  `p3b_egs_score.inc` and the `ausgab` in `p3b_egs_ausgab.inc` are the very files the standalone gfortran test drivers
  include). All pre-fixed checks pass: legacy output identical to `p3r_egs.f` without field/regions, scorer equal to the
  prototype's on the prototype's own recorded deposits (≤ 1e-13 MeV), per-history region ledger 0 violations, source
  uniformity p = 0.23, stop/resume byte-identical. Region outputs have the prototype's names, shapes and unit (MeV per
  batch). Run through `run_p3b2.run_egs` (not `run_p3r.run_egs`). Loop CPU: 68 → 124 µs/history with regions (×1.8);
  EGS5 start-up is ≈ 78 s, so use `CPU_seconds`, not wall time, for sizing.
- P3b-3 field PDD/OCR vs EGS5 (done, 2026-10-13): [RESULTS](../../validation/mv/p3b/RESULTS_P3B3.md),
  [DISCUSSION](../../validation/mv/p3b/DISCUSSION_P3B3.md). Mohan 10 MV, 10×10 cm² at the surface, divergent, 46M histories
  per code (count fixed from a 4M pilot ×1.5). **All 11 pre-fixed quantities are inside tolerance with the precision
  condition met**: 2×2 cm axis-box 5 cm blocks from 5 to 30 cm −0.44 to +0.28% of `Dref` (SEM 0.17–0.27%; tolerance 1%),
  central (±4 cm) profile means at 5/10/20 cm −0.18 to +0.05% (SEM 0.16–0.22%), 50% positions within 0.002 cm (SEM
  0.002–0.004 cm; tolerance 0.1 cm). Symmetry on the production data: not detected in either code (max |z| 1.85/1.91;
  detectable size 0.8–2.2% of the central mean) — P3b-1's B8 stays recorded as failed. Scope: block/central means and
  50% positions only; penumbra shape, build-up and per-bin agreement are not judged, and it is not an equivalence test.
  User decisions (2026-10-13): deep blocks start at a fixed 5 cm (the 97%-window rule is unstable for the small box:
  its end moved 2.25–3.75 cm at 2M), symmetry checked on production data. Report-only: whole-slab 0–2 cm deposit
  +0.18 ± 0.08% (the known build-up excess, still unexplained); 20–80% widths 0.006–0.010 cm narrower in the prototype
  (each within its ≈0.01 cm error). Pilot differences of ≈ −1.5% (≈2σ at 4M) shrank to −0.2/−0.3% in production. Runs go
  through `run_p3b3.py` (`pilot` / `production` / `analyse`; statistics in `analysis_p3b3.py`). One harness stop (a wrong
  shape assumption for the legacy depth arrays, fixed before any comparison value was seen; plan amendment 1). Time:
  prototype 167 µs/history wall on 2 processes, EGS5 123 µs CPU.
- P4 EGS5-independent checks (done, 2026-10-14): [P4a Fano](../../validation/mv/p4/RESULTS_P4A.md),
  [P4b/P4c kerma and range](../../validation/mv/p4/RESULTS_P4BC.md); skeleton and user decisions in
  [docs/ai/plans/2026-10-13-mv-p4-independent-checks.md](../../ai/plans/2026-10-13-mv-p4-independent-checks.md).
  **All pre-fixed quantities pass.** P4a: new kernel `photon_transport_p4.py` (= P3b + per-layer density along depth,
  mass-thickness scoring, an isotropic uniform-per-mass charged source, optional per-history RNG; byte-identical to P3b
  when no layers are given). Fano test with electrons of 0.1/1/10 MeV in two layered systems (density ratios
  1:1/2:2:1/4:1 and 1:1/1024:1): 18 layer doses within −0.46 to +0.35% of the analytic value (SEM 0.10–0.29%; tolerance
  1%, user decision). With identical random numbers, layered and uniform systems agree history by history
  (20,000/20,000 in all 12 pairs), and deliberately omitting the density factor is detected. **Read this narrowly: in
  this kernel the Fano test holds by construction** (boundaries only cut straight segments; no boundary-dependent step),
  so it checks density scaling and boundary crossing, *not* the accuracy of the hinge approximation, and only for a
  laterally infinite slab scored in depth. P4b: in effectively infinite water (2,000 cm slab, photon started at the
  centre, existing P3b kernel untouched) the summed collision kerma from NIST μen/ρ equals the incident energy within
  +0.02 to +0.24% at 0.5–10 MeV (SEM ≈ 0.08%; tolerance 1%); all five are positive, weighted mean +0.09 ± 0.04%
  with no visible energy dependence (χ² 4.3 / 4) — a small common offset of unknown origin, not a radiative-yield error
  (that would grow with energy; brems radiated fraction is 0.07% at 0.5 MeV and 2.6% at 10 MeV).
  Report-only: dose / collision kerma on the Mohan slab is 1.012 ± 0.001 over 5–25 cm. P4c: CSDA range from the P1
  electron kernel (hard collisions off, straight) vs the integral of 1/S: −0.009/−0.008/−0.001%; that integral vs NIST
  ESTAR's range table: within 0.02% (tolerance 0.1%). Not done: the interaction-point energy-transfer cross-check (B3),
  a photon-source Fano test, non-slab cavities, published measurements (user decision). Production kerma scoring had
  been switched off in all earlier runs (`score_kerma=False`), so no earlier saved run contains kerma.
- P5 GPU-oriented rewrite (in progress, 2026-10-14). Skeleton and user decisions:
  [docs/ai/plans/2026-10-14-mv-p5-gpu-rewrite.md](../../ai/plans/2026-10-14-mv-p5-gpu-rewrite.md) — **CuPy RawKernel (CUDA C++)**,
  first target the Mohan slab depth dose only, speed report-only, Colab is run by the user. One C++ source
  (`prototypes/mv_electron/gpu/`) is compiled by `clang++ -ffp-contract=off` locally (ctypes) and by NVRTC `--fmad=false` on
  Colab. Verification chain: chain 0 = the C++ kernel with MT19937 is byte-identical to the Numba kernel (only on the same
  machine); chain 1 = Philox (counter-based, keyed by seed and global history number) vs saved MT runs, statistically;
  chain 2 = C++ CPU vs GPU history by history (pass/fail on paired differences; the matched fraction is report-only).
  - [P5a0](../../validation/mv/p5/RESULTS_P5A0.md) (done): small trial on M3 and Colab T4. RNG, table read-out, stack and
    atomic sums are bit-identical CPU↔GPU; `--fmad=false` works (without it 16% of an arithmetic expression differs);
    GPU math functions differ in the last bit for 0.3–27% of inputs, and so do x86 g++ vs M3 clang. Toy speed GPU/CPU ≈ 4.7×.
  - [P5a](../../validation/mv/p5/RESULTS_P5A.md) (done): the full kernel ported to C++. Byte-identical to the Numba kernel
    (2×20k histories) **and reproduces saved P3r production batches 0 and 2 byte for byte**. Philox 10M histories vs saved
    P3r 10M: deep 5 blocks −0.003 to +0.139% of `Dref` (SEM 0.09–0.13%; tolerance 1%). Report-only: build-up 0–2 cm
    −0.33 ± 0.15% (2.2σ, single observation). CPU speed 164 vs 284 µs/history (Numba), cause not examined.
  - [P5b](../../validation/mv/p5/RESULTS_P5B.md) (done): GPU wrapper without editing the P5a kernel (each thread runs one
    history into its own local output, then merges by atomics), NumPy-only judging shared by Mac and Colab, Drive bundle.
    Local checks L0–L3 pass (incl. planted faults rejected as pre-registered); all 7 kernels compile on Colab T4 (no warnings).
    Note: `brems[1]` is accumulated mid-history, so per-history merging changes its summation order (≈1e-13 MeV) — compare it
    within a computed bound, everything else bitwise.
  - [P5c](../../validation/mv/p5/RESULTS_P5C.md) (done, Colab T4): all pre-fixed checks pass. 10M histories, same seed and
    history numbers as the saved P5a Philox run: **every history took the same path** (counts and steps identical in all 100
    batches); dose bins differ only in the last bits (max relative 2.9e-11), deep-block paired differences ~1e-15 % of `Dref`.
    So Numba → C++ CPU → T4 GPU is chained for the slab depth dose. Rough speed from file times: ≈80 µs/history on T4 vs
    164 on M3 C++ (superseded by P5d).
  - [P5d](../../validation/mv/p5/RESULTS_P5D.md) (done, report only): every timed run passed its correctness gate. µs/history:
    M3 Numba 257, M3 C++ 122 (P5a's 284/164 did not reproduce — use these), Colab Xeon C++ 358, **T4 79 (100k) / 77 (1M)**.
    T4 vs same-machine CPU ×4.5–4.7; vs M3 C++ only ×1.5–1.6. Diagnostics: atomic merge costs ~0 (0.1%), block size 64/128/256
    within 2.6%; the kernel uses 255 registers (the cap) and 30 KB local memory per thread → only 256 threads/SM resident.
    Claude's reading (not proven): register/local-memory pressure, with FP64, limits it. Speeding up changes the passed
    configuration → separate plan back through chain 2. Colab g++ needs `p5d_gxx_compat.h` (frozen kernel calls bare `isfinite`).
  - [P5e](../../validation/mv/p5/RESULTS_P5E.md) (done, exploration only): 12 wrapper/compile variants (static vs dynamic history
    fetch, stack 64/16, register cap none/168/128), all bit-identical to V0 per history, **all within ±2% of V0**. Dynamic fetch −1.7%
    (selected by the pre-fixed rule, **not adopted** by user decision; V0 stays the passed configuration). Doubling resident blocks
    did not help; sorting histories by step count made it 33% slower. Claude's reading: a shared throughput limit (local-memory
    traffic or FP64), not warp imbalance or latency (superseded by P5f-0). User decision 2026-10-15: rewrite the kernel itself.
  - [P5f-0](../../validation/mv/p5/RESULTS_P5F0.md) (done): ncu works on Colab T4. V0: **FP64 pipe 83% busy but only 2.1 of 32
    threads active per warp** (branch divergence: each lane follows a different particle/process); memory 5%, L2 hit 99%. Whole-kernel
    switches: moving the 72 MB `ys` reads into a cache-sized panel −0.4% (table is not the bottleneck); shrinking the dose path −8.6%.
    First rewrite candidate: make lanes in a warp do the same work.
  - [P5f-1](../../validation/mv/p5/RESULTS_P5F1.md) (done; **P5g adopted as the passed GPU configuration**, user decision 2026-10-16): kernel rewritten as a per-history state
    machine (`p5g_*`, P/C/E/H/D states) with the GPU driver advancing all lanes of one state together each round (ballot masks).
    Each history's operation order is unchanged: **CPU build byte-identical to the 100 saved P5a Philox batches; on T4, 20k histories
    bit-identical to V0 per history; P5c's C0–C4 (10M) all pass. T4 speed 79.2 → 42.5 µs/history (×1.87)**, same-session alternation.
    V0 divergence map (ncu source counters): 2.08 lanes/32 overall, 2.04 on FP64, and ~2 even on lines every history runs (Philox,
    step length) — consistent with lanes not reconverging inside the nested loops. P5g's own map not taken (gate-2 record fingerprints
    the measurement script). Lesson: never fingerprint launch/measurement tools in gate records; read the real artefact shape locally
    before shipping a Colab step. (Superseded as the passed configuration by P5i, then P5j — below.) P5g's map, taken later:
    4.80 lanes/32, warp instructions 9.36e10 → 4.49e10.
  - [P5h](../../validation/mv/p5/RESULTS_P5H.md) (done, **not adopted**, user decision 2026-10-16): binary searches replaced by a
    bucket table + verified short search; all gates pass (bit-identical) but T4 **≈1.5% slower**. Search lines were 20.3% of warp
    instructions, yet removing integer work did not buy time.
  - [P5i](../../validation/mv/p5/RESULTS_P5I.md) (done, **adopted** 2026-10-16): logs of table constants computed once per device
    with the same `log` (never host-computed). Gates 0–3 pass, bit-identical to P5g. Same-session T4 42.98 → 34.41 µs/history
    (×1.25); FP64 warp instructions −22%, all instructions −13%.
  - [P5j](../../validation/mv/p5/RESULTS_P5J.md) (done, adopted 2026-10-17; superseded by P5k F):
    removes the Møller constant-argument logs, `log1p(-d/tau)` (always exactly −0.5, guarded by an exact `==` check) and repeated
    `log(ev)` per energy update. Gates 0–3 pass, bit-identical to P5i. Same-session T4 33.12 → 30.83 µs/history (×1.074); FP64
    −12.7%, all instructions −4.0%. Claude's reading (n=3, hypothesis): time tracks FP64 instructions (P5h/P5i), less so once FP64
    share drops (P5j). Bit-preserving FP64 savings are now mostly used up; next candidate is cross-warp event queues.
    Compare speeds only within one session (the same P5i measured 34.41 and 33.12 in two sessions); "≈2.6× over V0" (79.2 → 30.8)
    is a cross-session ratio, roughly 2.5–2.7×. ncu CSV parsing inside Colab failed in all four runs; keep the raw export and
    parse locally.
  - [P5k](../../validation/mv/p5/RESULTS_P5K.md) (done, **variant F adopted as the current passed configuration**, user decision
    2026-10-17): new GPU drivers only, the P5j transition functions are `#include`d unchanged. F = when a lane's history ends it writes
    its outputs and fetches the next history index (`atomicAdd`), keeping the per-warp ballot; Q1/Q4/Q8/Q1-512 = block-local slots
    in global memory regrouped by state each round. Gates 0–3 pass (gate 1: every history of all 5 variants bit-identical to P5j over
    the 100 saved batches; gate 3 on F). Same-session T4: P5j 30.27, **F 16.51 µs/history (×1.83)**, Q8 19.20, Q4 19.71, Q1 23.98.
    F map: lanes 5.30 → 9.94, warp instructions −48.5%, FP64 −53.6% — the lost lanes were finished histories idling until the warp's
    longest one ended. Q beat F in the equal-cost CPU emulation (0.74 vs 0.39 utilisation) but lost on T4; not profiled (candidates:
    scattered slot access, block barriers, reloading values that F keeps in registers). ≈4.8× over V0 is a cross-session ratio.
    Q map taken later (report only): Q is slow because of the end-of-round `__syncthreads` — 37–47% of stall samples, issue-active
    5.29% (F) → 3.40% (Q1), although Q executes 7–14% fewer warp instructions with more lanes (12.98); memory is not the cause.
    Codex hit its usage limit at the final test run; Claude finished it by user instruction (tests, gates, docs; no source change).
  - [P5l](../../validation/mv/p5/RESULTS_P5L.md) (done, report only; **no variant beat F; not adopted**, user decision 2026-10-18 — P5k F stays the passed configuration): warp-local
    repacking (`__syncwarp` only, 64/128 slots per warp) and C repeated K=4 times. Gates 0–3 pass (gate 3 on W4K4). Same-session T4:
    F 17.41, FK4 19.32 (+11%), W2K4/W4K4 19.98, GK4 20.07, G1 20.93 µs/history. Map: FK4 lanes 9.94 → 8.29 (lanes that leave C wait
    inside the K loop; +11% warp instructions); W4K4 lanes 9.62 (repacking gain cancelled) and global slots add lg-throttle stalls
    (24%, RNG/driver lines). Equal-cost emulation predicted the opposite for both — do not trust it for K or pool size.
  - **GPU speed-up paused at P5k F** (user decision 2026-10-18). Codex consultation `data/logs/codex_p5_next2.md` (not tracked); lessons in docs/lessons_learned.md (2026-10-18). If resumed: compute resident blocks from registers (allocation granularity), shared memory and launch first — P5l at 130 registers was still 1 block/SM (registers and shared memory). Next stage (e.g. field PDD/OCR on GPU) to be decided with the user.
  - [P6](../../validation/mv/p6/RESULTS.md) field PDD/OCR on GPU (done, 2026-10-18; plan `docs/ai/plans/2026-10-18-mv-p6-gpu-field.md`): `gpu/p6_*` = P5j transitions + P5k F driver + divergent 10×10 source + P3b region scoring (P5 already carried xyz). Gate 1: the C++ MT build reproduces saved P3b-3 production batches 0/2/99 byte for byte (all 21 outputs). Gates 2–3 on T4 pass. **46M GPU histories vs saved EGS5 and vs saved P3b-3 MT: all 11 pre-fixed quantities inside tolerance** (axis-box blocks −0.25 to +0.31% of `Dref`, SEM ≤ 0.27%; 50% positions ≤ 0.005 cm). T4 24.8 µs/history (slab F 16.8 in the same session). Two plan amendments (rounding bound counts deposits; site 5 is checked by a fixed-state test — 0 hits in 1.38M histories). Colab first stopped because the notebook was fingerprinted (P5g lesson recurred); launch tools are now excluded from identity. Whole-slab deposit ratio GPU/EGS5 (report only, same method as P3b-3): 0–2 cm +0.18 ± 0.08% (the known build-up excess), deeper blocks within ±0.08%.
  - [P7](../../validation/mv/p7/RESULTS.md) field size / energy variants (done, 2026-10-18; plan `docs/ai/plans/2026-10-18-mv-p7-field-energy-variants.md`): 10 MV 5×5 (A), 10 MV 20×20 (B), Mohan 4 MV 10×10 (C; spectrum `mohan4.spectrum` from NRC EGSnrc, untracked in `~/.cache/chatcarlo/p7/spectra/`). `gpu/p7_*` = P6 + field half-width/SSD as arguments + common ±20 cm layout (3,420 bins); gate 1 replays saved P3b-3 batches byte for byte with the P6 layout. Central means use bins within 80% of the geometric half-width at that depth (user decision; differs from P6's ±4 cm at 10/20 cm). **GPU vs newly run EGS5: all 11 quantities inside tolerance for each condition** (N 20M/138M/26M from 4M pilots ×1.5; cap raised to 240M by user decision). Axis-box blocks −0.28 to +0.62% of `Dref` (B 10–20 cm +0.6% at ≈2.4σ each, not seen in the whole slab). Whole-slab 0–2 cm ratio A +0.03 ± 0.15, B +0.05 ± 0.05, C −0.07 ± 0.09% — the P3a–P6 build-up excess is not seen here (report only). Codex hit its usage limit; Claude implemented by user instruction, judgement code reviewed by Codex in 3 rounds before any verdict. Entry: `run_p7.py` (`local`/`gate1`/`pilot`/`egs`/`export`/`analyse`); EGS5 runs reuse `run_p3b2.run_egs` unchanged with scoped paths/spectrum SHA, one process per condition; EGS5 results are sealed (hashes) at completion.
  - P2 freeze list (2026-10-16): 137 regenerable P1b–P1f output `.npz` files were deleted (user ran `rm` on Claude's
    recommendation, without checking that `docs/validation/mv/p2/manifest.json` `protected` listed them), which stopped every
    input load. By user decision the code was left untouched and those 137 entries were removed from `protected`; their hashes
    are kept in `docs/validation/mv/p2/removed_2026-10-16.json` (831 files still checked). See docs/lessons_learned.md.
  - Colab hand-off: input bundles go to the user's gmail Drive `MyDrive/viveMonte/p5/` via the locally synced folder
    (`~/Library/CloudStorage/GoogleDrive-<gmail>/マイドライブ/...`); results are read back from it. Bundles are never committed.
    Check which account the Drive connector is signed into (it was the university one once) before concluding a file is missing.
    **Browser steps are done by the user** (user decision 2026-10-17, to save tokens): Claude gives the Colab URL, "T4", "Run all"
    and the Drive permission (gmail account) in a few lines, then watches the synced Drive folder for results. Keep every Colab
    step a single "Run all". The Drive mount often times out while the permission is pending; rerunning "Run all" fixes it.
    Large bundles (P7: ~750 MB) take minutes to upload: Colab started before the sync finished and stopped on a missing
    `p7_inputs_A.npz` (2026-10-18). Name the files the user should see in the Drive folder and ask them to wait until all
    are listed before "Run all" (the bundle check stops safely on a partial file, but the run is wasted).
  - Finished-stage cache data (2026-10-19, user decision): when a stage is closed, archive its large run directories under
    `~/.cache/chatcarlo` with `scripts/archive_run.sh <dir>` (tar + zstd -19 into `~/.cache/chatcarlo/archive/`, every file's SHA256
    recorded, extracted and compared before the original is moved to the Trash; restore command at the top of the script).
    Lossless, so recorded hashes still verify after restoring. Done for `p6` (1.9 GB → 728 MB), `p7/results` (2.04 GB → 733 MB) and
    `p7/work/pilot` (2.9 GB → 287 MB; the per-history `histories.npz` files are mostly zeros, 682 MB → 9 MB each). Also archived (2026-10-19): `p2/egs5` (1.4 GB → 276 MB, 1,741 files; checked by `docs/validation/mv/p2/manifest.json`, so P2 input loads stop until it is extracted), `p2b` (422 → 82 MB) and `p3a` (328 → 64 MB). Extract before rerunning
    `p6 analyse`, `run_p7 analyse` or a pilot-lock check. Keep `p7/spectra` (read by P7 code). Do not change `np.savez` in the shared
    saving helpers instead: P5-era frozen code uses them. A Drive-copied export bundle may be moved to the Trash after its manifest
    hashes are checked against the Drive copy.
- P3b skeleton and user decisions (2026-10-12; implemented in P3b-1 to P3b-3):
  [docs/ai/plans/2026-10-12-mv-p3b-field-pdd-ocr.md](../../ai/plans/2026-10-12-mv-p3b-field-pdd-ocr.md) — 10×10 cm² at the
  surface, divergent, 100 cm source-to-surface in vacuum, 2×2 cm axis box, profiles at 5/10/20 cm, depth bins 0.25 cm
  (coarse) / 0.1 cm (fine). **The slab kernels up to P3t do not carry lateral position for photons or for most
  secondaries** (only charged steps and Møller secondaries) — harmless for depth-only scoring, wrong for any lateral
  scoring; use the P3b/P4 kernels for anything lateral.
- Codex review originals for these plans are kept in `docs/ai/plans/reviews/`. Codex's sandbox cannot write outside the
  repository (`~/.cache`); checks that need it are run by Claude.

