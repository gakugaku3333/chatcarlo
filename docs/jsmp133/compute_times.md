# 抄録（登録用 v3）に使った計算の計算時間（2026-10-26 集計）

抄録では計算時間を扱わず、スライド発表に回す（使用者決定 2026-10-25）。そのための控え。
「記録」は保存された計測値。「見積もり」はログの時刻などからの推定で、計測値ではない。
GPU はどれも Google Colab の T4、EGS5 は手元の Mac の 1 プロセス。

| 抄録の項目 | 計算 | 入射光子数 | 時間 | 種類 | 出典 |
|---|---|---|---|---|---|
| 評価 1: 60 keV 水 | GPU | 5.0×10⁷ | 43 s（VM 上） | 記録 | `abstract_ja_draft.md` ③ 項目 7 |
| | EGS5 | 1.0×10⁸ | 120 s（ビルドと実行） | 記録 | `docs/egs5_crosscheck/revalidation_2026_10_07/egs5/pdd_phantom/metadata.json` |
| 評価 1: 10 MV 水 | GPU 入口（10-21） | 4.6×10⁷ | 1,199.5 s（2 台の VM の合計） | 記録 | `docs/validation/mv/entry/RESULTS.md` 所要時間 |
| | GPU 入口（10-23） | 5.4×10⁷ | 約 1,450 s | 見積もり（3 台目の VM が 16 バッチを 231.3 s で計算した速さから 100 バッチ分。1・2 台目の時間は残っていない） | `archive/mv_entry_b521f4485fa5.tar.zst` の `results/driver_status.json` |
| | EGS5 本番 | 4.6×10⁷ | CPU 5,656 s（経過 5,734 s） | 記録 | `docs/validation/mv/p3b/RESULTS_P3B3.md`、`p3b/egs5/main_egs/summary.json` |
| | EGS5 追加 × 2 | 2.7×10⁷ × 2 | CPU 3,638 s・3,648 s（経過 4,909 s・4,919 s） | 記録 | `docs/validation/mv/stats10mv/RESULTS.md` |
| 評価 1: 積層 | GPU P8f | 2.0×10⁷ | 5,513 s | 記録 | `docs/validation/mv/p8f/RESULTS.md` |
| | GPU P8g-3 | 8.0×10⁷ | 3,581.9 s（400 呼び出しの核の秒数の和） | 記録 | `~/.cache/chatcarlo/p8g3/colab/results/call_*.npz` の `seconds` |
| | EGS5 本番 | 6.0×10⁷ | CPU 5,632 s | 記録 | `docs/validation/mv/p8e/RESULTS.md` |
| | EGS5 追加 × 2 | 2.0×10⁷ × 2 | CPU 1,971 s・1,964 s | 記録 | `docs/validation/mv/statsstack/RESULTS.md` |
| 評価 2: IVR 室 | GPU | 1.0×10⁸ | 106 s（VM 上。Colab の準備と回収を含め約 13 分） | 記録 | `runs/ivr_room_box_gpu/run.log` |
| 評価 2: X 線画像 | GPU | 10⁴〜10⁸ の 5 段階 | 0.09・0.42・2.6・24・243 s（計 270 s） | 記録 | `docs/jsmp133/request_2_entry/RESULTS.md` |
| 評価 2: 前立腺シード | GPU（6 回、各 100 バッチ） | シードの有無の各 3.0×10⁸ | 1 回 約 85〜117 分、6 回で約 8.5〜11.7 時間 | 見積もり（ログの時刻から 1 バッチ 51〜70 s。VM 上の計測値は残っていない） | `data/logs/p9c_prod_prostate*.log` |

- 前立腺シードの 1 バッチの秒数: 1 回目 52.9 s（バッチ 4→17）、2 回目 63.5 s（4→94）、3 回目 50.9 s（3→17）。
  終了の時刻は 10 分おきの確認で見たもので、遅れを含みうる（1 回目の 4→終了は 61.1 s、3 回目は 69.7 s）。
- GPU の 1 履歴あたり（記録から）: 10 MV 水 26.1 µs（10-21）、積層 P8f 276 µs、P8g-3 44.8 µs。

## バッチごとの生データの扱い（2026-11-08、使用者決定）

`runs/ivr_room_box_gpu/colab/results/`（20 バッチ・260 MB）を
`~/.Trash/jsmp133_raw_batches_2026-11-08/ivr_room_box_colab_results` へ退避した。
退避の前に [ivr_room_box_compaction/](ivr_room_box_compaction/) を作り、`raw_compaction.json` に
元のハッシュ・受領票のハッシュ・検証の内容・失われるものを記録した
（方法は [compact_run_batches.py](compact_run_batches.py)、water60 と同じ形）。260 MB → **3.4 KB**。

**Why**: `dose.npz` が**全格子のボクセルごとのバッチ合計**（`kerma_keV`・`h10_track_pSv_cm3`）・
平均・SEM・相対誤差・`n_batches_hit` を持ち、`build` が生の合計と当たり数の一致を
`assert_array_equal` で確かめた。ボクセルごとの表は作っていない——
**1,137,600 ボクセルの 83.3% が当たっているので、表は生データより大きくなり、
公開リポジトリに約 36 MB の二重の写しを置くことになる**。
残したのはバッチごとのスカラー（格子の合計・`deposits` 7・`endings` 3・`scatter_sum`・`n_fluorescence`）。

**SEM が復元できることは実測した**（2026-11-08、ゴミ箱を空ける前に退避した写しから）。
20 本の生バッチから ボクセルごとの SEM を計算し直すと（`ddof=1`・`/√20`、平均は
`kerma_keV/20`・`h10_track_pSv_cm3/20`）、**信号のある 948,040 ボクセルすべてで
`rel_err_dose`・`rel_err_h10` を最大相対差 4.14×10⁻¹⁴・3.66×10⁻¹⁴ で再現**した（`rtol` 1e-8 以内）。
残る 189,560 ボクセルは `rel_err` が nan で `n_batches_hit == 0`——一度も当たっておらず、
復元すべき 2 次モーメントを持たない。

**How to apply**: 失われたのは**ボクセルごとのバッチの値**（ボクセル間の共分散、バッチの一部での再集計）。
ボクセルごとの 1 次・2 次モーメントは `dose.npz` から復元できる（上の実測）。
**「要約から復元できる」と書くなら、生データが消える前にその復元を 1 回やって数値を残す**——
合計の一致だけでは 2 次モーメントの主張の根拠にならない。
退避先を空にするのは使用者の作業で、それまでは元のファイルが戻せる。
