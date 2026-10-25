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
