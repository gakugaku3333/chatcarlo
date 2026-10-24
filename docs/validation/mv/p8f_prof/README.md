# P8f の積層の核の測定（2026-10-24、T4、報告のみ）

P8f の核は変更していない。P8f の書き出しは `~/.cache/chatcarlo/archive/p8f.tar.zst` から取り出し、manifest の全ハッシュと核の指紋を照合してから使った。結果の読み方と、それを受けた計画は [P8g の上位計画](../../../ai/plans/2026-10-24-mv-p8g-fast-stack-kernel.md) の §3。

| 測定 | スクリプト | 結果 |
|---|---|---|
| 1. 資源・速さ（水の P5k F と交互に 3 回、2 万履歴）・ncu `--set full`（両核 1 回ずつ） | `start_prof.sh` → `vm_prof.py`（`prof_target.py` を ncu の下で実行） | `prof.json`、`ncu_material_key.txt`、`ncu_water_key.txt`。P8f 約 333 µs/履歴、P5k F 約 23。命令 6.13×10¹⁰ 対 4.28×10⁹、有効スレッド/ワープ 1.64 対 7.53、達成占有率 11.7% |
| 2. 起動の大きさ 6 通り（核は無変更）と行の地図（ncu SourceCounters、`--generate-line-info` のビルド） | `start_line.sh` → `vm_line.py`（`line_target.py`） | `line_launch_results.json`（最良 40×256 で約 7%）、`linemap_summary.txt`（表の参照・補間を含む関数に停止の標本の約 6 割） |

- 行の地図の CSV は、初回の書き出しでソースを開けず全行が 1 行に集まった。NVRTC の一時ファイル名で `compiled.cu` を置いて書き出し直した（`reexport.py`）。生の CSV と ncu の記録は `~/.cache/chatcarlo/p8f_prof/line/`（`source2.csv`、`line_profile.ncu-rep`）。
- 行の情報付きのビルドは命令の総数（8.02×10¹⁰）が通常の測定（6.13×10¹⁰）と違う。理由は未確認なので、速さには使わない。停止の標本は待っている命令に付くため、原因の行とは限らない。
- `status_prof.py` は pgrep が自分自身に一致する不具合があり、使わない（完了はログの印で判定する）。
