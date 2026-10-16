# P5h 実装と手元の検査（2026-10-16）

計画 `docs/ai/plans/2026-10-16-mv-p5h-table-lookup.md` の §7、§6 を優先して実装した。
**関門0（CPU）・関門1 合格**（2026-10-16、Claude が手元の M3 で実行）。T4 の関門0・2・3、速さの交互の測定、ncu の地図は未実行。P5h の採用を意味しない。

## 関門1（M3、clang++、保存済みの P5a Philox 100バッチ、各10万履歴）

- 通常版・検査版とも、100バッチすべてで10出力がバイト一致。
- 検査版の全部の表引き（バッチ99の例）: 9か所すべてで新しい探索を通り、元の探索に戻した回数 0、添え字の不一致 0。呼び出しの回数は `loget`（密度）2,683.9万、`loget`（放射）2,685.4万、`brems_e` 2,581.7万、`pes`（光電）60.3万、`pes`（対生成2種）10.6万・6.5万、`rates` 2,683.9万、`ts`・`us` 各3,819.1万。
- 記録は `~/.cache/chatcarlo/p5/p5h/gate1.json`（Drive の `viveMonte/p5/p5h/` にも複製）。

以下は Codex の実装時の記録（関門1の前）。

## 成果物

すべて新規。既存ファイル・凍結ファイル・`p5g_*`・`chatcarlo/`・ルートの `tests/`・`scripts/`・`examples/` は変更していない。ステージング・commitなし。

`prototypes/mv_electron/gpu/`:

- `p5h_lookup.h`: CPU/GPU共通の候補範囲検証、短い探索、全範囲への復帰、検査カウンタ。
- `p5h_kernel.h`: P5gの遷移と、探索に至る関数を経路ごとに複製。探索以外の演算トークンを保存。
- `p5h_tables.py`: NumPyだけを使う区画生成と関門0の入力生成。
- `p5h_cpu.cpp`: 通常・添え字検査版のCPU駆動と、探索を直接呼ぶ関門0。
- `p5h_gpu.cu`: 通常・履歴ごと・添え字検査・関門0のGPU核。
- `p5h_core.py`: 計算ソース、元の入力、生成した区画・pair開始位置、コンパイル指定の指紋と判定。
- `p5h.py`: CPU関門0、手元の2,000履歴比較、明示的な100バッチ関門1、Colab用ソース書き出し。
- `p5h_colab.py`: T4関門0〜3、P5g/P5h交互測定、両構成の行情報あり/なしの出力・SASS・資源比較、ncu。
- `p5h_colab.ipynb`: 手書きの承認旗・コミット指定を不要にし、「すべてのセルを実行」で連続実行。

テスト: `prototypes/mv_electron/tests/test_p5h.py`。
結果の文書: このファイル。

## 区画

すべて点数の2倍。`ts`も他の表と同じ方式を使う。区画の表はグローバルメモリに置き、読み取り専用で使用する。

| 表 | 点数 | 区画数 | 比較 |
|---|---:|---:|---|
| us | 20,483 | 40,966 | 元の倍精度値 |
| brems_e | 3,100 | 6,200 | 元の倍精度値 |
| rates | 480 | 960 | 元の第1列 |
| loget | 167 | 334 | 元の対数格子 |
| pes | 711 | 1,422 | 実行先で計算したlog |
| ts | 49 | 98 | 元の倍精度値 |

計49,980区画。候補範囲のint32配列は399,840 bytes、メタデータは240 bytes、pair開始位置は8 bytes。
`ks`（9点）は意図して旧 `p5_bracket` → `lower_bound` を残す唯一の経路。
pair開始位置は `[489, 546]`（0始まり）。元と同じ `es[start] <= th` に相当する比較で生成する。
候補範囲は保証として信頼せず、§6の前後境界を実行先で検証する。失敗時・非有限値・範囲外では整数変換前に元の全範囲探索へ戻る。

## CPU関門0

各探索箇所について、全格子点・nextafter両隣、区画境界と両隣、NaN・±∞・±0・巨大値・負値、100万乱数を検査した。
さらにC++で計算した格子値（pesではlog）とnextafter両隣を加え、通常候補と意図的に狭めた候補の両方で旧添え字と比較した。
計18,638,112入力で不一致0。全9箇所で呼び出し・全範囲への復帰ともに非0。
重複列の先頭・末尾、重複の途中をstartにした場合、start=n、答え0/n/startも別テストで検査した。

## 2,000履歴の新旧一致

Philoxの既定seed、開始履歴番号7,000、n=2,000。
P5g CPUに対して、P5h通常版・添え字検査版の両方で、10出力に加えて診断軌跡と診断メタデータの計12項目がビット一致した。
各探索箇所のカウンタは次のとおり。補間の前に新旧添え字を比較し、不一致は最後にホスト側で判定する。

| 探索箇所 | 呼び出し数 | 全範囲への復帰 | 不一致 |
|---|---:|---:|---:|
| 密度/loget | 534,117 | 0 | 0 |
| 放射/loget | 534,375 | 0 | 0 |
| 制動放射/brems_e | 513,789 | 0 | 0 |
| 光電効果/pes | 11,828 | 0 | 0 |
| pair0/pes | 2,120 | 0 | 0 |
| pair1/pes | 1,339 | 0 | 0 |
| 散乱/rates | 534,117 | 0 | 0 |
| GS/ts | 760,234 | 0 | 0 |
| GS/us | 760,234 | 0 | 0 |

## 再実行とColabへの引き継ぎ

手元の記録と生成表・出力は `<作業フォルダ>/` に保存した。
`gate0_cpu.json`、`local_2000.json`、`p5h_tables.npz`、`ordinary.npz`、`checked.npz`、`p5g.npz`、ビルドログ、`p5h_sources.zip` を含む。
元の入力数値やスペクトルはリポジトリに追加していない。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p5h local --work <作業フォルダ>
.venv/bin/python -m prototypes.mv_electron.gpu.p5h export --work <作業フォルダ>
```

関門1のコマンド（明示的に指定した場合だけ100バッチを計算する。約2時間）:

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p5h gate1 \
  --work "$HOME/.cache/chatcarlo/p5/p5h" \
  --saved ~/.cache/chatcarlo/p5/p5a/philox
```

Colabには、出力した `p5h_sources.zip` と合格した `gate1.json` をDriveの `viveMonte/p5/p5h/` に置き、`p5h_colab.ipynb` をT4で全セル実行する。
入力束は既存の `viveMonte/p5/p5b/` を利用する。現ソースの100バッチ記録がなければGPU関門へ進まない。2,000履歴の記録を関門1として扱うことはない。
関門2で20,000履歴の全出力が通常・検査版ともP5gに一致することを要求し、続いてP5cと同じC0〜C4（1,000万履歴）を判定する。
速度は同一セッションで3巡交互測定する。地図には探索行の絶対命令数・割合・参加レーンとレジスタ/ローカルメモリ資源を記録する。
行情報あり/なしの出力・SASS・資源の同一性を示せなければ、「行の情報付きの構成の測定」と記録し、採否条件にはしない。

## テスト結果

| コマンド | 結果 |
|---|---|
| `.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5h.py -q` | 15 passed（29.48 s） |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | 860 passed、2 skipped（1,107.51 s） |
| `.venv/bin/python -m pytest tests/ -q` | 403 passed、既存の警告2件（111.63 s） |

スキップ2件は既存の `test_elsepa.py::test_actual_elsepa_transport_and_k5b`（旧ELSEPA GS不在）と `test_p5a0.py::test_gpu_toy_and_philox`（CuPy/GPU不在）。P5h専用テストのスキップはない。最終全体実行のログは `<作業フォルダ>/pytest_prototypes.log`。

追加のGPUソースC++構文検査も通った。CUDA組み込み関数を手元のスタブに替えた構文検査であり、NVRTCコンパイル・T4実行の代わりにはしていない。
専用テストは、旧探索への呼び出し残り、遷移と補間等の非探索演算のトークン保存、非有限値・重複列・不正な格子、関門記録の拒否、カウンタの拒否、Colab用ソースとノートブック、P5g参照駆動も検査する。
初回の専用テストで生成ツールの指紋混入を検出して修正した。修正前に収集した全体テストも同じ理由で1件失敗（857 passed、2 skipped）し、修正後に全体を再実行した。Colab駆動のP5g参照フラグも確認時に修正し、その回帰テストを追加した。
