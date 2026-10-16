# P5i — 補間表の定数対数の事前計算

2026-10-16。計画 `docs/ai/plans/2026-10-16-mv-p5i-precomputed-logs.md`（§7 > §6 > §1〜§5）に従う隔離試作。

CPU関門0と、n=2,000・start=7,000・固定Philox seedでのP5gとのビット一致を確認した。通常・検査ビルドとも、輸送10出力と診断2出力がすべて一致した。これは関門1の代替の小規模テストであり、100バッチの関門1合格を意味しない。関門1は使用者の指定により未実行。T4関門0・2・3、速度、命令計測も未実行であり、採用の判断は未了。


## 関門1（2026-10-16、Claude が手元の M3 で実行）

**合格。** 保存済みの P5a Philox 100バッチ（各10万履歴）で、通常版・検査版とも10出力がバイト一致。検査版の数え上げ（バッチ99）: 9つの配列すべてを通り、番兵を読んだ回数 0。読み出しの回数は `brems_e` 7,745万、`brems_v` 1億5,490万、`rates_ratio` 2,684万、`rates_values` 1億736万、`ts_ratio` 3,819万、`pes` 1,172万、`water2` 120万、`pair0` 21.2万、`pair1` 13.1万。記録は `~/.cache/chatcarlo/p5/p5i/gate1.json`（Drive の `viveMonte/p5/p5i/` にも複製）。

## 実装

P5gの遷移と探索を保持し、補間表の定数だけのlogを配列参照に置き換えた。変数を含むlog、exp、物理、乱数、算術の順序は保持した。探索の変更は取り込んでいない。CPUは同一翻訳単位の初期化関数、GPUは輸送と同じNVRTCソース・指定・ストリームの初期化核から対数を生成する。GPUへCPUで生成した対数を送らない。入力・ビルド・端末の構成ごとに生成し、各オブジェクトの寿命中保持する。

新規ファイルは次の10件。既存の凍結ファイル、p5g・p5h、本体のコードとテスト、scripts、examplesは編集していない。git add・commitは行っていない。

| ファイル（リポジトリからの相対位置） | 内容 |
|---|---|
| `prototypes/mv_electron/gpu/p5i_logs.h` | 配列レイアウト・同一端末での初期化・番兵・到達数・補助診断 |
| `prototypes/mv_electron/gpu/p5i_kernel.h` | P5gから複製した遷移と、定数対数を置き換えた関数 |
| `prototypes/mv_electron/gpu/p5i_cpu.cpp` | CPU ABI、初期化、元関数との比較 |
| `prototypes/mv_electron/gpu/p5i_gpu.cu` | 初期化・本番・履歴別・検査・関門0のCUDA核 |
| `prototypes/mv_electron/gpu/p5i_core.py` | 計算ソース・入力・レイアウトの指紋、判定、nspec=20 |
| `prototypes/mv_electron/gpu/p5i.py` | CPU関門0、小規模一致、明示的な関門1、ソースzip出力 |
| `prototypes/mv_electron/gpu/p5i_colab.py` | T4関門0〜3、交互計時、地図取得 |
| `prototypes/mv_electron/gpu/p5i_colab.ipynb` | Driveのソースzipから「すべてのセルを実行」する入口 |
| `prototypes/mv_electron/tests/test_p5i.py` | 専用テスト |
| `docs/validation/mv/p5/RESULTS_P5I.md` | 本報告 |

## 配列の仕様

float64の1本の`logs`に、下記9配列を元の添え字順で詰める。重複除去・並べ替えはしない。添え字と列番号は0始まり。現在の入力はbrems 3,100行、rates 480×3、ts 49点、pes 711点、water 711×7。表のstrideは元の入力配列に対するもの。

| ID・配列 | 元の配列・列・stride | 長さ（現在値） | 有効な添え字 | logs内offset |
|---|---|---|---|---:|
| 0 `brems_e` | brems_e、stride 1 | nbrems（3,100） | 0〜nbrems−1 | 0 |
| 1 `brems_v` | brems_vの列0・1・2、行stride 5、格納位置3i+k | 3×nbrems（9,300） | i=1〜nbrems−1、k=0〜2。先頭3要素はNaN | 3,100 |
| 2 `rates_ratio` | ratecols=3ならrates列0・stride 3、それ以外はts・stride 1。log(e1/e0) | n−1（479） | 0〜n−2 | 12,400 |
| 3 `rates_values` | ratesのshift・shift+1列、行stride ratecols、格納位置2i+k | 2n（960） | i=0〜n−1、k=0・1 | 12,879 |
| 4 `ts_ratio` | ts、stride 1。log(ts[i+1]/ts[i]) | nt−1（48） | 0〜nt−2 | 13,839 |
| 5 `pes` | pes、stride 1 | np（711） | 0〜np−1 | 13,887 |
| 6 `water2` | water列2、stride watercols（7） | np（711） | 0〜np−1 | 14,598 |
| 7 `pair0` | water列3・stride 7、thresholds[0] | np（711） | pes[j]>thresholds[0]。現在489〜710 | 15,309 |
| 8 `pair1` | water列4・stride 7、thresholds[1] | np（711） | pes[j]>thresholds[1]。現在546〜710 | 16,020 |

ratesではshift=(ratecols==3 ? 1 : 0)、n=(shift ? nrates : nt)。区間配列は必ずn−1要素。対生成は元と同じ`log(vs[j*stride]/((q*q)*q))`で、しきい値以下は評価せずNaNにする。bremsのi==0で使わない値も評価しない。実入力の有効な対数引数に0・負値は0件。補正やクリップは加えていない。

対数配列は計16,731要素・133,848 bytes。補助配列は`lengths`と`offsets`（各int32×9、合計72 bytes、ホスト上）と`counts`（uint64×9×2、144 bytes）。GPU上の追加常駐メモリは対数とカウンターを合わせ133,992 bytes。`extra_bytes`・計時報告の`additional_bytes`は対数配列のbytesを指す。

関門0の保存物`p5i_tables.npz`には、`logs`、`lengths`、`offsets`、`auxiliary`、`initialization_records`を含める。後二者はそれぞれfloat64×16,731×5で、列はlog、商、q、q³、対数の引数。元の式を書き直した診断と初期化経路の記録をビット比較する。これらは検査用で、輸送用の常駐メモリには含めない。通常の小規模一致の保存物は`ordinary.npz`、`checked.npz`、`p5g.npz`（圧縮保存）、`gate0_cpu.json`、`local_2000.json`。各輸送npzの配列はbatches、fine_batches、totals、maxima、violations、counts、maxdepth、nspec、step_batches、brems、diagnostic、diagnostic_meta、seconds。secondsは一致判定から除く。

## CPUの正式な関門0

CPU生成logsのSHA-256は`a2605db46014d2a25056cdea480e2faa0d9f4e0a93a89fe638d8185e5613106f`。計算ソースのSHA-256は`d27a906c4c274c6b644fb4d95d0b444488ea874e6c79f8def4913c40bb2a600d`（Colab補助ツールの変更は計算の指紋に含めない）。

環境はApple M3・Apple clang 21.0.0 (clang-2100.0.123.102)。指定は`-O2 -ffp-contract=off -std=c++11 -shared -fPIC -DP5_RNG_PHILOX -DP5_TRANSPORT_FAULT=0 -DP5_STACK_CAP=64`、通常は`-DP5I_CHECK=0`、検査は`-DP5I_CHECK=1`。

凍結関数を直接呼ぶ経路と新関数に同じ入力を与え、戻り値・出力引数・添え字・raw、GSでは乱数消費数と次の乱数もビット比較した。入力は格子点、そのnextafter両隣、範囲外、しきい値、各区間の乱数1,000個。GSはKと乱数の組も変える。

| 関数群 | 実入力の検査件数 | 結果 |
|---|---:|---|
| quantities（i==0区間を含む） | 3,108,308 | 一致 |
| table_rates | 480,448 | 一致 |
| gs_mu_cached | 48,155 | 一致 |
| log_interp | 712,141 | 一致 |
| pair_interp（両しきい値） | 712,141 | 一致 |
| coefficients | 712,141 | 一致 |

追加の合成入力でもratecols=2の分岐48,155件、log_interp・pair_interp・coefficients各4,023件を比較し一致した。合成pesに10.0とそのnextafter上隣を別行として残し、対数の丸めで同値になる格子を検査した。単体検査の全9配列の参照は0件を超え、番兵参照は0件。要素と商・q・q³・引数の補助診断も一致したが、§7に従い正式な合否条件には使っていない。

## n=2,000の輸送一致

通常・検査ビルドとも、P5gと全12出力でビット一致。検査ビルドの配列別参照数は次のとおり。すべて0件を超え、番兵参照はすべて0件。

| 配列 | 単体検査での参照数 | n=2,000輸送での参照数 |
|---|---:|---:|
| brems_e | 9,324,915 | 1,541,367 |
| brems_v | 18,643,812 | 3,082,734 |
| rates_ratio | 480,448 | 534,117 |
| rates_values | 1,921,792 | 2,136,468 |
| ts_ratio | 38,389 | 760,234 |
| pes | 32,726,108 | 231,218 |
| water2 | 2,845,782 | 23,560 |
| pair0 | 887,968 | 4,240 |
| pair1 | 659,388 | 2,678 |

この一致は固定したM3・clangの構成内に限る。CPUとGPU間、clangとg++間の一致は主張しない。

## テスト

| コマンド | 最終結果 |
|---|---|
| `.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5i.py -q` | 7 passed（31.90 s） |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | 868 passed, 2 skipped（1006.00 s） |
| `.venv/bin/python -m pytest tests/ -q` | 403 passed, 2 warnings（110.64 s） |

本体の警告2件はSpekPyの零放射条件と、SpekPy不在を模擬するテストでのKramersフォールバック。P5iテストでは、通常・検査ビルドの一致、元関数との出力比較、意図的に事前計算値を壊した場合の不一致検出、探索と算術の字句保持、対象logの輸送経路からの排除、指紋と関門1の拒否、Colabの構文・NumPyのみのimport・zip、地図取得失敗後の別構成の取得継続を検査した。

開発中の一度の再実行は診断npzの保存時にディスク空き容量不足で停止した。今回生成した一時npzを整理し、小規模一致の診断を圧縮保存に変更して再実行した。正式な数値比較の不一致による停止ではない。

## 未実行のT4測定

| 項目 | P5g | P5i |
|---|---|---|
| 全体の倍精度ワープ命令数 | 未測定 | 未測定 |
| 全ワープ命令数 | 未測定 | 未測定 |
| レジスタ数・ローカルメモリ | 未測定 | 未測定 |
| 輸送核の時間（交互に3回） | 未測定 | 未測定 |
| 対数初期化の時間 | 追加初期化なし | T4では未測定 |
| 追加対数配列 | 0 bytes | 133,848 bytes |

CPUでの初期化は一回の診断計測で約0.159 ms。性能を比較するための測定ではない。T4のCUDA runtime・driver・NVRTC版、ソース・入力・生成配列のハッシュ、起動指定を保存するコードを用意した。CPU関門1の完全な合格記録がなければGPUの輸送には進まない。T4では関門0→関門2→関門3の順を強制する。関門3とcheckpointの指紋にnspec=20を含める。

地図は初期化核を除外して名前で輸送核を選択する。履歴別20,000件、ウォームアップ1,000件、測定100,000件、seedとstartを固定し、行情報の有無で履歴出力・SASS・資源を比較する。構成ごとに取得・書き出し・読み取りの状態を別記録し、一方が失敗してももう一方の取得を試す。ncuの生記録とsource.csvを残す。命令数が減ったことだけから倍精度の回路が時間を決めるとは結論しない。実測前なので、速度向上・命令削減・律速要因についての結論はない。
