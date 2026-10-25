# P9c 実装と Colab への引き継ぎ

承認済み計画 `docs/ai/plans/2026-10-24-mv-p9c-seven-field-seeds.md` の
§5-2 と、T0／関門→本計算の Colab ランナー。GPU の合否と本計算結果は未取得。
§8 は変更していない。`p9c_cli.py` は作成していない。

P9b の G1-x は不合格のまま。シード近傍の線量は EGS5 と照合していない。
研究・教育用の試作である。

## CPU

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p9c.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m prototypes.mv_electron.gpu.p9c local --work runs/p9c_cpu_check
```

[CPU_GATES.json](CPU_GATES.json) はローカルの実行結果。テスト件数と再検査の記録は [TEST_RESULTS.md](TEST_RESULTS.md)。
V0（55例）、V1a（既知の6線分・3点）、V2a（7角度）、V3（既知140光線）、
V5（人工列・完全な対・途中再開・再送拒否・同一入力の7門CPU輸送）、
V6（58本・19針、5光線の18交点）、V6b（細・粗で8例）を通す。
テストにはさらにランダム40線分と、小規模な矩形／前立腺形の輸送がある。
配置は軸の両端が前立腺内となる58本。半径まで含めると6本がはみ出すという
計画の扱いを保つ。3本の評価線は指定のシード番号を実配置から固定する。

## 実行担当者の手順（ここでは未実行）

以下は既存キャッシュを読み、指定した作業先にだけ書く。`~/.cache` 配下に
作業先を置く実行は Claude 側で行う。Mohan の値や材料表は追跡ファイルへ入れない。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9c export --work ~/.cache/chatcarlo/p9c
```

生成される `export/` の `manifest.json`、`input.json`、`cpu_reference.npz`、
`material_0.npz`～`material_4.npz`、`source.zip` を Colab に渡す。
`source.zip` を作業ディレクトリに展開し、5材料とスペクトル・CPU関門・
核の識別情報が一致する export を使う。Colab の依存は NumPy、SciPy、
既にその VM に入っている CUDA 対応 CuPy。CuPy の別 CUDA 版を重ねて入れない。

T0（候補の細0.025 cm・粗0.2 cm、100バッチ、各門10万履歴を固定）:

```bash
python -u -m prototypes.mv_electron.gpu.p9c_colab /content/export /content/results_t0 --mode t0
```

V0・V1a・V1b・V4 と GPU の同一入力対を先に検査する。
V3 は独立の透視投影面積積分と受入率の3σ比較も行う。
その後、矩形と前立腺形の各7門・あり／なしを順に計算する。
`*_measurements.json` に各門・各側のカーネル秒数、棄却数、体に入らない数、
CuPyプールの確保量、CUDAの空き／総メモリー、ホストのピークRSS（Colab/LinuxのKiB）を残す。
`memory.json` はNVMLで50 msごとに測ったデバイス全体の使用量のピークも残す
（CuPyプール外のCUDAコンテキスト・ローカルメモリーも含む）。
初期化やチェックポイント保存時間はカーネル秒数に含まない。
`*_summary.json` に前立腺／直腸の平均、3本の線、代表門の3本の線、
影の平均の比・SEM・相関係数・独立扱いのSEMとその変化を残す。
影の集約は各指定シードの後端（z方向の側面）から0.1～0.5 cm下流の水セルを等重みで選ぶ。
未定義の統計量はJSONの `null`。線の実際のセル中心と有効マスクも保存する。

T0 の数字をユーザーに示してから、計画§8に格子と履歴数を確定する。
細格子は自動で粗くしない。以下の `N_PER_FIELD` は承認された各門の履歴数
（100の倍数）を入れる。履歴数の既定値はない。

```bash
python -u -m prototypes.mv_electron.gpu.p9c_colab /content/export /content/results_main \
  --mode production --t0 /content/results_t0 --histories-per-field N_PER_FIELD
```

必要な場合だけ、§8で確定した細格子間隔を `--fine-step-cm` で渡す。
本計算モードは一致する完了済みT0を要求し、初期関門に加えV1c（20万履歴、
P7の15領域と整列格子、自然に到達しない付与箇所だけ固定状態で補足）と
V2b（円柱の0／90度、各200万履歴）を先に行う。
V2bのSEM不足だけは計画どおりV2aに代え、統計不足として保存する。
関門の不合格、矩形での体への入射失敗、輸送のhealth／ledger／粗格子和の不一致は停止する。

## 保存と再開

- 一つのバッチを7門・両側とも完了してから、5モーメントと完了番号・入力識別情報を
  `rectangle.npz`／`prostate.npz` に一括保存（別名へ書き、fsync後にrename）。
  細・粗それぞれ5量、平均・線・代表門・影にも集約後の5量を持つ。
  100組の格子や各門の格子は保存しない。
- モーメントとバッチの集計はホスト上で行い、GPUでは各側の細・粗格子を門ごとに再利用する。
  ホストのピークもT0で確認する。格子の加算はGPUの倍精度global atomic。
- 同じコマンドと出力先で再開する。確定済み番号は飛ばし、途中の対は全体をやり直す。
  完了したバッチを再度commitする入口は拒否する。入力変更は拒否する。
- 各確定後のログは `P9C_BATCH <0〜99> beam <rectangle|prostate>`。
  全関門・両照射野の100バッチと `job.json` の保存完了後だけ `P9C_DONE` を出す。
  コマンドの全呼び出しに期限を設定し、10分ごとにpollして確定済み結果を回収し、終了時はColabをstopする。
- 収支はマスク前の粗格子全体で検査する。図・比はシードと交わるセルを除き、
  粗格子ではさらに体に完全に入らないセルを除く。接するセルも除く。
  `p9c_io.sections` は指定の5断面を実セル中心・有効マスク・線量とSEM付きで取り出す。
  線量は水の密度1 g/cm³で MeV/g/受入光子、比は無次元。

GPU関門と性能、T0のメモリー適否、本計算、最終の図と科学的な判定は、実行担当者の確認が残る。
