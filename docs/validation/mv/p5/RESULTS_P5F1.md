# P5f-1 状態機械の試作：実装と手元の検査

2026-10-15。計画 `docs/ai/plans/2026-10-15-mv-p5f1-warp-coherence.md` の §7 に従う新規実装。
**関門1の100バッチ、T4の関門2・3、速さ、参加レーンの実測は未実行。採用の判定はしていない。**

## 成果物

- `prototypes/mv_electron/gpu/p5g_kernel.h`：P/C/E/H/D の継続状態と CPU/GPU 共通の遷移。凍結ヘッダーの補助関数を読み取り専用で再利用する。新しい駆動は凍結 `history()` を呼ばない。粒子を取り出すとキャッシュを初期化し、C→Cでは保持する。失敗時は現在粒子とスタックを保持してHへ進む。
- `p5g_cpu.cpp` / `p5g.py`：`p5a_run` と同じCの契約。履歴を順に完走させ、途中の制動放射の加算も共有出力へ元と同じ時点で行う。100バッチはspawnの2プロセスで独立に計算し、親が保存・比較する。
- `p5g_gpu.cu`：128スレッド/ブロック、1レーン1履歴。周回の全状態マスクを先に確定し、P→C→E→Hの順に進める。無効・終了レーンもワープ全体の終了まで集合演算に参加する。本番、履歴別、遷移記録の核が同じ遷移とワープ駆動を使う。C3の故障版は本番の加算経路で最後の非ゼロ粗ビンを落とす。
- `p5g_core.py` / `p5g_colab.py` / `p5g_colab.ipynb`：ソース・全核・指定・起動条件・状態仕様・束・環境の指紋。関門1の現ソースでの100件の合格がなければGPUへ進めない。関門2が不合格なら関門3へ進めず、関門3が不合格なら比較測定へ進めない。いずれも自動採用しない。
- `prototypes/mv_electron/tests/test_p5g.py`：CPU 2,000履歴の10出力と診断のビット一致、通常でない遷移、1ビット差の拒否、記録の契約、集計の重み、import時の依存分離、ノートブックの停止条件。

ノートブックの `CODE_COMMIT` は空欄。Colab上の実行セルに実装の40桁コミットを設定してから実行する。`fetch` の後にdetach checkoutし、HEADと束の固定sha256を照合する。結果はDriveの `viveMonte/p5/p5g/` に保存する。各実行の承認フラグは独立で初期値はFalse。手元で使ったソースから変更した場合は関門1からやり直す。

## CPUで確認した範囲

Mohan入力、seed=20291960、start=7000、n=2,000で、凍結 `p5_cpu.cpp` と新CPUの全10出力、診断配列・診断件数がバイト一致した。これは関門1の100バッチの代替ではない。

通常でない経路の検査は、断面積・輸送条件を変更せず、検査専用のスペクトル端点を用いる。15–16 keVで光子処理中の打ち切り（maxsteps=2、100履歴）、9.99–10 MeVで荷電粒子処理中の打ち切り（maxsteps=1、100履歴）、同入力でスタック上限3（100履歴）、通常上限64（2,000履歴）。各条件で凍結CPUと全10出力が一致した。遷移の記録から、光子・荷電粒子の打ち切り、スタック失敗、硬い衝突・制動放射の退避と同じエネルギー・スタック位置の親粒子の再取り出し、陽電子の停止時消滅・飛行中消滅・領域外脱出を確認した。

## Claudeが実行する完全なコマンド

リポジトリのソースを確定した後、手元の通常環境で実行する。本作業では実行していない。

```bash
cd <リポジトリのルート>
.venv/bin/python -m prototypes.mv_electron.gpu.p5g gate1 \
  --work "$HOME/.cache/chatcarlo/p5/p5g" \
  --saved "$HOME/.cache/chatcarlo/p5/p5a/philox"
```

合格時の `~/.cache/chatcarlo/p5/p5g/gate1.json` をDriveの `viveMonte/p5/p5g/gate1.json` へコピーする。通常でないCPU経路を別途保存する場合：

```bash
cd <リポジトリのルート>
.venv/bin/python -m prototypes.mv_electron.gpu.p5g unusual \
  --work "$HOME/.cache/chatcarlo/p5/p5g" \
  --saved "$HOME/.cache/chatcarlo/p5/p5a/philox"
```

不一致・失敗・未完了の成果物は保持して停止する。削除処理は追加していない。凍結ファイルは編集していない。

## ncuの地図と未検証事項

NVRTCの行情報なし/ありでV0の20,000履歴の全出力を比較し、`cuobjdump` の対象関数のSASSとレジスタ・ローカル資源も比較する。差があれば行情報付き構成に限定して報告する。ソース全文と元ヘッダーへの対応、`.ncu-rep`、生のsource CSV、命令ごとの3計数、全命令/FP64別の行集計を保持する。平均参加レーンはΣスレッド命令/Σワープ命令。原因の二択判定はしない。

CLIの出力とSourceCountersは[NVIDIA CLI文書](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)と[2025.3 Customization Guide](https://archive.docs.nvidia.com/nsight-compute/2025.3/CustomizationGuide/index.html)に基づく。**この環境ではNVRTC・T4・ncu 2025.3.1の実出力は検証できていない。** 必要なCSV列・対応行・3計数が取れなければ生データを保持して停止し、地図を合格扱いにしない。CUDAの実コンパイル、ビット一致、C0–C4、速度・参加レーンの結果は後続の関門で測る必要がある。

## 指定pytestの結果（最終実装）

```text
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p5g.py -q
21 passed in 17.50s

.venv/bin/python -m pytest prototypes/mv_electron/tests -q
796 passed, 2 skipped in 909.09s (0:15:09)
```

既存のGPUソース・テストの追跡ファイルに差分がないことを `git diff --exit-code` で確認した。追加は計画で指定されたP5gの8ファイルと本報告書のみ。作業前から存在する無関係な未追跡ファイルは変更していない。
