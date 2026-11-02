# G5 A3 手元実装と検査（2026-10-31）

状態: A3 完了（2026-11-02、キャッシュは補足判定でユーザー承認）。下の手元実装の記録は当時のもの。
2026-11-01 Claude 再検証（cache を飛ばさない）: `test_t11_mv_dispatch` を `seed_ratio` の分岐の検査へ直した後、`tests/` 875 passed、試作 1808 passed・17 skipped。下の表は Codex の実行時の記録。
承認済み子計画 `2026-10-31-chatcarlo-gpu-g5-a3-entry.md` の §2e を優先した。
Colab/T4、`~/.cache` を使う検査、保存済み結果による判定は実行していない。

## 成果物

- `chatcarlo/gpu/therapy_entry.py` / `therapy_scene.py`: `therapy_mv/pdd_ocr` の
  validate → preview → approve → run。`seed_ratio` は既存の試作入口を維持。
  10 MV/4 MV とも P7（横方向 ±20 cm）。未知のキー・量の指定・他の組・蛍光の切替を拒否する。
- `therapy_cuda.py` / `t2_gpu.cu`: A2 の CUDA 起動役を製品側へ配置。
  `T2_SOURCE_CHECK=1` の検査ビルドだけで初期線源状態を出力する。
  本番の既定は無効。GPU のコンパイル・実行・速度は未確認。
- `contracts.py`: 診断・治療で共通の生成記録、承認、観測、受領書の照合。
  診断の条件形式は `chatcarlo-g5-entry-v1`。以前の承認は生成し直す。
- `therapy_observation.py`: 最終 native 引数の幾何・入力配列・材料表・種・履歴範囲・
  field half/SSD と、native が返す組の id/tag を観測し、承認条件と照合する。
- `therapy_run.py` / `therapy_health.py` / `therapy_checks.py`: 健全性の前提検査、
  2,048 履歴と 256×8 の和の比較、固定付与の採点検査、
  条件・組・バッチ番号・範囲・SHA256 で封印した外側 100 バッチ、検査記録付き再開。
  合計線量の統計的合否は判定しない。
- `therapy_colab.py`: 既存 `colab_driver` の分割転送・bootstrap 入口を利用。
  製品パッケージのソースを送付し、VM 上でも封印と承認を照合する。
  CLI 呼び出しを制限時間と再試行で囲み、回収・停止・heartbeat の独立監視を用意。
  偽 VM で正常完走・失敗・中断回収からの再開と CLI の全モードを検査。
- `therapy_layout.py`: P7 → P6 の切り出しと MeV → Gy/history の換算。
  P7 の bin 全要素を独立な座標ベースの期待添字と比較し、人工の P6 表を P7 の中央へ
  埋め込んで取り出す検査も実施。保存済み表の回帰は未実行。
- `therapy_inputs.py` / `therapy_input_hashes.json`: 実行時は封印済みの外部束だけを読む。
  A2 の記録 `a2_run/bundle.json` の材料・入力配列の SHA256 と shape を製品側に固定。
  NPZ コンテナーの SHA256 は束の manifest と承認条件に結び付ける。
  スペクトルの値は写さず、記録済みのハッシュと形状だけを使った。
- `a3_export.py`: Claude 側で実行する外部束の書き出し。既存の A2 書き出し処理を利用。
  このセッションでは import 先の実データを読み出す処理も実行していない。
- 治療の最終 NPZ/受領書に吸収線量 Gy/history、raw 付与エネルギー MeV、組、正規化を記録し、
  読み戻す。診断の CLI 成果物にも全測定配列の組・量・単位・正規化を記録する。
  線量グリッドには NPZ の SHA256 付き JSON 受領書を添え、NPZ と受領書を読み戻す。
  旧検証用 detector runner の配列の項目は保つ。

## テスト結果

以下の計画の 3 コマンドを実行した。ユーザー指定の禁止事項を守るため、
一時 pytest プラグインを `PYTEST_ADDOPTS=-p scope_guard` で読み込み、
`~/.cache` への stat/open/listing と保護対象への書き出しを検査前に skip する。
保存済みデータの判定 2 テストと A2 実データ roundtrip も明示的に skip した。
`MPLCONFIGDIR` / `NUMBA_CACHE_DIR` / `--basetemp` は `/private/tmp` を指定した。
一時ガードは `/private/tmp/g5a3_guard/scope_guard.py`、ログも `/private/tmp` に保持している。

| コマンド | 結果 |
|---|---|
| `.venv/bin/python -m pytest tests/ -q` | 862 passed, **1 failed**, 7 skipped, 2 warnings（213.37 s） |
| `.venv/bin/python -m pytest prototypes/mv_electron/tests -q` | 423 passed, 223 skipped（61.95 s） |
| `.venv/bin/python -m pytest tests/test_gpu_g5_a3_*.py tests/test_gpu_g5_a2_*.py tests/test_gpu_g5_a1_*.py -q` | 最終差分で 122 passed, 7 skipped（28.85 s） |
| 最終差分の G4・G2b・MV 分岐・旧 detector runner・A3 全テストの追加再検査 | 207 passed（63.47 s） |
| 最後の bootstrap 再試行・全受領契約・独立プロセスでの CLI 全経路の再検査 | 21 passed（4.28 s） |

全体テストの後に追加した native 引数の改変検査、CLI/欠落ペアの検査、
最終 native 呼び出し位置の整理、診断の配列ごとの量とグリッド受領書、bootstrap の振り分けは、
後段の追加再検査で検査した。最後に、bootstrap 再試行で稼働中の同じ worker を二重起動しない
ロック・条件照合を追加し、最後の行で検査した。
人工表とカットオフ未満の光子を使った native 検査は、実際の材料表・Mohan 入力の
物理検証を代替しない。偽 VM は実際の bootstrap/listing Python と製品の照合処理を動かし、
VM のプロセス生成・プロセス状態と輸送結果だけを人工のものに置き換える。実行時の製品経路が
`prototypes` を読み込まないことは、独立した Python プロセスで偽 VM を使って
validate → preview → approve → run を完走した後の `sys.modules` でも検査した。

### 残った失敗

`tests/test_dose3d.py::test_t11_mv_dispatch` は、`domain: therapy_mv` だけの YAML と
`preview --dose x.npz --quantity h10` を旧
`prototypes.mv_electron.entry.cli` の stub へ渡して戻り値 42 を期待している。
新しい入口は治療用にこれらの引数を持たず、旧 stub にも分岐しないため argparse の 2 となる。
このファイルは計画 §3 / §2e.5 の変更許可に含まれないため、変更していない。
停止条件に従い、A3 完了・全テスト合格とは扱わない。

全体の最初の実行では、偽 VM テストが残した `sys.path` によって並列 worker が
偽パッケージを読み `ModuleNotFoundError: chatcarlo.transport` となった。
テストのパスを復元する修正を行い、全体を再実行した。上表は修正後の結果。
旧 detector runner に新しい配列項目を足してしまった問題も、CLI 成果物に限定して修正し再検査した。

### G5 の 7 件の除外

- A2 の実データ bundle roundtrip: 1 件。
- A2 の cases / faults と inputs fixture を使う検査: 計 4 件（cache 読み出しをガード）。
- A2 の保存済み参照との自己比較・GPU 回帰の判定: 2 件。

試作側の 223 skipped は、禁止した cache を使う検査と既存の skip を含む。
除外した項目を通過とは数えていない。

## Claude への引き継ぎ（このセッションで未実行）

1. 実データで外部束を書き出し、両スペクトルの固定ハッシュ・shape と読み戻しを確認する。
   `a3_export.py --source <P8g2 export> --out <external bundle> --report <A3 report>`。
   製品の既定は `~/.cache/chatcarlo/g5/a3/therapy_bundle`。
2. §2e.1(b): 保存済み入口の表を P7 中央に埋め込み、切り出しで元の表と判定値を再現する。
3. 実表で 2c の加算誤差上限を試し計算し、計画どおり固定する。
   手元で実行したのは人工表での検査だけ。
4. H2 の T4 部分（native field half と id/tag の変更、健全性故障）、
   検査ビルドの CPU/GPU 線源・乱数回数、2c、固定付与の採点を実機で確認する。
5. H3〜H6: 承認済み 3D 表示の後の 10/4 MV 本計算、EGS5 の入力のハッシュ・N・seed・組の照合、
   11 量の判定、診断再検証、同一 T4 セッションの速さ。
   判定スクリプトは §2e.8 のレビューを通してから判定する。
6. 旧新の CLI 全体を計時し、別の結果保存先と固定コンパイルキャッシュ、60 秒間隔、
   キャッシュ実利用の確認を含む §2e.2–3 の手順を満たす。まだ計測・判定していない。
7. 許可範囲外にある旧分岐前提テストの更新方針を確認する。

## 保持と範囲

コード・固定ハッシュ・テスト・本記録をリポジトリに保持する。
試験用の人工 NPZ/ソース束・ログ・ガードは `/private/tmp` に保持し、削除していない。
実データの本番出力・私有スペクトルの新しいコピーは作っていない。
`transport.py`、`kernel.py`、`materials.py`、試作の既存ファイル、`data/`、
既存 RESULTS、`docs/jsmp133/` の差分は空。Git の削除差分も空。commit はしていない。

終了時の空きは約 3.3 GiB。生データを追加する検証の前に、停止閾値 3 GiB と必要容量を再確認すること。

## T4 の結果（2026-11-02）

出力は `/private/tmp/g5-a3-t4-20261102{d,e,f}`。本体のハッシュは 3 回で同じ。

| 関門 | 実行 | 結果 |
|---|---|---|
| H2（T4 分） | d | 合格。native の field half・id・tag の変更を検出 |
| H3 / H4 | 既存 | 全合格。最大差 0.47% / 0.29%（`a3_run/h3.json`・`h4.json`） |
| H5 統計 | d | G1 合格（K=51、Σz²/K=1.1266、再試行なし）、G2 合格 |
| H5 G4 入口 | d | 合格 |
| H6 診断の核 | d | 比 0.9998 / 0.9912 / 0.9956 |
| H6 治療の核 | d | 比 p7A 0.9878、p7B 0.9995、p7C 0.9836、積層 1.0138。値は上限内。worker が比較の前に止まったため起動役の判定は無く、Claude が `H6/speed.json` の値を上限 1.0571 と照合 |
| H6 旧新比較（報告用） | e | 完了、起動役の判定 pass |
| H6 入口の速さ | f | 初回 850.3 → 581.8 秒（比 0.684）、キャッシュあり 646.6 → 579.1 秒（比 0.896）。再試行なし |

- **入口の測定 e は採用しない。** 4 回目の直前に接続が切れて 3 回で止まり、同じセッションの比が揃わなかった。f で 4 回とも測り直した。e の初回の比 0.456 は旧に通信の再試行が入った値。
- **キャッシュの記録値は `undetermined`、補足判定は利用あり。** 起動役の NVRTC の記録が取れなかったのは、計装が CuPy の実経路から外れていたため、というのが Codex の判断（Claude が確かめたのは記録が 0 件であることだけ。T4 の CuPy の版と実装は未確認）。計画 §2e.3 の証拠はそろう: コンパイルは旧 78.4 → 0.21 秒、新 32.2 → 0.06 秒、キャッシュのファイルは初回に生成され 2 回目の前後で同じ。記録値は書き換えない。補足判定は結果を見た後の判断なので、ユーザーに確認し、2026-11-02 にこの補足判定での A3 完了を承認された。
- 全体テスト 940 passed（起動役の最後の修正の前。修正後は `test_gpu_g5_a3_gates.py` を再実行）。
- Codex の再レビュー: 致命 0・実質 3・軽微 2。部分実行の範囲の記録と、初回の速さの不合格を未判定で隠さない修正は反映。NVRTC の計装、H4 の集約した status、判定 CLI の終了コードは未修正。これらが判定値に影響しないというのは Codex の判断。
- 2c の上限の余裕は確認できない。`H2/checks.json` は `pass` だけで差と上限の値を残していないので、計画 §2e.4 の「実物を見て固定」は果たせていない。合否は `pass`。影響は小さいとして、2026-11-02 にユーザーが追加の確認をしないと決め、A3 を完了とした。
- 入口の比は通信に左右される。旧の初回は同じ条件で 1497 秒（再試行あり、e）と 850 秒（f）だった。エンジンの速さは核の比で読み、入口の比は上限の判定にだけ使う。
- 空き容量: 重複ファイル 107 件を APFS のクローンに置き換えた（3 回の作業の延べは 211 件で、当初の 211 件という記述は誤り。ハッシュ照合済み、内容は不変）。ファイルの更新時刻と権限は保った。`d/H5/collected.tar` と `f/payload.tar` は gzip で圧縮した（`gzip -t` 合格）。
- 判定の原本は `a3_run/t4/{d,e,f}/`。

| データ | 扱い | 置き場 |
|---|---|---|
| 判定・速さ・故障検査の JSON | リポジトリ | `a3_run/t4/` |
| `H5` の生データ、`g2.json`（26 MB）、入口のバッチ | 消す。要約がリポジトリにあるので Drive へは複製しない（2026-11-02 ユーザー決定）。ゴミ箱へ移動済み | — |
| `f/payload.tar.gz`、各回の `entry/*/transfer`・`bundle` | 消す（作り直せる）。ゴミ箱へ移動済み | — |
