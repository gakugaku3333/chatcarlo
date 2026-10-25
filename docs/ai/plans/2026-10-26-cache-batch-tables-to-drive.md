# 計画: 閉じた段階のキャッシュから外側のバッチの表を残し、生データを Drive へ移す（第 1 段階）

- 状態: implemented（2026-10-26。段階 1 は Codex、段階 2a・2b は Claude。結果は `docs/validation/mv/cold_storage/README.md`）
- 作成者: claude
- 実装担当: codex（表の抽出と再計算のスクリプト）。Drive への写しと Colab での照合は Claude（Codex のサンドボックスは `~/.cache` と Drive に書けないため）
- 日付: 2026-10-26
- 規模: 重（データの完全性）
- Codexレビュー: 2巡
  - 1 巡目（骨子）: 致命 0・実質 3・参考 3 をすべて反映。`data/logs/codex_cache_review1.md`
  - 2 巡目（詳細）: 致命 0・実質 1・軽微 6・参考 2 をすべて反映。`data/logs/codex_cache_review2.md`。
    Codex は、集計の部分だけを写して全 5 モードの出力がバイト一致することを、メモリー上で実際に確かめた。

## 目的

Mac の空きが少ない（2026-10-26 の時点で 8.7 GB）。閉じた段階の生データを gmail 側の Google Drive に移して、Mac の空きを増やしたい。
ただし、RESULTS の数値を手元で計算し直せる状態は残す（使用者決定 2026-10-26）。

そのために次の 2 つを行う。

1. **残すもの**: 各段階の集計スクリプト（`pool.py`）が実際に使っている「外側のバッチの表」（各計算 100 行 × ビン、float64）と、各計算の平均・SEM を、
   小さな派生ファイルとしてリポジトリに残す。
2. **移すもの**: 作り直すのに GPU の時間がかかり、作り直しても記録のハッシュと一致しない生データ（GPU の結果のバッチのファイル、束、関門の参照データ）だけを Drive に移す。
3. **消すもの**: 作り直しが簡単で、どの封印記録にも載っていないもの（EGS5 の `gsdist.dat`・`pgs5job.GSlst`、Colab に送った束の分割ファイル）は、
   Drive に送らずゴミ箱へ移す（使用者決定 2026-10-26。Drive への転送の時間を減らすため）。

この計画が答える問いは次のとおり。

> 派生ファイルだけで、両 RESULTS の比較の数値（通常・`detail`・`orig` の出力）がバイト単位で同じに出るか。
> Drive に置いた写しは、元と相対パスの集合も各ファイルの SHA256 も一致するか。

**保証しないこと**（参考 5 の区別）: 元データの検証を強めることではない。`pool.py` の読み込み関数が今している照合（積層 EGS5 は N・seed・グリッド、
P8f は各バッチのハッシュ・n・start・seed、P8g-3 は完了記録と各呼び出しのハッシュ、4 MV の GPU は `bundle.collect` の来歴・ハッシュ・健全性）を
そのまま通した配列を残す。過去の封印記録との完全な照合（`core.verify()`、`verify_sessions()`）を今回やり直すわけではない。

## 0. 前提事実（検証済み、2026-10-26）

### F2・F10 現物の大きさ

`du` の値。対象外のものも参考に並べている。

```
archive 5.6G | p9c2 4.3G | p8a 3.2G | p8e 2.2G (egs 1.9G, materials 278M) | mv_entry 1.7G (96782f068a81 1.1G, seed_ac2cd8a3e6f6 683M)
statsstack 1.2G (work 993M, p8f 220M) | p8g1 1.1G (cli 478M, cli_attempt1 478M, p8g_ref_gate2.npz 167M) | p8g3 890M (colab)
p8g2 645M (cli 478M, p8g2_ref_gate2.npz 167M) | elsepa 628M | p9a 611M | stats4mv 319M (work 312M)
```

- 10 MB 以上で中身が同じファイル 50 個は、2026-10-26 に APFS クローンにしてある（`STATUS.md` に記録）。
  - クローンは `du` では元の大きさのまま見えるが、データは共有している。
  - **したがって、空いた量は `du` の合計ではなく、使用者がゴミ箱を空にし、Drive の同期のキャッシュが解放された後の `df` で測る**（参考 6）。

### F7 集計スクリプトとの契約（コードを読んで確認）

**`docs/validation/mv/statsstack/pool.py`（積層、220 ビン）**

引数なしで全 5 計算、`orig` で EGS5 本番と P8f だけ、`detail` で層別の内訳を足す。

| 計算 | 読む場所 | 表 `q` |
|---|---|---|
| EGS5 本番 | `docs/validation/mv/p8e/egs_production.npz` の `dose`（リポジトリ） | 100 × 220 |
| EGS5 追加 × 2 | `~/.cache/chatcarlo/statsstack/out/egs_add_{820201,820202}.npz` の `dose` | 100 × 220 |
| GPU P8f | `statsstack/p8f/p8f/results/production/batch_{000..099}.npz` の `dose` | 100 × 220。各 `.json` の sha256・n・start・seed を照合している |
| GPU P8g-3 | `p8g3/colab/results/call_{000..399}.npz` の `dose` | 4 呼び出しの平均を外側の 1 バッチにした 100 × 220。`complete.json` の sha256 を照合している |

- 表のほかに、`bin_layer`・`edges_cm`・`mass_g_cm2`（EGS5 の 3 計算で一致を確かめている）を使う。
  - 層の境界をまたがない 2 ビンの統合、基準ビンの選択、深さと層の内訳の出力に要る（実質 1）。
- `ms(q)` は `q.mean(0)` と `q.std(0, ddof=1)/10`。
- 2 ビンの統合は、各バッチの中で行ってから SEM を取る。
  - **ビンごとの平均と SEM だけでは、統合後の SEM を再現できない。**
  - 外側のバッチの表を残す理由はこれ。

**`docs/validation/mv/stats4mv/pool.py`（4 MV、228 列 = PDD 120 + OCR 3×36）**

引数なしで全 5 計算、`orig` で元の 2 計算（P7 の EGS5 と `runs/` の GPU）だけ。

| 計算 | 読む場所 |
|---|---|
| EGS5 | `docs/validation/mv/p7/egs5/main_egs_C/result.npz`（リポジトリ）、`~/.cache/chatcarlo/stats4mv/out/egs5/add_C_{20292301,20292302}/result.npz` |
| GPU | `runs/mv_water_4mv_20261024/entry`（gitignore、ローカル）、`~/.cache/chatcarlo/mv_entry/96782f068a81`（`bundle.collect` と `compare.stacked` で 100 バッチの結果を読む） |

- 残す表は、`quantities()` の出力（線量への換算、4 方向の平均、内側の 36 ビンの切り出しを済ませた 100 × 228）。
  - `compare.stacked()` の出力そのものではない（実質 1）。
- ビンの統合（`merge(runs, k)`）もバッチの中で行う。
- 出力には各計算の名前・N・seed と並び順が出るので、これも残す。

**F10 出力がいま再現できることを確認した**（生データがある今の状態）:

```
$ PYTHONPATH=$PWD .venv/bin/python docs/validation/mv/statsstack/pool.py detail | diff - docs/validation/mv/statsstack/final.txt   -> 差なし
$ PYTHONPATH=$PWD .venv/bin/python docs/validation/mv/stats4mv/pool.py | diff - docs/validation/mv/stats4mv/final.txt              -> 差なし
$ pool.py の各モードの出力の行数: statsstack 通常 6 行・orig 2 行、stats4mv orig 3 行（どれも正常終了）
```

- `pool.py` はモジュールの最上位で計算を実行する。
  - `runpy.run_path(pool.py)` は、実行後の大域変数（`egs`、`gpu`、`layer`、`edges` など）を返す。
  - **このため、読み込み関数を写さずに、そのまま通した `q` を取り出せる。**

**他のコードからの参照（`grep` で確認。参考 4 の指摘を足した）**

| 移すもの | 読むコード | 復元が要る作業 |
|---|---|---|
| `p8e/egs` | `run_p8e.py`、`gpu/p8f.py:323`、`statsstack/run_add.py`（`production/identity.json`）、**`p9a.py:407` の `load_replay_reference()`（`p8e/egs/pilot`）** | P8e・P8f の再実行、積層の EGS5 の追加、P9a の `egs-replay`・`egs-pilot`・`egs` |
| `p8g1` | `gpu/p8g.py`、`p8g_cli/start_vm.sh` | P8g-1 の再実行 |
| `p8g2` | `gpu/p8g2.py`、`p8g3.py`、`p8g2_cli` | P8g-2・P8g-3 の再実行 |
| `p8g3/colab` | `gpu/p8g3.py`、`p8g3_cli`、`statsstack/pool.py` | P8g-3 の再実行と、`pool.py` の再実行（派生ファイルからの再計算では不要） |
| `statsstack/p8f`、`statsstack/work` | `statsstack/pool.py`、`run_add.py` | 同上 |
| `stats4mv/work` | `stats4mv/run_add.py` | 4 MV の EGS5 の追加 |
| `mv_entry/96782f068a81` | `stats4mv/pool.py` | `pool.py` の再実行 |

- 閉じた段階を計算し直すときに復元を求めることは、今の運用（`archive_run.sh` と STATUS の表）で認められている。
- P9a と P9c の材料の読み込みは `p8a` と `p8e/materials` だけを読む。どちらも対象外。

### F3 破壊的な操作

- 派生ファイルの作成は、新しいファイルを作るだけ。既存のファイルは変えない。
- **D の封印の確認**: 3 つの EGS5 の作業フォルダの 10 MB 以上のファイル 16 個（3.38 GB）を、同じフォルダの `complete.json`・`identity.json` のキーと照らした。
  - 結果は全部 `sealed=False`。中身は `gsdist.dat` 9 個、`pgs5job.GSlst` 5 個、`egs5job.out` 2 個。
  - `egs5job.out`（`stats4mv/work`、27 MB × 2）は EGS5 の本出力なので D に入れない。
  - 封印には、`run_p8e.verify_seal()` が求める `egs5job.out`・`pgs5job.pegs5dat`・`pgs5job.pegs5lst`・`config.dat`・`spectrum.dat` が含まれる。
- 生データについては、Drive への写し → Colab で全ファイルの SHA256 を照合 → 一致したら元を `~/.Trash` へ `mv` する。`rm` は使わない。
  - ゴミ箱を空にするのは使用者。
- 凍結の検査（`p9a_data.protected_paths()` の記録 `docs/validation/mv/p9a/protected_start.json`、62,042 件）は、いま 2 件が不一致。
  - 不一致は `prototypes/mv_electron/entry/cli.py` と `scene.py` で、コミット 1368eec で変わったもの。置き換えの前からある。
  - `verify_protected()` は最初の不一致で例外を投げるので、件数は返さない。
  - 対象のフォルダは、この検査の範囲に入っていない。

### F10 Drive の現物

- 写す先は `~/Library/CloudStorage/GoogleDrive-o.ayumu@gmail.com/マイドライブ/viveMonte/`。ストリーミングで同期されている。
- Drive のコネクタで見られるメタデータには、ハッシュ（md5）がない。
  - `p6/p6_inputs.npz` で確認した: 返る項目は fileSize・modifiedTime などだけ。
- Colab CLI（`colab`）には `drivemount` のコマンドがある（`colab --help` で確認）。**まだ使ったことがない**（未検証）。
  - 既存の起動役（`entry/colab_cli.py`、`p9c_cli.py`）は Drive をマウントせず、ファイルを直接送っている。
  - → 関門 D0 で、まず小さなファイルで試す。
- Drive for desktop は、アップロードが終わるまで写しをローカルのキャッシュに持つ。
  - 一度に大量に写すと Mac の空きが一時的に減る。

## 判定のルール（事前に決める）

1. **派生ファイル（関門 T）**:
   - 再計算のスクリプトの出力が、元の `pool.py` の出力とバイト単位で一致すること。両段階で計 5 つのモード（下の表）。
   - 基準の出力は、生データがある今のうちに、元の `pool.py` で作って保存しておく。
   - 計算ごとの `q` が、`pool.py` の読み込み関数が作った配列と、形・dtype・値（`np.array_equal`）で一致すること。

     | 段階 | モード |
     |---|---|
     | statsstack | 通常、`orig`、`detail` |
     | stats4mv | 通常、`orig` |

   - 再計算のスクリプトが計算の入力として読むファイルは、指定の npz と json だけであること。
     - 実行に必要なコードとライブラリの読み込みは除く。
     - 別プロセスで `sys.addaudithook` を使い、`~/.cache/chatcarlo`・`runs/`・Drive を開こうとしたら失敗にして確かめる。
   - 保存した `mean`・`sem` とメタデータを読み直し、`q` から計算した値や、抽出したときの値と一致すること。
     - 対象は、形、dtype、計算の順、`orig` の選び方、グリッド、N・seed。
   - 保証するのは、表の 5 つのモードを、今回の抽出と検証の環境で実行したときのバイト一致まで。
     - 将来の NumPy・SciPy・Python や別の CPU でも同じになるとは保証しない。
     - 抽出したときの Python・NumPy・SciPy の版を json に記録する。
     - 積層の `orig detail` の組み合わせは対象外。
2. **Drive の写し（関門 D）**:
   - 写す前に、元の全ファイルの相対パスと SHA256 の一覧を取る。
   - Colab で Drive 上の写しから同じ一覧を作り、相対パスの集合と各 SHA256 が全件一致すること。
   - 1 件でも違えば、そのフォルダの元は動かさない。
3. **凍結の検査（関門 P）**:
   - 移す前と後で、`protected_start.json` の全件について「不一致のパス・記録のハッシュ・今のハッシュ」の一覧を作り、2 つの一覧が完全に一致すること（実質 3）。
   - 既存の 2 件は認める。記録は書き換えない。

## 対象範囲

- **変更してよい・作ってよい**（Codex）:
  - `docs/validation/mv/statsstack/` と `docs/validation/mv/stats4mv/` に、次のものを新しく作る。
    - `extract_tables.py`: 元の `pool.py` を `runpy.run_path` で実行し、`q` とメタデータを書き出す。
    - `batch_tables.npz`、`batch_tables.json`: 派生ファイル。
    - `pool_tables.py`: 派生ファイルだけから同じ出力を出す再計算のスクリプト。
    - `baseline_{plain,orig,detail}.txt`: 元の `pool.py` の出力の基準。
  - `prototypes/mv_electron/tests/` に、上のスクリプトのテストを 1 ファイル足してよい。
- **変更してよい**（Claude）:
  - `docs/validation/mv/cold_storage/` に、次のものを置く。
    - 移した各フォルダの一覧（`<label>.sha256`）と、Colab の照合結果（`<label>.colab.sha256`）
    - 関門 P の前後の一覧（`protected_before.json`、`protected_after.json`）
    - 移す前の必須の検査の結果（`gate_T.txt`）
    - 復元の対応表（`README.md`）
  - `docs/validation/mv/STATUS.md` に 1 段落足す。
- **変更禁止**:
  - 既存の `pool.py`・`final.txt`・RESULTS・凍結のファイル・`protected_start.json`
  - `p8a`、`elsepa`、`p8e/materials`、`p9a`、`p7`
  - 使用中の `p9c2`、`p9c`、`mv_entry/seed_ac2cd8a3e6f6`、`jsmp133_request_d`
  - `archive/`、`runs/`
- **Drive に移すもの**（K。`~/.cache/chatcarlo` からの相対パス。1,340 ファイル、見かけ 2.29 GB）:

  | 対象 | 備考 |
  |---|---|
  | `p8g3/colab`（全体） | 結果は表に残す |
  | `statsstack/p8f`（全体） | 結果は表に残す |
  | `mv_entry/96782f068a81`（`transfer_parts/` を除く全体） | 結果は表に残す。`bundle/` は `pool.py` の `load_gpu` が読むので一緒に移す |
  | `p8g1/p8g_ref_gate2.npz`、`p8g2/p8g2_ref_gate2.npz` | 関門の参照（20,000 履歴）。CPU で作り直せるが時間がかかる |

- **Drive に送らず、ゴミ箱へ移すもの**（D。75 ファイル、見かけ 5.23 GB）:

  | 対象 | 消してよい理由（確認済み） |
  |---|---|
  | `p8e/egs/**`、`statsstack/work/**`、`stats4mv/work/**` の `gsdist.dat` と `pgs5job.GSlst` | EGS5・PEGS5 が実行のたびに作る表と一覧。各フォルダの `complete.json`・`identity.json` のどちらにも名前がない（下の F3 の出力）。ほかのファイル（`complete.json`、`identity.json`、`egs5job.out`、入力）は手元に残す |
  | `p8g1/cli/`、`p8g1/cli_attempt1/`、`p8g2/cli/` | Colab に送った束の分割ファイル。各段階の `export` で作り直せる。10-25 にも同じ種類の束を、圧縮せずにゴミ箱へ移している |
  | `mv_entry/96782f068a81/transfer_parts/` | `bundle/inputs.npz` と `scorer.npz` を送るために分割したもの。元の `bundle/` は K で Drive に移す |

  - P9a の `load_replay_reference()` が `p8e/egs/pilot` で照合するのは、`complete.json` に載っているファイル（`egs5job.out`・`pgs5job.pegs5dat`・`pgs5job.pegs5lst`・`config.dat`・`spectrum.dat` など）と `identity.json`、入力の文字列だけ。
    `gsdist.dat` と `GSlst` は含まない。
- **手元に残すもの**: 上の K と D 以外のすべて。
  - `statsstack/out` と `stats4mv/out`（合計 7 MB）、EGS5 の作業フォルダの小さなファイル、`p8g1`・`p8g2` の json など。
- **空く量（クローンを考えに入れて計算した値）**: D と K で 3.61 GB（D だけで 1.49 GB、K だけで 2.12 GB）。
  - 見かけの合計（7.5 GB）より小さいのは、2026-10-26 にクローンにした組が、1 組で 1 回分しか空かないため。
  - 空くのは、使用者がゴミ箱を空にし、Drive の同期のキャッシュが解放された後。

## 実装の手順

### 段階 1（Codex）: 表の抽出と再計算

1. **基準の出力を保存する。** 元の `pool.py` を各モードで実行し、出力を `baseline_*.txt` に保存する（PYTHONPATH はリポジトリのルート）。
   - statsstack の `detail` の出力は `final.txt` と一致することを確かめる（stats4mv の通常も同じ）。
2. **`extract_tables.py` を書く。**
   - 各段階で、`sys.argv` を通常のモードにしたうえで、`runpy.run_path(pool.py, run_name='__main__')` を 1 回だけ実行する。
     - `sys.argv` と作業フォルダは `finally` で元に戻す。
   - `egs` と `gpu` を、その順で保存する。
     - `orig` は、それぞれの先頭の 1 計算を選ぶ（Codex がこの選び方で出力の一致を確かめた）。
   - **読み込み関数は写さない。来歴の情報だけを補って読む。** `runpy` の戻り値には、積層の seed と元のファイルのハッシュがないため。
     - 積層の EGS5 の seed は、大域変数 `EGS` のタプルから取る。
     - GPU の seed とハッシュは、既存のバッチの json（P8f）と `complete.json`（P8g-3）から取る。
     - 4 MV の GPU は、バッチのメタデータから集める。
   - npz（`batch_tables.npz`）に保存するもの。キーは `egs_0_q`・`gpu_1_mean` のような安定した ID にする。
     - 各計算の `q`（float64、そのまま）
     - `mean = q.mean(0)`、`sem = q.std(0, ddof=1)/sqrt(100)`
     - statsstack は `bin_layer`・`edges_cm`・`mass_g_cm2`
   - json（`batch_tables.json`）に書くもの。
     - 各計算の ID、表示名、並び順、N、seed
       - 表示名は、出力に出る文字列をそのまま保存する（4 MV の絶対パスを含む名前も）。パスから作り直さない。
     - 単位と列の並びの説明（stats4mv は `NR` と、PDD 120 + OCR 3×36）
     - 元にした `pool.py` の SHA256、抽出したコミット、Python・NumPy・SciPy の版
     - 元のファイルの一覧（計算 ID → 相対パス → SHA256）
       - `pool.py` が照合しているもの。積層の EGS5 の npz は、SHA256 を新しく計算して記録する。
3. **`pool_tables.py` を書く。**
   - 集計の関数と出力の部分を写す。
     - 積層: `ms`・`pool`・`chi`・`compare` と、出力の部分（`merge`、`detail` を含む）
     - 4 MV: `merge`・`chi`・`compare_` と、出力の部分
   - 読み込みは派生ファイルに置き換える。AST による自動の抽出や、共通の集計の仕組みは作らない。
   - 引数は元と同じ（なし・`orig`・`detail`）。
4. **テスト**（`prototypes/mv_electron/tests/test_batch_tables.py`）: 判定のルール 1 をテストにする。
   - 5 つのモードをパラメータにする。
     - 各モードを `subprocess.run(..., check=True, capture_output=True)` で実行する。
     - `stdout` を基準のファイルの `read_bytes()` とバイト単位で比べる。
     - 基準のファイルが 5 つとも空でないことも調べる。
   - 入力の監視（監査フック）は、5 つのモードのすべてで常に行う。
     - 対照: 禁止したパスを `Path.read_bytes` で読もうとしたら拒否されること（監視自体の取り逃がしを見つけるため）。
   - 保存した `mean`・`sem`・メタデータを読み直して照合する（判定のルール 1）。
   - `q` と元のデータとの照合は、2 つのモードに分ける。
     - **移す前の必須のモード**（環境変数 `BATCH_TABLES_REQUIRE_RAW=1`）:
       全 10 計算の表を比べ終え、skip が 0 件であることを求める。
       読み込み関数の例外は広く捕まえて skip にしない。ハッシュの不一致・条件の不一致・壊れたファイルは失敗にする。
     - **通常のモード**: 生データを退避した後に限り、元のデータとの比較を skip できる。
   - 対照のテスト:
     - 必要な入力が 1 つ欠けたとき、必須のモードが失敗すること
     - 既存の読み込み関数が照合の例外を出したとき、失敗になること
     - 表の行を入れ替えたとき、`q` の照合が失敗すること（行を入れ替えても平均・SEM と出力は変わらないので、照合が働いていることの確認になる）

### 段階 2a（Claude）: D をゴミ箱へ

- 関門 T（移す前の必須の検査、skip 0・失敗 0）が通った後に行う。
- D の一覧（相対パス、大きさ、SHA256）を `cold_storage/deleted_20261026.sha256` に残してから、`~/.Trash/cache_deleted_<日時>/` へ相対パスのまま `mv` する。
- 移した後、D に入らないファイルが元の場所にすべて残っていることを確かめる。

### 段階 2b（Claude）: K の Drive への写しと照合（対象 1 つずつ）

0. **関門 D0**:
   - Colab CLI の `drivemount` で Drive をマウントし、既にある `viveMonte/p6/p6_inputs.npz` の SHA256 が手元の値と一致することを確かめる。
   - 通らなければ、ここで止めて報告する（Drive 側の照合を別の方法にするかは使用者が決める）。
1. 移す前に、次の 2 つを行う。
   - 関門 P の一覧を作る（`protected_before.json`）。
   - `BATCH_TABLES_REQUIRE_RAW=1` でテストを実行し、結果を `gate_T.txt` に残す。skip 0・失敗 0 でなければ、ここで止める。
2. フォルダごとに次を行う。
   - (a) 相対パスと SHA256 の一覧を `cold_storage/<label>.sha256` に作る。
   - (b) 写す前に空きを確かめる。「Mac の空き − そのフォルダの大きさ」が 3 GB 以上のときだけ写す。
     - 足りなければ、前のフォルダのアップロードが終わり、空きが戻るのを待つ。
   - (c) Drive の `viveMonte/cache_cold/<相対パス>/` へ `cp -Rp` で写す。
     - 途中で止めた写しは、未完了として扱う。写し直して全件を照合するまで、元は動かさない。
   - (d) 同期が終わるのを待つ。
   - (e) Colab で、Drive 上の写しから一覧を作る（`<label>.colab.sha256`）。
   - (f) 一覧が全件一致すれば、元を `~/.Trash/cache_cold_<label>_<日時>/` へ `mv` する。
3. 全部終えたら、関門 P の一覧を作り直し（`protected_after.json`）、前の一覧と比べる。
4. `cold_storage/README.md` に、復元の対応表を書く。
   - 移し元、Drive の場所、復元先、復元が要るコマンド（上の表）。
   - 復元の方法は、Drive から元の場所へ写して `shasum -a 256 -c <label>.sha256`。

## 受入条件

- [ ] 判定のルール 1 の 5 つのモードの出力が、基準とバイト単位で一致する（`subprocess` の `check=True` で、異常終了も失敗にする）。
- [ ] 各計算の `q` が、読み込み関数の配列と、形・dtype・値で一致する。
  - 移す前の必須のモードで、全 10 計算・skip 0 件。
- [ ] 保存した `mean`・`sem`・メタデータを読み直したものが、`q` からの計算値や抽出したときの値と一致する。
- [ ] `pool_tables.py` が、計算の入力として、指定の npz と json 以外を開かない（監査フック。対照の拒否も確かめる）。
- [ ] 対照のテスト 3 つ（入力の欠け、照合の例外、行の入れ替え）が、どれも期待どおり失敗になる。
- [ ] 関門 D0 が通る。
- [ ] K の全部（4 つの対象）で、相対パスの集合と SHA256 が全件一致する。
- [ ] D の一覧が残っていて、D 以外のファイルが元の場所にすべて残っている。
- [ ] 関門 P の前後の一覧が完全に一致する。
- [ ] 派生ファイルの合計が 5 MB 未満（見込みは約 1.8 MB）。

## テストコマンド

```bash
# 移す前（必須。skip 0 件であること）
BATCH_TABLES_REQUIRE_RAW=1 .venv/bin/python -m pytest prototypes/mv_electron/tests/test_batch_tables.py -q -rs
# 通常の回帰（移した後も通ること）
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_batch_tables.py -q -rs
# MV のテスト全体（既存のテストを壊していないこと）
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
```

## 停止条件

- 再計算の出力が基準と 1 バイトでも違う → その段階の生データは動かさず、報告する。
- 関門 D0 が通らない → 段階 2 に入らない。
- Drive の写しの一覧が 1 件でも違う、または Colab で照合できない → そのフォルダの元は動かさない。
- 写す前の「Mac の空き − フォルダの大きさ」が 3 GB を切る → そのフォルダは写さず、前のアップロードの完了を待つ。
- 移す前の必須の検査に、skip または失敗が 1 件でもある → 段階 2 に入らない。
- 関門 P の一覧が変わる → 全体を止める。

## 書かなかったこと（スコープ外）

- **第 2 段階（別の計画）**: `archive/`（6.0 GB）の中の段階。stats10mv（抄録の 10 MV）、P2〜P7 など。
  - 展開してから同じ方法で表を作るか、そのまま Drive に移すかは、第 1 段階の結果を見て決める。
- 使用中の `p9c2`・`mv_entry/seed_…`（抄録の締切 11-01 の後）。ボクセルのモーメント和なので、表の形が違う。
- `runs/`（1.9 GB）、`docs/`（4.2 GB）、`.git`。
- float32 への変換（使わない。表は float64 のままでも小さい）。
- 圧縮した形での写し（Codex の提案どおり、フォルダのまま写す。新しい形式を増やさない）。
- 平均・SEM は、`q` から計算できるので重複だが、使用者の希望で派生ファイルに入れる。再計算のスクリプトは `q` から計算し直す（保存した平均・SEM は読まない）。
