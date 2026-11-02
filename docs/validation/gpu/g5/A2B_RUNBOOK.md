# G5 A2b — Claudeへの引き渡し

実装の範囲は承認計画（特に§2e）。**公開CLIの`seed_ratio`は2026-11-03に新入口へ切り替え済み**（結果は A2B_LOCAL.md 末尾）。
B0〜B2、B3の全7量、B4、B5を確認するまで切り替えない。
ここにあるColab、T4、旧キャッシュの検査はCodexでは実行していない。

## 1. 手元で入力を封印する（Claude）

以下はリポジトリのルートから実行する。秘密のスペクトルをリポジトリへ置かない。
`A2B_WORK`は3 GiB以上の余裕がある保存先へ変えてよい。

```bash
export PYTHONPATH=.:docs/validation/gpu/g5
export A2B_WORK="$HOME/.cache/chatcarlo/g5/a2b"
mkdir -p "$A2B_WORK"
.venv/bin/python docs/validation/gpu/g5/a2b_physics.py
.venv/bin/python docs/validation/gpu/g5/a2b_export.py \
  "$HOME/.cache/chatcarlo/g5/a3/therapy_bundle" "$A2B_WORK/seed_bundle"
```

exportはP9aの凍結済みファイルのSHA256を照合する。`seed_agti`の33配列と5材料の密度を
`chatcarlo/gpu/therapy_input_hashes.json`へ追記する。既存4材料・既存inputのピンは変更しない。
その差分を確認する。ピンが未作成なら新経路は拒否する。旧P7はseed用の追記だけを読み飛ばす。

共通マスクは`a2b_masks.npz`と`a2b_masks.json`で固定済み。
近傍25,400、前立腺186,140、直腸193,480セル。0.5 mmセルの8つの旧セルがすべて
旧の水／臓器に属するものを使い、新格子の`water_mask`を掛け直さない。

## 2. 比較相手を固定する（計算前、Claudeが独立レビュー）

```bash
.venv/bin/python docs/validation/gpu/g5/a2b_judge.py freeze \
  "$A2B_WORK/reference.json" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate/prostate.npz" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate2/prostate.npz" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate3/prostate.npz" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate4/prostate.npz" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate5/prostate.npz" \
  "$HOME/.cache/chatcarlo/p9c2/prod_prostate6/prostate.npz"
```

このJSONには6本のSHA256、600バッチの明示リスト、入力identity、再集計した旧値・SEM・δが入る。
既存ファイルを上書きせず、6本のseedと幾何・束・履歴数が揃うことを要求する。
`combined.load()`の既定値は使わない。JSONと判定コードを計算前に独立レビューする。
旧の近傍SEMは旧0.25 mmセルを独立と仮定した値。新は50対バッチ間のSEMであり、算出法が異なる。
臓器の平均値は共通マスクで再計算し、旧SEMは既存aggregateの相対SEMを使う。

δ：前立腺比0.001、直腸比0.001、近傍比0.003、前立腺の各側0.5%、直腸の各側0.7%。
各側の量は付与の和÷受け入れ光子数÷マスク体積（MeV/cm³/光子）。比は和の比。
`|差|≤δ`かつ`3×合成SEM≤δ`を要求する。統計不足でもδは変えない。

## 3. 承認用の4ディレクトリを準備する

```bash
.venv/bin/python docs/validation/gpu/g5/a2b_launch.py prepare \
  --scene examples/therapy/prostate_seed_7field_g5_a2b.yaml \
  --bundle "$A2B_WORK/seed_bundle" --out "$A2B_WORK/entry"
```

`entry/{old_cold,new_cold,old_cached,new_cached}/preview.html`を使用者が3Dで確認する。
確認の後、対応する入口で承認する（計算前に人間の承認が必要）。

```bash
for phase in cold cached; do
  .venv/bin/python docs/validation/gpu/g5/a2b_legacy_adapter.py approve \
    examples/therapy/prostate_seed_7field_g5_a2b.yaml --bundle "$A2B_WORK/seed_bundle" \
    --preview "$A2B_WORK/entry/old_$phase"
  .venv/bin/python -m chatcarlo.gpu.seed_entry approve \
    examples/therapy/prostate_seed_7field_g5_a2b.yaml --bundle "$A2B_WORK/seed_bundle" \
    --preview "$A2B_WORK/entry/new_$phase"
done
```

旧側のadapterも0.5 mm・50バッチ・同じ各門71429光子に揃える。既存P9cファイルは変更しない。
旧baselineの条件には旧ソースのハッシュと`baseline_only`が入り、通常の新入口は受け入れない。
旧側にホストで補った履歴番号・物理ラベル・未使用採点欄はB5比較専用であり、
独立したnative観測の証拠にはしない。B1/B4の実引数観測は新CPU/GPUで行う。

## 3b. T4の前に手元で通す（必須、2026-11-02追加）

Colabでしか通らない経路を、CPU版をGPUの代わりに差し込んで通す。T4へ送る前に毎回実行する。
初回のT4実行では、この種類の不具合が5件、1回ずつ見つかった（コンパイラの選択、検査の格子、
G3の期待値、起動の照合時間、旧側adapterの差し替え範囲）。

```bash
W="$A2B_WORK/dryrun"
.venv/bin/python docs/validation/gpu/g5/a2b_cpu_dryrun.py gates --bundle "$A2B_WORK/seed_bundle" --work "$W"
# production は承認済みの新入口のディレクトリ（検査用に別に preview/approve したもの）を使う
.venv/bin/python docs/validation/gpu/g5/a2b_cpu_dryrun.py production --bundle "$A2B_WORK/seed_bundle" --work "$W" --root "$W/root"
```

`gates`はB0〜B2を最後まで（V1c・V2bの履歴ループは2チャンクに短縮、約3分）。
`production`は本番と同じ履歴数で2バッチ、途中喪失、再開で3バッチ（約16分）。
GPUにしかない振る舞い（`atomicAdd`の書き込み先、NVRTCのコンパイル、デバイスメモリ）は通らない。
CUDAでは`atomicAdd`の書き込み先はグローバルメモリに限る（スレッドのローカル変数は不可、P8g-1とA2bで踏んだ）。

## 4. T4実行

### CLIによるB3・B5の同一セッション実行

```bash
.venv/bin/python docs/validation/gpu/g5/a2b_launch.py execute \
  --scene examples/therapy/prostate_seed_7field_g5_a2b.yaml \
  --bundle "$A2B_WORK/seed_bundle" --reference "$A2B_WORK/reference.json" \
  --out "$A2B_WORK/entry"
```

T4を1台だけ所有し、旧／新の初回・キャッシュ済みの実CLIを測る。
A3のOwner、SharedClient、timed_main、cache_usedを再利用する。RawModuleの時間も記録し、
NVRTCとcubinの照合なしに「キャッシュ済み」とは判定しない。
初回旧の後、同じVMで旧新の核を交互に3回測る。核の比が1.0571を超えたら止めて報告する。
全てのColab呼び出しは既存のtimeout/retryを使い、60秒ごとにPID・出力を確認する。
別PIDのwatchdogがheartbeatを監視し、失敗時にも最終回収と`colab stop`を行う。

B0〜B2はVMごとに再実行する。V1c・V2b・V3は9502101→9502102の1回だけ再試行。
V2bだけは統計不足ならV2aで代える。歴史的なV3の90°例外は引き継がない。
V3は独立な投影面積に加え、実CUDA線源の受け入れ数・棄却数も確認する。
B3は9502001（再試行は9502002）。50バッチ、各門3,571,450、各側25,000,150光子。
「25,000,000/7」を丸めて履歴の欠落を作らない。

### ブラウザColab：「すべてのセルを実行」1回

```bash
.venv/bin/python docs/validation/gpu/g5/a2b_notebook.py "$A2B_WORK/notebook"
```

生成する`a2b_t4.ipynb`と`a2b_source.zip`、承認済み4ディレクトリをgmail Driveの
`MyDrive/viveMonte/g5/a2b/`へ置く。全ファイルの同期が終わってからColabで開き、
GPUをT4にし、「すべてのセルを実行」、gmail Driveを許可する。
承認用ディレクトリ内の`bundle`は手元では共有リンクになっている。Driveへはリンク先の実体を展開して転送する
（例：Pythonの`shutil.copytree(..., symlinks=False)`）。その複製の容量が足りなければCLI経路を使う。
コードarchiveのSHA256を確認してから取り込み、各workerのPIDとsnapshotを60秒ごとに確認・回収する。
同条件の再実行では最後に確定したバッチから再開する。
**notebookのelapsed_on_VM_sはVM内だけの時間**。入口全体のB5判定とキャッシュの立証には上のCLI測定を使う。
ブラウザ実行が終わったらランタイムを切断・削除する。

## 5. 回収・判定・停止

各batchの14門対が揃って健全性・native実引数が一致してから、1つのcheckpointを原子的に更新する。
途中の対は採用しない。再送・履歴欠落・範囲外・採点欠落は拒否する。49バッチでは未完了。
回収するtarは単一ファイルで、SHA256と安全なmember名を確認する。

```bash
.venv/bin/python docs/validation/gpu/g5/a2b_judge.py judge \
  "$A2B_WORK/reference.json" "$A2B_WORK/entry/new_cached/results/summary.json" \
  "$A2B_WORK/B3.json"
```

notebookの場合は`new_cached/snapshot.tar`をハッシュ照合して回収し、
`seed_colab.unpack()`でローカルの承認済み`new_cached/results/`へ展開する。
`seed_entry analyse`を実行してから同じ判定コマンドを使う。
判定スクリプトと旧参照値は判定前にも独立レビューする。

B1/B4不合格、B3差、統計不足なら旧入口を維持して報告する。B5超過は切り替える前に使用者へ諮る。
B3全項目がpassでも、この実装は自動で切り替えない。ClaudeがB4とB5を照合した後に分岐の変更を行う。
元の公開sceneは変更していない。シード近傍のEGS5精度は確立しておらず、P9bの不合格も残る。

## 6. データと再検査

| 出力 | 扱い | 用途 |
|---|---|---|
| `batch_table.npz`・summary・B3/B5 JSON・reference | 小さな表をリポジトリへ | 再判定、算出法と来歴 |
| `prostate.npz`（対の5モーメント・50行・native受領記録）・snapshot | Driveで全SHA256照合後、手元の生データを整理 | 格子の再解析、再開 |
| 転送parts・再生成可能なビルド | 参照されていないことを確認して整理 | 再転送で復元 |

空き3 GiBを下回る見込みなら止める。checkpoint保存時にも必要容量を確認する。

計画のテストコマンド（Claude側ではキャッシュ検査も含めてそのまま実行）：

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
```

後者が書き直す`docs/validation/mv/p8g1/faults_gate0.json`と
`p8g3/fault_observations.json`は実行前の内容と突き合わせ、テスト由来の一時パス差分だけを戻す。
