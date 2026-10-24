# P9a 第2回実装の引き継ぎ

実装対象は approved 計画の T0–T6・T8・T9。物理の合否は Claude による実データの生成・T4 の実行・レビュー後の判定で決める。pytest の合格は物理の合格ではない。

既存の EGS5 関数と `p9a_egs.f` は変更していない。`p9a.py` の変更は CLI のサブコマンド追加だけ。計算中の EGS5 を再起動する必要はない。

## ローカルの順序

すべてリポジトリのルートで実行する。標準のデータ保存先は `~/.cache/chatcarlo/p9a`。EGS5 の保存先は従来どおりその下の `egs`。ELSEPA は既存の `elsepa/z{22,47}_q-1/pass{1,2}/<float.hex()>` と `direct/<float.hex()>` を `case()` で読み、欠けた点だけ実行する。

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p9a.py -q
.venv/bin/python -m prototypes.mv_electron.p9a protect
.venv/bin/python -m prototypes.mv_electron.p9a photon
.venv/bin/python -m prototypes.mv_electron.p9a pegs
.venv/bin/python -m prototypes.mv_electron.p9a elsepa
.venv/bin/python -m prototypes.mv_electron.p9a bundle
.venv/bin/python -m prototypes.mv_electron.p9a check-data
.venv/bin/python -m prototypes.mv_electron.p9a check
```

`check` は `gpu.p9a local` と同じ CPU 部品検査・旧4材料回帰を実行する。別途実行するなら次のコマンド（同じ検査なので重複実行は不要）。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9a local
```

PEGS 診断は `P8A_SEED_AGTI` と検査用の仮 I を使う。正式な definition/constants/bundle の I は META の IEV から取得する。積層 EGS5 の `P9A_SEED` の入力には従来どおり I を入れない。

各段階の成果物は P9a の stage JSON に SHA256 を固定する。再実行は照合して戻る。失敗したデータを削除して許容値を広げる手順はない。`check-data` が失敗したらそこで止める。

## T4 部品検査（シード材だけの 2 cm の板、2000履歴）

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9a export \
  --work ~/.cache/chatcarlo/p9a/export_slab --layout slab \
  --phase component --n 2000 --seed 2026102102
```

export のフォルダ全体を Colab に渡す。`source.zip` はリポジトリの構造を含むので、Colab の作業ディレクトリに展開し、そのディレクトリからモジュールを実行する。CUDA の核・ヘッダーは P8g-2 の合格時の fingerprint と照合する。

```bash
unzip -o /content/export_slab/source.zip -d /content/p9a_code
cd /content/p9a_code
python -m prototypes.mv_electron.gpu.p9a_colab \
  /content/export_slab /content/p9a_results/slab
```

開始時に同じ2000履歴の CPU/GPU 比較と集計の保存則・異常・スタック容量を検査し、その結果を保存する。component は追加の輸送を行わず、その2000履歴の集計を封印する。`complete.json` とその参照する call/session の全ファイルを回収する。旧6層の配置も `--layout old --phase component --n 2000 --seed 2026102102` で受け付ける。

## 積層の GPU 試し計算と再開検査

板の部品検査の合格を確認した後に進む。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9a export \
  --work ~/.cache/chatcarlo/p9a/export_pilot --layout stack
```

Colab で export_pilot の source.zip を展開してから次を実行する。GPU は T4、40×256、試し計算は200000履歴・seed 920102。毎回開始時に2000履歴の CPU/GPU 比較を行う。

```bash
python -m prototypes.mv_electron.gpu.p9a_colab \
  /content/export_pilot /content/p9a_results/pilot
# 同じ出力先で再実行し、封印済みの最初の呼び出しを再計算して再開を照合する。
python -m prototypes.mv_electron.gpu.p9a_colab \
  /content/export_pilot /content/p9a_results/pilot
```

初回の `trial.json` は `restart-required`、2回目は照合・故障検出を通過した場合 `pass`。2回目は新しい session ID を保存し、既存の呼び出しを再保存しない。異なる VM で再開する場合も、output の全ファイルを先に渡す。`trial.json` には µs/履歴も記録する。

実行中は既存の運用どおり Colab CLI を timeout 付きで呼び、10分ごとにポーリングして完成済みの call/session を回収する。完了マーカーは `P9A_DONE`。呼び出しがタイムアウトしても回収してから `colab stop` する。回収先を `~/.cache/chatcarlo/p9a/gpu_pilot` とする。

## 履歴数の決定 → GPU 本計算 → 判定

EGS5 の本計算と GPU の試し計算・再開検査の完了を待つ。EGS5 の既存コマンドは変更していないので、実行中の手順を継続する。

```bash
.venv/bin/python -m prototypes.mv_electron.p9a allocate-gpu \
  --pilot-root ~/.cache/chatcarlo/p9a/gpu_pilot
```

`docs/validation/mv/p9a/gpu_projection.json` の `histories` を以下の `N_GPU` に入れる。`status` が `ready` の場合だけ進む。最大2.4億履歴、20万履歴の倍数。EGS5 本計算の SEM と GPU 試し計算の SEM を、同じ EGS5 試し計算の Dref で規格化する。全19ブロックの予測 SEM を記録する。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9a export \
  --work ~/.cache/chatcarlo/p9a/export_production --layout stack \
  --phase production --n N_GPU --seed 920101
```

export_production を Colab に渡して source.zip を展開する。試し計算の全 output も渡す。

```bash
python -m prototypes.mv_electron.gpu.p9a_colab \
  /content/export_production /content/p9a_results/production \
  --trial /content/p9a_results/pilot
```

回収した本計算の output 全体を `~/.cache/chatcarlo/p9a/gpu_production` に置く。判定コードは Claude がレビューした後に実行する。

```bash
.venv/bin/python -m prototypes.mv_electron.p9a judge \
  --pilot-root ~/.cache/chatcarlo/p9a/gpu_pilot \
  --gpu-root ~/.cache/chatcarlo/p9a/gpu_production
.venv/bin/python -m prototypes.mv_electron.p9a protect
```

`judgement.json` は、EGS5 と GPU の封印・履歴数・seed・配置・全 bin・材料・核・スペクトルを照合してから作る。GPU の生の沈着量を同じバッチ番号で全呼び出しにわたって合算し、バッチの総履歴数と面密度で割る。相関を保った100バッチを `judge_batches()` に渡す。0.05 cm の平均・差は同じ履歴の並行集計から報告する。

## ファイル

- `p9a_data.py`: 保護、光子、PEGS/ESTAR報告、ELSEPA/GS、束、データ検査。
- `gpu/p9a.py`: CPU ビルド・部品検査・回帰・export。
- `gpu/p9a_spec.py`: Colab と共有する形状スキーマ。
- `gpu/p9a_store.py`: 可変形状の call/session の保存・照合・再開。
- `gpu/p9a_colab.py`: T4 起動時比較・試し計算・故障検出・本計算。
- `p9a_judge.py`: 生集計の変換、逐次の履歴数決定、入力照合、判定。

既存の `p9a.py` は CLI だけ、`tests/test_p9a.py` はオフラインテストだけを追加。実データでの実行記録（RESULTS・STATUS）は実行担当が記入する。
