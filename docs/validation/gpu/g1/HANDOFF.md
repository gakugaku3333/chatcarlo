# G1 の引き継ぎ

実装は承認済み G1 の範囲。関門 1・2 と実際の Colab は未実行で、Claude が実行する。関門 0 の結果と環境は `RESULTS.md` / `environment.json`、NumPy の基準は `baseline_manifest.json` と `baseline_s1.npz` / `baseline_s2.npz`。基準は実装前に指定コミット `636b2cffd3c8839e7d0f50b73f5496289d6db18a` の `chatcarlo/` を一時フォルダへ `git archive` で展開して作った。

全コマンドはリポジトリの最上位で実行する。作業フォルダは任意の一時フォルダにでき、`~/.cache` は使わない。

```bash
export XDG_CACHE_HOME=/private/tmp/chatcarlo-g1-cache
export NUMBA_CACHE_DIR=/private/tmp/chatcarlo-g1-numba
export MPLCONFIGDIR=/private/tmp/chatcarlo-g1-mpl
export PYTHONPYCACHEPREFIX=/private/tmp/chatcarlo-g1-pycache
export PYTEST_ADDOPTS='-o cache_dir=/private/tmp/chatcarlo-g1-pytest-cache'
export PYTHONPATH=.
G1_WORK=$(mktemp -d /private/tmp/chatcarlo-g1.XXXXXX)
```

## 関門 0 の再確認

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -q -k gpu
.venv/bin/python -m chatcarlo validate examples/chest_room.yaml
```

`test_gpu_g1_functions.py` は各物理関数に 10⁵ 件と全元素の格子・吸収端の隣接値を与え、戻り値・作業配列・MT の 624 語と index・後続 16 個をバイト比較する。`test_gpu_g1_history.py` は指定の 6 ケース各 10⁴ 履歴の鎖を比較する。許容差への置換はしない。`test_gpu_g1_geometry.py` は指定の線分 10⁵ 本と端を、傾斜面も含めて比較する。

## 関門 1（Claude、手元）

```bash
.venv/bin/python docs/validation/gpu/g1/run_gates.py gate1 --out "$G1_WORK/gate1"
```

各シーン・各側で 10 本 × 10⁶ 履歴、batch_size 2×10⁵。NumPy は seed 5000+i・workers 4、双子は輸送 6000+i・線源 7000+i・workers 1。ROI・判定・一度だけの再試行（5100+i / 6100+i / 7100+i）は計画の固定値。全 3 シーンをまとめた K と Σz²/K、各量の平均・SEM・z・検出できる差の目安を `gate1.json` に書く。途中の各本は `--out` に保存して再開でき、シーン・ソース・束が変わった結果の再利用は拒否する。

S3 は依頼 D のコミット済み YAML を参照し、`run_gates.py` の SHA256 と照合する。シーンや ROI を変更して通し直さない。関門 1 が合格するまで Colab に進まない。

## 関門 2（Claude、Colab T4）

```bash
.venv/bin/python docs/validation/gpu/g1/run_gates.py export --out "$G1_WORK/bundles"
.venv/bin/python -m chatcarlo.gpu.colab_driver launch \
  --bundle "$G1_WORK/bundles/s2" --job "$G1_WORK/bundles/s2_gate2_job.json" \
  --out "$G1_WORK/gate2/s2" --gate1 docs/validation/gpu/g1/gate1.json --session g1-s2
.venv/bin/python -m chatcarlo.gpu.colab_driver launch \
  --bundle "$G1_WORK/bundles/s3" --job "$G1_WORK/bundles/s3_gate2_job.json" \
  --out "$G1_WORK/gate2/s3" --gate1 docs/validation/gpu/g1/gate1.json --session g1-s3
.venv/bin/python docs/validation/gpu/g1/run_gates.py gate2-merge --out "$G1_WORK/gate2"
```

起動役は 50 MB 分割、各片と全体の SHA256 照合、VM 上の目録照合後に `nohup` で起動する。VM には NumPy・CuPy と C++ コンパイラが必要。Colab に同梱の CuPy を使い、別 CUDA 世代の CuPy はインストールしない。xraylib・SciPy・SpekPy は不要。CPU の双子も GPU もその VM 上の同じ線源配列を使う。

関門 2a は関門 0 と同じ固定入力を Philox で回し、整数・乱数消費数を完全比較、浮動小数の一致率を記録する。2b は 2,048 履歴、分岐履歴の最初の 5 件の異なる歩について、歩・位置・材料・消費数・エネルギー・光学的厚みを再実行で記録する。2c は 10 バッチ × 10⁶ 履歴、輸送 seed 8000・線源 seed 8001。装置での集計は各材料に履歴ごとに 1 回加算する。区分別の画素ごとの検出件数も装置で数え、画像・ROI の実際の加算回数を得る（画像を合計する加算も含む）。GPU−CPU の符号つき寄与について、両側の加算回数の和 × 2⁻⁵² × 両側の寄与の絶対値の和を丸め上限とし、上限外では対応するバッチ差の SEM を使う。別途 10⁵ 履歴を記録して集め直し、装置の集計と照合する。

10 分ごとに確認・回収する。本体と識別情報つき受領票の両方を照合してからローカルへ確定する。`.pending` は回収しない。セッションが消えたら停止し、回収済みを残す。同じコマンドで再開すると照合済みバッチを再送し、その線源抽選だけを進めて輸送を省く。終了時・失敗時とも `colab stop` を試みる。

`gate2.json` の合否と、CPU/GPU の一致率・各量の SEM・速さを別々に読む。関門 2 が不合格なら依頼 D は GPU で回さない。

## 依頼 D の画像

手元で CPU の双子を小さく確認する方法（テストもこの 10⁴ 履歴を使う）:

```bash
.venv/bin/python -m chatcarlo.gpu.run_detector_image \
  --n 1e4 --seed 101 --workers 1 --tag A --out "$G1_WORK/image" --device cpu
```

`--device cuda` は関門 2 の合格記録を要求する。API の装置引数は設計 §1-2 の `gpu_device='cpu'/'cuda'`。受入条件の例にある `device='cpu'` は採用していない。CLI 本体は変更していない。

Colab での依頼 D は、**本計算前に依頼 D の事前記録へエンジン変更を記録してから**、エクスポート済み S3 の束で次の job を作る。`N`、輸送 seed、tag は依頼 D の固定値を使い、線源 seed は別に記録する。省略時の API と同じ規則は `int(np.random.SeedSequence(seed).generate_state(1,np.uint64)[0])`。

```json
{"mode":"batches","device":"cuda","n_histories":10000,"batch_size":1000,"seed":2000,"source_seed":12345}
```

上の `source_seed` は形式例であり、本計算では採用した固定値に置き換えて事前記録する。`batch_size=min(200000,ceil(N/10))`。起動方法は関門 2 と同じで、`--gate2 docs/validation/gpu/g1/gate2.json` も渡す。回収後はホストで、受領票を再照合し、既存の `DetectorTally` にバッチ順で足して依頼 D と同じ形式へ保存する:

```python
import json
from chatcarlo.gpu.run_detector_image import collect_image
job = json.load(open("job.json"))
collect_image("bundles/s3", job, "collected_batches", "images", tag="A")
```

出力 `N000010000_A.npz` は元ランナーと同じ 10 キー・配列形状。JSON には engine・装置・輸送 seed・線源 seed・実効 worker 数・束の目録を追加する。回収後の集計では実行時間を `null` とする（各バッチの実測時間からは復元できないため）。
