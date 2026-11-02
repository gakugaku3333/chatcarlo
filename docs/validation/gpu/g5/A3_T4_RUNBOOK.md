# A3 H2 / H5 / H6 の T4 起動手順

この手順は approved 計画 `docs/ai/plans/2026-10-31-chatcarlo-gpu-g5-a3-entry.md`
の残りの **H2 T4 部分、H5、H6** を実行する。H3/H4 の本計算とその承認は別であり、
この起動の成功だけでは A3 完了にしない。実機未実行。判定の閾値は変更していない。

## 前提と入力

リポジトリのルートで `.venv/bin/python` を使用する。Colab CLI の認証と T4 の利用枠が
必要。VM 側には CuPy と NumPy/SciPy/PyYAML/Numba が必要。
VM には xraylib/spekpy が無く、pip もしない前提。診断の参照計算・表の生成・判定は
xraylib/spekpy を備えたホスト側で行い、VM は書き出し済みの表とジョブを読む。既存の CuPy CUDA 13 環境に CUDA 12 用 CuPy を重ねて入れない。
依存が不足したら停止して環境を整える。起動役はインストールしない。

入力はすべて既存の封印済みディレクトリを **読むだけ**。

| `prepare` 引数 | 内容 / 既存の候補 |
|---|---|
| `--therapy-bundle` | A3 外部束 `~/.cache/chatcarlo/g5/a3/therapy_bundle` |
| `--p7` | P7 export `~/.cache/chatcarlo/p7/work/export` |
| `--p8g2` | P8g2 export `~/.cache/chatcarlo/p8g2/export` |
| `--old-diagnostic` | A1 の旧診断束。直下に `g2b/`, `g4/`。候補 `~/.cache/chatcarlo/g5/a1/old_bundles`（19c281e 時点で封印した比較相手） |
| `--g2-reference` | G2 固定参照 `~/.cache/chatcarlo/g5/a0/g2_ref` |

旧入口の既存入力（P5/ELSEPA/Mohan10）と P3b-3 の EGS5 比較入力も読み出せること。
欠落・アーカイブ状態なら停止する。この起動役は `~/.cache` を展開・作成・更新・削除しない。
スペクトルの数値をリポジトリに置かない。旧診断束は manifest の全 SHA256 と旧新の
シーン条件を照合する。治療束は製品の固定ハッシュで照合する。

出力先はリポジトリと `~/.cache` の **外** に置く。例は `/private/tmp/g5-a3-t4-20261101`。
同名の出力や `STARTED.json` が残る場所は使い回せない。
3 GiB の空き下限を維持する。準備前に4組のプレビュー束を見積もる。
`prepare` は H5 の G1/G2 ジョブと表を `H5_jobs/`、小さい診断新束と
`water_A/B/C.npz` を `generated/` へ書き出す。VM 作成前にすべて揃える。
`payload()` は入力を複製せず、元のパスから arcname を付けて tar に入れる。
`payload/` には `diagnostic/new` と `water_*.npz` だけを置く。
`payload/` の mkdir 前に、全メンバーの PAX ヘッダー・パディングを含む tar 実サイズ
＋小さい生成物の staging 分＋結果用2 GiBで reserve する。容量不足なら `payload/` と
`payload.tar` は残らない。送付断片はローカルの `upload_part` 1個を上書きし、
遠隔では従来どおり `upload_0000` から連番にする（断片は結果用の余裕内）。
大きな入力では十分な空きのある外部領域を使う。
この見積もりで停止したら空きのある出力先へ移し、途中の証拠を勝手に消さない。

## 1. ローカル準備（T4 を作らない）

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/validation/gpu/g5/a3_t4_gates.py prepare \
  --output /private/tmp/g5-a3-t4-20261101 \
  --therapy-bundle ~/.cache/chatcarlo/g5/a3/therapy_bundle \
  --p7 ~/.cache/chatcarlo/p7/work/export \
  --p8g2 ~/.cache/chatcarlo/p8g2/export \
  --old-diagnostic ~/.cache/chatcarlo/g5/a1/old_bundles \
  --g2-reference ~/.cache/chatcarlo/g5/a0/g2_ref
```

生成する表示をブラウザーで確認する。

- `entry/old/cold/preview.html`, `entry/old/cached/preview.html`
- `entry/new/cold/preview.html`, `entry/new/cached/preview.html`
- `g4/preview.html`

入口は全て Mohan10、水30 cm、10×10 cm²、SSD100 cm、4,600,000 履歴、種20261101。
旧入口は P6、新入口は承認済み方針の P7 ±20 cm。cold/cached も別々の生成・承認記録を持つ。
G4 は既存の `scene_portrait_detector.yaml`、種2040、CLI既定100,000履歴。

**使用者が5表示を確認し承認した後だけ**、以下で承認を記録する。この操作を起動役は
自動実行しない。`run` は全承認を検証してから VM を作る。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from pathlib import Path
import os, sys
sys.path.insert(0, 'docs/validation/gpu/g5')
from a3_t4_gates import no_cache_writes
sys.addaudithook(no_cache_writes)
os.environ['MPLCONFIGDIR'] = '/private/tmp/a3-approval-mpl'
os.environ['NUMBA_CACHE_DIR'] = '/private/tmp/a3-approval-numba'
from prototypes.mv_electron.entry import cli
from chatcarlo.gpu import contracts, therapy_entry, approval
root = Path('/private/tmp/g5-a3-t4-20261101')
for phase in ('cold', 'cached'):
    assert cli.main(['approve', str(root/f'old_{phase}.yaml'),
                     '--preview', str(root/'entry/old'/phase)]) == 0
    contracts.approve(root/'entry/new'/phase, therapy_entry.verify)
approval.approve(root/'g4')
PY
```

この改修前の準備結果には `H5_jobs/` と `generated/` が無いため、新しい出力先で
`prepare` と表示の承認をやり直す。
準備と起動の間に本体のソースを変えると条件が変わる。起動は拒否する。
既存の H3/H4 の承認・結果をこの速度試験へ流用しない。

## 2. 全関門を1回のコマンドで起動

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/validation/gpu/g5/a3_t4_gates.py run \
  --output /private/tmp/g5-a3-t4-20261101
```

このコマンドだけが T4 を1台作成する。最後まで同一セッションを使用し、子の旧新入口や
G4が要求する new/stop は既存VMへの貸出として処理する。終了・例外時は所有側が stop する。
全 Colab 呼出しにタイムアウトを付ける。長い検査は nohup worker で動かし、60秒ごとに
ログと完了済みファイルを回収する。失敗なら次段へ進まない。転送・環境失敗後も、
有効な速度測定を追加実行しない。`STARTED.json` は payload 完成後、VM 作成の直前に
書き、再起動を拒否する。

所有側 `Owner.call` は upload/download と明示した冪等な状態確認 exec のみ、
`TimeoutExpired` / `CalledProcessError` に対して60秒間隔・最大6試行とする。
stop は別途同じ60秒間隔・最大6試行。new、Popen を含む exec、ワーカー起動、
セットアップの exec は再試行しない。旧新入口は既存の状態確認テンプレートだけを
再試行対象にする。G4 は凍結した起動役の外側の試し直しが重ならないよう、
Owner の試行を使い切ったら停止する。各試行の番号・所要時間・例外型・retry を
`transport_timings.json` に保存する。

| 順序 | 実施内容 | 保存先 |
|---|---|---|
| H2 | 最終 T4 検査ビルドの CPU/GPU 線源・RNG、2048対256×8、固定付与採点。実際の native 引数19（field half）改変、native ID/tag のみの改変を検出。A2 のパス・健全性故障も最終 T4 で再実行 | `H2/entry_faults.json`, `checks.json`, `cases_t4.json`, `faults_t4.json` |
| H5 統計 | VM は `a1_speed.worker_batches` でホスト書き出しジョブの GPU バッチだけを生成。回収後、ホストで `a1_gate1.run(device='cuda', remote=...)` によるG1の51量、`a1_g2vox.run(..., remote=...)` による固定G2参照を再判定。既存の種・履歴数・許容差・限定再試行規則を使用 | `H5/gate1/`, `H5/g2vox/`, `H5/g1.json`, `H5/g2.json`, `H5/tmp/` と計算表 |
| H5 入口 | G4の実際の CLI preview→approve→Colab run→受領・出力まで | `g4/` |
| H6 核 | `a1_speed` の診断 g2b（500k/100k）・g4、`a2_speed.build_new` と A2 の測定順による p7A/B/C・stack。ウォームアップ後、有効1回。旧と最終コードを交互に同一T4で測定 | `H6/diagnostic/measurements.json`, `H6/speed.json`, `H6/verdict.json` |
| H6 直接比較 | 同じセッション・種・履歴範囲で旧新各100外側バッチ。P7 の dose/compare、stack_dose/judge は `a2_oldnew.py` と同じ既存関数。これは報告用でEGS5統計関門を置き換えない | `H6/oldnew_*.npz`, `H6/oldnew.json` |
| H6 入口 | old cold → new cold → old cached → new cached。それぞれ4.6M全履歴を再計算 | `entry_speed.json`, 各 `entry/*/*/results/` |

H5 は書き出し済みの初回・再試行の両 seed 組をVMで生成する。再試行組を判定へ
採用するかはホストの既存判定だけが決める。NumPy参照計算をVMへ送らない。
ホストの参照計算・判定表・一時束はすべて `--output` 以下、固定G2参照と `~/.cache` は読むだけ。

A1/A2の既存ファイルは編集しない。`a2_t4.water` で入力を書き出し、A2検査駆動の `T4` と速度構築の `a2_speed.NewCore` を
実行中だけ最終製品クラスへ差し替える。
`a1_speed.install_runtimes` は最終本体を古い束で上書きするため呼ばず、旧パッケージだけ
VMの別名 `chatcarlo_g5_old` へ展開する。
`a2_oldnew.py` は import 時に旧記録へ書き出すので実行/importせず、その比較関数を再利用する。

## 入口全体の計時とキャッシュ

`a3_entry_speed.cli_measure` は旧 CLI `main(['run',...])` **全体の前後**を計時する。
メモリー上だけで `colab_cli.run` を包み client を注入し、凍結ファイルは変えない。
送付・VM起動待ち・60秒間隔の確認・回収・受領照合・解析を含む。新側も CLI main 全体を
計時する。通信の再試行と60秒の待ちも全体の時間に含む。
各計時行に `transport_retried` と `transport_retry_attempts` を残し、試行明細を
`transport` に保存する。再試行を理由に速度測定をやり直さず、比の基準も変えない。
新 CLI 自体には EGS5解析がないため、同じ P3b-3 定義の `a3_judge.judge10`
による解析時間を別記して全体へ加える。速度用4.6Mを H3の46M判定と取り違えない。

毎回、手元もVMも別の結果保存先。バッチ再利用はしない。旧新ごとに専用の
`/content/<session>/compile_cache/old|new` を固定する。2回目へ引き継ぐのは
CuPyコンパイルキャッシュだけであり、結果・関門記録・生バッチを移さない。

各VM workerの `sitecustomize.py`（検証側だけ）で `RawKernel.compile` の時間と
NVRTC呼出しを採取する。coldで実際のソースコンパイルを観測し、cachedでその呼出しがなく、
2回目のコンパイル時間を得られ、coldの `.cubin` 群がcached前後でハッシュ一致したときだけ
キャッシュ利用を確認済みにする。計測器が使えない、証拠が欠ける、再コンパイルした等は
**未判定（undetermined）**。時間が短い・ファイルが存在するだけでは通さない。

各速度比は **new / old ≤ 1.0571**。入口はcold/cached双方の全体比を保存する。
キャッシュ証拠を確認できない場合、比が小さくても H6入口は未判定、A3完了にしない。
判定の緩和や測定の追加をしない。`summary.json` の pass は今回要求された関門に限る。
H3/H4等の完了を含まない。CLI終了値は pass=0、未判定=2、例外・不合格=非0。

## 証拠・失敗時・保持

`prepared.json`, `STARTED.json`, `source_hashes.json`, `transport_timings.json`,
`summary.json` と上記JSON/NPZ/ログを保持する。`source_hashes.json` は送った最終ソース、
`prepared.json` は承認時のGPU本体・CLI・シーンのハッシュを含む。
結果・私有束・payload・送付断片は外部の出力先に残す。自動削除やcommitはしない。
`~/.cache` へのホスト側の書込み・削除は audit guard で拒否し、Numba/Matplotlibキャッシュと
Pythonバイトコードも外部出力へ向ける／停止する。

失敗後は `summary.json`, 各 `worker.log`, `failure.json`, 入口の回収済み結果を見る。
Colab通信自体が失われると最終回収やstopも失敗し得る。その場合は `STARTED.json` の
セッション名を使ってColabの状態を確認し停止する。すでに測った値を都合よく採り直さない。

小さい結果表・判定JSONをリポジトリへ採用する作業、重い生データのDrive保全と整理は
実行後に別途行う。私有スペクトルを含む束やtarをGitへ追加しない。

## ローカル単体検査

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/a3-mpl NUMBA_CACHE_DIR=/private/tmp/a3-numba \
  .venv/bin/python -m pytest tests/test_gpu_g5_a3_gates.py -q -p no:cacheprovider
```

このテストは偽clientで関門順・失敗時停止・main全体へのclient注入・60秒・保存先分離・
キャッシュ証拠・比の境界・書込み禁止に加え、通信例外2種の再試行範囲・回数・証跡、
入力を複製しないtar、実サイズの見積もり、容量不足時のpayload未作成、断片の固定名、
ホストのremote判定を検査する。新規Pythonプロセスで xraylib/spekpy を遮断し、
h2/h5/speed・治療速度・直接比較・旧新入口VMモジュールの経路も検査する
（GPUと私有入力を要する境界は代替する）。CuPy/T4/NVRTC・実物の転送速度・物理の
正しさは確認しない。本計画§2e.8の判定コードのCodexレビューは今回の実装時に実施。
独立したClaudeの検証と実機による判定は未実施。

## 途中からの再実行（2026-11-02 追加）

- `run --remaining`: H2・H5・核の速さの記録がある実行の後に、旧新比較と入口の速さだけ回す。
- `run --entry-only`: 入口の速さの 4 回だけを、新しい 1 セッションで回す。比較は同じセッションの中でしか成り立たないので、途中で切れたら 4 回ともやり直す。
- どちらも省いた関門の結果は引き継がない。`summary.json` の `scope` を見て、省いた分は前の実行を記録に書く。本体のハッシュ（`prepared.json` の `sources`）が前の実行と同じことを確かめる。
- VM の tar の組み立ては裏で走り、状態を確認で読む（`colab exec` は約 60 秒で打ち切られる）。
- 出力先に置いた `payload.tar` は、全ファイルの名前と SHA256 が一致するときだけ使い、違えば止まる。追記した後の版を優先して照合する。容量の確認は通らないので、自分で空きを確かめる。
- `cache_used` の NVRTC の記録は計装が CuPy 14.0.1 の実経路から外れていて取れない。`undetermined` のときは 2 回目のコンパイル時間とキャッシュのファイル（計画 §2e.3）で補足判定する。
