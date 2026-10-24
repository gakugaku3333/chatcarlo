# P9b の実行と引き継ぎ

計画: [approved plan](../../../ai/plans/2026-10-24-mv-p9b-3d-geometry.md)。
`p9b_cli.py` は作成していない（Claude の担当）。Colab 内の実行本体は
`gpu/p9b_colab.py`。1 回の起動で G3 → G4 → G5 → G1_x → G1_y → G2 を実行し、
全ジョブの保存後、標準出力の最後に `P9B_DONE` を書く。失敗時には書かず、例外で停止する。
G3〜G5 が不合格なら G1/G2 のディレクトリも作らない。既存出力への上書きは拒否する。

## 手元

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_p9b.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m prototypes.mv_electron.gpu.p9b local
.venv/bin/python -m prototypes.mv_electron.gpu.p9b export
```

既定の作業先はリポジトリ内の `runs/p9b/`（Git 対象外）。材料・スペクトル・封印済み結果は
読み取りのみ。`local.json` が G0/G3 CPU の結果、`export/` が転送用の束。
export は G0/G3 を再実行し、3 条件各 2,000 履歴の CPU 参照、G3 の固定方向版と実乱数版の
参照、全材料、スペクトル、入力・ソースのハッシュを格納する。
P7 と共用する C++ 見出しの依存をすべて同梱し、Python の物理モジュールの import に必要な
既存の P1 データ 3 ファイルも source.zip に含める。Mohan の値をソースへ転記しない。

`~/.cache` に配置したい場合の操作は Claude が行う（本実装では書き込んでいない）:

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9b export --work ~/.cache/chatcarlo/p9b
```

export は既存の束の上書きを拒否する。別の空の作業先を指定すること。

## Colab T4（Claude の CLI 駆動で行う操作）

`export/` 全体を `/content/p9b_input/` に送り、`source.zip` を
`/content/p9b_source/` に展開する。`manifest.json` の全ファイルが必要。
既存の CuPy と CUDA の組合せを使い、別 CUDA 版の CuPy を追加しない。
Python は numpy/scipy/numba を使用する。

`/content/p9b_source/` をカレントディレクトリにして、次の 1 コマンドを起動する:

```bash
python -u -m prototypes.mv_electron.gpu.p9b_colab \
  /content/p9b_input /content/p9b_results > /content/p9b.log 2>&1
```

CLI 駆動側でバックグラウンド起動・各呼出しの時間制限・10 分ごとの監視と回収・最後の
`colab stop` を実装すること。完了はプロセス名ではなくログの `P9B_DONE` で判断する。
失敗時もログと出力を回収して VM を止める。

- G3: CPU と同じ 161 境界ケース、7 拒否ケース。23 成分の入力を直接置き、
  面に合わせた処理直前・処理後の状態、距離、次の領域、物性の材料、実際の再計算回数、
  RNG 呼出し数を返す。固定方向・実乱数とも CPU 参照と比較する。
- G4: z 積層、x 積層、水中シード各 2,000 履歴。`p8f_core.compare_histories` をそのまま使用。
  深さ・横方向の採点の一致率と最大差は報告のみ（合否には使わない）。
  深さの採点は世界座標の選択軸。横方向の採点は、線源の発散と独立した旗で全 3 条件に有効。
- G5: シード条件 2,000 履歴で、100×3,420 集計を履歴の和と比較。
  P7 c3 と同じ `(実加算回数 + バッチ内履歴数) * 2^-52 * 絶対値和` の上限。
  GPU の加算カウンターは必ずグローバルメモリー。
  付与の箇所ごとの個数・エネルギー、全水平板の採点、収支を照合。
  到達しにくい箇所は begin を通さない固定状態から本番輸送を呼ぶ。
  シード内の既知の点・線分は P7 の 7 区間へ各 1 MeV、深さは包む水層の bin 50 へ 1 MeV と完全一致。
- G1: +x / +y 各 13,600,000 履歴。種 930101 / 930102。
- G2: 30 cm 水、5×5 cm²、SSD 100 cm、20,000,000 履歴。種 930103。
  G1/G2 は各呼出し 200,000 履歴で、対応する 100 バッチを後で加算する。
  横方向の外枠は ±1,000,000 cm。横からの脱出数も保存。

`job.json` は完了後のみ作成し、出力ファイルの SHA256 を持つ。
GPU 未実行時点では P9b 全体の合格とはしない。

## 結果回収と判定

`/content/p9b_results/` を `runs/p9b/results/` に回収（`~/.cache` を使った場合はその作業先の
`results/`）。ログも保管して次を実行する:

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p9b analyse
# cache 配置なら:
.venv/bin/python -m prototypes.mv_electron.gpu.p9b analyse --work ~/.cache/chatcarlo/p9b
```

G1 は `p9a_judge.normalize_calls` → 既存 `judge_batches`。`judge`/`load_gpu` は呼ばない。
19 ブロックすべての pass を両方向で要求する。
G2 は `p7_core.unpack_regions` → `analysis_p7.dose/compare`。
封印済み P7 EGS5 条件 A を読み取り検証し、`pilot_lock_A.json` の固定 Dref を使用する。
J1〜J3 の全項目に加え、symmetry の状態「偏りは検出されず」を必須にする。
`judgement.json` を得た後、[RESULTS.md](RESULTS.md) に実際の数値と判定を記録する。
不合格・統計不足・未実行を合格へ書き換えない。
