# 計画: G5 A4u — H6（T4 の 1 セッション）の起動役

- 状態: implemented（手元まで。T4 は未実施）／approved（**ユーザー承認 2026-11-06**。群 A/B/C の三分割、群 B で実機の穴 3 つを塞ぐ、報告専用を除く関門が fail か未実施なら H6 を合格にしない、時間の見込みは撤回して最初の 1 本で測り直す）
- Codexレビュー: 1巡（骨子、2026-11-06。致命 0・実質 5・軽微 2・参考 2。原文 `data/logs/codex_a4u_h6_review1.md`。実質 5 件と軽微 2 件を反映済み）
- 作成者: claude
- 実装担当: codex（起動役 `a4u_h6.py` と VM 側の仕事の一覧）。T4 の運転・採点・記録は Claude
- 日付: 2026-11-06
- 親計画: [2026-11-03-...-a4u-unified-physics-set.md](2026-11-03-chatcarlo-gpu-g5-a4u-unified-physics-set.md) の H6（行 111）
- 規模: **中**（骨子レビュー 1 巡。新規の数値判定は作らず、既存の判定器を順に呼ぶ起動役）

## 目的

**答える問い**: 統合した物理の組（`chatcarlo-photon-electron-v1`）を **NVRTC で実機にコンパイルしても、
手元の CPU 双子と同じ経路を通り、診断側の既存の関門が落ちないか。**

H6 は「物理が正しいか」を問う段ではない（それは H3a・H4・H5・中心軸 PDD が EGS5 で済ませた）。
**手元では踏めない経路＝NVRTC のコンパイル・実機のレジスター割り当て・実機の速さ**だけを問う。

## 0. 前提事実（検証済み、2026-11-06）

いずれもコマンドで確かめた。推測は「未測定」と書いた。

| # | 事実 | 確かめ方 |
|---|---|---|
| F1 | **MV（10 MV・4 MV）と `seed_ratio` は H6 の対象外**（ユーザー決定 2026-11-06）。MV は変更分の上限が水 500 keV で 0.26%（許容 1%）で A3 の合格を引き継ぐ。`seed_ratio` の封印済み参照は EGS5 でなく**旧 P9c 試作**（`a2b_run/reference.json` の `runs[].identity.sources` が `p9c_*.h`・`p5_*.h`、`physics_set` の欄なし）で、同一物理の移植試験だったため物理を変えた後に当てても情報が出ない | `A4U.md`「H6 を縮小した」、`reference.json` を読んだ |
| F2 | **試験専用の旗は両方の特殊化にある**: 診断側 `-DG5_TEST_LEGACY_LOCAL`（`core.py:58`、引数 `_test_legacy_physics`）、治療側 `-DG5_TEST_LEGACY_ELECTRON`（`therapy.py:102`・`therapy_cuda.py:16`・`seed_core.py:23`・`seed_cuda.py:16`）。**D10 (1) は実装できる** | `grep` |
| F3 | `physics_sets.declaration()`（引数なし）は**統合後の組の `local-deposition`** を返す（`physics_sets.py:27`、`specialization = LOCAL if name is None`）。`a1_g2gate3.verify_inputs` の `physics_sets.check(...,declaration())` は**新しい物理で通る** | `physics_sets.py` と `a1_g2gate3.py:28` を読んだ |
| F4 | D8 (3) の条件は `a1_g2gate3.CONDITIONS = {'pdd':(3101,5000000),'thin':(3102,500000),'phantom':(3103,500000)}`。モードは `prepare`/`execute`/`compare` で、`execute` は `--bundle` を取り旧 `compare.py`（`g2/run_egs5_60kev.py`）に委譲。**EGS5 は封印済みハッシュ 3 件を照合するだけで回さない** | `a1_g2gate3.py` を読んだ |
| F5 | G2 のボクセルの関門は `a1_g2vox.py` に **`--remote`/`--export` があり**、`a1_remote.export_job`/`collect` で VM に出して回収する形。既に T4 で使った経路 | `a1_g2vox.py:36,94,112`、`a1_remote.py` |
| F6 | **G4 に関門スクリプトは無い**。`docs/validation/gpu/g4/` にあるのは `scene_portrait_detector.yaml` と `local_baseline.json` と `colab_A_detector.png` だけで、関門は `chatcarlo run ... --engine gpu [--gpu-device colab] --seed 2040` の CLI を 2 回走らせて画像を比べる形 | `ls` と CLAUDE.md の例 |
| F7 | G1 の 51 量は `a4u_s1_gate1.py` が既にあるが **`gpu_device='cpu'`（CPU 双子）固定**（`a4u_s1_gate1.py:38`）。T4 で回すには device を渡せるようにする必要がある | 該当行を読んだ |
| F8 | T4 を 1 セッションで使い回す道具は `a3_entry_speed.py` の `SharedClient`・`POLL_SECONDS`・`SITE_CUSTOMIZE`・`cache_used`・`ratio`。`a3_t4_gates.py` が `prepare`（手元）→`run`（T4 を 1 つだけ持つ）の形で使っている | 両ファイルを読んだ |
| F9 | 束は既に書き出し済みで照合済み: `~/.cache/chatcarlo/g5/a3/therapy_bundle/manifest.json` の digest `9f54db5a...dc119`、`physics_set` は id 2・`electron-transport`・tag 6882153074982124678、4 ファイルの SHA が実体と一致 | 2026-11-05 に照合 |
| F10 | ディスクの空きは **5 GiB**（床は 3 GiB）。MV を外したので A3 式の 2 GiB 確保は不要 | `df -g /` |
| F11 | (c) 殻の注入の試験は `tests/test_gpu_g5_a4u_s2.py::test_shell_injection_ck_branches_energy_and_faults` に既にある。故障の仕掛けは `core.Core(..., _test_photon_fault=1\|2)`（`photon_physics.h:77`）で、新しい仕掛けは作らない | 親計画の表（行 235・236）と S2 のテスト |
| F12 | VM 1 本あたりの固定費（起動・コンパイル・通信）は**未測定**。A3 の 460 万の 1 本から固定費 27 秒・26.85 µs/履歴に分離できたのは MV の核だけで、診断側の固定費は別 | `A4U.md`「さらに訂正」 |

## 事前登録の判定ルール（計算の前に固定する）

**新しい数値基準は作らない。** 既存の判定器（`validation.paired_judge`・`gate2a`・`gate2`、G2 の
ボクセルの許容）を launcher から呼ぶ。Codex のレビュー 1 巡を受けて**仕事を 3 群に分けた**。

### 群 A — 同じ新しい物理・同じ束での CPU 双子 対 T4（既存の数値判定で合否）

参照を**旧モデル・旧参照に取らない**（指摘 2）。旧モデルとの比較はモデル差を CUDA の不良と誤認する。
`transport.py`（NumPy）との 51 量の比較は **H3a で済んでいる CPU 側の回帰**で、VM では回さない。

| # | 中身 | 判定 |
|---|---|---|
| A1 | 診断の 51 量。同じ束・同じ seed を CPU 双子と T4 に渡す（3 シーン） | **合否**: `validation.paired_judge(cpu,gpu,bounds)`（`validation.py:55`）。`independent_judge` は使わない |
| A2 | ボクセルの格子 | **合否**: G2 の決まり——ボクセルごとの相対許容＋総量の保存（丸めの差は最大 ~3e-9 相対。`.claude/rules/chatcarlo.md`）。`a1_g2vox.py` の凍結参照 `ref_*_batches.npz` には当てない |
| A3 | G4 の検出器画像（`g4/scene_portrait_detector.yaml`、seed 2040） | **合否**: `gate2a` の整数の照合（RNG の語数・格子番号・検出器の整数列）**と** `gate2` の対比較・記録からの再集計（`validation.py:191,234`）。**報告のみ**にするのは `per_history_exact`・`float_match`・`integer_field_match`（`validation.py:218`）だけ（指摘 3） |

A1〜A3 はいずれも「同じ入力を 2 つの装置に渡して対で比べる」形にする。**`local_baseline.json` や
A1 の旧結果との直接比較には使わない**（表のハッシュが変わると `inapplicable`、`validation.py:245,262`）。

### 群 B — 小さな実機経路の確認（到達の証拠。統計の判定ではない）

MV と `seed_ratio` を外したことで**実機でしか踏めない経路に 3 つ穴が空いた**（指摘 1）。
本番規模の履歴数も新しい EGS5 も使わずに塞ぐ。合否は「到達数が正」「完走」「収支」「出力が有限」。

| # | 穴 | 塞ぎ方 | 判定 |
|---|---|---|---|
| B1 | 治療の特殊化の**低エネルギー** | `therapy.py`/`therapy_cuda` を**直接呼ぶ**（入口は 20 keV を設計で通さず、`cpu_local` は 20 万で上限。途中にユーザーの 3D 承認が入るので CLI を使わない）。20 keV と 60 keV の両方を残す（指摘 3・5） | **合否**: Rayleigh・束縛の棄却・打ち切りの局所沈着の**到達数が 3 つとも正**、収支、完走。履歴数を増やすより到達を保証する小 fixture を使う |
| B2 | 治療の特殊化の**高エネルギー**（対生成・陽電子消滅・制動放射） | 本番の治療 build で既存の注入 fixture を `device='cuda'` で実行（指摘 1） | **合否**: 対象分岐の到達数が正、エネルギー収支 |
| B3 | **seed 専用の CUDA 核**（`seed_cuda.py:16` は `-DT2_SEED=1` の別モジュールで、独自の幾何・転送・集計。`13e4305` で `t2_seed_kernel.h:221` に Rayleigh が入った） | 本番の `seed_cuda` を**コンパイルし**、1 門・数千履歴の短い輸送を CPU 対 T4 で回す。**旧 P9c の比には当てない** | **合否**: コンパイル成功・完走・境界・集計の健全性。判定の再現は求めない |
| B4 | 殻の注入 (c) | **CUDA の `g5_relax_probe_kernel`** で殻を注入し、既存の分岐判定を当てる。`tests/.../test_shell_injection_...` は `device` 未指定の ctypes なので、VM で走らせても CPU 試験のまま（指摘 4） | **合否**: 放射・CK の分岐確率・遷移先・エネルギー予算 |
| B5 | 故障注入 | **輸送側は故障 3・4・5 と容量制限・二光子 fixture を `device='cuda'` で**（`tests/test_gpu_g5_a4u_s2.py:73`）。CK の 1・2 は B4 の relax probe で（`_test_photon_fault=1,2` は CK を壊す仕掛けで、一般の輸送健全性検査が拒否するとは限らない。指摘 4） | **合否**: 故障が検出されること |
| B6 | NVRTC の経路 | **コンパイルする module を数え上げる**: 診断の `core`（`G5_SET_ID=1`）、`therapy_cuda`（id 2）、`seed_cuda`（`T2_SEED=1`）、C2 の旗つき build 2 種 | **合否**: 5 つ全部が `backend='nvrtc'` で通る |

### 群 C — 報告と測定

| # | 中身 | 扱い |
|---|---|---|
| C1 | D8 (3) 60 keV の分類（`a1_g2gate3.py` の `pdd` 3101/500 万・`thin` 3102/50 万・`phantom` 3103/50 万） | **報告のみ**（スクリプト冒頭が "Reporting-only"）。A1 の 37・6・4 の隣に置く。`SEALED` と `CONDITIONS` は変えない。EGS5 は回さない |
| C2 | D10 (1) 旧物理の旗つき build 対 旧実装の回帰 | **合否: 比 ≤ 1.0571**。負荷・旧実装・回数を下に固定する |
| C3 | D10 (2) 物理を足した本番 build の速さ | **報告のみ。1.3 を超えたら停止して相談**（勝手に物理を削らない） |
| C4 | 時間 | **総時間と、直接測れるコンパイル・核・通信の時間を別々に記録する。「共通の固定費」とは呼ばない**（1 本の経過時間から分離できないし、治療核の測定を診断核・別 build に一般化できない。指摘 7） |

**C2 の固定（これが無いと受入条件を測れない）**: 1.0571 は p7A で**同じコードを繰り返し測った幅**
s = 2.86% から決めた `1+2s` なので、**s を測ったのと同じ負荷でしか当てられない**。
- 診断側: A1 の登録済みの負荷をそのまま使う——`old-g2b:500000`・`new-g2b:500000`・`old-g2b:100000`・
  `new-g2b:100000`・`old-g4:100000`・`new-g4:100000` を 3 つの入れ替えた順で（`a1_speed.py:21`）
- 治療側: A0 の p7A
- 旧実装の運び方: `a1_old_bundles.py`（凍結した 19c281e の木を `cwd` にして束を作る）と
  `a3_t4_gates.py --old-diagnostic`（`verify_old_diagnostic` でハッシュを照合）の既存の経路
- **同じ VM セッションの中で**測る。別セッションの数字と比べない

## 手元でやること（T4 の前）

| # | 中身 | 理由 |
|---|---|---|
| L1 | **H0 (4) を埋める**: Mohan 2 つの Rayleigh・束縛の寄与の割合を XCOM（`fetch_materials_p8a.prepare_xcom`、1〜10 MeV。xraylib は約 1 MeV で止まる）で計算する | 承認済みの親計画 D9 に登録されているのに `A4U_H0.md:65` は**やることとして書かれたまま空**。MV を外す根拠はいま水 500 keV の 1 点しかない。GPU は要らない |
| L2 | **A4U.md の言い過ぎを直す**: 「変更分の上限 0.26%」は**断面積に占める割合**で、線量差の数学的な上限ではない（指摘 8）。「`seed_ratio` は情報が出ない」も広すぎる——**旧参照を正解にした判定**には情報が無いが、新物理同士の CPU/T4 比較（B3）には情報がある（指摘 9） | 記録が残るので先に直す |
| L3 | **親計画を 2026-11-06 の決定で更新する**: D9、H6 の行（111）、§7 の停止条件（184）、受入条件（152） | CLAUDE.md は「`状態: approved` の計画の対象範囲・受入条件に厳密に従う」と定めている。古い範囲のまま H7 が「H6 が合格」を照合してしまう |
| L4 | B1〜B5 の fixture を**先に CPU で固定**し、seed と期待する到達数を JSON に書く | 到達数を受入条件にするので、実機で初めて走らせてはいけない |

## 対象範囲

**変更してよい**
- 新規 `docs/validation/gpu/g5/a4u_h6.py`（`prepare`＝手元、`run`＝T4 を 1 つだけ持つ、`judge`＝手元）
- 新規 `docs/validation/gpu/g5/a4u_h6_fixtures.py`（B1〜B5 の入力・seed・期待する到達数）
- 結果の置き場 `docs/validation/gpu/g5/a4u_h6_run/`、`A4U.md` への節、L1〜L3 の文書

**変更禁止**
- `a4u_h4_gate.py`・`a4u_h5_gate.py`・`a4u_h5.json`・`a4u_h4.json`（H4/H5 が `source_sha256` で参照）
- `a1_g2gate3.py` の `SEALED`・`CONDITIONS`、`a1_g2vox.py` の参照と基準、`a1_speed.py` の負荷、
  `chatcarlo/gpu/validation.py` の判定、`tests/test_gpu_g5_a4u_s2.py` の既存の試験
- `docs/validation/mv/p1,p1b,p1c,p1d/`（P1e/P1f が凍結）
- `~/.cache/chatcarlo/g5/a3/`（**A3 の本番 2 本 1.8 GB は消さない**）
- `data/`（`data/logs/` を除く）。Mohan のスペクトル値を追跡下のファイルに書かない
- `physics_sets.py` の登録簿（組の識別子・tag）

**スコープ外**
- MV と `seed_ratio` の**物理の再判定**（旧参照に当てること）。`seed_ratio` の参照の取り直し
  （シード近傍の EGS5 照合）は別段
- H7 の判定、D14 の文書更新、H1 の鉛の継ぎ目 +1.13% の判定

## 停止条件

- **報告専用（C1・C3・C4）を除く関門が fail または未実施なら、H6 を合格にしない**（指摘 6 で一本化）。
  実機資源の回収と `colab stop` は、その判定とは別に必ず行う
- **T4 は 1 セッション**（親計画の計算量の上限 (2)）。VM を失ったら、その時点までの結果を回収して報告する。
  再試行はユーザーの判断
- C2 が 1.0571 を外れたら統合の無駄として直す。C3 が 1.3 を超えたらレジスター数・占有率・経過時間と
  ともに報告して採否を聞く
- A1〜A3・B1〜B6 が落ちたら**実装の誤り**として止めて報告する（物理の判定ではないので、基準は結果を見て変えない）
- ディスクの空きが 3 GiB を割りそうなら止める（今 5 GiB）。出力は**リポジトリと `~/.cache` の外**
- 起動スクリプトの中に `rm` を入れない（削除はユーザーの作業）
- `colab` の呼び出しには全部に時間の上限を付け、見張りは **PID と `kill -0`**（`pgrep -f` を使わない）。
  10 分ごとに poll し、終わったものは poll ごとに回収する

## 最終報告の書き方（最小方針との整合）

有限の試験で「全経路が同じ」とは言えないので、**「事前に指定した分岐・構成で CPU と T4 の一致と
健全性を確認した」**と限定して書く。「同等性の証明」とは書かない。

## テストコマンド

```
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python docs/validation/gpu/g5/a4u_h6.py --help          # import と --help で計算しない
.venv/bin/python docs/validation/gpu/g5/a4u_s1_gate1.py --help     # 既存の形が変わっていない
```

## 時間の見込み

**「約 10 分」は撤回する。** 固定費が未測定で、群 B の実機 fixture と 5 つの NVRTC コンパイルが増えたため。
内訳: C1 の 3 条件（核は 3.5〜14 秒）＋ A1〜A3 ＋ 群 B の小 fixture ＋ C2 の 6 タスク × 3 順。
**核の計算時間は全部で 1 分に満たないが、支配するのは VM の立ち上げと 5 本のコンパイルで、それは測っていない。**
セッションの最初に総時間とコンパイル時間を記録し、そこから残りを見積もり直す（C4）。

## 計算データの行き先（PLAN_TEMPLATE の欄）

| 出力 | 行き先 |
|---|---|
| `a4u_h6_run/*.json`（判定・速さ・時間・到達数） | **リポジトリ**（小さい表と条件） |
| G4 の PNG 1 枚と要点のログ | **リポジトリ** |
| 中間の npz・バッチ・VM のフルログ | **消す**（判定後。削除はユーザー） |

## 実装の記録（2026-11-06）

Codex が実装し、Claude が独立に検証した。`a4u_h6.py`（570 行、`prepare`/`run`/`judge`/`_worker`）と
`a4u_h6_fixtures.py`（324 行）。CPU の B1〜B5 は全部通り、到達数を `a4u_h6_run/b_fixtures_cpu.json` に封印した。

### 計画の範囲外の修正 3 件（記録する）

**`chatcarlo/gpu/seed_cuda.py`: `physics_sets` を module 級の import に上げた。**
`GPU.observe()`（78 行）が `physics_sets` を参照していたが、import は `__init__` の中のローカルで、
`observe` からは見えず**呼べば必ず `NameError`**。A2b（`ffe20b9`）から残っていた。CUDA 専用なので
CPU の試験は一度も到達せず、手元の全テストが通っていた。**群 B の B3 が止まる位置**なので直した
（1 行、`from . import seed_geometry as core,therapy,physics_sets`）。Codex が実装中に見つけ、
範囲外として報告したもの。教訓は lessons_learned の「CPU だけの関門では GPU の不具合を捕まえられない」に
3 回目として足した。

**`a4u_h6.py` の `source_files()`: シンボリックリンクを除外した（2026-11-07）。**
`run` の 1 回目が `ValueError: payload requires regular files` で止まった。
`docs/validation/gpu/g5/a4_run/egs_code/` の 5 本（`batch_kernels_p3b.py` ほか）は
`prototypes/mv_electron/` を指すリンクで、`p.is_file()` はリンク先を見て True を返すが、
`tarfile.gettarinfo` はリンクとして見るので `isfile()` が False になる。
**実体はいずれも `source/prototypes/mv_electron/...` として既に payload に入っている重複**で、
しかも絶対パスを指すため VM 側では壊れたリンクになる。だから除外が正しい（`not p.is_symlink()` の 1 語）。
`identity()` が 3384→3379 件に変わるので `prepare` をやり直した。
**根本原因: `payload()` は手元のテストで一度も走っていなかった**——`--help` と群 B の fixture しか
動かしておらず、`prepare` までで止まる経路しか確かめていなかった。
止まったのは `STARTED.json` を書く**前・VM 作成前**で、T4 は消費していない。
修正後に payload 全体 3,730 件を数え直してリンク 0・tar 1.26 GiB を確認した。

### 承認後に変えた 1 件（ユーザー判断 2026-11-07）

**`H6_POLL_SECONDS` を 600 秒から 60 秒（A3 と同じ）に戻した。**
計画に「10 分ごとの poll で回収する」と書いたが、`run` のループは
`launch` →即時 `status` →`sleep` の順で、**最初の確認で完了していることはまずない**ため、
6 タスクそれぞれが必ず 1 回は丸ごと待つ。**下限 6×600 秒 = 60 分**で、その大半は純粋な待ちだった。
計画の「時間の見込み」は VM の立ち上げと 5 本のコンパイルだけを数えており、この待ちの下限を
数えていなかった——**承認の根拠にした見込みが誤り**だったので、ユーザーに内訳を出して判断を仰いだ。
60 秒なら下限 6×60 秒 = 6 分。払った代償は `prepare` のやり直し 1 回（手元 CPU 約 4 分、T4 は 0）。
payload の組み立ての poll は元から `POLL_SECONDS`（60 秒）だったので、変わったのは 6 タスクの待ちだけ。

### 計画から戻した 1 件

**`a4u_s1_gate1.py` の `--device` は戻した**（`sys.path` の修正だけ残した）。計画の「変更してよい」に
書いていたが、**レビュー後の群 A は `a4u_h6.py` が自前で `paired_judge` を呼ぶ**ので誰も使わない。
しかも `identity` とファイル名に device が入らないため、同じ `--out` で**CPU の結果を cuda として
再利用してしまい、再開の照合が通ってしまう**（H6 の中心の主張が CPU 対 CPU になって空で合格する）。
「誰も求めていないケースのために設計しない」に従って削除した。

### 検証した項目

| 項目 | 結果 |
|---|---|
| 禁止範囲 | `git status` で変更なし（`a4u_h4/h5_gate`・`a1_g2gate3`・`a1_g2vox`・`a1_speed`・`validation.py`・`physics_sets.py`・`mv/p1*`・`tests/test_gpu_g5_a4u_s2.py`） |
| 群 A | `judge_pairs` が `validation.paired_judge`（331 行）、G2 は `grid_paired_judge`、G4 は `gate2` ＋ `gate2a` を合否（501 行）。**`independent_judge` も NumPy 比較も使っていない** |
| 群 B の到達数 | B1 の `cutoff_local`・B2 の `annihilation_rest`/`flight` が 1 件だが、**注入した 1 粒子の決定論的な fixture**（0.007 MeV を 1 本、`old.fixed_cases()` を n=1）なので統計のゆらぎで 0 にならない |
| B6 | module を 5 本数える: `local`・`electron`・`seed`・`local_legacy`・`electron_legacy`、全部 `backend == 'nvrtc'`（358 行） |
| C2 | `a1_speed.load_configs/measure/summarize` と登録済みの `a1_speed_configs.json`、治療側は `a0_configs.json` の p7A。旗は `_test_legacy_physics=(version=='new')`＝新コード＋物理を戻した build で、D10 (1) の定義どおり |
| 削除 | `rm`・`rmtree`・`unlink` の呼び出しなし |
| colab | 全呼び出しに時間の上限（new/upload/download 900 s、stop 180 s、他 300 s）、poll は 10 分 |
| `--help` | `a4u_h6.py`・`a4u_s1_gate1.py` ともに計算せず終了（再実行して確認） |

### 全テストの通し実行（2026-11-06、ゴミ箱を空にしてから）: **合格**

| コマンド | 結果 |
|---|---|
| `pytest tests/ -q` | **1100 passed**・不合格 0（300.7 s、警告 2 は既存の spekpy のもの） |
| `pytest prototypes/mv_electron/tests -q` | **1804 passed・17 skipped**、不合格 4（1242.6 s） |
| 不合格 4 件の再実行 | **4 passed（8.48 s）** |

不合格 4 件（`test_p9a.py` の roundtrip 3 件と `drive_limit` 1 件）は**すべて
`OSError: [Errno 28] No space left on device`** で、空きが 0 GiB になったため。
空きを戻して同じ 4 件を回したら 8.48 秒で合格したので、**物理やコードの不合格ではない**。

**運用上の発見**: 両スイートの通し実行は一時出力に**約 6 GiB** を使い、**計画の 3 GiB の床を自力で
突き抜ける**（6 → 2 → 0 GiB）。MV スイート単独で約 3.3 GiB。T4 のセッションの前にテストを通すなら、
**先に 7 GiB 以上空けてから**にする。退避先 `~/.Trash/pytest-temp-2026-11-06b/`（空にするのはユーザー）。

**あふれた容量はコードが使っていない**（ユーザーの問い 2026-11-06）。pytest の `tmp_path` に作られる
捨て置きの出力で、中身はテストが作り直す（再実行が 8.48 秒で合格したのが証拠）。記録の中に同じ経路が
引用されているのは**ビルドした場所の由来**としてで、生きたデータとしてではない。
**`~/.cache/chatcarlo` の 25 GB は別物で、ほぼ参照されている**——`p8a` 3.2 GiB は検証の記録から
62,429 箇所、`archive` 5.6 GiB と `p9c2` 4.3 GiB は計画と `docs/jsmp133` の抄録から参照。
捨て置きではないので消す候補にしない。

### T4 セッション 1（2026-11-07）: **A1 のコンパイルで不合格。実機でしか出ない不具合を 1 件捕まえた**

| 項目 | 値 |
|---|---|
| session | `g5-a4u-h6-5d0065fca78c` |
| colab の操作 | `new` 1（**5.8 秒**）・`upload` 65（**計 約 9 分**、20 MiB × 65 断片 = 1.26 GiB）・`exec` 12・`download` 3・`stop` 1、**失敗 0** |
| 総時間 | 12:41:48 → 12:53:35（**11 分 47 秒**）。うち転送が約 9 分 |
| 結果 | A1 で停止。`finally` が VM を停止（`vm_stopped` 真）。以降の A2・A3・群 B・C1・C2 は**未実施** |
| H6 の判定 | **合格にしない**（停止条件どおり。報告専用を除く関門が未実施） |

止まった原因は**実機の NVRTC だけで出るコンパイル失敗**で、CPU の双子・手元の全テスト 1100 件・
`--help`・群 B の CPU fixture では一度も見えなかった。**これは H6 が存在する理由そのものである。**

```
79c07d5d....cubin.cu(136): error: identifier "NAN" is undefined
          if(e>hi && int(tail[0])==0)return NAN;
```

### 原因（最初の説明は 2 箇所間違っていたので、検証してから書き直した）

`chatcarlo/gpu/photon_physics.h:32` の `g5_total` は、表の上端を超えたエネルギーで `NAN` を返す。
NVRTC は host の `<cmath>` を持たないので `NAN` は未定義になる。**この定義は元から入口の `.cu` で守られていた**——
`t2_gpu.cu:2` と `t2_seed_gpu.cu` が `#ifndef NAN` で同じ式を置いている。だが診断の入口 `g1_gpu.cu` には無く、
**A4u 以前の診断の鎖には `NAN` を使う箇所が 1 つも無かった**。A4u が `photon_physics.h` を診断側に入れたことで、
初めて必要になった。したがって**壊れていたのは local-deposition の組だけ**で、electron-transport と seed は
元から守られていた。

**最初に書いた説明 2 件は誤りだった。** (1)「`t2_logs.h` の `p8g_initialize` も GPU で一度もコンパイルされて
いなかった」——偽。`p8g_initialize` は A4u 以前から治療の組に入っており、`t2_gpu.cu` の守りで通っていた。
(2)「治療と seed も落ちていた」——偽。検出力の試験で `#define NAN ...` という**文字列を全置換してしまい、
`t2_gpu.cu` の正当な守りまで消していた**ための誤り。A4u 直前の木（`13e4305^`）を別の worktree に出して
前処理し直し、治療・seed は bare NAN が **0 件**であることを確かめて訂正した。

### 直した場所と、直したことの確かめ方

`chatcarlo/gpu/g5_base.h`: `INFINITY` の守りの隣に `NAN` の守りを足した（`#if defined(P5_GPU) && !defined(NAN)`）。
ここを選んだのは、**診断の鎖の `INFINITY` が既にここにある**から。治療・seed では入口の守りが先に来るので
この定義は飛ばされる。ビット列 `0x7ff8000000000000` は標準の quiet NaN（`isnan` 真を確認）。

手元で 5 つの組すべてを C 前処理器にかけ、`P5_GPU` の枝に host 専用の綴りが残らないことを確かめた
（local-deposition・electron-transport・seed × legacy 0/1）。**守りを外すと local-deposition だけが
bare NAN 1 件になる**——これが検出力の証拠で、治療・seed は 0 件のまま。

### 教訓をテストに昇格させた（4 回目なので文章を足さない）

[tests/test_gpu_nvrtc_macros.py](../../../tests/test_gpu_nvrtc_macros.py)（7 件、0.27 秒）。
起動役が CuPy に渡すのと**同じ組み立て関数・同じ `-D` の組**から源を作り、前処理して
host 専用のマクロ 19 個・`#include <...>`・`std::` が `P5_GPU` の枝に残らないことを見る。
**テスト自身の検出力も 1 件で守る**（守りを外したら落ちることを確かめる。落ちない検査は、
本物の不具合を同じだけ静かに通してしまう）。

## 未了

**T4 のセッション 2**。セッション 1 は A1 のコンパイルで止まり、`STARTED.json` があるので
`~/chatcarlo_h6_2026-11-07c` では `run` を回せない（1 セッション 1 回の決まり）。
**この出力先はゴミ箱に入れない**——今回の発見の証拠そのものである。

やる順序（`prepare` を**最後**にする。源の木を触ると `prepared()` が照合で拒否するため）:

1. コードの変更をすべて終える（`g5_base.h` の `NAN` の守り、`a4u_h6.py` の 2 件は済み）
2. **コンパイルだけの下見**を 1 回（計画に無い追加。ユーザーの判断を仰ぐ）。
   源の tar だけなら **92,764,160 バイト＝断片 5 個**で、セッション 1 の 65 個に対し転送が約 9 分 → 約 1 分。
   5 つの組と探査核を**最初の失敗で止めずに全部**コンパイルし、残りのコンパイル誤りを 1 回で洗い出す。
   手元の前処理器の検査はマクロと標準ヘッダーしか見ないので、NVRTC 本体の拒否はこれでしか分からない。
3. `prepare` を新しい出力先でやり直す（手元 CPU 約 4 分）
4. `run`（poll 60 秒なので待ちの下限 6 分）→ `judge`

**実行の前に空きを 7 GiB 以上にする**（上の運用上の発見）。
