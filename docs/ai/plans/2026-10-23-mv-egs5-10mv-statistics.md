# 計画: 10 MV 水ファントム PDD・OCR の EGS5 側の統計を増やす（追加計算して合算）

- 状態: implemented（計画どおりではない。下の「実施の記録」）
- 作成者: claude
- 実装担当: codex（計画レビューも Codex。`codex-kyodo` の手順）
- 日付: 2026-10-23
- Codexレビュー: 1巡（2026-10-23。致命 1・実質 5・軽微 2・参考 3、すべて反映。原文 `data/logs/codex_review_stats10mv_1.md`）

## 0. 前提事実（検証済み、2026-10-23）

- EGS5 本番の設定（`docs/validation/mv/p3b/egs5/main_egs/config.json` の先頭）: `"N": 46000000, "seed": 20291763, "E": 10.0, "IUNRST": 0, "noflight": 0, "thickness": 30.0, "batches": 100, "user_code": "p3b_egs", "field": [5.0, 100.0]`。
  `summary.json`: `"status": "合格"`、`"CPU_seconds": 5656.455712000001`、`"wall_seconds": 5733.777`。
- `result.npz` の配列（一部）: `coarse (100,150)`, `fine (100,600)`, `axis_coarse (100,120)`, `axis_fine (100,300)`, `slab_coarse (100,120)`, `profile (100,2,3,72)`, `profile_fine (100,2,3,144)`, `batchin (100,)`, `N ()`, `seed ()`。先頭の次元がバッチ。
- 乱数: `p3b_egs.f` の 128–129 行 `luxlev=1` / `inseed=seed`（RANLUX）。
- EGS5 ソースの最終変更: `git log -- prototypes/mv_electron/p3b_egs.f prototypes/mv_electron/p3b_egs_common.inc prototypes/mv_electron/p3b_egs_score.inc prototypes/mv_electron/p3b_egs_ausgab.inc docs/egs5_crosscheck/egs5` → `da3e1f4`（P3b-2/P3b-3 追加のコミット）だけ。
- `run_p3b2.run_egs(name, N, seed, edges, cum, field=None, regions=None, stop_after=None, user_code='p3b_egs', resume=False)`。
  作業 `CACHE/'egs5'/name`、出力 `OUT/'egs5'/name`（`OUT = ROOT/'docs/validation/mv/p3b'` 決め打ち）。`name_check` は N が 100 の倍数であることを要求。
  `stop_after` は 1〜99 の整数で途中停止。ビルドは `cwd=work` で `egs5run comp`。`refuse_failed()` は `checks_p3b2.json` に不合格があれば止める。
- 種: `grep -rlE "\b20292[0-9]{3}\b" docs prototypes examples chatcarlo` → P4a の 3 ファイルだけ、使われている値は 20292881・20292882・20292883・20292982・20292983。20292101・20292102 は未使用。
- CPU: `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` → 4 / 4。
- 抄録の数値の再現: 作業用 `req1.json`（`gpu_pdd`/`egs_pdd` 各 120 点と SEM、`gpu_ocr`/`egs_ocr` 3×36 点と SEM、`Dref`）から、
  誤差 PDD 平均 1.29%・最大 1.74%、OCR 70 点 平均 1.28%・最大 2.00% を再現。相対 SEM は PDD で GPU 0.732%・EGS5 0.734%、OCR で各 0.958%。
  `req1.json` を作ったスクリプトはリポジトリにない（別セッションの作業用）。そのため G0 で保存データから作り直す。

## 目的（1〜3行）

JSMP133 抄録 Results (1) の 10 MV 水ファントム（10×10 cm²）の 1 点ごとの比較は、統計誤差が平均 1.3% と差（PDD 平均 1.5%）と同じ大きさで、
差が統計で読めない。EGS5 の保存済み本番（4,600 万履歴）に、種だけを変えた同じ計算を足して合算し、EGS5 側の誤差を下げる。
GPU 側の追加は別の計画（下の「書かなかったこと」）。

## 「追加して加算」でよいか（結論: よい。ただし条件つき）

- 既存の `docs/validation/mv/p3b/egs5/main_egs/`（N=46,000,000、100 バッチ × 460,000、種 20291763、`p3b_egs`、RANLUX luxlev=1）と
  **履歴数と種以外をすべて同じ**にした計算を足す。どの本も中身は独立な同じ確率の計算なので、本ごとの平均を履歴数で重みづけして足せばよい
  （下の「合算」。使用者の決定で追加の本は 1 本 2,700 万なので、バッチの大きさが既存の 46 万と違い、全バッチを同じ重みで並べる方法は使えない）。
- 足す計算の EGS5 ソース・PEGS 入力・スペクトル・照射野・領域の SHA256 が `main_egs/config.json` と一致することを、起動前に照合する
  （`p3b_egs*.f`/`.inc` と `docs/egs5_crosscheck/egs5` は 2026-10-13 の da3e1f4 以降変更なし。2026-10-23 に git log で確認）。
- 最初から全部計算し直す案は、既存の 1.57 時間分を捨てるだけで利点がないので採らない。

## 追加量と統計誤差の見込み（2026-10-23 に計算済み）

抄録の数値を出した作業用データ（`req1.json`。PDD は 0.25 cm 刻み 120 点、EGS5 の最大のビン 2.375 cm で規格化、OCR は 3 深さの中心の 20% 以上の
70 点、各点の誤差は 点と基準ビンの SEM の二乗和平方根）から、元の数値（PDD 平均 1.29%・最大 1.74%、OCR 平均 1.28%・最大 2.00%）を
再現したうえで、SEM ∝ 1/√N で見積もった。現状は GPU と EGS5 の相対 SEM がほぼ同じ（PDD 各 0.73%、OCR 各 0.96%）。

| GPU の倍率 | EGS5 の倍率 | PDD 誤差 平均 / 最大 | OCR 誤差 平均 / 最大 |
|---|---|---|---|
| 1 | 1（現状） | 1.29 / 1.74% | 1.28 / 2.00% |
| 1 | 2 | 1.12 / 1.52% | 1.12 / 1.71% |
| 1 | 4 | 1.02 / 1.39% | 1.02 / 1.54% |
| 1 | 9 | 0.96 / 1.32% | 0.97 / 1.45% |
| 1 | ∞ | 0.91 / 1.25% | 0.92 / 1.36% |
| 4 | 4 | 0.65 / 0.87% | 0.64 / 1.00% |
| 4 | 9 | 0.55 / 0.75% | 0.55 / 0.84% |
| 9 | 9 | 0.43 / 0.58% | 0.43 / 0.67% |
| 16 | 16 | 0.32 / 0.44% | 0.32 / 0.50% |

**EGS5 だけを増やしても、合成の誤差は最大で約 3 割しか下がらない**（∞ でも平均 0.91%）。誤差を半分にするには両方を約 4 倍、
3 分の 1 にするには両方を約 9 倍にする必要がある。PDD の 2σ 超 18 点の原因（基準の 1 ビンの比が 1.7σ ずれる）も、GPU 側の分は EGS5 を増やしても残る。

**所要時間の見込み**（EGS5 は 1 プロセス 123 µs/履歴 = 4,600 万で約 1.6 時間、P3b-3 の実測。この Mac は高性能コア 4・高効率コア 4）

| EGS5 の倍率 | 追加の本数 | 手元の経過時間の見込み（高性能コアで 4 本並列） | GPU を同じ倍率にした場合の Colab T4（26.1 µs/履歴） |
|---|---|---|---|
| 4 | 3 | 約 1.6 時間（1 巡） | 追加 1.38 億で約 60 分 |
| 9 | 8 | 約 3.2 時間（2 巡） | 追加 3.68 億で約 2.7 時間 |
| 16 | 15 | 約 6.4 時間（4 巡） | 追加 6.9 億で約 5.0 時間 |

並列時の速さは未測定（下の関門 G1 で測ってから全体を起動する）。

### 使用者の決定（2026-10-23）

**EGS5 の入射光子を合計 1.00×10⁸ 個にする。追加は 2 本**（各 27,000,000 = 100 バッチ × 270,000、種 20292101・20292102、2 本を同時に実行）。
46,000,000 + 2×27,000,000 = 100,000,000。上の表の延長（2026-10-23 に同じ方法で計算）:

| GPU | EGS5 | PDD 誤差 平均 / 最大 | OCR 誤差 平均 / 最大 |
|---|---|---|---|
| 4.6×10⁷（今のまま） | 1.0×10⁸ | 1.10 / 1.50% | 1.10 / 1.68% |
| 1.0×10⁸（後で GPU も足した場合） | 1.0×10⁸ | 0.88 / 1.18% | 0.87 / 1.36% |

所要時間の見込み: 1 本 2,700 万 × 123 µs = 約 55 分の CPU 時間。2 本の同時実行で経過時間は約 1 時間（G1 で確かめる）。
2 プロセスは MV のルールの並列の上限の内側。

## 対象範囲

- 変更してよい:
  - 新規 `prototypes/mv_electron/run_stats10mv.py`（追加計算の起動・照合・合算・比較の駆動）
  - 新規 `prototypes/mv_electron/tests/test_stats10mv.py`
  - 新規 `docs/validation/mv/stats10mv/`（RESULTS.md・各本の `config.json`/`summary.json` の写し・`result.npz` の SHA256 の一覧・合算と比較の結果の JSON）
  - `docs/validation/mv/STATUS.md` に 1 行
- 変更禁止:
  - **`prototypes/mv_electron/run_p3b2.py`・`run_p3b3.py`・`run_p3a.py`・`analysis_p3b3.py`**（`run_p3b2.SOURCES` に `run_p3b2.py` 自身が入っており、
    触ると `source_hashes` が `main_egs` と一致しなくなる。Codex レビュー 1 巡目の致命の指摘）
  - `docs/validation/mv/p3b/` の既存ファイル（`main_egs`・`e3` を含む。読むだけ）
  - `prototypes/mv_electron/entry/`（`compare.py`・`verify_comparison` は 46,000,000・種 20291763・100 バッチを決め打ちしており、承認済みの入口の結果の照合に使う。合算は別の経路にする）
  - `chatcarlo/`、`p3b_egs*.f`/`.inc`、`docs/egs5_crosscheck/egs5/`
  - `docs/jsmp133/` の抄録本文（数値の差し替えは結果を見てから使用者が決める）

## 実装方針

1. **既存の起動役を変えずに場所だけ切り替える**（Codex の代替案を採用）。`run_p7.egs_scope` と同じ形で、`run_p3b2` のモジュール変数
   `CACHE`・`OUT`・`CHECKS` を一時的に差し替えて `run_p3b2.run_egs` をそのまま呼ぶ:
   `CACHE = OUT = ~/.cache/chatcarlo/stats10mv`、`CHECKS = ~/.cache/chatcarlo/stats10mv/checks_none.json`（存在しないファイル）。`SPECTRUM_SHA` は既定のまま（`main_egs` と同じ）。
   作業は `~/.cache/chatcarlo/stats10mv/egs5/<name>`、結果は `~/.cache/chatcarlo/stats10mv/egs5/<name>/{config.json,summary.json,result.npz}`。
   2 本は別々の子プロセス（`multiprocessing` の spawn か `subprocess`）で動かし、モジュール変数を共有しない。
   Codex が読んで確認済み: `egs5run comp`・PEGS・チェックポイントは作業ディレクトリに書き、共有の `data/` は読むだけ。名前が違えば 2 本同時でも衝突しない。
2. **種と名前**: `add_20292101`・`add_20292102`（各 N=27,000,000、100 バッチ）。G1 の再現は `repro_e3`（下記）。
3. **起動前の照合**: 各本の `config` が `run_p3b3.egs_config(27000000, seed, edges, cum)` と一致し、かつ `N`・`seed` 以外のすべての項目が
   `main_egs/config.json` と一致すること（`egs_input_hashes`・`source_hashes` を含む）。Codex が今のコードで確認済み（差は `N`・`seed` だけ）。
4. **読み込みの検査**: `run_p3b3.validate(data, 'egs', N, seed, summary)` に加え、`summary['tests']` の項目名の集合が `main_egs/summary.json` と
   完全に同じであること（`validate` は「ある項目がすべて true」しか見ず、項目の欠けを検出しないため）。
5. **単位**: 保存配列（`axis_coarse`・`profile` など）は領域ごとの MeV の積算で、まだ Gy／入射光子ではない。各本を
   `analysis_p3b3.dose(data)` で Gy／入射光子に換算してから扱う（`dose` は本ごとの N で割るので、N が違っても同じ単位になる）。旧形式の `coarse`・`fine` は混ぜない。
6. **合算**: 各本 i（既存 N=46,000,000、追加 2 本 N=27,000,000）について、換算後の 100 バッチから点ごとの平均 m_i と SEM s_i（標本標準偏差 ÷ 10）を出し、
   重み w_i = N_i / 100,000,000（0.46・0.27・0.27）で 合算の平均 = Σ w_i m_i、合算の SEM = √Σ w_i² s_i²。
   OCR の 4 方向の平均（G0 で特定した抄録と同じ平均）は**バッチごとに**取ってから m_i・s_i を出す。規格化（PDD の最大のビン、OCR の中心）は合算の後に行う。
7. **比較**: 抄録と同じ定義（PDD は EGS5 の最大のビンで両者を規格化、OCR は中心のビン、EGS5 が 20% 以上の 70 点、各点の誤差は点と基準ビンの SEM の
   二乗和平方根）。GPU は既存の入口の 100 バッチ（`~/.cache/chatcarlo/mv_entry/9734038057e4/`、SHA256 を入口の添え書きと照合）のまま。
   この処理を新スクリプトに書き、まず現状の EGS5 だけで抄録の数値を再現する（G0）。

## 関門と受入条件

- [ ] **G0 再現**: `req1.json` を使わず保存データ（GPU の入口 100 バッチ、`main_egs/result.npz`）から、抄録の 6 個の数値
      （PDD 差の平均 1.51%・最大 3.56%・誤差の平均 1.29%、OCR 差の平均 0.62%・最大 1.91%・誤差の平均 1.28%）を小数第 2 位まで再現する。
      Codex がレビュー中に再現済み（1.506012 / 3.557365 / 1.291472、0.624478 / 1.914834 / 1.280337%）。
- [ ] **G1 ビルドの再現**: 保存済み `docs/validation/mv/p3b/egs5/e3`（N=200,000、種 20291751、`p3b_egs`、照射野 [5,100]、領域と `source_hashes`
      を含めて `N`・`seed` 以外が `main_egs` と同じ設定）を、上の切り替えた場所で `repro_e3` として計算し直す。`config.json` が保存済みの `e3` と完全に一致し、
      `result.npz` の `run_p3b2.EQUALITY_KEYS` の全配列がバイト単位で一致すること。一致しなければ追加計算を起動せずに止めて報告する。
      （ビルドに使う gfortran や EGS5 の環境が 10-13 から変わっていないことの確認。所要約 2 分）
- [ ] **G2 追加計算**: 2 本を同時に起動し、2 本とも上の 3・4 の照合と検査を通る。各本の `CPU_seconds` と `wall_seconds` を記録する（速さの合否は付けない。見込みは 1 本 CPU 約 55 分）。
      途中で止まった場合の再開は `run_p3b2.run_egs(..., resume=True)`（既存の仕組み）。再開した本の速さは参考値として扱う。
- [ ] **G3 合算前の整合**: 換算後の絶対線量（Gy／入射光子）で、PDD 120 点（`axis_coarse`）と OCR 3 深さ × 36 点（方向平均後）の計 228 点を 1 つの統計量にまとめる:
      χ² = Σ (m_a − m_b)² / (s_a² + s_b²)、自由度 228。組は 既存–追加1、既存–追加2、追加1–追加2 の 3 つ。**3 つとも p ≥ 0.001 で合格**。
      p 値は名目値（隣り合う点の相関を無視している）として報告する。**不合格が出たら、自動で計算し直さずに止めて報告し、扱いは使用者が決める**。
- [ ] **G4 合算と比較**: 合算した EGS5（1.00×10⁸ 履歴）と GPU（4.6×10⁷ 履歴）で、G0 と同じ定義の数値（差の平均・最大、誤差の平均・最大、2σ 超の点数と偶然の見込み〔名目値〕）を出す。
      誤差の平均は見込み（PDD 1.10%、OCR 1.10%）と相対 10% 以内で合う（合わなければ原因を調べて報告。判定の不合格にはしない）。
- [ ] 判定のスクリプトは、解析の前に入力を照合する: 各本の `result.npz`・`summary.json`・`config.json` の SHA256（完了時に `~/.cache/chatcarlo/stats10mv/seal.json` に記録し、以後の読み込みで照合）、
      履歴数・種・設定、同じ本を 2 回入れていないこと（種の重複を拒否）、GPU の入口の 100 バッチの SHA256。
- [ ] 照合の指紋（ハッシュ）の対象はコードと入力データだけにし、計画ファイルと RESULTS は入れない。
- [ ] テスト（`test_stats10mv.py`、EGS5 を起動しない小さな作り物のデータで）:
  - 履歴数の違う 3 本の既知データで、合算の平均・SEM が手計算と一致する。
  - MeV からの換算 → OCR の方向平均（バッチごと）→ 合算 → 規格化 の順序を守っている（順序を入れ替えると値が変わる作り物で検出できる）。
  - 種・N・入力ハッシュの不一致、同じ種の重複、`summary['tests']` の項目の欠けを、解析の前に拒否する。
  - 場所の切り替えが終わった後、`run_p3b2.CACHE`・`OUT`・`CHECKS` が元の値に戻る（例外で抜けた場合も）。2 本の作業先・保存先が別である。
  - G0 の再現（保存データがある環境だけで実行。ない環境では skip と表示）。

## テストコマンド（実装完了の定義）

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests/test_stats10mv.py prototypes/mv_electron/tests/test_p3b3.py -q
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m prototypes.mv_electron.run_stats10mv reproduce   # G0
.venv/bin/python -m prototypes.mv_electron.run_stats10mv gate1       # G1（約 2 分）
.venv/bin/python -m prototypes.mv_electron.run_stats10mv run         # G2（約 1 時間。Claude がバックグラウンドで実行。Codex のサンドボックスは ~/.cache に書けない）
.venv/bin/python -m prototypes.mv_electron.run_stats10mv analyse     # G3・G4
```

## 書かなかったこと（スコープ外を明示）

- **GPU 側の追加**。上の表のとおり、誤差を大きく下げるには GPU も増やす必要がある。GPU の追加は、入口の条件ハッシュが種で変わるため、
  「承認した条件 = 実計算の条件」の主張との関係を含めて別の計画にする。
- **抄録本文の差し替え**。次の 2 点は使用者の判断が要る（結果を見てから決める）:
  - Results の「依頼 1 … 結果は (1) の 10 MV の行のとおり」は (1) の行を承認済みの入口の計算に結びつけている。EGS5 を合算した数値（または後で GPU を足した数値）に差し替えると、この文の書き方を変える必要がある。
  - 「入射光子 4.6×10⁷ 個」は、両コードで数が違うので書き分けが要る。
- 積層ファントムと 60 keV の行の追加計算。
- 規格化の定義（基準 1 ビン）の見直し。
- 1 巡目の計画にあった「同じ種で `stop_after=1` の 1 バッチを本番と比べる」G1 と「4 本並列の速さの関門」は採らない（途中停止では `result.npz` ができず読み出しの仕組みが別に要る、G2 と状態が衝突する。Codex 指摘 3・4・8）。ビルドの同一性は G1 の `e3` の再計算で、速さは記録だけで扱う。
- `run_p3b2.run_egs` に `out` 引数を足す案は採らない（`source_hashes` が変わる。Codex 指摘 1）。

## 実施の記録（2026-10-23、Claude が追記）

- Codex レビューは 1 巡（原文 `docs/ai/plans/reviews/2026-10-23-mv-egs5-10mv-statistics-review1-codex.md`）。2 巡目は起動したが、ユーザーの指示「周りくどいことしなくていいから早く計算した方がよくない？」で途中で止めた（出力なし）。承認の手続きは取っていない。
- 実装は計画の `run_stats10mv.py`・テスト・関門 G0〜G4 の形ではなく、作業用のスクリプト（`docs/validation/mv/stats10mv/` の `run_add.py`・`pool.py`・`final.py`・`g3detail.py`）で行った。
  - G0（抄録の数値の再現）、G1（`e3` の再計算でバイト一致）、G3（合算前の χ²）は実施した。
  - G2 の照合（N・種以外の設定の一致、検査の項目名の集合）は実施した。
  - 新規のテスト（合算の式・順序・拒否）は書いていない。
- 範囲の変更（ユーザー決定）:
  - GPU も合計 1×10⁸ 個に増やした（入口で 5,400 万、種 20292201）。
  - 比較は 0.5 cm 刻み（2 ビンの平均）にした。
  - 抄録の (1) の 4 MV の行を 10 MV に置き換えた。
- 途中で入口の不具合（VM が消えると関門の記録を失う）が見つかり、`colab_cli.py` を直した（計画の範囲外。ユーザー指示「根本解決を行なって」）。
- 結果: [docs/validation/mv/stats10mv/RESULTS.md](../../validation/mv/stats10mv/RESULTS.md)。
