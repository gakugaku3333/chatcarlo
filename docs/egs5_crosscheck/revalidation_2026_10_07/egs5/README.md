# EGS5側の再実行

事前登録は親ディレクトリの `PREREGISTRATION.md`。KEK配布EGS5 1.0.8を
変更せず、以前の実績コードをコピーして密度・光子下限・線源位置のみ整合した。
`run_egs5.py` は別の新規保存ディレクトリにコピーしてから実行する（既存ランの
上書きを禁止している）。計算本体・結合Fortran・実行ファイルは一時ディレクトリで
生成し、終了後に消去する。`egs5run comp` と実行ファイルのforeground起動を分離し、
正常終了コードを確認する。

```bash
.venv/bin/python docs/egs5_crosscheck/revalidation_2026_10_07/egs5/run_egs5.py
.venv/bin/python docs/egs5_crosscheck/revalidation_2026_10_07/egs5/parse_results.py
```

`pdd_repeat_a/b` は10,000 histories・seed 1の反復。一次・二次モーメントから
算出された全47ビンの平均/SEM/相対SEMと全ファントム収支が完全一致してから
本計算を開始した。乱数・幾何・カットオフは本計算と同じで、これらの小標本結果を
本計算へ混ぜていない。`reproducibility.json` に一致判定とscoring行のSHA256を保存。

本計算は `bsf_thinslab`（8M、seed 6）、`bsf_phantom`（8M、seed 7）、
`pdd_phantom`（100M、seed 1）。平均/SEMは各historyの付与エネルギー和を
一次・二次モーメントとして集計した標本平均/標本SEM（1σ）。計算量・seedは
結果によって変更していない。

絶対Gy/historyへの換算は `MeV/history × 1.602176634e-13 J/MeV / mass(kg)`。
質量は水密度1.000でBSF20 g、PDD4 g、OCR2 g。BSF_wは独立runの比なので
相対SEMの二乗和から比のSEMを計算する。PDD/OCRは絶対値比較で、表面/中央による
形状正規化は判定量ではない。EGS5既存出力書式による表示桁の丸め
（PDDのmean/SEMは7有効桁）を含む。全ビンは `pdd_bins.csv`、BSFと全ビンの
数値・警告一覧は `results.json`。

PEGS5の出力では電子fit150区間とRayleigh分布fit100区間について、既定相対精度
1%を達成できなかった警告が出る。光子輸送断面積のfitは69区間で終了し、この
警告は出ない。INCOH=1のS(q)生成にも同警告は出ない。電子はECUT=1.5 MeVにより
輸送されないが、Rayleigh角度分布fitの精度上限は残存する系統的不確実性として
記録し、数値合格と同一視しない。許容基準や入力を結果後に変更していない。
各runの `pgs5job.pegs5lst`、`egs5job.out`、`build.log`、`run.log` を一次資料とする。

初回10k計装のコンパイルは密度変数名を `rho` と書き、IMPLICIT NONEでエラーに
なった。COMMON MEDIAの正しい `rhom` に修正してから再実行した。
`pdd_repeat_a_build_failed` はその失敗の証跡で、物理結果には使用しない。
成功runでは `metadata.json` のcode/input SHA256、EGS5配布物ツリーSHA256、
コンパイラ、実行前後の事前登録SHA256、終了コード、実行時刻を保存する。
