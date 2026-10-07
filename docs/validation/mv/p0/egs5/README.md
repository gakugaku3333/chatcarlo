# P0 EGS5 実行手順

本体: `docs/egs5_crosscheck/egs5/` (1.0.8)。変更しない。
`.claude/agents/egs5-operator.md` と再検証の `run_egs5.py` のforeground/scratchレシピを使う。
`prepare_inputs.py`は既存再検証PDDのMAINを複製し、MV・採点セル・収支へ拡張する。
`run_egs5.py`は新scratchで `egs5run comp`、その後直接exeを起動する。
コンパイラのプロファイルは既存のまま。EGS5配布物への書き込みはない。

成功run（先行試運転を本計算へ混ぜない）:

|run|用途|
|---|---|
|coupled_repeat_a|10k単色10 MeV、初期COMP/default density effect試運転|
|coupled_repeat_c/d|10k単色10 MeV、最終MIXT/EPSTFL=1設定の同seed反復|
|coupled|200k単色10 MeV、電子輸送あり、J3/J5|
|kerma|200k同条件、電子局所停止、J3/J5|
|stopping_v3|PEGS-only SPIONE照合、CALLはDECK前、J4|
|spectrum_pilot|100k Mohan10、AUSGAB計時を含む負荷測定|
|spectrum_no_timer|100k Mohan10、計時を外した同seed反復|
|spectrum_200k|200k Mohan10、計時を外した履歴数・SEM・性能測定|
|spectrum_no_pdd|200k Mohan10、PDD採点とその一次/二次モーメントだけを無効にしたCPU差分測定|

固定seedは20261007、10×10 cm²平行照射、水30×30×30 cm³、中央2×2 cm²を2 mm binで採点。
MIXTの質量比は取得したESTAR組成から読み取る。密度効果fixtureはEGS5配布の
`data/density_corrections/compounds/water_liquid.density`をread-onlyでコピーする。
fixtureの比較3点のdensity effectは取得ESTAR応答と一致することを`check_estar.py`で確認する。

初回生成時に使用したコマンドの形（既存ディレクトリがあると安全に停止する）:

```bash
.venv/bin/python docs/validation/mv/p0/fetch_photon_data.py
.venv/bin/python docs/validation/mv/p0/fetch_estar.py
.venv/bin/python docs/validation/mv/p0/fetch_spectrum.py
.venv/bin/python docs/validation/mv/p0/check_photon_data.py
.venv/bin/python docs/validation/mv/p0/check_spectrum.py
.venv/bin/python docs/validation/mv/p0/egs5/prepare_inputs.py coupled --histories 200000
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py coupled
.venv/bin/python docs/validation/mv/p0/egs5/prepare_inputs.py kerma --histories 200000 --kerma
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py kerma
.venv/bin/python docs/validation/mv/p0/egs5/prepare_stopping.py
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py stopping_v3
.venv/bin/python docs/validation/mv/p0/analyze_egs5.py
.venv/bin/python docs/validation/mv/p0/check_estar.py
.venv/bin/python docs/validation/mv/p0/measure_kernel.py
.venv/bin/python docs/validation/mv/p0/egs5/prepare_spectrum.py spectrum_200k --histories 200000 --no-timer
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py spectrum_200k
.venv/bin/python docs/validation/mv/p0/egs5/prepare_spectrum.py spectrum_pilot --histories 100000
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py spectrum_pilot
.venv/bin/python docs/validation/mv/p0/egs5/prepare_spectrum.py spectrum_no_timer --histories 100000 --no-timer
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py spectrum_no_timer
.venv/bin/python docs/validation/mv/p0/egs5/prepare_spectrum.py spectrum_no_pdd --histories 200000 --no-timer --no-pdd
.venv/bin/python docs/validation/mv/p0/egs5/run_egs5.py spectrum_no_pdd
.venv/bin/python docs/validation/mv/p0/check_performance.py
```

再MC計算は、保存runの上書きを避けて新規名（例`coupled_reproduction`）を指定する。
同seed・同Nの物理出力は再現するが、CPU/wall/RSSは再現値ではなく再計測値である。
公開元のmaster URLは将来変わり得るので、再取得時のSHA256をmanifestの値と比較する。
結果文書に採用した基準系は同配布EGS5ソースツリーSHA256で固定されている。

失敗証跡も残す: coupled_pilotのF77宣言列数、coupled_pilot_v2のtime wrapper、
coupled_repeat_bのps制限、stoppingのRM探索、stopping_v2のDECK後設定リセット。
これらを成功結果と混ぜない。再解析のexit 0はJ1b留保やJ6予算超過の解除ではない。

`os.wait4`でrun単独のpeak RSSを取得する。Darwinのbyte単位。
ビルドを含む子プロセス全体の高水位RSSもmetadataにあるが、性能表には使わない。
計時つきAUSGABのCPUはclock呼び出し負荷を含むので、採点そのものの値に使わない。

rawデータ、参照parser、epstar fixture、PEGS生成データ、cacheはP0の`.gitignore`で
追跡対象から外している。取得・入力・結果のハッシュだけをmanifestに記録する。
