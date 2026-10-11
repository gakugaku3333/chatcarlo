# P1f データと再生成

ELSEPA elsepa-2020、Mendeley Data doi:10.17632/w4hm5vymym.1。
取得済み `~/.cache/chatcarlo/elsepa/elsepa-2020.zip` のSHA256:
`1066d4d4a008d4e65c087f1bdf5e348e5c0a8b2d0dc088748f403ea9fdbaa7e5`。
ネットワークを使わず、P1eの展開物とビルドをアーカイブの全ファイルに照合して読み取りだけで使用する。
ELSEPAのソースは変更しない。P1eの保存物は消さない。

掲載情報の利用条件の表記は、P1eの取得時に記録された「CC BY NC 3.0」。
同梱ソース/readmeに別の記述がないことはP1eのlicense_scan.jsonを参照する。
CC BY-NC 3.0条文そのものと生成表への適用は未確認。配布方法の決定は対象外。

自由原子 MUFFIN=0、Fermi核分布 MNUCL=3、Dirac–Fock電子密度 MELEC=4、
電子Furness–McCarthy交換 MEXCH=1（陽電子0）、MCPOL=0、MABS=0。
核分布・電子密度・電子交換はEGS5説明と共通。EGS5のMCPOL/MABS/IHEF/陽電子交換の実際の指定は未確定。
P1fは全エネルギーでIHEF=0を指定する。P1eの不合格を見たあとに変更したこと、
変更の根拠がEGS5に依存しない検査であることを事前登録している。
格子・補間・倍率・その他のポテンシャルは結果に合わせて調整しない。

表: `~/.cache/chatcarlo/elsepa/p1f/tables/z{Z}_q{charge}.npz`。
10 keV〜10 MeV、対数49点に0.02/2 MeVを追加して51点（0.1/1 MeVは既存の格子点を正確な値で置換）。
ELSEPAの606角度点、DCS、ERROR、級数側/表積分側の別々の断面積、警告、
入力・プログラムのハッシュ、収束の出力行を保存する。値を丸めない。
各元素の(Z+1)/Zと数密度で水を合成し、補間はP1eの自然境界三次スプラインを使う。
陽電子の表は検査だけで輸送には使わない。

生出力: `~/.cache/chatcarlo/elsepa/p1f/cases/`。
1ケース1空ディレクトリ、全キーワード明示、入力・要求モデル・606行・両端を照合する。
入力、stdout、stderr、dcs、その他の出力とハッシュ・計算時間を保存する。
収束の原文はstdoutとelsepa.dat等にも残す。
P1eのパーサは多行の警告を見出しと継続行の両方で収録するので、T2では原文ブロックを全て残し、
同じ警告の継続行だけを同一事象として数える。警告を結果を見たあとで除外しない。
「表が整合していない」警告は、表/級数の全断面積差0.5%・第1輸送断面積差0.1%以内のみ受入。
全断面積0.5%は既知の約0.25%を見て事前に置いた上限（ELSEPA閾値の2.5倍）。
それ以外の警告は1件でも不合格。ERROR最大0.001以下。

P1eのビルド来歴: `~/.cache/chatcarlo/elsepa/p1e/build/build.json`。
コンパイラ版、引数、実行ファイル・source・database全ファイルのハッシュはこれを再確認する。
P1fの実行環境と再確認した来歴は runtime_provenance.json、表と各ケースのハッシュは manifest.json。
GS: `~/.cache/chatcarlo/elsepa/p1f/gs.npz`。上端10 MeV。
T2で検査した表のハッシュを、GS表と各runのレシートに明示する。IHEF/表のハッシュが違えば拒否する。
P1eの表への暗黙の切り替えは行わない。入射10 MeVは許可、範囲外は輸送実行前に拒否する。

最初にPREREGISTRATION.mdとinitial_hashes.json、既存4 backendの回帰beforeを固定する。
その後のコマンド（完了済みケース/runは再実行せずハッシュ照合し、未完了だけ同条件で再開）:

```bash
.venv/bin/python prototypes/mv_electron/generate_elsepa_p1f.py --generate
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --build-gs
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --freeze
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --calculate
# T0/T1/T2/T4が合格した場合だけ。K9(i)不合格/統計不足なら輸送へ進まない
.venv/bin/python prototypes/mv_electron/run_p1f.py
.venv/bin/python -m pytest tests/ -q
git status --short -- chatcarlo tests scripts examples
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1f.py --verify-untouched
```

輸送seed=20261013、K9(i)試作seed=20261014。履歴数と条件はPREREGISTRATION.md。
輸送はrun単位でpending npz→完了レシート→確定npzの順に保存。
レシート済みpendingはハッシュ照合して確定し、再計算しない。
生成途中の実行済みケースはexecution.jsonで変換から再開する。
`--from-saved`は生成の二巡・直接計算の保存物を読み取りだけで再確認し、ELSEPA/輸送/EGS5を起動しない。
環境変数CHATCARLO_ELSEPA_CACHEによる外部保存先の変更に対応する。

ELSEPAのstdoutでは第2輸送断面積をDCS積分から計算して印字している。P1eを継承した
`series_xs`の第3欄はこのstdout値であり、独立な部分波級数の第2モーメントではない。
T2の警告受入判定は、級数からの独立な値がある全断面積・第1輸送断面積だけを使う。

IHEF=0でも、部分波次数間のスプライン補間（ISCH=2）、次数上限25000、
約0.5°での級数評価方式の切り替えなど内部の数値近似は残る。IHEF=0を「近似が一切ない」と解釈しない。

輸送runは独立したプロセスで実行し、各関数の固定seedで開始する。
各runの保存先・progress・完了レシートは独立し、親だけが全run完了後に共有の報告を作る。
輸送単価の測定は全輸送run完了後に逐次実行し、並行実行の負荷を含めない。
