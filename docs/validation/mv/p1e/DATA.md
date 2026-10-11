# P1e データと再生成

入手元: Mendeley Data、doi:10.17632/w4hm5vymym.1、elsepa-2020.zip。
取得済みファイルは `~/.cache/chatcarlo/elsepa/elsepa-2020.zip`。
SHA256: `1066d4d4a008d4e65c087f1bdf5e348e5c0a8b2d0dc088748f403ea9fdbaa7e5`。
一致したためネットワークを使わず、再取得しなかった。

計画に記録された掲載情報のライセンス表記は「CC BY NC 3.0」。
展開したソースとreadmeで `licen`、`copyright`、`commercial` の別記述を検索する。
CC BY-NC 3.0条文そのものと生成した表への適用は未確認。生成表を配布する判断はしない。

自由原子 MUFFIN=0、Fermi核分布 MNUCL=3、Dirac–Fock電子密度 MELEC=4、
電子Furness–McCarthy交換 MEXCH=1（陽電子0）、MCPOL=0、MABS=0、IHEF=1。
核分布・電子密度・電子交換はEGS5説明と共通であることを確認した選択。
MCPOL/MABS/IHEF/陽電子交換はEGS5の実際の指定が確認できていない今回の固定選択。
一致を良くするために変更しない。

入力は列1〜6のキーワード、列8〜19の数値。elscata.fが読む全16キーワードを明示する。
VMOLは内部変数で入力キーワードではない。最初のEVより後はEV以外を読まない。
1ケース1空ディレクトリ、固定出力名の衝突を避ける。
陽電子の交換の説明行は出力されないため、電子交換の記述がないことを確認する。
因子分解後は級数側の断面積が印字されない。この欠測は `series_available=false`、
`series_xs=NaN` で明示し、値は補わない。表積分側は `table_xs` に別々に保存する。

生成表は外部 `~/.cache/chatcarlo/elsepa/p1e/tables/z{Z}_q{charge}.npz`。
83点: 81点の対数格子と追加0.02/2 MeV。角度は印字された606点のMU/THETAをそのまま使う。
読み取りはdcslibと同じ自然境界三次スプライン（対数エネルギー、対数DCSとMU）。
水は元素ごとに(Z+1)/Zを掛け数密度で合成。使用範囲0.01〜20 MeV以外を拒否する。
S2不合格の表で輸送を開始しない。

```bash
.venv/bin/python prototypes/mv_electron/generate_elsepa.py --prepare
.venv/bin/python prototypes/mv_electron/generate_elsepa.py --generate
# GSを使用する輸送を行う場合は、凍結前に--build-gsで同一backendのGSを準備する
# 今回S2停止時はGSを使用しないため、この準備・S4判定は未実行
.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --freeze
.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --calculate
.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --from-saved
.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --verify-untouched
```

環境変数 `CHATCARLO_ELSEPA_CACHE` で外部保存先を変更できる。
この実装は取得済みZIPを照合する。ZIPがない場合は自動取得を行わずエラーにする。
コンパイラ・引数・Python依存版・database全ファイル・実行ファイルのハッシュは
外部 `p1e/build/build.json` と `manifest.json`。生出力、入力、レシートは外部 `p1e/cases/`。
各ケースのexecution.jsonを先に保存し、出力変換の中断からは実行を再開せず解析を再開する。
各巡の全配列はshape・dtype・バイト列で一致を照合する（欠測NaNも同一表現）。
輸送は各runのnpzと条件・seed・SHA256の完了レシートで再開する。
