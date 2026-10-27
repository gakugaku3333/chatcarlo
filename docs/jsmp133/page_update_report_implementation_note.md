# 実装と物理データの補足図解

2026-10-27。公開版29の最新ソースに、2章「仕組み」の補足として「計算コードと物理データの役割」を追加、公開版30へ反映。

Claude Codeによる開発支援→Pythonによる条件・データ・実行管理→GPU/CUDAの粒子輸送、の3段階を図示。光子のEPDL系xraylib（60keV側）とNIST XCOM（高エネルギー側）、電子のELSEPAの役割を短く説明。断面積・弾性散乱・最大10MeVの意味も補足。実装がPythonだけでGPU輸送を行うという誤解を避け、CUDAの役割を区別した。

ローカルのCLAUDE.md、MVのdata_sources.md、ELSEPAの常設ルールと、NIST XCOM・ELSEPA公開元で出典を確認。データやコードの再配布は行っていない。

参考：
- https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html
- https://data.mendeley.com/datasets/w4hm5vymym/1

追記のブロックと専用CSS以外に差がないことを確認。抄録本文・数値・既存図・フォーム・送信処理・用語解説は不変。ローカル表示、ビルド、公開後の表示確認を完了。viveMonteでcommit/pushなし。

- 公開URL: https://chatcarlo-jsmp133-abstract-review.ayumu-raku.chatgpt.site/#implementation-data
- Sites version: 30
- Source commit: `f0a6e0a9307d9c3a1058705c782ea16c502292cf`
- Deployment: `appgdep_6adfb375fd1481919971549d2fc82418`（succeeded）
- 表示確認: `runs/jsmp133-implementation-note.png`
- 保全HTML: `runs/jsmp133_site_v30/review.html`
