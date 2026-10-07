# T1/T5 データ供給経路

参照値を手入力しない。`fetch_photon_data.py`, `fetch_estar.py`, `fetch_spectrum.py`
で公式公開元から取得し、各取得記録のUTC時刻、URL、SHA256を`manifest.json`へ集約する。
Rawデータと第三者の参照コードは`data/`に保存し、再配布条件が未確定のため
この新規ディレクトリの`.gitignore`で追跡対象から外している。

|候補|版・公開元|形式と範囲|利用条件／本調査での扱い|選定|
|---|---|---|---|---|
|NIST XCOM|公式配布Fortran 3.1（ソースヘッダ1999-06-23）、SRD8。Web版1.5は2010年更新|MDATX3.001–100（Z=1–100）、全係数とcoherent/incoherent/photoelectric/nuclear pair/electron pair。1 keV–100 GeV、今回の照会1 keV–20 MeV|配布プログラムの冒頭にnon-commercial使用許可。公開ダウンロードは明示。SRDのcopyrightは維持され、無条件な再配布許可とは読まない。非商用研究で使用、出典表示、raw再配布は未確認|T1に採用。元素表・化合物／混合物計算機を元のままscratchでビルドし、公式補間結果を取得。補正・外挿なし|
|xraylib|環境の4.2.1、EPDL系|既存API、約800 keVまで。対生成の関数なし|本環境にインストール済み、既存の低エネルギー照合のみ|T1のMV供給源には不適。100–800 keV照合参照として使用|
|同梱NIST XAAMDI 水CSV|既存`chatcarlo/data/nist_xaamdi/water.csv`|μ/ρ・μen/ρ、1 keV–20 MeV|既存ファイルは読取りのみ|全12点のMV全減弱係数照合。部分係数を提供しないためMV輸送用単独供給源ではない|
|NIST ESTAR|公式text formの実応答とwater composition|HTML、今回0.1/1/10 MeVのCLOSS/RLOSS/TLOSS/DELTA|SRDの再配布は未確認。取得スクリプトとハッシュを保存|J4参照として使用|
|Mohan et al. 10 MV|NRC公開EGSnrc配布`HEN_HOUSE/spectra/egsnrc/mohan10.spectrum`|20 bins、0–10 MeV、0.5 MeV幅、mode=1 (counts/MeV)|NRC配布LICENCEはAGPLv3。既存論文由来のspectrumの独立した再配布条件は未確認のためraw追跡せず、取得元・ハッシュで再現|T5第一候補を取得できた。代替への切替不要|

出典:

- [XCOM公式配布](https://physics.nist.gov/PhysRefData/Xcom/Text/download.html)
- [XCOM版の履歴](https://physics.nist.gov/PhysRefData/Xcom/Text/version.shtml)
- [NISTのSRD・その他データ・ソフトウェアの利用条件](https://www.nist.gov/open/license)
- [ESTAR text form](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR-t.html)
- [ESTAR水組成](https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=276)
- [ESTAR空気組成](https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=104)
- [NRC Mohan10実ファイル](https://github.com/nrc-cnrc/EGSnrc/blob/master/HEN_HOUSE/spectra/egsnrc/mohan10.spectrum)
- [NRC spectrum file parserとbin定義](https://github.com/nrc-cnrc/EGSnrc/blob/master/HEN_HOUSE/egs%2B%2B/egs_spectra.cpp)
- [NRC配布利用条件](https://github.com/nrc-cnrc/EGSnrc/blob/master/LICENCE)

## J1の扱い

J1aは同じ基礎データを共有するデータ経路の照合であり、独立の物理検証ではない。
J1bは数値の全表を`photon_comparison.csv`、閾値と合成の値を`photon_check.json`に保存。
光電効果の最大相対差1.885%は補正せず保持する。計画に部分係数の数値許容差がないので
「整合する」の確定判定は留保する。核場・電子場対生成の閾値確認は印刷エネルギーの
丸めで2つの行が同じ表示になるため、**入力エネルギーの順番**で行を対応付けている。

水はH2Oとして直接計算した値と、配布ATWTS.DATから得たH/Oの質量比合成を比較。
空気はESTARで取得したC/N/O/Ar質量比で直接mixture計算し、独立に元素値を合成する。
直接と合成は表の4有効桁で丸められるため厳密なビット一致ではない。
空気の公表質量比の和は丸めで1と異なるため、XCOMの標準機能と同じく正規化して使う。

## J7のbin定義

ヘッダ`20,0,1`の2番目の値が最小エネルギー、mode=1がcounts/MeV。
次の20行の1列目は**上端**であり、代表点ではない。bin内は一様分布。
EGS spectrum parserの`en_array[0]=dum`とmode=1の説明で確定した。
このP0の性能用EGS5入力では、取得したcounts/MeV×bin幅を規格化したCDFでbinを選び、
bin内を一様にサンプルする。元ファイルを20個の単色線へ置換していない。
最初の0–0.5 MeV binのPCUT未満の光子もサンプルし、EGS5の残存エネルギー沈着規則で
処理する。将来の両コード比較ではこの低エネルギーカットオフ処理を同じにする必要がある。

公表データはNRC配布物としてMohanに帰属するものを確保したが、原論文の数表との
逐項比較、特定の現代linacへの適合や測定PDDとの比較はP0の対象外で未実施。
EGSnrcは線源ファイルと形式の資料だけを取得し、第二の輸送コードは導入していない。
