**文法上は問題なしです。数値も提示された集計に沿っていますが、平均値であることを明示する修正を推奨します。**

- **【推奨】平均の範囲を明示する。** “local dose” は局所線量として妥当ですが、“up to 3.3%” との組み合わせでは、個々のボクセル・方向・シードでの最大増加率にも読めます。実際は58本をまとめた距離別平均なので、それを明示すると主張の範囲が正確になります。
- **【任意】“decreasing to” と距離表現を明確にする。** 文意は通じますが、減るのは線量そのものではなく**増加率**です。“at 2 mm” も通常は前の “from their surface” を引き継ぎますが、“approximately 2 mm from the surface” とすれば、表面基準と距離の概数性が明確になります。

両点を反映した修正案です。

> The dose increase averaged over all 58 seeds was 3.3% about 0.2 mm from the seed surface and about 0.3% approximately 2 mm from the surface; the mean prostate dose increased by 0.5%.

**語数：原文30語 → 修正案33語（＋3語、空白区切り）。**

データとの対応では、約0.2 mmは層の代表距離、約0.3%は区間中心2.14 mmでの＋0.31%、前立腺平均0.5%は＋0.525%の丸めとして問題ありません。ただし、[RESULTS.md](/Users/oishifamily/Projects/viveMonte/docs/validation/mv/p9c/RESULTS.md:57) は旧集計＋3.34%と再計算＋3.39%の未解明の差を記録しています。**3.3%は旧集計に対応し、再計算値を小数第1位に四捨五入すれば3.4%**です。これは英訳の誤りではなく、根拠側に残る留保です。

ファイルは変更していません。