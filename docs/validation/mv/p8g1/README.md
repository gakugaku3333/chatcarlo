# P8g-1 A1 の実装と引き継ぎ

対象は承認済み `docs/ai/plans/2026-10-24-mv-p8g1-precomputed-logs.md`。
表の定数logを材料別の配列へ置き換え、CPU/GPUがそれぞれの端末で初期化する。
格子、輸送の物理、変数log・exp・powと演算順は維持する。

- `log_sites.md`: 呼出グラフ、全log出現、18配列、故障の固定位置、未参照理由。
- `original_copy.diff`: 凍結ソースと新しいコピーの比較（名前のみ正規化）。
- `gate0.json`: 現在のソース指紋と関門0の結果。
- `gate0_reads.json`: 2,000履歴、seed=2026102102、start=7000の材料別読み出し／無効読み出し数。
- `faults_gate0.json`: 故障ごとのビルド、到達、拒否。ライブラリはpytestの一時ディレクトリに作成される。
- `RESULTS.md`: Codexが実行した検証の結果と未実行範囲。

関門0の通常版・検査版・P8fは全履歴出力をバイト比較する。
部品比較は元節点・nextafter両隣・表外・吸収端・対生成しきい値・z==1の分岐を含む。
重複と同じlogに丸まる合成格子は、左右それぞれの探索を凍結関数と直接照合する。
CUDAのホスト再生は起動境界・非ゼロstart・連続呼出しと100群の全出力集計を検査するが、T4の関門2ではない。

## Claude が実行するコマンド

以下はコードを用意しただけで、Codexは実行していない。

```bash
.venv/bin/python -m prototypes.mv_electron.gpu.p8g gate1
.venv/bin/python -m prototypes.mv_electron.gpu.p8g export
```

既定の作業先は `~/.cache/chatcarlo/p8g1`、P8f参照先は
`~/.cache/chatcarlo/p8f_prof/p8f/export`。必要なら `--work` / `--reference` を指定する。
P8fのmanifest・全ファイルのハッシュ・ソース指紋を確認し、保存済み参照の封印はP8fの読み出し関数で照合する。
gate1は1a/1bとf2の輸送での拒否を確認する。1bの集計は元のrun経路を使い、secondsのみ除外する。
exportはgate0/1の現在の指紋と合格を必須とし、元のP8fの束を変更せず別フォルダへ複写する。

T4のVMではexport内の全ファイルをそろえ、`p8g_source.zip`を展開し、そのルートをPYTHONPATHにして実行する。
このモジュールはCLIとして直接実行でき、Driveやブラウザを必要としない。

```bash
python -m prototypes.mv_electron.gpu.p8g_colab /path/to/export /path/to/results
```

Colab CLIの既存手順に従い、T4、nohup実行、10分ごとの確認・完成ファイル収集を行い、最後にVMを停止する。
関門2a〜fに順に合格してから、同じVMでP8f/P8g、40×128/40×256を交互に3回ずつ測定する。
20,000と200,000履歴の核の時間、初期化時間、レジスタ／ローカル／共有メモリ、SMあたり常駐ブロック数と中央値比を保存する。
測定に行情報を追加しない。新しい本計算は実行しない。

最後の全体テストもClaude担当：

```bash
.venv/bin/python -m pytest prototypes/mv_electron/tests -q
```

関門1・2は未実行であり、P8fの判定A・Bの継承や採否はまだ判断していない。
