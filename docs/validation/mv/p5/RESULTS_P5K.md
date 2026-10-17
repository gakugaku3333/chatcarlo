# P5k — ブロックの中で履歴を状態ごとに詰め直す（事象ごとのキュー）

2026-10-17。計画 `docs/ai/plans/2026-10-17-mv-p5k-phase-queues.md`（§7 優先）。

**現在の状態: 手元の関門0・関門1 合格、T4 の関門0・2・3・速さ・地図は未実行。採用の判断は未了で、P5j が合格した構成のまま。**

## 実装の経緯

Codex（`gpt-6.1-sol`）が実装し、最後の全体テストの途中で利用上限に達して止まった。ユーザーの指示（2026-10-17「今はあなたの方がトークンに余裕があるので、代わりに実装を進めて」）で、残り（全体テストの実行、手元の関門0、関門1の起動、この文書）を Claude が引き継いだ。ソースは Codex が書いたものから変えていない。計画した者が実装の仕上げも行ったので、Codex の上限が戻ったら Claude の担当分を Codex が読み取り専用で確認する。

## 成果物（すべて新規）

`prototypes/mv_electron/gpu/`:

- `p5k_driver.h`: スロット（`P5kSlot`、29,872 バイト）、初期化・1仕事（遷移を1〜K回）・終わりの書き出し・分類。CPU と GPU で共通。
- `p5k_cpu.cpp`: P5j の CPU の翻訳単位をそのまま取り込み、P5j の参照（`p5j_history` を履歴ごとに包んだもの）、5変種の模擬（複数ブロック、段階ごと）、スロット管理の合成の検査。
- `p5k_gpu.cu`: `p5k_kernel`（F はレーンの中に状態を持つ補充、Q 系はブロックの中の固定 AoS スロットを ballot と前置和で詰め直す）、合成の検査の核、大きさの核、時間の内訳の参考の核。
- `p5k_core.py`、`p5k.py`（local・gate1・export）、`p5k_colab.py`・`p5k_colab.ipynb`（「すべてのセルを実行」1回）、`p5k_emulate.py`（計画 §0 の模擬）、`p5k_pytest_keep.py`（pytest の一時フォルダの自動削除を止める任意の仕組み）。

`prototypes/mv_electron/tests/test_p5k.py`（41件）。凍結したファイル・既存のファイルは変えていない（`git diff` 空）。

## 手元の関門0（M3、Apple clang、2026-10-17、Claude が実行）

`p5k local --work data/p5k_local`（27 秒）: **通常版・検査版とも合格。**

- スロット管理: 5変種 × ブロック数 1〜3 × n（1、31、R−1、R、R+1、B×R−1、B×R、B×R+1、2B×R+17、続けて呼ぶ）の165件で全履歴がちょうど1回・行と履歴番号の対応が正しい。植えた故障3種（2回実行、1つ飛ばし、記録なし）×5変種の15件はすべて不合格として検出。リストの長さ 0・1・31・32・33・255・256・257・511・512 の端も正しい。
- 輸送: n=2,000、start=7,000 で、5変種すべての各履歴の全項目が P5j の参照とビット一致。監査の記録（あふれ 0、1履歴の最大 4,614 記録）も一致。`maxsteps`=1・2 で P→H・C→H→D の失敗の経路を通し、同じ領域で n=529→1→529 と続けて呼んでも一致。容量不足の記録は不合格になる。
- 模擬の再現（報告のみ、4,096 履歴）: 今の方式 0.187、補充 0.39〜0.42、ブロックで詰め直し R=256 0.739〜0.764、R=512 0.860〜0.887、R=1024 0.917〜0.934。計画 §0-2 の Claude の数値（0.187・0.739 など）を独立の実装で再現した。

## テスト（Claude が実行し直した）

| コマンド | 結果 |
|---|---|
| `pytest prototypes/mv_electron/tests/test_p5k.py -q` | 41 passed（27 秒） |
| `pytest prototypes/mv_electron/tests -q` | 917 passed, 2 skipped（17 分16 秒。スキップは既存の2件） |
| `pytest tests/ -q` | 403 passed, 2 warnings（既存） |

ログ: `data/p5k_prototype_tests.log`、`data/p5k_local.log`。Colab 用ソース: `data/p5k_export/p5k_sources.zip`。

## 関門1（M3、2026-10-17、Claude が実行。合格）

`p5k gate1 --work ~/.cache/chatcarlo/p5/p5k --saved ~/.cache/chatcarlo/p5/p5a/philox`（約6時間40分）。**100バッチすべてで、5変種 × 通常版・検査版の各履歴の全項目が P5j の参照とビット一致。** 添字の順に足した値は保存済みバッチと一致（`brems[1]` 以外はビット。`brems[1]` の差はバッチ99で 4.5e-13、上限 2.4e-9）。検査版の `log_counts` は参照と同じ値。記録は `~/.cache/chatcarlo/p5/p5k/gate1.json`（Drive の `viveMonte/p5/p5k/` に複製、`p5k_sources.zip` と同じ場所）。
