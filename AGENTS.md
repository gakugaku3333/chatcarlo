# AGENTS.md

AIエージェント（Claude Code、OpenAI Codex、Google Antigravity など）がこのリポジトリで作業するときの共通の入口。

このリポジトリの技術的な正典は [CLAUDE.md](CLAUDE.md)。作業前に必ず読むこと（アーキテクチャ、コマンド、既知の落とし穴、物理・データの出典など、どのAIにも共通で従う内容はすべてそちらに書かれている）。内容をこのファイルに複製しない。

親ディレクトリに `../AGENTS.md` がある場合は、それを先に読むこと。

## 使用者がシミュレーションを頼んできたとき

ChatCarloは、使用者がAIとの対話で自分の意図したモンテカルロ計算を行うためのシステムである。計算・検査・可視化は `chatcarlo` のコマンドが行い、AIは聞き取り、条件の組み立て、結果の説明を担う。進め方は次のスキルに書いてある。

| 場面 | スキル |
|---|---|
| 「〜の被ばくを計算したい」など、scene.yamlがまだない・要望が曖昧 | `vive-interview`（聞き取りで要件を確定し、scene.yamlを作る） |
| scene.yamlがあり、確認しながら実行する | `vive-check`（形状確認→軌跡確認→本計算→結果確認。各関門で使用者の承認を待つ） |
| 成果物に誤りがないか確かめる | `vive-audit`（作業した本人とは別の立場で監査する） |

- スキルの本体は `.claude/skills/<名前>/SKILL.md` にある。`.agents/skills/<名前>` は同じフォルダへのリンクで、内容は1つである。
- スキルを自動で読み込まないAIでは、上の `SKILL.md` を直接読んで、その手順に従うこと。
- スキルは特定のAIの機能を前提にしない。サブエージェントや選択肢つき質問の機能がある場合の使い方と、ない場合の代わりの手順が、各スキルに書いてある。
- いきなりscene.yamlを書き始めない。先に `vive-interview` で要件を確定する。

## 環境の準備

Pythonの仮想環境はリポジトリ内に作る（グローバル環境にパッケージを入れない）。コマンドは CLAUDE.md の「Commands」にある。準備ができたら `.venv/bin/python -m pytest tests/ -q` が通ることを確認する。

## 開発の決まり

- `docs/ai/plans/` に `状態: approved` の計画ファイルがある場合は、その対象範囲・受入条件・テストコマンドに厳密に従う。書かれていない変更をしない。
- 断面積や線量換算係数などの数値を記憶から書かない。出典と取得スクリプトは CLAUDE.md の「Cross-section/dose-coefficient data provenance」を参照。
- 現状は研究・教育用である。線量を患者被ばくや遮蔽の判断に使う前に、確立された計算コードとの照合が必要（README.md の警告を参照）。

## Claude Code専用のもの

次は研究用で、Claude Code以外では使わない。公開する利用手順には含めない。

- `.claude/skills/vive-crosscheck/`: EGS5との相互検証。
- `.claude/agents/egs5-operator.md`: EGS5の操作役。
- `.claude/agents/vive-auditor.md`: 監査役のサブエージェント定義。中身は `.claude/skills/vive-audit/auditor.md` を参照するだけである。
