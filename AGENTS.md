# AGENTS.md

このリポジトリの技術的な正典は [CLAUDE.md](CLAUDE.md)。作業前に必ず読むこと（アーキテクチャ、既知の落とし穴、物理・データの出典など、Codex/Claude共通で従う内容はすべてそちらに書かれている）。

親ディレクトリの共通ルールは [../AGENTS.md](../AGENTS.md) を先に読むこと。

## Codex固有の差分

- `.claude/skills/vive-check/` 等の Claude Code Skill は Codex からは呼び出せない。同等の作業（ジオメトリー確認→軌跡確認→本計算→結果確認の段階実行）を行う場合は、CLAUDE.md の Commands セクションにある CLI コマンドを同じ順序で人間の承認を挟みながら手動で実行すること。
- `docs/ai/plans/` に `状態: approved` の計画ファイルがある場合はそれに厳密に従う。対象範囲外の変更をしない。
