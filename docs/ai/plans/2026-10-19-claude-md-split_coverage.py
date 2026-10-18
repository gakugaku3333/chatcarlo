"""旧 CLAUDE.md の数値・リンク・`コード名` が、新しい置き場所のどこかにあるかを数える
（計画 docs/ai/plans/2026-10-19-claude-md-split.md の受入条件の機械的な照合）。

使い方（リポジトリの根で）:
    git show cf95727:CLAUDE.md > /tmp/old_claude.md
    .venv/bin/python -I docs/ai/plans/2026-10-19-claude-md-split_coverage.py . /tmp/old_claude.md

限界: 「docs」に分類された値は、docs などのどこかに同じ文字列があるというだけで、
文脈まで一致するとは限らない（短い数値は無関係な箇所でも当たる）。意味の照合は Codex の
レビュー（reviews/2026-10-19-claude-md-split-review-1.md）で行った。"""
import re,sys,pathlib
R=pathlib.Path(sys.argv[1]); old=pathlib.Path(sys.argv[2]).read_text().splitlines()
new_files=[R/'CLAUDE.md',R/'AGENTS.md',R/'.claude/rules/mv.md',R/'.claude/rules/chatcarlo.md',R/'docs/validation/mv/STATUS.md']
status=(R/'docs/validation/mv/STATUS.md').read_text()
# STATUS.md の原文の節は除いて「新しく書いた部分」だけを第一の照合先にする
status_new=status.split('## 段階ごとの要約（原文')[0]
primary='\n'.join([p.read_text() for p in new_files[:4]]+[status_new])
verb=status.split('## 段階ごとの要約（原文')[1]
other=''
for pat in ('docs/**/*.md','docs/**/*.txt','docs/**/*.json','chatcarlo/**/*.py','tests/**/*.py','prototypes/**/*.py','scripts/*','examples/*.yaml','README.md'):
    for f in R.glob(pat):
        if f.name=='STATUS.md': continue
        try: other+=f.read_text(errors='ignore')+'\n'
        except Exception: pass
num=re.compile(r'(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?(?:\s?(?:%|σ|µs|MB|GB|KB|cm|keV|MeV|M|s|×))?')
code=re.compile(r'`([^`]{3,80})`')
link=re.compile(r'\]\(([^)#]+)')
rows=[]
for i,l in enumerate(old,1):
    toks=set()
    for m in num.finditer(l):
        t=m.group(0).strip()
        if re.fullmatch(r'[-+−]?\d', t): continue
        toks.add(('num',t))
    for m in code.finditer(l): toks.add(('code',m.group(1)))
    for m in link.finditer(l): toks.add(('link',m.group(1)))
    for kind,t in toks:
        core=re.sub(r'^[-+−]','',t).rstrip('%σµsMBGKcmkeVMeV× ').strip() or t
        if kind=='num': core=re.sub(r'\s?(%|σ|µs|MB|GB|KB|cm|keV|MeV|M|s|×)$','',t).lstrip('+-−')
        where='primary' if (t in primary or core in primary) else 'verbatim' if (t in verb or core in verb) else 'docs' if (t in other or core in other) else 'MISSING'
        rows.append((i,kind,t,where))
from collections import Counter
print(Counter(w for *_,w in rows))
inmv=lambda i:38<=i<=318
print('MV節(38-318)以外で primary にないもの:')
for i,k,t,w in rows:
    if not inmv(i) and w!='primary': print(f'  L{i} {k:4s} {t!r:40s} -> {w}')
print('MV節で STATUS 原文にもないもの:')
for i,k,t,w in rows:
    if inmv(i) and w in ('MISSING','docs') and t not in verb: print(f'  L{i} {k} {t!r} -> {w}')
