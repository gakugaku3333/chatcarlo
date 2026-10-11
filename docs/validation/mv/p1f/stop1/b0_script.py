"""B0: read-only hashing and byte-for-byte preservation before amendment edits."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from prototypes.mv_electron.p1f_common import OUT, WORK, ROOT, CONFIGS, display, resolve, sha, write, verify
stop=OUT/'stop1'
if stop.exists():raise RuntimeError('stop1 already exists; refuse overwrite')
freeze=verify()
old=json.loads((OUT/'manifest.json').read_text())['files']
mismatch=[dict(path=p,expected=h,actual=sha(resolve(p)) if resolve(p).is_file() else None) for p,h in old.items() if not resolve(p).is_file() or sha(resolve(p))!=h]
assert [r['path'] for r in mismatch]==[display(OUT/'progress.json')],mismatch
names=['RESULTS.md','summary.json','k5b.json','progress.json','manifest.json','freeze_hashes.json','initial_hashes.json','untouched.json','commands.json']
paths=[OUT/n for n in names]+sorted(OUT.rglob('*.log'))
stop.mkdir()
for p in paths:
 dest=stop/p.relative_to(OUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 assert sha(p)==sha(dest)
for name in ['test_p1f.py']:
 p=ROOT/'prototypes/mv_electron/tests'/name;dest=stop/'baseline_code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for name in ['p1f_common.py','run_checks_p1f.py']:
 p=ROOT/'prototypes/mv_electron'/name;shutil.copyfile(p,stop/'baseline_code'/name)
fixed=[OUT/'elsepa/runs'/f'{n}{ext}' for n in CONFIGS for ext in ('.npz','.json')]
fixed += [OUT/'elsepa/fixed'/f'{length}_{c}{ext}' for length in (.05,.2) for c in (False,True) for ext in ('.npz','.json')]
fixed += [OUT/f't{i}.json' for i in range(1,5)] + [OUT/'t4_checkpoint.json']
fixed += sorted((WORK/'tables').glob('*.npz'))+[WORK/'gs.npz']
assert len(fixed)==32
write(stop/'immutable_hashes.json',{display(p):sha(p) for p in fixed})
write(stop/'preservation.json',dict(stage='B0',freeze1=freeze,copied={display(p):sha(p) for p in paths},immutable_files=len(fixed),manifest_registered_files=len(old),manifest_mismatches=mismatch,reason='progress.jsonはClaudeが再開コマンドを実行した際に更新された（運用の記録・承認済み追補による）。成果物だけでは実行者を証明できない。',T6_known_before_amendment=0.007502233691723832))
shutil.copyfile(Path(__file__),stop/'b0_script.py')
print(json.dumps(dict(freeze1=freeze,preserved=len(paths),immutable=len(fixed),manifest_mismatches=mismatch),ensure_ascii=False))
