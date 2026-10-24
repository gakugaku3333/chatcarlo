"""Run only the three authorized local gates in a disposable source copy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'docs/validation/mv/p8g3'
PYTHON=ROOT/'.venv/bin/python'


def main():
    before=json.loads((OUT/'protected_before.json').read_text())
    def check():
        changed=[p for p,d in before.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=d]
        if changed:raise AssertionError('protected files changed: '+repr(changed))
    check();rows=[]
    with tempfile.TemporaryDirectory(prefix='p8g3-regression-') as tmp:
        copy=Path(tmp)
        shutil.copytree(ROOT/'prototypes',copy/'prototypes',ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('p0','p8g1','p8g2','p8f'):
            shutil.copytree(ROOT/'docs/validation/mv'/name,copy/'docs/validation/mv'/name)
        for source in (ROOT/'docs/validation/mv').iterdir():
            target=copy/'docs/validation/mv'/source.name
            if source.name!='p8g3' and not target.exists():
                target.symlink_to(source,target_is_directory=source.is_dir())
        # The tests' P8g-3 collection artifact must also stay in the copy.
        (copy/'docs/validation/mv/p8g3').mkdir(parents=True)
        for module in ('p8g2','p8g','p8f'):
            command=[str(PYTHON),'-m','prototypes.mv_electron.gpu.'+module,'local']
            begin=time.monotonic()
            with (OUT/(module+'_local.log')).open('w') as log:
                result=subprocess.run(command,cwd=copy,stdout=log,stderr=subprocess.STDOUT)
            check()
            rows.append(dict(command=command,exit_code=result.returncode,seconds=time.monotonic()-begin,
                             execution='temporary source copy',original_hashes_unchanged=True))
            (OUT/'regression.json').write_text(json.dumps(dict(runs=rows,protected_files=len(before)),indent=2)+'\n')
            print(module,result.returncode,flush=True)
            if result.returncode:raise SystemExit(result.returncode)
    check()

if __name__=='__main__':main()
