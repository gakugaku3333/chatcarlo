"""A1 paths, disk guard and hash-bound result identities."""
import importlib.util
import json
from pathlib import Path
import shutil
from a0_speed import sha
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
if (ROOT/'chatcarlo').is_dir() and str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
CACHE=Path.home()/'.cache/chatcarlo/g5/a1'
SUMMARY=HERE/'a1_run'


def space(path):
    path=Path(path)
    while not path.exists():path=path.parent
    if shutil.disk_usage(path).free<3*1024**3:raise RuntimeError('A1 disk space below 3 GiB')


def write(path,value):
    space(Path(path).parent);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if hasattr(v,'item') else v.tolist())+'\n')
    pending.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def code_identity():
    names=['chatcarlo/'+n for n in ('transport.py','kernel.py','physics.py','geometry.py','source.py','spectrum.py','materials.py','detector.py','tally.py')]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'chatcarlo/gpu').glob('*')) if p.suffix in ('.py','.h','.cu','.cpp')]
    names += [str(p.relative_to(ROOT)) for p in sorted(HERE.glob('a1_*.py'))]
    return {n:sha(ROOT/n) for n in names}


def checked_cached(path,identity,compute):
    path=Path(path)
    if path.exists():
        value=json.loads(path.read_text())
        if value['identity']!=identity:raise ValueError('A1 result identity mismatch: '+str(path))
        return value['result']
    space(path.parent);result=compute();write(path,dict(identity=identity,result=result));return result
