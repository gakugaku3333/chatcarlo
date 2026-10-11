"""P1f checkpoints and the exact T0 file sets (never plans/other documents)."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .p1d_common import CONFIGS, KEYS, LABELS, aggregate, status
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/validation/mv/p1f'
CACHE = Path(os.environ.get('CHATCARLO_ELSEPA_CACHE', '~/.cache/chatcarlo/elsepa')).expanduser()
if CACHE.resolve().is_relative_to(ROOT):
    raise ValueError('ELSEPA cache must be outside the repository')
WORK = CACHE / 'p1f'
SEED, FIXED_SEED = 20261013, 20261014
TABLES = {'rutherford': ROOT/'docs/validation/mv/p1/data/gs.npz',
          'dcslib': ROOT/'docs/validation/mv/p1b/dcslib/data/gs.npz',
          'eedl': ROOT/'docs/validation/mv/p1b/eedl/data/gs.npz',
          'mf': ROOT/'docs/validation/mv/p1c/mf/data/gs.npz'}

def display(path):
    p = Path(path).resolve()
    if p.is_relative_to(ROOT): return p.relative_to(ROOT).as_posix()
    if p.is_relative_to(Path.home()): return '~/' + p.relative_to(Path.home()).as_posix()
    # External override: portable relative path to the configured cache.
    if p.is_relative_to(CACHE.resolve()): return '$CHATCARLO_ELSEPA_CACHE/' + p.relative_to(CACHE.resolve()).as_posix()
    raise ValueError('path outside registered roots')

def resolve(name):
    if name.startswith('~/'): return Path(name).expanduser()
    if name.startswith('$CHATCARLO_ELSEPA_CACHE/'): return CACHE/name.split('/',1)[1]
    return ROOT/name

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def sanitize(text):
    return text.replace(str(ROOT), '.').replace(str(Path.home()), '~').replace(str(CACHE.resolve()), '$CHATCARLO_ELSEPA_CACHE')

def write(path, value):
    p = Path(path)
    if not p.is_absolute(): p = OUT/p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(sanitize(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))+'\n')
    tmp.replace(p)

def save(path, **values):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.pending.npz')
    with tmp.open('wb') as f: np.savez(f,**values)
    tmp.replace(p)

def targets():
    paths=list((ROOT/'prototypes/mv_electron').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))
    for n in ('p1','p1b','p1c','p1d','p1e'):
        paths.extend(p for p in (ROOT/'docs/validation/mv'/n).rglob('*') if p.is_file())
    paths += [OUT/'PREREGISTRATION.md'] + list(TABLES.values())
    paths += [ROOT/'docs/egs5_crosscheck/egs5/data/dcslib'/f'{prefix}dx{z:03}.tab' for prefix in ('eel','pel') for z in (1,8)]
    paths += list((WORK/'tables').glob('*.npz'))
    if (WORK/'gs.npz').exists(): paths.append(WORK/'gs.npz')
    return {display(p):sha(p) for p in sorted(set(paths))}

def verify_initial():
    initial=json.loads((OUT/'initial_hashes.json').read_text()); now=targets()
    allowed={'prototypes/mv_electron/'+name for name in ('elsepa.py','gs_backend.py','single_table.py','run_transport.py')}
    changed=[p for p,h in initial.items() if p not in allowed and now.get(p)!=h]
    added=[p for p in now if p not in initial and not p.startswith('prototypes/mv_electron/') and not p.startswith(display(WORK)+'/')]
    if changed or added: raise RuntimeError('T0 initial preservation mismatch: '+str(changed+added))
    return dict(status='pass',initial_files=len(initial),authorized_modified=[p for p in allowed if now.get(p)!=initial.get(p)],added_code=[p for p in now if p not in initial and p.startswith('prototypes/mv_electron/')])

def verify():
    if (OUT/'amendment1_hashes.json').exists():
        from .run_p1f_amendment1 import verify_freeze2
        return verify_freeze2()
    frozen=json.loads((OUT/'freeze_hashes.json').read_text()); now=targets()
    changed=[p for p,h in frozen.items() if now.get(p)!=h]; added=sorted(set(now)-set(frozen))
    result=dict(status='fail' if changed or added else 'pass',files=len(frozen),changed=changed,added=added)
    if changed or added: raise RuntimeError('T0 frozen input mismatch: '+str(result))
    return result

def regression(phase):
    from .transport import run
    results={}
    for b,path in TABLES.items():
        target=OUT/'regression'/f'{b}_{phase}.npz'
        if not target.exists():
            with np.load(path) as g:
                kw={} if b=='rutherford' else dict(backend_rates=g['rates'],backend_single=g['single'])
                r=run(2.,10000,SEED,.05,.025,np.array([0.,1.2]),.01,*(g[k] for k in ('T','K','u','y')),**kw)
                save(target,**dict(zip(KEYS,r)))
        if phase=='after':
            with np.load(target) as a,np.load(target.with_name(f'{b}_before.npz')) as z:
                results[b]=set(a.files)==set(z.files) and all(np.array_equal(a[k],z[k]) for k in a.files)
    if phase=='after':
        write('regression.json',dict(status='pass' if all(results.values()) else 'fail',checks=results,N=10000,seed=SEED,K=.025))
        if not all(results.values()): raise RuntimeError('T0 regression mismatch; stop')
    return results

def table_hashes():
    return {p.name:sha(p) for p in (WORK/'tables').glob('*.npz')}

def provider():
    from .elsepa import Elsepa
    record=OUT/'t1.json'
    expected=json.loads(record.read_text())['table_hashes']
    if table_hashes()!=expected:raise RuntimeError('T1 table hashes changed')
    return Elsepa(WORK/'tables',expected_ihef=0,expected_hashes=expected)

def validate_gs(g,energy=None):
    expected=json.loads((OUT/'t2.json').read_text())
    if expected['status']!='pass':raise RuntimeError('T2 not passed; no transport')
    if table_hashes()!=expected['table_hashes']:raise RuntimeError('T2 table hash mismatch')
    if str(g['backend'])!='elsepa' or int(g['IHEF'])!=0:raise RuntimeError('ELSEPA GS identity mismatch')
    if json.loads(str(g['elsepa_table_hashes']))!=expected['table_hashes']:raise RuntimeError('GS/T2 source hash mismatch')
    if energy is not None:provider().validate_energy(energy)
    return expected['table_hashes']
