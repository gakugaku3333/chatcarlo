"""P2 atomic checkpoints, portable provenance, and three explicit freeze sets."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/validation/mv/p2'
CACHE = Path(os.environ.get('CHATCARLO_P2_CACHE', '~/.cache/chatcarlo/p2')).expanduser()
ELSEPA = Path('~/.cache/chatcarlo/elsepa/p1f').expanduser()
EGS = ROOT/'docs/egs5_crosscheck/egs5'
TABLES = {'dcslib': ROOT/'docs/validation/mv/p1b/dcslib/data/gs.npz',
          'mf': ROOT/'docs/validation/mv/p1c/mf/data/gs.npz',
          'elsepa': ELSEPA/'gs.npz'}
ENERGIES = (1.25, 2., 6., 10.)
SEED = 20261020
KEYS = ('batches','ledger','counts','cross_sum','cross_sum2','cross_n','maxdepth')
if CACHE.resolve().is_relative_to(ROOT):
    raise ValueError('P2 work/cache must be outside repository')

def portable(p):
    p=Path(p).resolve()
    if p.is_relative_to(ROOT): return p.relative_to(ROOT).as_posix()
    if p.is_relative_to(CACHE.resolve()): return '$CHATCARLO_P2_CACHE/'+p.relative_to(CACHE.resolve()).as_posix()
    if p.is_relative_to(Path.home()): return '~/'+p.relative_to(Path.home()).as_posix()
    raise ValueError('unregistered path')

def resolve(p):
    if p.startswith('$CHATCARLO_P2_CACHE/'): return CACHE/p.split('/',1)[1]
    if p.startswith('~/'): return Path(p).expanduser()
    return ROOT/p

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def sanitize(s):
    return s.replace(str(CACHE.resolve()),'$CHATCARLO_P2_CACHE').replace(str(ROOT),'.').replace(str(Path.home()),'~')

def write(p,v):
    p=Path(p)
    if not p.is_absolute(): p=OUT/p
    p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=p.parent,prefix=p.name+'.',suffix='.tmp',delete=False) as f:
        f.write(sanitize(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))+'\n');q=Path(f.name)
    q.replace(p)

def save(p,**v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',dir=p.parent,prefix=p.stem+'.',suffix='.pending.npz',delete=False) as f:
        np.savez(f,**v);q=Path(f.name)
    q.replace(p)

def verify(group='protected'):
    m=json.loads((OUT/'manifest.json').read_text())
    frozen=m[group]
    if group=='production_frozen' and not frozen:raise RuntimeError('P2 production freeze has not been made')
    changed=[p for p,h in frozen.items() if not resolve(p).is_file() or sha(resolve(p))!=h]
    result=dict(status='合格' if not changed else '不合格',group=group,files=len(frozen),changed=changed)
    if changed: raise RuntimeError('Q0 freeze mismatch; stop: '+str(result))
    return result

def regression(phase):
    from .transport import run
    checks={}
    for b in ('dcslib','mf'):
        target=OUT/'regression'/f'{b}_{phase}.npz'
        if not target.exists():
            with np.load(TABLES[b]) as g:
                r=run(2.,10000,SEED,.05,.025,np.array([0.,1.2]),.01,
                      *(g[x] for x in ('T','K','u','y')),
                      backend_rates=g['rates'],backend_single=g['single'])
                save(target,**dict(zip(KEYS,r)))
        if phase=='after':
            with np.load(target) as a,np.load(target.with_name(b+'_before.npz')) as z:
                checks[b]={k:bool(np.array_equal(a[k],z[k])) for k in KEYS}
    if phase=='after':
        import difflib
        code=ROOT/'prototypes/mv_electron/transport.py'
        baseline=OUT/'regression/transport_before.py'
        (OUT/'regression/code_diff.patch').write_text(''.join(difflib.unified_diff(baseline.read_text().splitlines(True),code.read_text().splitlines(True),fromfile='transport_before.py',tofile='transport.py')))
        result=dict(code_sha256=sha(code),status='合格' if all(all(x.values()) for x in checks.values()) else '不合格',N=10000,seed=SEED,K1_max=.025,f_E=.05,checks=checks)
        write('q0_regression.json',result)
        if result['status']!='合格': raise RuntimeError('Q0 electron regression failed; stop')
        return result

def progress(stage,state,**details):
    p=OUT/'progress.json';v=json.loads(p.read_text()) if p.exists() else dict(stages={})
    v['stages'][stage]=dict(state=state,**details)
    v['resume_command']='.venv/bin/python prototypes/mv_electron/run_checks_p2.py --resume'
    v['production_runs_started']=v.get('production_runs_started',0)
    write(p,v)

def is_freeze_input(path):
    name=Path(path).name.lower()
    return name not in ('readme','readme.txt','egsnrc_license.txt','nist_license.html',
                        'xcom_download.html','xcom_version.html','pegs_manual.txt',
                        'user_manual.txt','amendment1_report_source.txt') and not name.endswith(('.md','.pdf'))

def t0():
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    if not (OUT/'manifest.json').exists():
        # Enumerate ONCE, then verify only the recorded explicit list; never documents/plans.
        paths=[]
        for stage in ('p0','p1','p1b','p1c','p1d','p1e','p1f'):
            paths.extend(p for p in (ROOT/'docs/validation/mv'/stage).rglob('*')
                         if p.is_file() and p.suffix in ('.npz','.json','.csv','.f','.inp','.input','.txt','.html','.gz','.DAT')
                         and 'baseline_code' not in p.parts)
        paths.extend((ROOT/'prototypes/mv_electron').glob('*.py'))
        paths=[p for p in paths if p.name not in ('p2_common.py','run_checks_p2.py')]
        paths.extend([EGS/'egs'/f for f in ('egs5_pair.f','egs5_hatch.f','egs5_photon.f','egs5_brems.f','egs5_photo.f','egs5_compt.f','egs5_annih.f')])
        paths.extend([EGS/'pegs/pegs5.f',ROOT/'docs/validation/mv/p0/data/XCOM/XCOM.f',ROOT/'docs/validation/mv/p0/data/XCOM/ATWTS.DAT',ROOT/'chatcarlo/data/nist_xaamdi/water.csv'])
        paths.extend(TABLES.values());paths.extend((ELSEPA/'tables').glob('*.npz'))
        paths.extend(EGS/'data/dcslib'/f'{prefix}dx{z:03}.tab' for prefix in ('eel','pel') for z in (1,8))
        # Existing electron loop is a potentially mutable code target with baseline copy.
        mutable=ROOT/'prototypes/mv_electron/transport.py'
        paths=[p for p in paths if p!=mutable and is_freeze_input(p)]
        missing=[portable(p) for p in paths if not p.is_file()]
        if missing: raise RuntimeError('T0 missing inputs: '+str(missing))
        write('manifest.json',dict(protected={portable(p):sha(p) for p in sorted(set(paths))},
              mutable_before={portable(mutable):sha(mutable)},production_frozen={},
              freeze_policy='Explicit lists only. No plans, markdown documents or runtime-directory scans during verification.'))
        (OUT/'regression').mkdir(exist_ok=True)
        (OUT/'regression/transport_before.py').write_bytes(mutable.read_bytes())
        write('initial_git_status.json',dict(note='Pre-existing work is not reverted; status is stored separately.'))
    verify();progress('T0','running',pending='electron baseline and production-suite baseline')
    regression('before')
    progress('T0','complete',protected=verify(),regression_N=10000,regression_seed=SEED,production_freeze='pending until implementation and pilots are complete')
