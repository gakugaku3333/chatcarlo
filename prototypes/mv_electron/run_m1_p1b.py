"""M1 exact regression against frozen before-change arrays and input hashes."""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
from .physics import ROOT
from .component_checks import SEED
P=ROOT/'docs/validation/mv/p1b'

def verify_p1():
    initial=json.loads((P/'p1_initial_hashes.json').read_text())
    actual={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'docs/validation/mv/p1').rglob('*')) if f.is_file()}
    missing=sorted(initial.keys()-actual.keys());added=sorted(actual.keys()-initial.keys());changed=sorted(k for k in initial.keys()&actual.keys() if initial[k]!=actual[k])
    r={'status':'pass' if initial==actual else 'fail','file_count':len(initial),'missing':missing,'added':added,'changed':changed}
    (P/'p1_untouched.json').write_text(json.dumps(r,indent=2)+'\n')
    if initial!=actual:raise RuntimeError(r)
    return r

def main():
    from .transport import run
    g=np.load(ROOT/'docs/validation/mv/p1/data/gs.npz');t=time.perf_counter()
    results=run(2.,10000,SEED,.05,.05,np.array([0.,1.2]),.01,*(g[x] for x in ['T','K','u','y']))
    keys=['batches','ledger','counts','cross_sum','cross_sum2','cross_n','maxdepth'];baseline=np.load(P/'m1_baseline.npz')
    checks={k:bool(np.array_equal(baseline[k],v)) for k,v in zip(keys,results)}
    np.savez(P/'m1_after.npz',**dict(zip(keys,results)),wall_seconds=time.perf_counter()-t)
    (P/'m1_identity.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('M1 mismatch: stop')
    return checks
