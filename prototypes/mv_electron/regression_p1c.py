"""Freeze/replay N1 before touching any existing provider code; relative paths only."""
import hashlib,json
from pathlib import Path
import numpy as np
from .physics import ROOT
from .component_checks import SEED
P=ROOT/'docs/validation/mv/p1c'
KEYS=['batches','ledger','counts','cross_sum','cross_sum2','cross_n','maxdepth']
def write(name,value):
    target=P/name;temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temporary.replace(target)
def verify():
    initial=json.loads((P/'initial_hashes.json').read_text())
    now={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() if (ROOT/f).exists() else None for f in initial}
    changed=[f for f in initial if now[f]!=initial[f]]
    frozen=(P/'preregistration.sha256').read_text().strip()==hashlib.sha256((P/'PREREGISTRATION.md').read_bytes()).hexdigest()
    r={'status':'pass' if not changed and frozen else 'fail','tracked_files':len(initial),'changed':changed,'preregistration_unchanged':frozen}
    write('untouched.json',r)
    if r['status']!='pass':raise RuntimeError(r)
    return r
def numerical(name):
    from .dcs import get_backend
    from .gs_backend import coefficients,single_cdf,distribution
    from .single_table import build
    b=get_backend(name);G,total=coefficients(b,2.,1024);y,cdf=single_cdf(b,2.)
    u,rates,single,info=build(b)
    result=dict(coefficients=G,total=np.array(total),single_y=y,single_cdf=cdf,u=u,rates=rates,single=single)
    for k in [.0005,.001,.003125,.00625,.0125,.025,.05,.1,.2]:
        L=1024
        while True:
            try:a=distribution(b,2.,k,L);break
            except ValueError:
                L*=2
                if L>32768:raise
        result[f'gs_{k}']=np.interp(u,a[1],a[0]);result[f'p0_{k}']=np.array(a[2])
    return result
def main(before=False):
    from .transport import run
    verify();d=P/'regression';d.mkdir(exist_ok=True);checks={}
    for name in ['rutherford','dcslib','eedl']:
        path=ROOT/('docs/validation/mv/p1/data/gs.npz' if name=='rutherford' else f'docs/validation/mv/p1b/{name}/data/gs.npz')
        g=np.load(path);kw={} if name=='rutherford' else dict(backend_rates=g['rates'],backend_single=g['single'])
        target=d/f'{name}_before.npz'
        if before and target.exists():continue
        r=run(2.,10000,SEED,.05,.05,np.array([0.,1.2]),.01,*(g[x] for x in ['T','K','u','y']),**kw)
        values=dict(zip(KEYS,r))
        if name=='rutherford':
            old=np.load(ROOT/'docs/validation/mv/p1b/m1_baseline.npz')
            assert all(np.array_equal(old[k],values[k]) for k in KEYS)
        if before:np.savez(target,**values)
        else:
            old=np.load(target);checks[name]={k:bool(np.array_equal(old[k],v)) for k,v in values.items()}
            np.savez(d/f'{name}_after.npz',**values)
        if name!='rutherford':
            target=d/f'{name}_table_before.npz'
            if before and target.exists():continue
            values=numerical(name)
            if before:np.savez(target,**values)
            else:
                old=np.load(target);checks[name+'_generated']={k:bool(np.array_equal(old[k],v)) for k,v in values.items()}
                np.savez(d/f'{name}_table_after.npz',**values)
    if before:write('n1_before.json',{'status':'saved','seed':SEED,'N':10000,'T':2.})
    else:
        passed=all(all(v.values()) for v in checks.values());write('n1.json',{'status':'pass' if passed else 'fail','checks':checks,'untouched':verify()})
        if not passed:raise RuntimeError('N1 mismatch: stop')
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--before',action='store_true');main(a.parse_args().before)
