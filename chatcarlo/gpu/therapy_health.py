"""Transport prerequisites; no dose or speed judgement is made here."""
import numpy as np


def check(r,arrays,fg):
    n=len(r['visits']);nb,nc,nf=map(int,arrays['sizes'][1:4]);f=nb+nc+nf
    for k,v in r.items():
        if not np.isfinite(v).all():raise ValueError('nonfinite '+k)
    if not np.all(r['visits']==1):raise ValueError('history output count')
    d=r['groups'];i=r['group_ints']
    if d.shape!=(100,f+37) or i.shape!=(100,79+nb):raise ValueError('group shape')
    if np.any(i[:,:3]) or np.any(i[:,8:10]) or np.any(d[:,f+3]):raise ValueError('capacity/maxsteps/unresolved/violations')
    if np.any(d[:,:f]<0) or np.any(r['field']<0) or np.any(r['ledger']<0):raise ValueError('negative deposition')
    t=d[:,f:f+6].sum(axis=0)
    if t[0]<=0:raise ValueError('nonpositive source energy')
    if abs(t[0]-t[1:].sum()+t[5])>1e-9*max(t[0],1e-30) or d[:,f+6].max()>1e-11:raise ValueError('energy balance')
    if r['hits'][60:62].any():raise ValueError('secondary/parent position')
    for j,(off,size) in enumerate(zip(fg.offsets,fg.sizes)):
        values=r['field'][off:off+size];ops=int(r['field_additions'][off:off+size].sum())
        bound=(2*ops+2*n+size)*np.finfo(float).eps*np.abs(values).sum()
        if abs(values.sum()-r['ledger'][j])>bound:raise ValueError('region ledger '+str(j))
    if abs(d[:,:nb].sum()-t[1])>1e-9*max(t[0],1e-30):raise ValueError('depth deposition ledger')
    return dict(status='pass',histories=n)


def split_check(whole,parts):
    """2c: exact integer counts, explicit summation bound for positive scores."""
    for k in ('visits','field_additions'):
        summed=np.concatenate([p[k] for p in parts]) if k=='visits' else sum(p[k] for p in parts)
        if not np.array_equal(whole[k],summed):raise ValueError('split integer '+k)
    operations=int(whole['field_additions'].sum())
    for k in ('field','ledger','groups'):
        # Depth/field sums only. Group maxima are not additive records.
        x=whole[k].sum(axis=0)[:300] if k=='groups' else whole[k]
        y=sum(p[k].sum(axis=0)[:300] for p in parts) if k=='groups' else sum(p[k] for p in parts)
        bound=max(operations,1)*2**-53*(np.abs(x)+np.abs(y))
        if np.any(np.abs(x-y)>bound):raise ValueError('split summation '+k)
