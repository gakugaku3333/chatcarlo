"""Finite/nonnegative, visits, energy, region ledger and bounded group checks."""
import numpy as np
import a2_common
from prototypes.mv_electron.gpu import p8g2_core as core


def unpack(result,g,arrays):
    nb,nc,nf=g.nb,g.nc,g.nf;n=len(result['visits'])
    h=core.unpack(result['d'],result['i'],nb,nc,nf)
    ag=core.reduce_records(result['groups'],result['group_ints'],nb,nc,nf,n,arrays['mass_g_cm2'])
    ag.update(_groups_d=result['groups'],_groups_i=result['group_ints'])
    return h,ag


def check(result,g,arrays,*,expected_failure=False):
    core.check_visits(result['visits'])
    h,ag=unpack(result,g,arrays)
    for name,a in result.items():
        if not np.isfinite(a).all():raise AssertionError('nonfinite '+name)
    for name in ('histories','coarse','fine','counts','violations','additions','dep','sources'):
        if np.any(h[name]<0):raise AssertionError('negative '+name)
    if np.any(result['field']<0) or np.any(result['ledger']<0):raise AssertionError('negative field/ledger')
    source=h['sources'];dep=h['dep'];tot=h['totals']
    if np.any(np.abs(tot[:,0]-tot[:,1:].sum(axis=1)+tot[:,5])>1e-12*np.maximum(source,1e-30)):
        raise AssertionError('energy balance')
    if np.any(np.abs(h['histories'].sum(axis=1)-dep)>1e-12*np.maximum(source,1.)):
        raise AssertionError('local bins / deposition ledger')
    failures=int(h['counts'][:,5:7].sum())
    if expected_failure:
        if not failures or not np.any(tot[:,3]>0):raise AssertionError('expected failure/unresolved missing')
    elif failures or np.any(tot[:,3]) or np.any(h['violations']):
        raise AssertionError('capacity/maxsteps/violations')
    core.check_group_records(ag,h,len(source))
    return dict(status='pass',histories=len(source),failures=failures,max_balance=float(np.max(h['maxima'][:,0],initial=0.)))


def ledger(result,geometry):
    for j,(off,n) in enumerate(zip(geometry.offsets,geometry.sizes)):
        values=result['field'][off:off+n]
        operations=int(result['field_additions'][off:off+n].sum())
        bound=(2*operations+2*len(result['visits'])+len(values))*np.finfo(float).eps*np.abs(values).sum()
        if abs(values.sum()-result['ledger'][j])>bound:raise AssertionError('region ledger '+str(j))
    return True
