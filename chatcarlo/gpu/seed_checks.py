"""Seed history health; used before any batch is committed."""
import numpy as np

def history_health(r):
    finite=all(np.isfinite(v).all() for v in r.values())
    return bool(finite and np.all(r['histories']>=0) and not np.any(r['counts'][:,5:7]) and not np.any(r['violations']) and not np.any(r['brems'][:,2:4]) and not np.any(r['totals'][:,3]) and np.all(r['maxima'][:,0]<=1e-12) and np.all(r['p8_max'][:,0]<=1e-12))

def raw(d,i,ledger,counts,start):
    from .seed_core import unpack
    if np.shape(d)!=(len(d),37) or np.shape(i)!=(len(d),79) or np.shape(ledger)!=(len(d),6):raise ValueError('raw history record shape')
    if not np.isfinite(ledger).all() or np.any(ledger<0):raise ValueError('invalid ledger')
    h=unpack(d,i)
    if np.any(h['sources']<=0):raise ValueError('missing source energy record')
    if not history_health(h):raise ValueError('history health/capacity/unresolved/zero boundary')
    if counts.shape!=(len(d),4) or not np.array_equal(counts[:,2],np.arange(start,start+len(d))) or not np.all(counts[:,3]==1):raise ValueError('history duplicate/missing')
    if np.any(counts[:,:2]<0) or np.any(counts[:,1]):raise ValueError('source missed body')
    if np.any(abs(ledger.sum(1)-h['dep'])>1e-12*h['sources']):raise ValueError('six-site ledger')
    return h


def local(work):
    from . import seed_geometry_checks as c,seed_validation as v
    from .seed_core import CPU
    common=dict(densities=[1.]*5,grids=(c.SMALL,c.SMALL))
    e=CPU(work,**common,cylinders=((2,2,2,.05,.25,4),))
    actual=CPU(work,**common)
    return dict(V0=c.v0(e),V1a=c.v1a(e),V2a=c.v2a(e),V3=c.v3(e),V6=c.v6(actual),G3_grazing=c.grazing_seed(actual),V5=v.v5(work),V6b=v.v6b())


def compare_histories(cpu,gpu):
    # Same criterion as P9c: source equality and paired RNG-call differences.
    from .seed_comparison import c2_calls
    calls=c2_calls(dict(diag_used=cpu['diagnostic_count'],diagnostic=cpu['diagnostic']),dict(diag_used=gpu['diagnostic_count'],diagnostic=gpu['diagnostic']))
    checks=dict(cpu_health=history_health(cpu),gpu_health=history_health(gpu),source=bool(np.array_equal(cpu['sources'],gpu['sources']) and np.array_equal(cpu['nspec_index'],gpu['nspec_index'])),calls=calls)
    return dict(status='pass' if checks['cpu_health'] and checks['gpu_health'] and checks['source'] and calls['status']=='pass' else 'fail',checks=checks)
