"""Spawn-importable kernels for P3u; no worker writes files."""
from collections import namedtuple
import os
import platform
import time
import llvmlite
import numba
import numpy as np
import scipy

Kernel = namedtuple('Kernel','run keys healthy fingerprint check_params')


def versions():
    return dict(python=platform.python_version(),numpy=np.__version__,
                numba=numba.__version__,llvmlite=llvmlite.__version__,
                scipy=scipy.__version__,machine=platform.machine())


def slab_params(params,n):
    if not isinstance(params,dict) or set(params)!=set(('E','backend','brems_mode','spectrum')):
        raise ValueError('slab params must be E, backend, brems_mode, spectrum')
    from .p2_common import TABLES
    if (params['backend'] not in TABLES or params['brems_mode'] not in (1,2) or
        params['spectrum'] not in (None,'mohan') or n%100):
        raise ValueError('invalid slab params or batch size')
    try:E=float(params['E'])
    except (TypeError,ValueError) as exc:raise ValueError('invalid E') from exc
    if not np.isfinite(E) or not .01<=E<=10.:raise ValueError('E outside 10 keV–10 MeV')


def spectrum(params):
    if params['spectrum'] is None:return None,None
    from .run_p3a import load_spectrum
    edges,prob,cum=load_spectrum()
    return edges,cum


def slab_run(n,seed,params):
    from . import run_p3s as s, run_p3r as r, photon_transport_p3t as t
    edges,cum=spectrum(params)
    values=t.run_coupled(float(params['E']),n,seed,.05,.025,30.,*s.inputs(params['backend']),
        annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
        brems_mode=params['brems_mode'],score_kerma=False,score_pair=False,**s.rule_kwargs('own'))
    if len(values)!=15:raise RuntimeError('P3t must return 15 arrays')
    return {k:np.asarray(v.sum(axis=0) if k in r.KEYS[:5] else v)
            for k,v in zip(r.KEYS,values)}


def slab_healthy(batch):
    from .run_p3r import healthy
    return healthy(batch)


def slab_fingerprint(params):
    from . import run_p3a as a, run_p3s as s, bremsstrahlung_p3r as b
    edges,cum=spectrum(params)
    return dict(tables=a.table_hashes(params['backend'],s.inputs(params['backend'])),
                spectrum=a.spec_config(edges,cum,a.SPECTRUM_SHA),
                brems_arrays=[a.array_hash(x) for x in b.tables()],**versions())


def fake_params(params,n):
    allowed={'fingerprint','nan_seed','unhealthy_seed','exception_seed','exit_seed',
             'sleep','sleep_seed','worker_fingerprint'}
    if not isinstance(params,dict) or set(params)-allowed:raise ValueError('unknown fake params')
    if 'sleep' in params and (not np.isfinite(params['sleep']) or params['sleep']<0):
        raise ValueError('invalid sleep')


def fake_run(n,seed,params):
    # The environment switch lets a test remove an interruption's cause while
    # keeping the immutable run config. It has no effect on normal fake output.
    faults=os.environ.get('CHATCARLO_P3U_FAKE_FAULTS','1')=='1'
    if faults and seed==params.get('exception_seed'):raise RuntimeError('fake kernel exception')
    if faults and seed==params.get('exit_seed'):os._exit(1)
    if 'sleep' in params and seed==params.get('sleep_seed',seed):time.sleep(params['sleep'])
    rng=np.random.default_rng(seed)
    value=rng.normal(size=(2,3))
    violations=np.zeros(2,dtype=np.int64)
    counts=rng.integers(0,n+1,size=4,dtype=np.int64)
    if faults and seed==params.get('nan_seed'):value[0,0]=np.nan
    if faults and seed==params.get('unhealthy_seed'):violations[0]=1
    return dict(value=value,violations=violations,counts=counts)


def fake_healthy(batch):
    return not np.any(batch['violations'])


def fake_fingerprint(params):
    return dict(input=params.get('fingerprint'),**versions())


# Importing KEYS reads no external GS/photon cache and generates no tables.
from .run_p3r import KEYS
KERNELS={'p3t_slab':Kernel(slab_run,KEYS,slab_healthy,slab_fingerprint,slab_params),
         'fake':Kernel(fake_run,('value','violations','counts'),fake_healthy,fake_fingerprint,fake_params)}

# Field adapter stays light; transport and tables are imported inside its calls.
from . import batch_kernels_p3b as field
KERNELS['p3b_field']=Kernel(field.run,field.KEYS,field.healthy,field.fingerprint,field.check_params)

# P8e imports only code; material files are read/verified by its adapter calls.
from . import photon_transport_p8e as p8e
KERNELS['p8e_stack']=Kernel(p8e.batch_run,p8e.KEYS,p8e.healthy,p8e.fingerprint,p8e.check_params)
