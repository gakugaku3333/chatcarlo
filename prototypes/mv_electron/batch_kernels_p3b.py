"""Lazy P3b field adapter for the unchanged P3u spawn runner."""
import numpy as np
from .run_p3r import KEYS as SLAB_KEYS

REGION_KEYS=('axis_coarse','axis_fine','slab_coarse','profile','profile_fine','region_ledger')
KEYS=SLAB_KEYS+REGION_KEYS
SOURCES=('photon_transport_p3b.py','scoring_p3b.py','batch_kernels_p3b.py')
PARAMS=('E','backend','brems_mode','spectrum','field_half','ssd','dz','dz_fine','regions')


def check_params(params,n):
    from .batch_kernels_p3u import slab_params
    from .photon_transport_p3b import expand_regions
    if not isinstance(params,dict) or set(params)!=set(PARAMS):raise ValueError('invalid field params keys')
    slab_params({key:params[key] for key in ('E','backend','brems_mode','spectrum')},n)
    half=params['field_half'];ssd=params['ssd']
    try:valid=np.all(np.isfinite([half,ssd])) and half>0 and ssd>0
    except (TypeError,ValueError) as exc:raise ValueError('invalid field source') from exc
    if not valid:raise ValueError('invalid field source')
    if not isinstance(params['regions'],list):raise ValueError('regions must be numeric list')
    expand_regions(params['regions'],30.,params['dz'],params['dz_fine'])


def reduce(values):
    return {key:np.asarray(value.sum(axis=0) if key in SLAB_KEYS[:5]+REGION_KEYS[:5] else value)
            for key,value in zip(KEYS,values[:21])}


def run(n,seed,params):
    from . import run_p3s as s, photon_transport_p3b as t
    from .batch_kernels_p3u import spectrum
    edges,cum=spectrum(params)
    values=t.run_coupled(float(params['E']),n,seed,.05,.025,30.,*s.inputs(params['backend']),
        annihilate_in_flight=True,spectrum_edges=edges,spectrum_cum=cum,
        brems_mode=params['brems_mode'],score_kerma=False,score_pair=False,
        field_half=params['field_half'],ssd=params['ssd'],dz=params['dz'],
        dz_fine=params['dz_fine'],regions=params['regions'],**s.rule_kwargs('own'))
    return reduce(values)


def healthy(batch):
    from .batch_kernels_p3u import slab_healthy
    return slab_healthy(batch) and not np.any(batch['region_ledger'])


def source_hashes():
    from . import run_p3a as a
    return {name:a.sha(a.CODE/name) for name in SOURCES}


def fingerprint(params):
    from .batch_kernels_p3u import slab_fingerprint
    return dict(slab_fingerprint(params),field_sources=source_hashes())

