"""Local Python adapter, keeping the production NumPy path untouched."""
import tempfile
from pathlib import Path
import numpy as np
from . import core

def gpu_engine_compatible(scene,n_workers=1,dose_grid=False):
    raw=scene.raw;src=raw.get('source',{})
    if raw.get('domain','diagnostic')=='therapy_mv':return False,'G1 supports diagnostic photons only'
    if n_workers!=1:return False,'G1 requires n_workers=1'
    for g in raw.get('geometry',[]):
        if g.get('shape')!='box':return False,'G1 supports box only: '+str(g.get('shape'))
    for key in ('mas','ctdi_vol_mGy','rotation'):
        if src.get(key) is not None:return False,'G1 does not support '+key
    if src.get('heel_effect'):return False,'G1 does not support heel_effect'
    if src.get('field',{}).get('shape','rect') not in ('rect','cone','parallel'):return False,'G1 field must be rect, cone or parallel'
    if not ('kvp' in src or src.get('spectrum')):return False,'G1 requires kvp or spectrum'
    return True,''

def run(scene,n_histories,seed,batch_size,dose_grid,n_workers,track_uncertainty,detector,detector_roi,gpu_device,source_seed,grid_resolution_cm=5.0):
    from ..source import sample_source_photons
    from .bundle import create,load
    ok,reason=gpu_engine_compatible(scene,n_workers,dose_grid)
    if not ok:raise ValueError(reason)
    if seed is None:raise ValueError('G1 requires an explicit transport seed')
    if gpu_device not in ('cpu','cuda'):raise ValueError('gpu_device must be cpu or cuda')
    if detector is None and detector_roi is not None:raise ValueError('detector_roi requires detector')
    if n_histories<=0 or batch_size<=0:raise ValueError('n_histories and batch_size must be positive')
    if source_seed is None:source_seed=int(np.random.SeedSequence(seed).generate_state(1,np.uint64)[0])
    from .aggregate import aggregate
    from .approval import check_observation
    from .source_host import sample_source_photons as replay_source
    import time
    grid = None
    if dose_grid:
        from .condition import grid_spec, grid_for
        grid = grid_for(grid_spec(scene, detector, grid_resolution_cm), track_uncertainty)
    rng=np.random.default_rng(source_seed)
    replay_rng=np.random.default_rng(source_seed)
    elapsed=0.
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g1-') as temp:
        create(scene,detector,temp,g2_tally=dose_grid)
        a,src,spec,settings,manifest=load(temp)
        if dose_grid:
            from .g2_core import Core
            driver=Core(a,gpu_device,grid=grid,build_dir=Path(temp)/'build')
        else:
            driver=core.Core(a,gpu_device,build_dir=Path(temp)/'build')
        def batches():
            nonlocal elapsed
            for start in range(0,n_histories,batch_size):
                n=min(batch_size,n_histories-start);o,d,e=sample_source_photons(scene.raw['source'],n,rng)
                t0=time.perf_counter()
                r,observed=driver.run(o,d,e,seed=seed,start=start,fluorescence=settings['fluorescence'],observe=True)
                elapsed+=time.perf_counter()-t0
                eo,ed,ee=replay_source(src,n,replay_rng,spec)
                if dose_grid:
                    from .g2_core import expected_observation
                    from .g2_tables import config
                    expected = expected_observation(a, config(a, grid), eo, ed, ee, seed, start, settings['fluorescence'], gpu_device)
                else:
                    expected = core.expected_observation(a, eo, ed, ee, seed, start, settings['fluorescence'], gpu_device)
                check_observation(observed, expected, f'batch_{start//batch_size:06d}')
                yield n,r
        result=aggregate(batches(),settings['material_names'],detector,detector_roi,track_uncertainty,grid=grid)
    result.grid=grid
    result.gpu_metadata=dict(engine='gpu',gpu_device=gpu_device,seed=int(seed),source_seed=int(source_seed),n_workers=1,batch_size=batch_size,bundle=manifest)
    result.gpu_metadata['timing']=dict(elapsed_s=elapsed,new_elapsed_s=elapsed,reused_elapsed_s=0.,new_batches=(n_histories+batch_size-1)//batch_size,reused_batches=0)
    return result
