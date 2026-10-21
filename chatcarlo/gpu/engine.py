"""Local Python adapter, keeping the production NumPy path untouched."""
import tempfile
from pathlib import Path
import numpy as np
from . import core

def gpu_engine_compatible(scene,n_workers=1,dose_grid=False):
    raw=scene.raw;src=raw.get('source',{})
    if raw.get('domain','diagnostic')=='therapy_mv':return False,'G1 supports diagnostic photons only'
    if n_workers!=1:return False,'G1 requires n_workers=1'
    if dose_grid:return False,'G1 does not support voxel dose_grid tally'
    for g in raw.get('geometry',[]):
        if g.get('shape')!='box':return False,'G1 supports box only: '+str(g.get('shape'))
    for key in ('mas','ctdi_vol_mGy','rotation'):
        if src.get(key) is not None:return False,'G1 does not support '+key
    if src.get('heel_effect'):return False,'G1 does not support heel_effect'
    if src.get('field',{}).get('shape','rect') not in ('rect','cone','parallel'):return False,'G1 field must be rect, cone or parallel'
    if not ('kvp' in src or src.get('spectrum')):return False,'G1 requires kvp or spectrum'
    return True,''

def run(scene,n_histories,seed,batch_size,dose_grid,n_workers,track_uncertainty,detector,detector_roi,gpu_device,source_seed):
    from ..transport import TransportResult
    from ..detector import DetectorTally
    from ..source import sample_source_photons
    from ..tally import ScalarMoments
    from .bundle import create,load
    ok,reason=gpu_engine_compatible(scene,n_workers,dose_grid)
    if not ok:raise ValueError(reason)
    if seed is None:raise ValueError('G1 requires an explicit transport seed')
    if gpu_device not in ('cpu','cuda'):raise ValueError('gpu_device must be cpu or cuda')
    if detector is None and detector_roi is not None:raise ValueError('detector_roi requires detector')
    if n_histories<=0 or batch_size<=0:raise ValueError('n_histories and batch_size must be positive')
    if source_seed is None:source_seed=int(np.random.SeedSequence(seed).generate_state(1,np.uint64)[0])
    rng=np.random.default_rng(source_seed);tally=DetectorTally(detector,track_uncertainty,detector_roi) if detector is not None else None
    moments=ScalarMoments() if track_uncertainty else None
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g1-') as temp:
        create(scene,detector,temp);a,src,spec,settings,manifest=load(temp);driver=core.Core(a,gpu_device,build_dir=Path(temp)/'build')
        names=settings['material_names'];deposits={name:0. for name in names};endings=np.zeros(3);scatter=fluor=0.
        for start in range(0,n_histories,batch_size):
            n=min(batch_size,n_histories-start);o,d,e=sample_source_photons(scene.raw['source'],n,rng)
            r=driver.run(o,d,e,seed=seed,start=start,fluorescence=settings['fluorescence'])
            batch={name:float(v) for name,v in zip(names,r['deposits'])}
            for name,value in batch.items():deposits[name]+=value
            if moments is not None:moments.add_batch(batch,n)
            endings+=r['endings'];scatter+=r['scatter_sum'];fluor+=r['n_fluorescence']
            if tally is not None:
                tally.category_fluence+=r['images'][:4];tally.photon_count+=r['images'][4];tally.energy_sum_keV+=r['images'][5];tally.energy_sum2_keV2+=r['images'][6];tally.end_batch(n)
    result=TransportResult(n_histories,{k:v/1000 for k,v in deposits.items()},float(endings[0]/n_histories),float(endings[1]/n_histories),float(scatter/n_histories),n_fluorescence=int(fluor),n_batches=moments.n_batches if moments is not None else 0,energy_deposited_sem_MeV={k:v/1000 for k,v in moments.standard_errors().items()} if moments is not None else {},energy_deposited_rel_err=moments.relative_errors() if moments is not None else {},detector=tally)
    result.gpu_metadata=dict(engine='gpu',gpu_device=gpu_device,seed=int(seed),source_seed=int(source_seed),n_workers=1,batch_size=batch_size,bundle=manifest)
    return result
