"""Sealed 50-batch, equal-integer-door seed conditions; no entry switch yet."""
from pathlib import Path
from . import contracts,physics_sets,therapy,seed_geometry,seed_analysis,seed_core,seed_scene
from .condition import sha256
from .therapy_bundle import sha
BATCHES=50
PER_BATCH_PER_FIELD=71429
SEEDS=(9502001,9502002)


def condition(scene,manifest,materials,inputs):
    seed_scene.validate(scene)
    cond=dict(format='chatcarlo-g5-seed-entry-v1',scene=scene,physics_set=physics_sets.declaration(therapy.NAME, specialization=physics_sets.ELECTRON),
        sources=seed_core.source_hashes(),bundle_manifest_sha256=sha256(manifest),
        masks_sha256=sha(seed_analysis.MASK_PATH),batches=BATCHES,per_batch_per_field=PER_BATCH_PER_FIELD,
        histories_per_field=BATCHES*PER_BATCH_PER_FIELD,histories_per_side=7*BATCHES*PER_BATCH_PER_FIELD,
        angles=list(seed_geometry.ANGLES),grids=[g.record() for g in (seed_geometry.FINE,seed_geometry.COARSE)],
        seed=scene['run']['seed'],engine='colab_t4',spectrum='mohan10')
    from .seed_run import expected
    cond['native_templates']=[dict(field=f,with_seeds=s,observed=expected(cond,materials,inputs,0,f,s)) for f in range(7) for s in (True,False)]
    return cond


def identity(cond,batch):
    if type(batch) is not int or not 0<=batch<cond['batches'] or cond['batches']!=50:raise ValueError('batch range/count')
    n=cond['per_batch_per_field']
    return dict(condition_sha256=sha256(cond),batch=batch,start=batch*n,count=n,accepted_per_side=7*n)


def verify(out):
    import json
    from .seed_bundle import load
    out=Path(out);g=contracts.verify(out,('condition.json','preview_snapshot.json','preview.html','bundle/manifest.json'))
    c=json.loads((out/'condition.json').read_text());materials,inputs,manifest=load(out/'bundle');seed_analysis.load_masks()
    if c!=g['condition'] or condition(c['scene'],manifest,materials,inputs)!=c:raise ValueError('seed condition/bundle/source mismatch')
    return g
