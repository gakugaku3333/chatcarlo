"""Sealed outer batches, resumable only after the verification build passed."""
import json
from pathlib import Path
import numpy as np
from . import contracts, physics_sets, therapy_health, therapy_layout
from .condition import sha256
from .therapy_bundle import sha
from .therapy_observation import descriptor


def identity(cond,batch):
    count=cond['histories']//100
    return dict(condition_sha256=sha256(cond),physics_set=cond['physics_set'],batch=batch,start=batch*count,count=count)


def expected(cond,arrays,fg,batch):
    k=identity(cond,batch)
    return descriptor(arrays,fg,(5.,5.),100.,cond['seed'],k['start'],k['count'],cond['physics_set'])


def gate_key(cond):
    # No gate result can be carried across seeds, bundles, or source changes.
    return dict(condition_sha256=sha256(cond),physics_set=cond['physics_set'])


def read_batch(out,cond,batch,arrays,fg):
    out=Path(out);p=out/f'batch_{batch:03d}.npz';rp=p.with_suffix('.json')
    if not p.exists() and not rp.exists():return None
    if not p.exists() or not rp.exists():raise ValueError('incomplete batch/receipt')
    r=json.loads(rp.read_text())
    if r.get('identity')!=identity(cond,batch) or r.get('sha256')!=sha(p):raise ValueError('batch identity/SHA256 mismatch')
    contracts.receipt(r,cond,r.get('observed',{}).get('physics_set'))
    contracts.check_observation(r.get('observed'),expected(cond,arrays,fg,batch),p.name)
    with np.load(p,allow_pickle=False) as z:values={k:z[k] for k in z.files}
    therapy_health.check(values,arrays,fg)
    return values


def preflight(out,cond,arrays,fg):
    out=Path(out)
    expected_names={f'batch_{b:03d}{s}' for b in range(100) for s in ('.npz','.json')}
    if any(p.name not in expected_names for p in out.glob('batch_*')):raise ValueError('unexpected/duplicate batch name or range')
    if any(out.glob('batch_*')):
        g=json.loads((out/'gates.json').read_text()) if (out/'gates.json').exists() else {}
        if g!=dict(gate_key(cond),status='pass'):raise ValueError('resume requires matching gate record')
    return [read_batch(out,cond,b,arrays,fg) for b in range(100)]


def run(out,cond,arrays,fg,driver,checks):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    existing=preflight(out,cond,arrays,fg)
    if not any(x is not None for x in existing):
        checks()
        contracts.write(out/'gates.json',dict(gate_key(cond),status='pass'))
    for b,old in enumerate(existing):
        if old is not None:continue
        key=identity(cond,b)
        r=driver(key['count'],cond['seed'],key['start'])
        observed=driver.observation()
        contracts.check_observation(observed,expected(cond,arrays,fg,b),'native batch')
        physics_sets.check(observed['physics_set'],cond['physics_set'])
        therapy_health.check(r,arrays,fg)
        p=out/f'batch_{b:03d}.npz';np.savez_compressed(p,**r)
        contracts.write(p.with_suffix('.json'),dict(identity=key,sha256=sha(p),observed=observed,
            condition_sha256=sha256(cond),physics_set=cond['physics_set'],quantities={'field':'deposited-energy', 'unit':'MeV','normalization':'raw sum'}))
    return collect(out,cond,arrays,fg)


def collect(out,cond,arrays,fg):
    out=Path(out);batches=preflight(out,cond,arrays,fg)
    if any(r is None for r in batches):raise ValueError('missing completed batch')
    field=np.stack([r['field'] for r in batches]);meta=therapy_layout.metadata(cond['histories'])
    p=out/'dose.npz'
    np.savez_compressed(p,field_MeV=field,absorbed_dose_Gy=therapy_layout.dose(field.sum(axis=0),cond['histories']),
                        metadata=json.dumps(meta,sort_keys=True),physics_set=cond['physics_set']['name'])
    record=dict(condition_sha256=sha256(cond),physics_set=cond['physics_set'],observed=cond['physics_set'],metadata=meta,sha256=sha(p))
    contracts.write(out/'receipt.json',record)
    # Read back both final artifacts, including quantity, unit and normalization.
    saved=json.loads((out/'receipt.json').read_text());contracts.receipt(saved,cond,saved['observed'])
    with np.load(p,allow_pickle=False) as z:
        if json.loads(str(z['metadata']))!=saved['metadata'] or str(z['physics_set'])!=cond['physics_set']['name']:raise ValueError('output metadata mismatch')
        if not np.array_equal(z['absorbed_dose_Gy'],therapy_layout.dose(z['field_MeV'].sum(axis=0),cond['histories'])):raise ValueError('output dose conversion mismatch')
    return saved
