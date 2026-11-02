"""Five-material seed bundle. First four array pins remain the A2/A3 pins."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import catalog,contracts,physics_sets,therapy,therapy_bundle

NAMES=(*therapy_bundle.NAMES,'seed_agti')
PINS=Path(__file__).with_name('therapy_input_hashes.json')

def create(path,materials,inputs,densities,generators):
    path=Path(path).resolve();repo=Path(__file__).resolve().parents[2]
    if path==repo or repo in path.parents:raise ValueError('private bundle must stay outside repository')
    if len(materials)!=5 or np.shape(densities)!=(5,) or not np.isfinite(densities).all() or np.any(np.asarray(densities)<=0):raise ValueError('five materials/densities required')
    if (path/'manifest.json').exists():raise ValueError('bundle exists')
    therapy.space(path);path.mkdir(parents=True,exist_ok=True)
    fields=[];blobs={}
    for name,m in zip(NAMES,materials):
        cat,blob=catalog.p8_material(m,name,generators);key='p8_'+name
        for row in cat['fields']:row['blob']=key
        fields+=cat['fields'];blobs[key]=blob
    therapy.space(path,sum(v.nbytes for v in blobs.values())+sum(np.asarray(v).nbytes for v in inputs.values())+16*1024**2)
    np.savez(path/'tables.npz',**blobs);np.savez(path/'inputs.npz',**inputs,densities=np.asarray(densities,float))
    contracts.write(path/'catalog.json',dict(format=catalog.FORMAT,fields=fields))
    contracts.write(path/'settings.json',dict(material_names=list(NAMES),physics_set=physics_sets.declaration(therapy.NAME),generators=generators))
    manifest=dict(format='chatcarlo-g5-seed-bundle-v1',physics_set=physics_sets.declaration(therapy.NAME),files={k:therapy_bundle.sha(path/k) for k in sorted(therapy_bundle.MEMBERS)})
    contracts.write(path/'manifest.json',manifest);return manifest

def load(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text());settings=json.loads((path/'settings.json').read_text())
    if manifest.get('format')!='chatcarlo-g5-seed-bundle-v1' or set(manifest['files'])!=therapy_bundle.MEMBERS or settings['material_names']!=list(NAMES):raise ValueError('seed bundle members/material IDs')
    physics_sets.check(manifest['physics_set'],physics_sets.declaration(therapy.NAME));physics_sets.check(settings['physics_set'],manifest['physics_set'])
    for k,v in manifest['files'].items():
        if therapy_bundle.sha(path/k)!=v:raise ValueError('bundle SHA256 '+k)
    cat=json.loads((path/'catalog.json').read_text())
    with np.load(path/'tables.npz',allow_pickle=False) as f:blobs={k:f[k] for k in f.files}
    if set(blobs)!={'p8_'+n for n in NAMES}:raise ValueError('seed material blobs')
    catalog.verify(cat,blobs);restored=catalog.reconstruct(cat,blobs)
    if set(restored)!={n+'.'+f for n in NAMES for f in catalog.P8_DEFINITIONS}:raise ValueError('missing/extra seed material field')
    materials=[SimpleNamespace(**{f:restored[n+'.'+f] for f in catalog.P8_DEFINITIONS}) for n in NAMES]
    expected=[]
    for n,m in zip(NAMES,materials):
        c,_=catalog.p8_material(m,n,settings['generators'])
        for row in c['fields']:row['blob']='p8_'+n
        expected+=c['fields']
    catalog.verify(cat,blobs,dict(format=catalog.FORMAT,fields=expected))
    with np.load(path/'inputs.npz',allow_pickle=False) as f:inputs={k:f[k] for k in f.files}
    pins=json.loads(PINS.read_text())
    required={n+'.'+f for n in NAMES for f in catalog.P8_DEFINITIONS}|{'seed_input.densities'}
    if not required<=pins.keys():raise ValueError('seed pins missing: run validation-side a2b_export first')
    for key,pin in pins.items():
        if key.startswith('seed_input.'):value=inputs[key[11:]]
        elif key.startswith('input.'):value=inputs[key[6:]]
        else:
            name,f=key.split('.',1);value=getattr(materials[NAMES.index(name)],f)
        if catalog.digest(value)!=pin['sha256'] or list(np.shape(value))!=pin['shape']:raise ValueError('pinned input SHA256 '+key)
    return materials,inputs,manifest
