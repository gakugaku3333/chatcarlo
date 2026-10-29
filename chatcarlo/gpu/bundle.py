"""Hash-verified diagnostic bundles. Physics is imported only during creation."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from . import core, catalog, physics_sets

HERE=Path(__file__).resolve().parent
RUNTIME=(*core.SOURCES,'__init__.py','catalog.py','physics_sets.py','core.py','source_host.py','bundle.py','colab_driver.py','validation.py','g2_core.py','g2_tables.py','run_gates.py','run_egs5_60kev.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('format')!='chatcarlo-g5-v1':raise ValueError('unknown bundle format')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or not isinstance(digest,str):raise ValueError('invalid bundle member')
        if sha(path/name)!=digest:raise ValueError('bundle SHA256 mismatch: '+name)
    required={'catalog.json','tables.npz','geometry.npz','spectrum.npz','source.json','settings.json',*RUNTIME}
    if 'g2_tables.npz' in manifest['files']:required.add('g2_tables.npz')
    if set(manifest['files'])!=required:raise ValueError('incomplete/unexpected bundle members')
    settings=json.loads((path/'settings.json').read_text())
    physics_sets.check(settings['physics_set'],physics_sets.declaration())
    a={}
    for filename in ('tables.npz','g2_tables.npz'):
        if filename not in manifest['files']:continue
        with np.load(path/filename,allow_pickle=False) as z:
            allowed={'g2_table','g2_sizes'}|catalog.G2_METADATA if filename=='g2_tables.npz' else set(z.files)
            if set(z.files)-allowed:raise ValueError('unexpected G2 metadata')
            a.update({k:z[k] for k in z.files if k not in catalog.G2_METADATA})
    cat=json.loads((path/'catalog.json').read_text())
    generators=settings['catalog_generators']
    for gen in generators.values():
        local={'chatcarlo/kernel.py':None,'chatcarlo/gpu/g2_tables.py':'g2_tables.py'}.get(gen['path'])
        if local is not None and gen['sha256']!=sha(path/local):raise ValueError('catalog generator SHA256 mismatch')
    expected=catalog.diagnostic(a,generators,settings['material_names'])
    catalog.verify(cat,dict(tables=a['data'],g2_tables=a.get('g2_table',np.array([])),g2_sizes=a.get('g2_sizes',np.array([],dtype=np.int64))),expected)
    return manifest

def create(scene,plane,path,*,bbox_margin_cm=50.0,g2_tally=False):
    from ..kernel import bake_scene_materials
    from ..geometry import Geometry
    from ..spectrum import _default_spectrum
    from .engine import gpu_engine_compatible
    ok,reason=gpu_engine_compatible(scene,1,False)
    if not ok:raise ValueError(reason)
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    raw=scene.raw;src=raw['source'];geoms=raw['geometry'];background='air'
    names=list(dict.fromkeys([background]+[g['material'] for g in geoms]));t=bake_scene_materials(names)
    arrays=core.pack(t);np.savez(path/'tables.npz',**arrays)
    geom=Geometry(geoms,detector_plane=plane,bbox_margin_cm=bbox_margin_cm)
    det=np.array([]) if plane is None else np.r_[plane.center_cm,plane.normal,plane.u_axis,plane.v_axis,plane.size_cm,plane.shape]
    np.savez(path/'geometry.npz',centers=np.array([g['center'] for g in geoms],float).reshape(-1,3),halves=np.array([g['size_cm'] for g in geoms],float).reshape(-1,3)/2,materials=np.array([names.index(g['material']) for g in geoms],np.int64),background=np.array(names.index(background)),bbox_min=geom.bbox_min,bbox_max=geom.bbox_max,world=np.r_[(geom.bbox_min+geom.bbox_max)/2,(geom.bbox_max-geom.bbox_min)/2],detector=det)
    if src.get('spectrum'):
        e=np.array([s['energy_keV'] for s in src['spectrum']],float);w=np.array([s['weight'] for s in src['spectrum']],float);w=w/w.sum()
    else:e,w=_default_spectrum(src['kvp'],src.get('filtration_mm_al',2.5),src.get('anode_angle_deg',12.0))
    np.savez(path/'spectrum.npz',energy=e,weight=w)
    (path/'source.json').write_text(json.dumps(src,sort_keys=True)+'\n')
    settings=dict(physics_set=physics_sets.declaration(),material_names=names,fluorescence=raw.get('physics',{}).get('fluorescence',True),scene_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),provenance=dict(cross_sections='xraylib (kernel.bake_scene_materials)',spectrum='explicit' if src.get('spectrum') else 'spectrum._default_spectrum'))
    settings['catalog_generators']={'diagnostic':catalog.generator('chatcarlo/kernel.py',HERE.parents[1])}
    if g2_tally:settings['catalog_generators']['g2']=catalog.generator('chatcarlo/gpu/g2_tables.py',HERE.parents[1])
    (path/'settings.json').write_text(json.dumps(settings,sort_keys=True)+'\n')
    if g2_tally:
        from .g2_tables import pack
        np.savez(path/'g2_tables.npz',**pack(names))
    if g2_tally:
        with np.load(path/'g2_tables.npz') as z:arrays.update({k:z[k] for k in z.files})
    cat=catalog.diagnostic(arrays,settings['catalog_generators'],names)
    (path/'catalog.json').write_text(json.dumps(cat,indent=2,sort_keys=True)+'\n')
    for name in RUNTIME:
        source = HERE/name
        if name in ('run_gates.py','run_egs5_60kev.py'):
            source = HERE.parents[1]/'docs/validation/gpu/g2'/name
        shutil.copyfile(source,path/name)
    manifest=dict(format='chatcarlo-g5-v1',files={name:sha(path/name) for name in sorted({'catalog.json','tables.npz','geometry.npz','spectrum.npz','source.json','settings.json',*RUNTIME})})
    if g2_tally:manifest['files']['g2_tables.npz']=sha(path/'g2_tables.npz')
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return manifest

def load(path):
    path=Path(path);manifest=verify(path);a={}
    for f in ('tables.npz','geometry.npz', *(['g2_tables.npz'] if 'g2_tables.npz' in manifest['files'] else [])):
        with np.load(path/f,allow_pickle=False) as z:a.update({k:z[k] for k in z.files if f != 'g2_tables.npz' or k.startswith('g2_')})
    with np.load(path/'spectrum.npz',allow_pickle=False) as z:spectrum=(z['energy'],z['weight'])
    return a,json.loads((path/'source.json').read_text()),spectrum,json.loads((path/'settings.json').read_text()),manifest
