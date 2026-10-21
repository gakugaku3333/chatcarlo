"""Hash-verified diagnostic bundles. Physics is imported only during creation."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from . import core

HERE=Path(__file__).resolve().parent
RUNTIME=(*core.SOURCES,'__init__.py','core.py','source_host.py','bundle.py','colab_driver.py','validation.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(path):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('format')!='chatcarlo-g1-v1':raise ValueError('unknown bundle format')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or not isinstance(digest,str):raise ValueError('invalid bundle member')
        if sha(path/name)!=digest:raise ValueError('bundle SHA256 mismatch: '+name)
    required={'tables.npz','geometry.npz','spectrum.npz','source.json','settings.json',*RUNTIME}
    if set(manifest['files'])!=required:raise ValueError('incomplete/unexpected bundle members')
    return manifest

def create(scene,plane,path):
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
    geom=Geometry(geoms,detector_plane=plane)
    det=np.array([]) if plane is None else np.r_[plane.center_cm,plane.normal,plane.u_axis,plane.v_axis,plane.size_cm,plane.shape]
    np.savez(path/'geometry.npz',centers=np.array([g['center'] for g in geoms],float).reshape(-1,3),halves=np.array([g['size_cm'] for g in geoms],float).reshape(-1,3)/2,materials=np.array([names.index(g['material']) for g in geoms],np.int64),background=np.array(names.index(background)),bbox_min=geom.bbox_min,bbox_max=geom.bbox_max,world=np.r_[(geom.bbox_min+geom.bbox_max)/2,(geom.bbox_max-geom.bbox_min)/2],detector=det)
    if src.get('spectrum'):
        e=np.array([s['energy_keV'] for s in src['spectrum']],float);w=np.array([s['weight'] for s in src['spectrum']],float);w=w/w.sum()
    else:e,w=_default_spectrum(src['kvp'],src.get('filtration_mm_al',2.5),src.get('anode_angle_deg',12.0))
    np.savez(path/'spectrum.npz',energy=e,weight=w)
    (path/'source.json').write_text(json.dumps(src,sort_keys=True)+'\n')
    settings=dict(material_names=names,fluorescence=raw.get('physics',{}).get('fluorescence',True),scene_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),provenance=dict(cross_sections='xraylib (kernel.bake_scene_materials)',spectrum='explicit' if src.get('spectrum') else 'spectrum._default_spectrum'))
    (path/'settings.json').write_text(json.dumps(settings,sort_keys=True)+'\n')
    for name in RUNTIME:shutil.copyfile(HERE/name,path/name)
    manifest=dict(format='chatcarlo-g1-v1',files={name:sha(path/name) for name in sorted({'tables.npz','geometry.npz','spectrum.npz','source.json','settings.json',*RUNTIME})})
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');return manifest

def load(path):
    path=Path(path);manifest=verify(path);a={}
    for f in ('tables.npz','geometry.npz'):
        with np.load(path/f,allow_pickle=False) as z:a.update({k:z[k] for k in z.files})
    with np.load(path/'spectrum.npz',allow_pickle=False) as z:spectrum=(z['energy'],z['weight'])
    return a,json.loads((path/'source.json').read_text()),spectrum,json.loads((path/'settings.json').read_text()),manifest
