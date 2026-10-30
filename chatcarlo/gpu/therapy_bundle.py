"""Hash-verified G5 therapy bundles, independent of diagnostic bundle loading."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import catalog, physics_sets, therapy

NAMES=('water','lung','bone','lead')
MEMBERS={'catalog.json','tables.npz','inputs.npz','settings.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def create(path,materials,arrays,generators):
    """Contains private spectrum inputs: caller must choose an external cache."""
    path=Path(path).resolve()
    repo=Path(__file__).resolve().parents[2]
    if path==repo or repo in path.parents:
        raise ValueError('therapy bundle with private spectra must stay outside repository')
    cat,blobs=catalog.therapy(materials,generators)
    therapy.space(path,sum(a.nbytes for a in blobs.values())+16*1024**2)
    path.mkdir(parents=True,exist_ok=True)
    if (path/'manifest.json').exists():raise ValueError('bundle already exists')
    np.savez(path/'tables.npz',**blobs)
    np.savez(path/'inputs.npz',**arrays)
    settings=dict(physics_set=physics_sets.declaration(therapy.NAME),material_names=list(NAMES),generators=generators,
                  material_hashes={n:catalog.digest(blobs['p8_'+n]) for n in NAMES})
    (path/'settings.json').write_text(json.dumps(settings,sort_keys=True)+'\n')
    (path/'catalog.json').write_text(json.dumps(cat,sort_keys=True)+'\n')
    manifest=dict(format='chatcarlo-g5-v1',physics_set=settings['physics_set'],files={n:sha(path/n) for n in sorted(MEMBERS)})
    (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
    return manifest


def verify(path,*,expected_declaration=None):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m.get('format')!='chatcarlo-g5-v1' or set(m['files'])!=MEMBERS:raise ValueError('invalid therapy bundle format/members')
    want=physics_sets.declaration(therapy.NAME) if expected_declaration is None else expected_declaration
    physics_sets.check(m['physics_set'],want)
    for name,digest in m['files'].items():
        if sha(path/name)!=digest:raise ValueError('bundle SHA256 mismatch: '+name)
    settings=json.loads((path/'settings.json').read_text())
    physics_sets.check(settings['physics_set'],want)
    if settings['material_names']!=list(NAMES):raise ValueError('material IDs/order')
    cat=json.loads((path/'catalog.json').read_text())
    with np.load(path/'tables.npz',allow_pickle=False) as z:blobs={n:z[n] for n in z.files}
    if set(blobs)!={'p8_'+n for n in NAMES}:raise ValueError('missing material blob')
    catalog.verify(cat,blobs)
    restored=catalog.reconstruct(cat,blobs)
    materials=[]
    for name in NAMES:
        if catalog.digest(blobs['p8_'+name])!=settings['material_hashes'][name]:raise ValueError('material swap: '+name)
        try:materials.append(SimpleNamespace(**{f:restored[name+'.'+f] for f in catalog.P8_DEFINITIONS}))
        except KeyError as exc:raise ValueError('missing material field') from exc
    expected,_=catalog.therapy(materials,settings['generators'])
    catalog.verify(cat,blobs,expected)
    return m,materials,settings


def load(path,*,expected_declaration=None):
    m,materials,settings=verify(path,expected_declaration=expected_declaration)
    with np.load(Path(path)/'inputs.npz',allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
    return materials,arrays,settings,m
