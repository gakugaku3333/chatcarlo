"""Read sealed therapy inputs; the A2 content hashes pin tables and private spectra.

Only digests are distributed. Exporting the bundle is a validation-side action.
"""
import json
from pathlib import Path
import numpy as np
from . import therapy_bundle, catalog
from .therapy import FieldGeometry

DEFAULT_BUNDLE=Path.home()/'.cache/chatcarlo/g5/a3/therapy_bundle'
PINS=Path(__file__).with_name('therapy_input_hashes.json')


def load(path, spectrum):
    materials,inputs,settings,manifest=therapy_bundle.load(path)
    pins=json.loads(PINS.read_text())
    for name,expected in pins.items():
        if name.startswith(("seed_agti.","seed_input.")):continue
        if name.startswith('input.'):
            actual=inputs[name[6:]]
        else:
            material,field=name.split('.',1)
            actual=getattr(materials[therapy_bundle.NAMES.index(material)],field)
        if catalog.digest(actual)!=expected['sha256'] or list(np.shape(actual))!=expected['shape']:raise ValueError('pinned input SHA256 mismatch: '+name)
    fg=FieldGeometry()
    for name in ('lo','hi','origins','widths','axes','sizes','offsets'):
        getattr(fg,name)[:]=inputs['field_'+name]
    # P7 geometry and probabilities come from the sealed bundle, never a prototype.
    if isinstance(spectrum,dict):edges=np.asarray(spectrum['edges'],float);cum=np.asarray(spectrum['cum'],float)
    else:edges=inputs[spectrum+'_edges'];cum=inputs[spectrum+'_cum']
    bins=np.arange(301,dtype=float)*.1
    arrays=dict(faces=np.array([0.,30.]),rho=np.array([1.]),bins=bins,
                strength=inputs['strength'],edges=edges,cum=cum,ids=np.array([0],np.int64),
                owner=np.zeros(300,np.int64),constants=inputs['constants'],
                sizes=np.array([1,300,150,600,len(cum)],np.int64),mass_g_cm2=np.diff(bins))
    arrays['_table_hashes']={str(mid)+'.'+k:catalog.digest(getattr(m,k)) for mid,m in enumerate(materials) for k in catalog.P8_DEFINITIONS}
    return materials,arrays,fg,inputs['field_volumes'],manifest
