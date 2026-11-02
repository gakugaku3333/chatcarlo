"""Claude-side B0 export. Explicitly verifies frozen P9a archives before pinning."""
import argparse
import json
from pathlib import Path
import numpy as np
from chatcarlo.gpu import seed_bundle,catalog,contracts,therapy_bundle,therapy_inputs


def export(a3, out, pins=seed_bundle.PINS):
    from prototypes.mv_electron import p9a_data
    # load_bundles verifies the P9a source seals, including the frozen seed archive.
    materials,identity=p9a_data.load_bundles()
    original=Path(__file__).resolve().parents[4]/'docs/validation/mv/p9a/bundle.json'
    frozen=json.loads(original.read_text())
    for name,digest in frozen.items():
        if therapy_bundle.sha(Path(name))!=digest:raise ValueError('P9a frozen archive SHA256')
    therapy_inputs.load(a3,'mohan10')
    _,inputs,settings,_=therapy_bundle.load(a3)
    seed_bundle.create(out,materials,inputs,identity['densities'],settings['generators'])
    saved=json.loads(Path(pins).read_text())
    # Pin the values as restored from the bundle (catalog dtypes), as the existing A2/A3 pins are.
    cat=json.loads((Path(out)/'catalog.json').read_text())
    with np.load(Path(out)/'tables.npz',allow_pickle=False) as f:blobs={k:f[k] for k in f.files}
    restored=catalog.reconstruct(cat,blobs)
    for f in catalog.P8_DEFINITIONS:
        v=restored['seed_agti.'+f];saved['seed_agti.'+f]=dict(sha256=catalog.digest(v),shape=list(np.shape(v)))
    v=np.asarray(identity['densities'],float);saved['seed_input.densities']=dict(sha256=catalog.digest(v),shape=list(v.shape))
    Path(pins).write_text(json.dumps(saved,indent=2,sort_keys=True)+'\n')
    return seed_bundle.load(out)[2]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('a3_bundle',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();export(a.a3_bundle,a.out)
