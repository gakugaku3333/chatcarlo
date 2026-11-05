"""D2 deterministic rebuild: existing sealed XCOM only; never runs XCOM or MC.

.venv/bin/python -m docs.validation.gpu.g5.a4u_d2_update --out /tmp/chatcarlo-a4u-d2
The output bundles contain private spectra and must stay outside the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from chatcarlo.gpu import catalog,therapy_bundle,seed_bundle
from chatcarlo.gpu.generate_photon_tables import ensure
from . import a4u_d15_bundle,a4u_d15_fixtures,a4u_h1

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


def check_physics():
    actual=a4u_d15_fixtures.physics_hashes()
    expected=json.loads((HERE/'a4u_d2_fixed.json').read_text())['current']
    if actual!=expected:raise ValueError('D2 source hash mismatch')
    return dict(status='pass',functions=actual)


def rebuild(out):
    out=Path(out).resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('private bundle must be outside repository')
    # Raw P9a material inputs are verified by the existing frozen loader.
    from prototypes.mv_electron import p9a_data
    original,identity=p9a_data.load_bundles();materials=[];sources=[]
    for m in original:
        p=ensure(m,sources=sources)
        materials.append(SimpleNamespace(**{key:p if key=='photon' else getattr(m,key) for key in catalog.P8_DEFINITIONS}))
    pins_path=ROOT/'chatcarlo/gpu/therapy_input_hashes.json';pins=json.loads(pins_path.read_text())
    old_pins=json.loads((HERE/'a4u_d2_before.json').read_text())['photon_pins']
    for name,m in zip(seed_bundle.NAMES,materials):pins[name+'.photon']=dict(sha256=catalog.digest(m.photon),shape=list(m.photon.shape))
    pins_path.write_text(json.dumps(pins,sort_keys=True,indent=2)+'\n')
    # Preserve the earlier source expectations; only this model-change snapshot
    # becomes the current hash oracle. No short MC fixtures are regenerated.
    before=json.loads((HERE/'a4u_d15_fixed.json').read_text())['physics']
    (HERE/'a4u_d2_fixed.json').write_text(json.dumps(dict(previous=before,current=a4u_d15_fixtures.physics_hashes()),indent=2)+'\n')
    result=a4u_d15_bundle.run(out=out/'therapy_bundle',summary=None)
    restored,inputs,settings,manifest=therapy_bundle.load(out/'therapy_bundle')
    seed_manifest=seed_bundle.create(out/'seed_bundle',materials,inputs,identity['densities'],settings['generators'])
    seed_bundle.load(out/'seed_bundle')  # verify new field hashes and all pins
    h1=a4u_h1.report(restored,therapy_bundle.NAMES,inputs['constants'])
    (HERE/'a4u_d2_h1.json').write_text(json.dumps(h1,indent=2,allow_nan=False)+'\n')
    unique={s['path']:s for s in sources}
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'chatcarlo/gpu').iterdir()) if p.suffix in ('.py','.h','.cpp','.cu','.json')}
    evidence=dict(command='.venv/bin/python -m docs.validation.gpu.g5.a4u_d2_update --out '+str(out),
        xcom_execution=False,new_acquisition=False,raw_columns=dict(energy=0,coherent=1,incoherent=2,photo=3),
        xcom_sources=[unique[p] for p in sorted(unique)],old_photon_pins=old_pins,
        new_photon_pins={name+'.photon':pins[name+'.photon'] for name in seed_bundle.NAMES},
        bundles=dict(therapy=dict(path=str(out/'therapy_bundle'),manifest=manifest,manifest_sha256=therapy_bundle.sha(out/'therapy_bundle/manifest.json')),
                     seed=dict(path=str(out/'seed_bundle'),manifest=seed_manifest,manifest_sha256=therapy_bundle.sha(out/'seed_bundle/manifest.json'))),
        sources=source_hashes,roundtrip_status=result['status'],source_observation='new four-column photon trailer',
        private_input_values='kept only in the external bundles, never in repository evidence')
    (HERE/'a4u_d2_rebuild.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    result=rebuild(args.out);print(json.dumps({k:v['manifest_sha256'] for k,v in result['bundles'].items()},indent=2))
