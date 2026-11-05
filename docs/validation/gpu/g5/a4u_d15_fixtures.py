"""D15 small fixture regeneration and schema witnesses; no long MC gates.

Run: .venv/bin/python -m docs.validation.gpu.g5.a4u_d15_fixtures --regenerate
Historical G4 evidence is read-only; old and new values are stored together.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
FIXED=HERE/'a4u_d15_fixed.json'


def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def physics_hashes():
    helper=module('a2b_physics');old=ROOT/'prototypes/mv_electron/gpu';new=ROOT/'chatcarlo/gpu'
    pairs=(('p5a0_common.h','g5_base.h'),('p5_rng.h','t2_rng.h'),('p5_kernel.h','t2_physics.h'),('p8g_logs.h','t2_logs.h'),('p8g_functions.h','t2_functions.h'),('p8g_materials.h','t2_materials.h'),('p8g_kernel.h','t2_transport.h'))
    unchanged={};changed={}
    for a,b in pairs:
        x,y=helper.functions(old/a),helper.functions(new/b)
        for name,digest in x.items():
            if name in ('toy','p8g_history','p8g_record_history','p8g_clear'):continue
            if name not in y:y.update(helper.functions(new/'t2_rng.h'))
            if name not in y:raise ValueError('missing helper '+name)
            if y[name]==digest:unchanged[name]=digest
            else:changed[name]=dict(old=digest,new=y[name])
    # Only the standalone photon's channel count/cutoff/history body changes
    # here. Electron, pair, RNG and elastic scalar helpers remain byte hashes.
    if set(changed)!={'history'}:raise ValueError('unexplained helper changes: '+repr(changed))
    return dict(unchanged=unchanged,changed=changed,
        shared_photon=helper.functions(new/'photon_physics.h'))


def check_physics():
    expected=json.loads(FIXED.read_text())['physics']
    actual=physics_hashes()
    if actual!=expected:raise ValueError('D15 scalar/photons source hash mismatch')
    return dict(status='pass',functions=actual)


def p8_roundtrip(out):
    """Original 33 arrays remain exact; field 34 and native append are explicit."""
    from docs.validation.gpu.g5 import a4u_d15_bundle
    return a4u_d15_bundle.run(out=out,summary=None)


def regenerate():
    from chatcarlo.gpu import bundle,core,catalog,physics_sets
    from chatcarlo.scene import load_scene
    from chatcarlo.source import sample_source_photons
    from chatcarlo.transport import run_transport
    from chatcarlo.detector import DetectorPlane
    sys.path.insert(0,str(ROOT/'tests'))
    import test_gpu_g4 as g4
    old_g4=json.loads((ROOT/'docs/validation/gpu/g4/local_baseline.json').read_text())
    old_g2=dict(deposits='d424085911c924d0b70c2678da6caf91991a0f5a1ad5753d2c9e5cdbbae36e06',endings='46d27f6b8fc6ff1a71c1acb951457a9a9c49c26b0c5cdc39e4c55391a3d9b9c9',scatter_sum='6602cf082886b358e25715c49ab10047eb8871d8c2ea5e3340dcc0b76065f69e',n_fluorescence='af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc',d='5a28e04a585df9951c0e807e6f949474b10402fb3bac1a5e3672e33df292d071',i='8391c9b32ff975bf86558de646d1111834b311f1fabb66acc3df3da98e9b108c')
    scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    with tempfile.TemporaryDirectory(prefix='a4u-d15-') as tmp:
        bundle.create(scene,None,tmp);a,*_=bundle.load(tmp)
        o,d,e=sample_source_photons(scene.raw['source'],128,np.random.default_rng(713))
        r=core.Core(a).run(o,d,e,seed=712,record=True)
        bundle.create(scene,None,Path(tmp)/'g2',g2_tally=True)
        schema=dict(old=dict(diagnostic_offsets=20,diagnostic_with_g2=26,p8_fields=33,therapy_transport_low_keV=10.),
                    new=dict(diagnostic_offsets=len(a['offsets']),diagnostic_with_g2=len(json.loads((Path(tmp)/'g2/catalog.json').read_text())['fields']),p8_fields=len(catalog.P8_DEFINITIONS),therapy_transport_low_keV=physics_sets.REGISTRY[physics_sets.NAME]['specializations'][physics_sets.ELECTRON]['descriptor']['energy_band_keV'][0]))
    new_g2={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in r.items()}
    new_g4={}
    for n in (1,2):
        result=run_transport(load_scene(str(ROOT/f'docs/validation/gpu/g1/scenes/s{n}.yaml')),n_histories=1000,seed=42,batch_size=400,engine='gpu',detector=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60)))
        new_g4['gpu_s'+str(n)]=g4.fingerprint(result)
    import test_gpu_g5_a4u_s2 as s2
    witnesses={}
    for name in ('lead','water'):
        a=s2.arrays(('air',name));a['halves']=np.array([[10.,10.,.025]])
        a['data'][int(a['offsets'][19])]=0.
        source=(np.zeros((8192,3)),np.tile([0.,0.,1.],(8192,1)),np.full(8192,100.))
        result={}
        for label,legacy in (('old',True),('new',False)):
            r=core.Core(a,_test_legacy_physics=legacy).run(*source,seed=9504015)
            result[label]=dict(fluorescence=float(r['n_fluorescence']),escaped=float(r['endings'][1]),created=float(r['created']))
        witnesses[name]=dict(histories=8192,seed=9504015,energy_keV=100.,thickness_cm=.05,**result)
    payload=dict(command='.venv/bin/python -m docs.validation.gpu.g5.a4u_d15_fixtures --regenerate',
        schema=schema,direction_witnesses=witnesses,g2=dict(old=old_g2,new=new_g2),g4=dict(old={k:old_g4[k] for k in new_g4},new=new_g4),physics=physics_hashes())
    FIXED.write_text(json.dumps(payload,indent=2,allow_nan=True)+'\n')
    return FIXED

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--regenerate',action='store_true',required=True)
    parser.parse_args();print(regenerate())
