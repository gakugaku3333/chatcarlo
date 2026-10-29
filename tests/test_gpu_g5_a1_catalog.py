"""A1a-6: unchanged numeric blobs and inner catalogue corruption checks."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from chatcarlo import kernel
from chatcarlo.gpu import bundle,catalog,core
from chatcarlo.scene import load_scene

ROOT=Path(__file__).resolve().parents[1]


def refresh(path,name):
    m=json.loads((path/'manifest.json').read_text());m['files'][name]=bundle.sha(path/name)
    (path/'manifest.json').write_text(json.dumps(m))


@pytest.fixture
def packed(tmp_path):
    scene=load_scene(str(ROOT/'examples/water_phantom_pdd_ocr.yaml'))
    bundle.create(scene,None,tmp_path,g2_tally=True)
    return tmp_path


@pytest.mark.parametrize('scene_path',[
 'docs/validation/gpu/g1/scenes/s1.yaml','docs/validation/gpu/g1/scenes/s2.yaml',
 'docs/jsmp133/request_d_history_noise/scene_portrait.yaml','examples/water_phantom_pdd_ocr.yaml'])
def test_original_pack_and_geometry(tmp_path,scene_path):
    old=tmp_path/'old_core.py';old.write_bytes(subprocess.check_output(['git','show','19c281e:chatcarlo/gpu/core.py']))
    spec=importlib.util.spec_from_file_location('g5_old_pack',old);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    scene=load_scene(str(ROOT/scene_path));path=tmp_path/'bundle';bundle.create(scene,None,path,g2_tally=True)
    a,_,_,settings,_=bundle.load(path)
    t=kernel.bake_scene_materials(settings['material_names']);expected=mod.pack(t)
    for k,v in expected.items():assert v.dtype==a[k].dtype and v.tobytes()==a[k].tobytes()
    from chatcarlo.geometry import Geometry
    geom=Geometry(scene.raw['geometry'],bbox_margin_cm=50.)
    geoms=scene.raw['geometry']
    geometry=dict(centers=np.array([g['center'] for g in geoms],float).reshape(-1,3),
                  halves=np.array([g['size_cm'] for g in geoms],float).reshape(-1,3)/2,
                  materials=np.array([settings['material_names'].index(g['material']) for g in geoms],np.int64),
                  world=np.r_[(geom.bbox_min+geom.bbox_max)/2,(geom.bbox_max-geom.bbox_min)/2],
                  bbox_min=geom.bbox_min,bbox_max=geom.bbox_max,background=np.array(0),detector=np.array([]))
    for k,v in geometry.items():assert a[k].tobytes()==v.tobytes()
    cat=json.loads((path/'catalog.json').read_text());assert len(cat['fields'])==26
    for f in cat['fields']:
        blob={'tables':a['data'],'g2_tables':a['g2_table'],'g2_sizes':a['g2_sizes']}[f['blob']]
        assert catalog.digest(blob[f['offset']:f['offset']+f['size']])==f['data_sha256']
    log=next(f for f in cat['fields'] if f['name']=='log_e')
    assert log['energy_band_keV']==catalog.energy_band(catalog.diagnostic_arrays(a))


@pytest.mark.parametrize('damage',['byte','duplicate','missing','offset','shape','grid','swap','band'])
def test_inner_catalog_rejects_after_outer_reseal(packed,damage):
    path=packed;cat=json.loads((path/'catalog.json').read_text());fields=cat['fields'];name='photo'
    photo=next(f for f in fields if f['name']=='photo')
    if damage=='byte':
        with np.load(path/'tables.npz') as z:a={k:z[k] for k in z.files}
        a['data'].view(np.uint8)[photo['offset']*8]^=1
        np.savez(path/'tables.npz',**a);refresh(path,'tables.npz')
    elif damage=='duplicate':fields.append(copy.deepcopy(photo))
    elif damage=='missing':fields.remove(photo)
    elif damage=='offset':photo['offset']=10**12
    elif damage=='shape':photo['shape']=[photo['size']+1]
    elif damage=='grid':photo['grid_kind']='incoh_q'
    elif damage=='band':photo['energy_band_keV']=[1.,151.]
    elif damage=='swap':
        compt=next(f for f in fields if f['name']=='compt')
        for k in ('offset','data_sha256'):photo[k],compt[k]=compt[k],photo[k]
    (path/'catalog.json').write_text(json.dumps(cat));refresh(path,'catalog.json')
    with pytest.raises(ValueError,match=name):bundle.verify(path)


def test_outer_hash_and_old_format(packed):
    p=packed/'catalog.json';p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='bundle SHA256'):bundle.verify(packed)
    refresh(packed,'catalog.json');m=json.loads((packed/'manifest.json').read_text());m['format']='chatcarlo-g1-v1'
    (packed/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='unknown bundle format'):bundle.verify(packed)


def test_g2_scoring_metadata_allowed(packed):
    with np.load(packed/'g2_tables.npz') as z:a={k:z[k] for k in z.files}
    a.update(boxes=np.empty((0,6)),water=np.array(1),seed=np.array(3101),box_names=np.array([],dtype=str),
             n_histories=np.array(100),batch_size=np.array(10),depth=np.array(20.),grid_origin=np.zeros(3),grid_shape=np.ones(3,int),grid_resolution=np.array(2.))
    np.savez(packed/'g2_tables.npz',**a);refresh(packed,'g2_tables.npz');bundle.verify(packed)


def test_p8_small_scalar_3d_4d_roundtrip():
    values={f:np.array(1.) for f in catalog.P8_DEFINITIONS}
    for f in ('stop_T','gs_T','photo_E','pair_E','brems_E'):values[f]=np.array([.01,1.,20.])
    values.update(gs_y=np.arange(24.).reshape(2,3,4),lines=np.arange(64.).reshape(2,4,2,4),
                  rates=np.array([[.01,1.,2.],[10.,3.,4.]]))
    m=SimpleNamespace(**values)
    generators={d['generator_path']:dict(path=d['generator_path'],sha256='0'*64,options={}) for d in catalog.P8_DEFINITIONS.values()}
    cat,blob=catalog.p8_material(m,'water',generators);r=catalog.reconstruct(cat,{'p8':blob})
    assert len(r)==33
    for f,v in values.items():assert r['water.'+f].shape==v.shape and r['water.'+f].tobytes()==v.tobytes()
    for f in ('ne','gs_K','gs_u'):assert next(e for e in cat['fields'] if e['name']=='water.'+f)['energy_band_keV'] is None
