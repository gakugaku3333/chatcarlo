"""Gates 0d/0e/0g: source identity, world bounds and VM independence."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from chatcarlo.scene import load_scene
from chatcarlo.source import sample_source_photons
from chatcarlo.spectrum import _default_spectrum
from chatcarlo.geometry import Geometry
from chatcarlo.detector import DetectorPlane
from chatcarlo.gpu import bundle,core,source_host

DOC=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'
PORTRAIT=DOC.parents[3]/'docs/jsmp133/request_d_history_noise/scene_portrait.yaml'

def spectrum(src):
    if src.get('spectrum'):
        e=np.array([s['energy_keV'] for s in src['spectrum']],float);w=np.array([s['weight'] for s in src['spectrum']],float);return e,w/w.sum()
    return _default_spectrum(src['kvp'],src.get('filtration_mm_al',2.5),src.get('anode_angle_deg',12.))

@pytest.mark.parametrize('scene_id',('s1','s2','s3'))
@pytest.mark.parametrize('explicit',[False,True])
def test_gpu_g1_0d_sources(scene_id,explicit):
    src=load_scene(str(PORTRAIT if scene_id=='s3' else DOC/f'scenes/{scene_id}.yaml')).raw['source']
    spec=spectrum(src)
    if explicit:
        src=copy.deepcopy(src);src.pop('kvp',None);src['spectrum']=[dict(energy_keV=float(e),weight=float(w)) for e,w in zip(*spec)];spec=spectrum(src)
    check_source(src,spec)


def check_source(src,spec):
    a=np.random.default_rng(2026102103);b=np.random.default_rng(2026102103);resume=np.random.default_rng(2026102103)
    for batch in range(5):
        x=sample_source_photons(src,10000,a);y=source_host.sample_source_photons(src,10000,b,spec)
        for aa,bb in zip(x,y):assert aa.shape==bb.shape and aa.tobytes()==bb.tobytes()
        assert a.bit_generator.state==b.bit_generator.state
        z=source_host.sample_source_photons(src,10000,resume,spec)
        if batch>=2:
            for aa,bb in zip(x,z):assert aa.tobytes()==bb.tobytes()
    assert a.bit_generator.state==resume.bit_generator.state

@pytest.mark.parametrize('shape',['rect','cone','parallel'])
@pytest.mark.parametrize('direction',[(.3,.2,-1.),(np.sqrt(1-.9989**2),0.,-.9989),(np.sqrt(1-.9991**2),0.,-.9991)])
def test_gpu_g1_0d_basis(shape,direction):
    src=load_scene(str(DOC/'scenes/s1.yaml')).raw['source'];d=np.array(direction);d/=np.linalg.norm(d);src['direction']=d.tolist()
    src['field']=dict(shape=shape,size_cm=[10,10],sid_cm=100,diameter_cm=20)
    check_source(src,spectrum(src))

@pytest.mark.parametrize('plane_on',[True,False])
def test_gpu_g1_0e_bundle_world(tmp_path,plane_on):
    scene=load_scene(str(DOC/'scenes/s2.yaml'));plane=DetectorPlane([50,-60,-10],[0,0,1],[1,0,0],(30,30),(60,60)) if plane_on else None
    bundle.create(scene,plane,tmp_path);a,*_=bundle.load(tmp_path);g=Geometry(scene.raw['geometry'],detector_plane=plane)
    assert a['bbox_min'].tobytes()==g.bbox_min.tobytes()
    assert a['bbox_max'].tobytes()==g.bbox_max.tobytes()
    # Compare the canonical center and half, so no reverse-rounding is introduced.
    assert a['world'].tobytes()==np.r_[(g.bbox_min+g.bbox_max)/2,(g.bbox_max-g.bbox_min)/2].tobytes()
    changed=bytearray((tmp_path/'tables.npz').read_bytes());changed[-1]^=1;(tmp_path/'tables.npz').write_bytes(changed)
    with pytest.raises(ValueError,match='SHA256'):bundle.load(tmp_path)


def test_gpu_g1_0g_vm_import(tmp_path):
    scene=load_scene(str(DOC/'scenes/s1.yaml'));bundle.create(scene,None,tmp_path/'bundle')
    code='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('xraylib','scipy','spekpy'):raise ImportError('blocked '+fullname)
sys.meta_path.insert(0,Block())
from chatcarlo.gpu.bundle import load
from chatcarlo.gpu.core import Core
from chatcarlo.gpu.source_host import sample_source_photons
import numpy as np
a,s,sp,settings,_=load(sys.argv[1]);o,d,e=sample_source_photons(s,1000,np.random.default_rng(11),sp)
r=Core(a).run(o,d,e,seed=12)
assert r['endings'].sum()==1000
assert not any(m.split('.')[0] in ('xraylib','scipy','spekpy') for m in sys.modules)
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path/'bundle')],check=True,env=os.environ.copy(),capture_output=True)
