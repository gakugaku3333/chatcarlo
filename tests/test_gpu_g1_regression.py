"""Gate 0h: values frozen before implementation at the requested commit."""
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from chatcarlo.scene import load_scene
from chatcarlo.detector import DetectorPlane
from chatcarlo.transport import run_transport

HERE=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'

@pytest.mark.parametrize('number',[1,2])
def test_gpu_g1_0h_numpy_unchanged(number):
    manifest=json.loads((HERE/'baseline_manifest.json').read_text())
    assert manifest['commit']=='636b2cffd3c8839e7d0f50b73f5496289d6db18a'
    scene_path=HERE/f'scenes/s{number}.yaml';baseline_path=HERE/f'baseline_s{number}.npz'
    for path,key in ((scene_path,'scene_sha256'),(baseline_path,'baseline_sha256')):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['scenes'][f'S{number}'][key]
    r=run_transport(load_scene(str(scene_path)),n_histories=200000,seed=42,n_workers=1,batch_size=100000,track_uncertainty=True,detector=DetectorPlane([0,0,0],[0,0,1],[1,0,0],(30,30),(60,60)))
    arrays={};fields={}
    for f in dataclasses.fields(r):
        v=getattr(r,f.name)
        if f.name=='detector':
            for df in dataclasses.fields(v):
                vv=getattr(v,df.name)
                if isinstance(vv,np.ndarray):arrays['detector.'+df.name]=vv
                elif df.name!='plane':fields['detector.'+df.name]=vv
        else:fields[f.name]=v
    arrays['fields_json']=np.array(json.dumps(fields,sort_keys=True,allow_nan=True))
    with np.load(baseline_path,allow_pickle=False) as fixed:
        assert set(arrays)==set(fixed.files)
        for name,a in arrays.items():
            b=fixed[name]
            assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
