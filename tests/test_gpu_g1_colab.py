"""Colab protocol tests use an executable fake; no real sessions or cache."""
import json
from pathlib import Path
import os
import sys
import numpy as np
import pytest
from chatcarlo.gpu import colab_driver as c,bundle
from chatcarlo.scene import load_scene

DOC=Path(__file__).resolve().parents[1]/'docs/validation/gpu/g1'

FAKE=r'''#!/usr/bin/env python3
import sys,os,pathlib,json,shutil,subprocess
args=sys.argv[1:];state=pathlib.Path(os.environ['G1_FAKE_ROOT']);state.mkdir(exist_ok=True)
with (state/'commands').open('a') as f:f.write(json.dumps(args)+'\n')
mode=os.environ.get('G1_FAKE_MODE','normal');cmd=args[0]
if cmd=='new':print('fake session');sys.exit(0)
if cmd=='stop':print('stopped');sys.exit(0)
if cmd in ('upload','download'):
 src=pathlib.Path(args[-2]);dst=pathlib.Path(args[-1]);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 if mode=='tamper' and cmd=='upload' and '.part' in dst.name:dst.write_bytes(dst.read_bytes()+b'X')
 sys.exit(0)
text=sys.stdin.read()
if 'G1_LAUNCHED' in text:
 text=text.split("with (base/'worker.log')")[0]
 exec(text)
 if mode=='loss':
  from chatcarlo.gpu.bundle import load
  from chatcarlo.gpu.core import Core
  from chatcarlo.gpu.source_host import sample_source_photons
  from chatcarlo.gpu.colab_driver import identity,write_result
  import numpy as np
  job=json.loads((base/'job.json').read_text());a,src,sp,settings,m=load(target)
  o,d,e=sample_source_photons(src,job['batch_size'],np.random.default_rng(job['source_seed']),sp)
  r=Core(a).run(o,d,e,seed=job['seed']);write_result(base/'results','batch_000000.npz',r,identity(m,job,0,0,job['batch_size']))
 else:
  from chatcarlo.gpu.colab_driver import worker
  worker(target,base/'job.json',base/'results')
 print('G1_LAUNCHED');sys.exit(0)
if 'G1_RESULT ' in text and mode=='loss':
 polls=state/'polls';num=int(polls.read_text()) if polls.exists() else 0;polls.write_text(str(num+1))
 if num>0:print('session not found');sys.exit(1)
 print('G1_RESULT batch_000000.npz\nG1_RUNNING');sys.exit(0)
exec(text)
'''

@pytest.fixture
def prepared(tmp_path,monkeypatch):
    path=tmp_path/'bundle';bundle.create(load_scene(str(DOC/'scenes/s1.yaml')),None,path)
    exe=tmp_path/'colab';exe.write_text(FAKE.replace("#!/usr/bin/env python3", "#!"+sys.executable));exe.chmod(0o755)
    monkeypatch.setenv('G1_FAKE_ROOT',str(tmp_path/'fake-state'))
    monkeypatch.setattr(c,'REMOTE',str(tmp_path/'remote'));monkeypatch.setattr(c,'PART_BYTES',100000)
    monkeypatch.setattr(c.time,'sleep',lambda seconds:None)
    gate=tmp_path/'gate.json';gate.write_text('{"status":"pass"}')
    return path,exe,gate,dict(mode='batches',seed=71,source_seed=72,n_histories=100,batch_size=40,device='cpu')


def test_gpu_g1_colab_parts(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'tiny').write_bytes(b'abcdefg');(source/'large').write_bytes(bytes(range(256))*100)
    files=c.split_bundle(source,tmp_path/'parts',part_bytes=103);assert len(files['large']['parts'])>1
    c.assemble(tmp_path/'parts',tmp_path/'assembled',files)
    for name in files:assert (source/name).read_bytes()==(tmp_path/'assembled'/name).read_bytes()
    (tmp_path/'parts'/files['large']['parts'][0]['name']).write_bytes(b'bad')
    with pytest.raises(ValueError,match='SHA256'):c.assemble(tmp_path/'parts',tmp_path/'corrupt',files)


def test_gpu_g1_colab_full_and_resume(prepared,tmp_path):
    path,exe,gate,job=prepared;out=tmp_path/'results'
    c.launch(path,job,out,gate1=gate,gate2=gate,colab=str(exe),poll_seconds=0)
    manifest=bundle.verify(path)
    for b,start,n in ((0,0,40),(1,40,40),(2,80,20)):assert c.read_result(out,f'batch_{b:06d}.npz',c.identity(manifest,job,b,start,n)) is not None
    saved={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in out.glob('*.npz')}
    c.launch(path,job,out,gate1=gate,gate2=gate,colab=str(exe),poll_seconds=0)
    for name,(data,mtime) in saved.items():assert (out/name).read_bytes()==data and (out/name).stat().st_mtime_ns==mtime
    commands=[json.loads(l) for l in (tmp_path/'fake-state/commands').read_text().splitlines()]
    assert sum(a[0]=='stop' for a in commands)==2


def test_gpu_g1_colab_tamper_stops(prepared,tmp_path,monkeypatch):
    path,exe,gate,job=prepared;monkeypatch.setenv('G1_FAKE_MODE','tamper')
    with pytest.raises(Exception):c.launch(path,job,tmp_path/'results',gate1=gate,gate2=gate,colab=str(exe),poll_seconds=0)
    assert not list((tmp_path/'results').glob('*.npz'))
    assert '"stop"' in (tmp_path/'fake-state/commands').read_text()


def test_gpu_g1_colab_loss_preserves_recovery(prepared,tmp_path,monkeypatch):
    path,exe,gate,job=prepared;monkeypatch.setenv('G1_FAKE_MODE','loss');out=tmp_path/'results'
    with pytest.raises(Exception):c.launch(path,job,out,gate1=gate,gate2=gate,colab=str(exe),poll_seconds=0)
    manifest=bundle.verify(path);saved=c.read_result(out,'batch_000000.npz',c.identity(manifest,job,0,0,40));assert saved is not None
    first=(out/'batch_000000.npz').read_bytes();monkeypatch.setenv('G1_FAKE_MODE','normal')
    c.launch(path,job,out,gate1=gate,gate2=gate,colab=str(exe),poll_seconds=0)
    assert (out/'batch_000000.npz').read_bytes()==first
    full=tmp_path/'continuous';c.worker(path,tmp_path/'remote/job.json',full)
    for name in ('batch_000001.npz','batch_000002.npz'):
        with np.load(full/name) as a,np.load(out/name) as b:
            for key in a.files:assert a[key].tobytes()==b[key].tobytes()


def test_gpu_g1_receipt_needs_both_files(tmp_path):
    key=dict(seed=1);c.write_result(tmp_path,'a.npz',dict(x=np.array([1.])),key)
    (tmp_path/'a.npz.receipt.json').unlink();assert c.read_result(tmp_path,'a.npz',key) is None
    (tmp_path/'stray.pending').write_text('partial')
    c.write_result(tmp_path,'a.npz',dict(x=np.array([1.])),key)
    with pytest.raises(ValueError,match='identity'):c.read_result(tmp_path,'a.npz',dict(seed=2))
    (tmp_path/'a.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='SHA256'):c.read_result(tmp_path,'a.npz',key)
