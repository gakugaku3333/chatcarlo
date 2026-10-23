"""All commands here use a fake executable; the installed colab is never invoked."""
import json
import os
from pathlib import Path
import zipfile
import pytest
from prototypes.mv_electron.entry import colab_cli as cli, bundle
from prototypes.mv_electron.gpu import p5b_core as core


@pytest.fixture
def fake(tmp_path, monkeypatch):
    program = tmp_path/'fake_colab'
    program.write_text('''#!/usr/bin/env python3
import os,sys,json,pathlib,shutil,subprocess,importlib.metadata
root=pathlib.Path(os.environ['FAKE_ROOT']);root.mkdir(exist_ok=True)
cmd=sys.argv[1];args=sys.argv[2:];mode=os.environ.get('FAKE_MODE','normal')
with (root/'calls.jsonl').open('a') as f:f.write(json.dumps([cmd,*args])+'\\n')
if cmd=='new':sys.exit(0)
if cmd=='stop':sys.exit(0)
if cmd=='upload':
    if mode=='upload_fail':sys.exit(1)
    src=pathlib.Path(args[-2]);dst=pathlib.Path(args[-1]);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);sys.exit(0)
if cmd=='download':
    shutil.copyfile(args[-2],args[-1]);sys.exit(0)
code=sys.stdin.read()
if "state='running'" in code:
    count_file=root/'polls';count=int(count_file.read_text()) if count_file.exists() else 0;count_file.write_text(str(count+1))
    if mode=='lost' and count>0:sys.exit(1)
    files=['batch_000.npz','observation_000.json','batch_000.json']
    for name in files:(root/'results'/name).write_text('synthetic completed batch')
    print('MV_ENTRY_JSON '+json.dumps(dict(state='running' if mode=='lost' else 'complete',files=files)));sys.exit(0)
if 'importlib.metadata' in code:
    class Distribution:
        metadata={'Name':'cupy-cuda12x'}
    importlib.metadata.distributions=lambda: [Distribution()] if mode=='cupy12' else []
    class Popen:
        def __init__(self,*a,**kw):self.pid=12345
    subprocess.Popen=Popen
exec(code)
''')
    program.chmod(0o755)
    remote = tmp_path/'remote'
    monkeypatch.setenv('FAKE_ROOT', str(remote))
    monkeypatch.setattr(cli, 'REMOTE', str(remote))
    directory = tmp_path/'bundle'; directory.mkdir()
    with zipfile.ZipFile(directory/'source.zip', 'w') as z:
        z.writestr('prototypes/__init__.py', '')
    (directory/'payload').write_bytes(b'0123456789'*30)
    manifest = dict(files={n: core.file_sha(directory/n) for n in ('source.zip', 'payload')})
    core.write_json(directory/'manifest.json', manifest)
    core.write_json(directory/'condition.json', {})
    core.write_json(directory/'materials.json', {})
    monkeypatch.setattr(bundle, 'load', lambda d: ({}, manifest))
    monkeypatch.setattr(bundle, 'existing', lambda root, *a: {0} if (Path(root)/'batch_000.json').exists() else set())
    monkeypatch.setattr(cli, 'SPLIT_THRESHOLD', 200)
    monkeypatch.setattr(cli, 'PART_BYTES', 50)
    return cli.Client(str(program)), directory, remote, tmp_path/'results'


def calls(remote):
    return [json.loads(line) for line in (remote/'calls.jsonl').read_text().splitlines()]


def test_entry_colab_split_assemble_resume(fake):
    client, directory, remote, results = fake
    cli.run(directory, results, client=client, sleep=lambda _: pytest.fail('unexpected sleep'))
    assert (remote/'payload').read_bytes() == (directory/'payload').read_bytes()
    assert len(list((remote/'parts').glob('payload.part.*'))) == 6
    assert (results/'batch_000.npz').exists()
    before = len(calls(remote))
    cli.run(directory, results, client=client, sleep=lambda _: None)
    uploads = [c for c in calls(remote)[before:] if c[0] == 'upload']
    assert any(c[-1].endswith('/results/batch_000.npz') for c in uploads)
    assert calls(remote)[-1] == ['stop', '-s', 'mventry']


@pytest.mark.parametrize('mode,reason', [('upload_fail', '2 attempts'), ('cupy12', 'cupy-cuda12x'), ('lost', 'session missing')])
def test_entry_colab_stops(fake, monkeypatch, mode, reason):
    client, directory, remote, results = fake
    monkeypatch.setenv('FAKE_MODE', mode)
    sleeps = []
    with pytest.raises(core.Stop, match=reason):
        cli.run(directory, results, client=client, sleep=sleeps.append)
    assert calls(remote)[-1] == ['stop', '-s', 'mventry']
    if mode == 'lost':
        assert (results/'batch_000.npz').exists()
        assert sleeps == [600, 30, 30]  # 失敗した確認は 3 回まで、30 秒おいて再試行する
    if mode == 'upload_fail':
        assert sum(c[0] == 'upload' for c in calls(remote)) == 2
    if mode == 'cupy12':
        assert not (remote/'driver.pid').exists()


def test_entry_colab_part_corruption(fake):
    client, directory, remote, results = fake
    client.execute(f"import pathlib,json\nr=pathlib.Path({str(remote)!r})\n[(r/p).mkdir(parents=True,exist_ok=True) for p in ('parts','results')]\nprint('MV_ENTRY_JSON '+json.dumps(dict(status='ready')))")
    uploads, assembly = cli.transfer_files(directory, results/'parts')
    for local, target in uploads:
        client.upload(local, target)
    part = next((remote/'parts').glob('payload.part.*')); part.write_bytes(b'corrupt')
    result = client.execute(cli.bootstrap(assembly, 'fake-session'))
    assert result['status'] == 'failed' and 'part SHA256' in result['reason']
    assert not (remote/'driver.pid').exists()
