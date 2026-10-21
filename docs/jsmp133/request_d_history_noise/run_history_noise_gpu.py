"""依頼 D（改訂 2）: 9 段階 × 2 本を ChatCarlo の GPU エンジン（G1）で Colab の T4 に回し、依頼 D の形式で保存する。

使い方（リポジトリの最上位で）:
  PYTHONPATH=. .venv/bin/python docs/jsmp133/request_d_history_noise/run_history_noise_gpu.py \
      --work WORK --out ~/.cache/chatcarlo/jsmp133_request_d/gpu
条件は PREREGISTRATION.md（改訂 2）。Colab のセッションは 1 つで、VM 上で 18 本を順に回す。
10 分ごとに確認し、終わった本から回収する。各バッチは受領票（SHA-256 と識別情報）で照合してから
ホストで足し合わせる（chatcarlo.gpu.run_detector_image.collect_image）。同じコマンドで再開すると、
回収済みの本は飛ばす。解析は analyze_history_noise.py --data OUT。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

from chatcarlo.detector import DetectorPlane
from chatcarlo.gpu import bundle
from chatcarlo.gpu.colab_driver import REMOTE, digest, split_bundle
from chatcarlo.gpu.run_detector_image import DET, collect_image
from chatcarlo.scene import load_scene

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCENE = HERE / "scene_portrait.yaml"
GATE2 = ROOT / "docs/validation/gpu/g1/gate2.json"
POLL_SECONDS = 600


def jobs():
    out = []
    for k in range(9):
        n = int(round(10 ** (4 + 0.5 * k)))
        for tag, seed in (("A", 2000 + 10 * k), ("B", 2001 + 10 * k)):
            source_seed = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])
            job = dict(mode="batches", device="cuda", n_histories=n, batch_size=int(min(200000, -(-n // 10))),
                       seed=seed, source_seed=source_seed)
            out.append((f"N{n:09d}_{tag}", tag, job))
    return out


VM_RUNNER = r'''
import json,pathlib,subprocess,sys,tarfile,time
base=pathlib.Path(BASE);names=json.loads((base/'order.json').read_text())
(base/'tars').mkdir(exist_ok=True)
for name in names:
    if (base/'tars'/(name+'.tar')).exists():continue
    t0=time.time()
    r=subprocess.run([sys.executable,'-m','chatcarlo.gpu.colab_driver','worker','--bundle',str(base/'chatcarlo'/'gpu'),'--job',str(base/'jobs'/(name+'.json')),'--out',str(base/'results'/name)],cwd=base)
    t1=time.time()
    if r.returncode!=0:(base/'FAILED').write_text(name);sys.exit(1)
    pending=base/'tars'/(name+'.tar.pending')
    with tarfile.open(pending,'w') as tf:tf.add(base/'results'/name,arcname=name)
    (base/'tars'/(name+'.timing.json')).write_text(json.dumps(dict(start_unix=t0,end_unix=t1,wall_s=t1-t0)))
    pending.replace(base/'tars'/(name+'.tar'))
(base/'ALL_DONE').write_text('ok')
'''


def bootstrap(files, job_shas):
    # 送った束を組み立て、目録で照合してから、順に回す役を nohup で起動する。
    return f'''import pathlib,hashlib,json,subprocess,sys
base=pathlib.Path({REMOTE!r});parts=base/'parts';target=base/'chatcarlo'/'gpu';target.mkdir(parents=True,exist_ok=True)
(base/'chatcarlo'/'__init__.py').write_text('')
files=json.loads({json.dumps(files)!r})
for name,item in files.items():
 pending=target/(name+'.pending')
 with pending.open('wb') as dest:
  for part in item['parts']:
   src=parts/part['name'];assert hashlib.sha256(src.read_bytes()).hexdigest()==part['sha256'],'part SHA256'
   dest.write(src.read_bytes())
 assert hashlib.sha256(pending.read_bytes()).hexdigest()==item['sha256'],'assembled SHA256'
 pending.replace(target/name)
for name,h in json.loads({json.dumps(job_shas)!r}).items():
 assert hashlib.sha256((base/'jobs'/(name+'.json')).read_bytes()).hexdigest()==h,'job SHA256'
sys.path.insert(0,str(base))
from chatcarlo.gpu.bundle import verify
verify(target)
(base/'runner.py').write_text({VM_RUNNER.replace('BASE', repr(REMOTE))!r})
with (base/'runner.log').open('ab') as log:
 p=subprocess.Popen(['nohup',sys.executable,str(base/'runner.py')],cwd=base,stdout=log,stderr=log,start_new_session=True)
(base/'runner.pid').write_text(str(p.pid))
print('D_LAUNCHED')
'''


LISTING = f'''import pathlib,os
base=pathlib.Path({REMOTE!r})
for t in sorted((base/'tars').glob('*.tar')):print('D_TAR '+t.name[:-4])
if (base/'ALL_DONE').exists():print('D_DONE')
elif (base/'FAILED').exists():print('D_FAILED '+(base/'FAILED').read_text())
else:
 try:os.kill(int((base/'runner.pid').read_text()),0);print('D_RUNNING')
 except (OSError,ValueError):print('D_LOST')
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session", default="request-d")
    ap.add_argument("--colab", default="colab")
    a = ap.parse_args()
    if json.loads(GATE2.read_text()).get("status") != "pass":
        raise ValueError("gate 2 must pass before request-D CUDA execution")
    work = a.work.expanduser();out = a.out.expanduser();out.mkdir(parents=True, exist_ok=True)
    bpath = work / "bundle"
    if not (bpath / "manifest.json").exists():
        bundle.create(load_scene(str(SCENE)), DetectorPlane(**DET), bpath)
    manifest = bundle.verify(bpath)
    todo = [(name, tag, job) for name, tag, job in jobs() if not (out / f"{name}.npz").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    if not todo:
        return

    def cli(*args, input=None):
        r = subprocess.run([a.colab, *args], input=input, text=True, capture_output=True, check=True)
        return r.stdout

    jdir = work / "jobs";jdir.mkdir(parents=True, exist_ok=True)
    for name, _, job in todo:
        (jdir / f"{name}.json").write_text(json.dumps(job, sort_keys=True) + "\n")
    (work / "order.json").write_text(json.dumps([n for n, _, _ in todo]) + "\n")
    with tempfile.TemporaryDirectory(prefix="request-d-transfer-") as tmp:
        tmp = Path(tmp);files = split_bundle(bpath, tmp / "parts")
        try:
            cli("new", "-s", a.session, "--gpu", "T4")
            print(time.strftime("%H:%M:%S"), "session created", flush=True)
            cli("exec", "-s", a.session, input=f"import pathlib\nb=pathlib.Path({REMOTE!r})\nfor d in ('parts','jobs','results'):(b/d).mkdir(parents=True,exist_ok=True)\n")
            for part in sorted((tmp / "parts").iterdir()):
                cli("upload", "-s", a.session, str(part), f"{REMOTE}/parts/{part.name}")
            for name, _, _ in todo:
                cli("upload", "-s", a.session, str(jdir / f"{name}.json"), f"{REMOTE}/jobs/{name}.json")
            cli("upload", "-s", a.session, str(work / "order.json"), f"{REMOTE}/order.json")
            shas = {name: digest(jdir / f"{name}.json") for name, _, _ in todo}
            if "D_LAUNCHED" not in cli("exec", "-s", a.session, input=bootstrap(files, shas)):
                raise RuntimeError("bundle verification on the VM did not launch the runner")
            print(time.strftime("%H:%M:%S"), "runner launched", flush=True)
            pending = {name: (tag, job) for name, tag, job in todo}
            while pending:
                time.sleep(POLL_SECONDS)
                listing = cli("exec", "-s", a.session, input=LISTING)
                for line in listing.splitlines():
                    if not line.startswith("D_TAR "):
                        continue
                    name = line.split(" ", 1)[1]
                    if name not in pending:
                        continue
                    tag, job = pending[name];stage = work / "collected" / name
                    stage.parent.mkdir(parents=True, exist_ok=True)
                    tar = work / "collected" / f"{name}.tar"
                    cli("download", "-s", a.session, f"{REMOTE}/tars/{name}.tar", str(tar))
                    cli("download", "-s", a.session, f"{REMOTE}/tars/{name}.timing.json", str(out / f"{name}.vm_timing.json"))
                    with tarfile.open(tar) as tf:
                        tf.extractall(work / "collected", filter="data")
                    meta = collect_image(bpath, job, stage, out, tag=tag)  # 受領票で全バッチを照合
                    print(time.strftime("%H:%M:%S"), name, "collected", json.dumps(dict(detected=meta["detected"], n_batches=meta["n_batches"])), flush=True)
                    del pending[name]
                if pending and "D_RUNNING" not in listing and "D_DONE" not in listing:
                    raise RuntimeError("VM runner stopped: " + listing.strip().splitlines()[-1])
        finally:
            try:
                cli("stop", "-s", a.session)
            except subprocess.CalledProcessError:
                pass


if __name__ == "__main__":
    main()
