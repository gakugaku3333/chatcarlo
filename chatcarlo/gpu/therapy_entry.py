"""G5 P7 water-field CLI. seed_ratio remains at the frozen prototype entry."""
import argparse
import html
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from . import contracts, therapy_scene, therapy_inputs, therapy_layout, therapy_run, physics_sets
from .condition import sha256
from .therapy_bundle import sha
from .therapy import NAME, CPU


def source_hashes():
    root=Path(__file__).parent
    return {p.name:sha(p) for p in sorted(root.iterdir()) if p.suffix in ('.py','.json','.h','.cpp','.cu')}


def condition(scene,manifest):
    return dict(format='chatcarlo-g5-therapy-entry-v1',physics_set=physics_sets.declaration(NAME),
                scene=scene,bundle_manifest_sha256=sha256(manifest),sources=source_hashes(),
                spectrum=scene['source']['spectrum'],histories=scene['run']['histories'],seed=scene['run']['seed'],
                engine=scene['run']['engine'],layout='p7',lateral_extent_cm=20)


def geometry_json(spectrum):
    from ..preview import scene_to_json
    objects=[dict(name='水30 cm（無限板の表示範囲±30 cm）',shape='box',material='water',center=[0,0,15],size_cm=[60,60,30]),
             dict(name='中心軸2×2 cm',shape='box',material='scoring',center=[0,0,15],size_cm=[2,2,30])]
    for axis in (0,1):
        for depth in (5,10,20):
            size=[2,2,1];size[axis]=40
            objects.append(dict(name=f'{"xy"[axis]} ±20 cm / 深さ{depth} cm',shape='box',material='scoring',center=[0,0,30-depth],size_cm=size))
    raw=dict(geometry=objects,source=dict(position=[0,0,130],direction=[0,0,-1],spectrum=[dict(energy_keV=10000 if spectrum=='mohan10' else 4000,weight=1)],field=dict(shape='rect',size_cm=[10,10],sid_cm=100)))
    data=scene_to_json(SimpleNamespace(raw=raw,warnings=[]));data['beam']['energy_label']='Mohan '+('10' if spectrum=='mohan10' else '4')+' MV'
    return data


def prepare(out,cond,bundle_path):
    from ..preview import _TEMPLATE, _SHARED_CANVAS_JS
    import shutil
    out=Path(out)
    if (out/'generation.json').exists() or (out/'approval.json').exists():raise ValueError('generated run directory cannot be overwritten')
    # Private probabilities may only be copied outside the repository.
    repo=Path(__file__).resolve().parents[2]
    if out.resolve()==repo or repo in out.resolve().parents:raise ValueError('private therapy bundle must stay outside repository')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(bundle_path,out/'bundle')
    contracts.write(out/'condition.json',cond)
    data=geometry_json(cond['spectrum']);contracts.write(out/'preview_snapshot.json',dict(condition=cond,geometry=data))
    text=_TEMPLATE.replace('__TITLE__','ChatCarlo G5 治療領域の承認').replace('__DATA__',json.dumps(data,ensure_ascii=False)).replace('__SHARED_CANVAS_JS__',_SHARED_CANVAS_JS)
    for marker in ('__DOSE_HEADER__','__DOSE_LEGEND__','__DOSE_JS__','__DOSE_DRAW__','__APPROVAL_PANEL__'):text=text.replace(marker,'')
    label=f"{cond['spectrum']} / 水30 cm / 10×10 cm² / SSD100 cm / 採点±20 cm / {NAME}\n履歴数 {cond['histories']} / 種 {cond['seed']} / {cond['engine']}\n条件 SHA256: {sha256(cond)}"
    (out/'preview.html').write_text(text.replace('<body>','<body><pre style="position:relative;z-index:10">'+html.escape(label)+'</pre>',1))
    contracts.seal(out,cond,('condition.json','preview_snapshot.json','preview.html','bundle/manifest.json'))


def verify(out):
    out=Path(out);g=contracts.verify(out,('condition.json','preview_snapshot.json','preview.html','bundle/manifest.json'));c=json.loads((out/'condition.json').read_text())
    _,_,_,_,m=therapy_inputs.load(out/'bundle',c['spectrum'])
    if c!=g['condition'] or condition(therapy_scene.validate(c['scene']),m)!=c:raise ValueError('generated condition/bundle/source mismatch')
    return g


class LocalDriver:
    def __init__(self,cpu):self.cpu=cpu
    def __call__(self,n,seed,start):return self.cpu.histories(n,seed,start=start,slots=256)
    def observation(self):return self.cpu.last_observation


def execute(out,cond,inputs,client=None):
    out=Path(out);contracts.require(out,cond,verify)
    materials,arrays,fg,volumes,_=inputs
    if cond['engine']=='cpu_local':
        from .therapy_checks import run as checks
        cpu=CPU(out/'build',materials,arrays,field_geometry=fg,half=(5.,5.))
        return therapy_run.run(out/'results',cond,arrays,fg,LocalDriver(cpu),lambda:checks(cpu,arrays,fg,cond['seed']))
    from .therapy_colab import run
    run(out,cond,arrays,fg,client=client)
    return therapy_run.collect(out/'results',cond,arrays,fg)


def main(argv=None):
    p=argparse.ArgumentParser(prog='chatcarlo (therapy_mv, G5)');sub=p.add_subparsers(dest='command',required=True)
    for name in ('validate','preview','approve','run','trace','animate','plot'):
        q=sub.add_parser(name);q.add_argument('scene')
        if name in ('preview','run'):q.add_argument('--out',type=Path)
        if name=='approve':q.add_argument('--preview',required=True,type=Path)
    raw=list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] in ('trace','animate','plot'):
        print('[エラー] 治療領域では未対応',file=sys.stderr);return 1
    args=p.parse_args(raw)
    try:
        s=therapy_scene.load(args.scene)
        if args.command in ('trace','animate','plot'):raise ValueError('治療領域では未対応')
        if args.command=='validate':
            print(f'OK: {args.scene} は有効な治療領域シーンです（{s["source"]["spectrum"]}、水30 cm、10×10 cm²、SSD 100 cm、採点±20 cm）');return 0
        inputs=therapy_inputs.load(therapy_inputs.DEFAULT_BUNDLE,s['source']['spectrum'])
        cond=condition(s,inputs[-1]);out=getattr(args,'out',None) or Path.home()/'.cache/chatcarlo/g5/a3/runs'/sha256(cond)[:12]
        if args.command=='approve':
            g=verify(args.preview)
            if g['condition']!=cond:raise ValueError('condition mismatch')
            contracts.approve(args.preview,verify);print('承認記録: '+str(args.preview/'approval.json'));return 0
        if args.command=='preview':prepare(out,cond,therapy_inputs.DEFAULT_BUNDLE);print('3D表示: '+str(out/'preview.html'));return 0
        execute(out,cond,inputs);print('結果: '+str(out/'results/dose.npz'));return 0
    except (OSError,ValueError,KeyError,RuntimeError) as exc:
        print('[エラー] '+str(exc),file=sys.stderr);return 1
