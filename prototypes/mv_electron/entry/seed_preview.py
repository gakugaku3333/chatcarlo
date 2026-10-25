"""Direct meshes in kernel coordinates, bound to a reproducible approval snapshot."""
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .condition import digest, check_equal
from . import seed_condition as condition
from .seed_scene import prostate_diameters
from ..gpu import p9c_core as core
from ..gpu.p5b_core import write_json

EXCLUDED = '入口の外: 強度変調・MLC・半影・人体形状・シード内部構造・EGS5 との照合'


def cylinder(center, radii, length, n=64):
    rings = [[[center[0]+radii[0]*np.cos(t), center[1]+end*length/2,
               center[2]+radii[1]*np.sin(t)] for t in np.arange(n)*2*np.pi/n] for end in (-1,1)]
    a, b = rings
    edges = [[ring[j], ring[(j+1)%n]] for ring in rings for j in range(n)]
    edges += [[a[j],b[j]] for j in range(0,n,4)]
    faces = [[a[j],a[(j+1)%n],b[(j+1)%n],b[j]] for j in range(n)]+rings
    return edges, faces


def geometry_json(scene):
    from chatcarlo.preview import _sphere_mesh
    objects = []
    def add(name, material, color, center, mesh, **metadata):
        edges, faces = mesh
        objects.append(dict(name=name, material=material, color=color, label_at=center,
                            edges=edges, faces=faces, **metadata))
    body = scene['body']; ellipse = [v/2 for v in body['ellipse_cm']]
    add('体（水）', 'water', '#60a5fa', [0,0,0], cylinder([0,0,0],ellipse,body['length_cm']),
        radii_cm=ellipse, length_cm=body['length_cm'], axis='y')
    diam = prostate_diameters(scene)
    edges, _ = _sphere_mesh([0,0,0],1,n=64)
    edges = [[[p[k]*diam[k]/2 for k in range(3)] for p in e] for e in edges]
    add('前立腺（水中の評価領域）', 'water', '#fbbf24', [0,0,0], (edges,[]), diameters_cm=diam)
    radius = scene['target']['rectum_diameter_cm']/2
    add('直腸（水中の評価領域）', 'water', '#f97316', [0,0,3], cylinder([0,0,3],[radius]*2,diam[1]),
        radius_cm=radius, length_cm=diam[1], axis='y')
    for j,(x,y,z,r,h,_) in enumerate(core.seeds()):
        add(f'シード {j+1}', 'ag_ti_mixture', '#c084fc', [x,y,z], cylinder([x,y,z],[r,r],2*h,24),
            radius_cm=r, half_length_cm=h, axis='y')
    sources = []
    for angle in scene['source']['gantry_deg']:
        s, direction, _, _ = core.frame(angle)
        hit = core.entry(s,direction)
        sources.append(dict(angle=angle, position=s.tolist(), body_entry=hit.tolist(), distance_to_surface_cm=float(np.linalg.norm(hit-s))))
        mesh = ([[s.tolist(),[0,0,0]]],[])
        add(f'線源 {angle}° SAD 100 cm', 'source', '#ef4444', s.tolist(), mesh)
    return dict(objects=objects, beam=dict(source=sources[0]['position'], corners=[], energy_label='Mohan 10 MV',
                field_label='prostate_projection', sid=100, source_label='SAD 100 cm'),
                sources=sources, center=[0,0,0], radius=105, warnings=[], trajectories=[])


def snapshot(cond):
    return dict(condition=cond, condition_hash=digest(cond), bundle_hash=digest(cond['manifest']),
                t0_hash=digest(cond['t0']), geometry=geometry_json(cond['scene']))


def render(snap):
    from chatcarlo.preview import _TEMPLATE, _SHARED_CANVAS_JS
    text = (_TEMPLATE.replace('__TITLE__', 'ChatCarlo — 前立腺シード7門 承認用プレビュー')
            .replace('__DATA__', json.dumps(snap['geometry'],ensure_ascii=False))
            .replace('__SHARED_CANVAS_JS__',_SHARED_CANVAS_JS)
            .replace(' @ SID ', ' @ SAD ')
            .replace('poly(b.corners,', 'if (b.corners.length) poly(b.corners,')
            .replace('X線ビーム（照射野 ${DATA.beam.field_label} cm',
                     'X線ビーム（照射野 ${DATA.beam.field_label}'))
    for marker in ('__DOSE_HEADER__','__DOSE_LEGEND__','__DOSE_JS__','__DOSE_DRAW__','__APPROVAL_PANEL__'):
        text = text.replace(marker,'')
    run = snap['condition']['scene']['run']; n = run['histories_per_field']
    description = (f'条件ハッシュ: {snap["condition_hash"]}\n束のハッシュ: {snap["bundle_hash"]}\nT0 のハッシュ: {snap["t0_hash"]}\n'
        f'Mohan 10 MV、7 門、SAD 100 cm、各門 {n} 光子、各側の合計 {7*n} 光子、種 {run["seed"]}\n'
        'シード 58 本、銀・チタン混合物: 密度 2.47 g/cm³、銀 59.6%・チタン 40.4%\n'
        'x=左右、y=頭尾、z=前→後。前立腺・直腸は水中の評価領域。\n'+EXCLUDED)
    return text.replace('<body>','<body><pre style="position:relative;z-index:10;white-space:pre-wrap">'+html.escape(description)+'</pre>',1)


def write(scene, directory):
    cond = condition.identities(scene); snap = snapshot(cond)
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'condition.json',cond)
    write_json(directory/'preview_snapshot.json',snap)
    (directory/'preview.html').write_text(render(snap))
    return snap


def verify_preview(directory, cond):
    directory = Path(directory); snap = snapshot(cond)
    check_equal(condition.read(directory/'preview_snapshot.json'),snap,'preview_snapshot')
    text = (directory/'preview.html').read_text()
    visible = re.sub(r'<script\b[^>]*>.*?</script>','',text,flags=re.S|re.I)
    visible = re.sub(r'<!--.*?-->','',visible,flags=re.S)
    for key in ('condition_hash','bundle_hash','t0_hash'):
        if snap[key] not in visible:
            raise ValueError('preview.html: visible hash missing')
    check_equal(text,render(snap),'preview.html')


def approve(scene, directory):
    cond = condition.identities(scene); directory = Path(directory)
    verify_preview(directory,cond)
    record = dict(condition_hash=digest(cond), bundle_hash=digest(cond['manifest']),t0_hash=digest(cond['t0']),
                  snapshot_sha256=condition.io.sha(directory/'preview_snapshot.json'),
                  html_sha256=condition.io.sha(directory/'preview.html'),time=datetime.now(timezone.utc).isoformat())
    write_json(directory/'approval.json',record)
    return record


def require_approval(directory, cond):
    directory = Path(directory)
    if not (directory/'approval.json').is_file():
        raise ValueError('approval.json: 使用者による3D表示の承認が必要です')
    record = condition.read(directory/'approval.json')
    check_equal(record['condition_hash'],digest(cond),'approval condition_hash')
    for key,value in (('bundle_hash',digest(cond['manifest'])),('t0_hash',digest(cond['t0'])),
                      ('snapshot_sha256',condition.io.sha(directory/'preview_snapshot.json')),
                      ('html_sha256',condition.io.sha(directory/'preview.html'))):
        check_equal(record[key],value,'approval '+key)
    verify_preview(directory,cond)
    return record
