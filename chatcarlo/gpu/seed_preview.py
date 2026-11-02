"""Fixed geometry preview using shared approval contracts."""
import json
import html
from pathlib import Path
import shutil
import numpy as np
from . import seed_geometry as core,contracts
from .seed_scene import prostate_diameters
from .condition import sha256
EXCLUDED='入口の外: 強度変調・MLC・半影・人体形状・シード内部構造・EGS5との照合'

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


def prepare(out,cond,bundle):
    from chatcarlo.preview import _TEMPLATE,_SHARED_CANVAS_JS
    out=Path(out);repo=Path(__file__).resolve().parents[2]
    if out.resolve()==repo or repo in out.resolve().parents:raise ValueError('private bundle must stay outside repository')
    if (out/'generation.json').exists() or (out/'approval.json').exists():raise ValueError('cannot overwrite generated preview')
    from .therapy import space
    space(out,1024**2)
    out.mkdir(parents=True,exist_ok=True)
    # One sealed external bundle backs all four B5 previews, without four copies.
    (out/'bundle').symlink_to(Path(bundle).resolve(),target_is_directory=True)
    contracts.write(out/'condition.json',cond)
    data=geometry_json(cond['scene']);contracts.write(out/'preview_snapshot.json',dict(condition=cond,geometry=data))
    text=_TEMPLATE.replace('__TITLE__','ChatCarlo A2b 前立腺シード7門').replace('__DATA__',json.dumps(data,ensure_ascii=False)).replace('__SHARED_CANVAS_JS__',_SHARED_CANVAS_JS).replace('poly(b.corners,','if (b.corners.length) poly(b.corners,')
    for marker in ('__DOSE_HEADER__','__DOSE_LEGEND__','__DOSE_JS__','__DOSE_DRAW__','__APPROVAL_PANEL__'):text=text.replace(marker,'')
    label=f"条件 SHA256: {sha256(cond)}\nMohan 10 MV / SAD100 cm / シード58本 / 前立腺形7門 / 0.5 mm\n50バッチ、各門 {cond['histories_per_field']}、各側 {cond['histories_per_side']}、種 {cond['seed']}\n{EXCLUDED}"
    (out/'preview.html').write_text(text.replace('<body>','<body><pre style="position:relative;z-index:10;white-space:pre-wrap">'+html.escape(label)+'</pre>',1))
    contracts.seal(out,cond,('condition.json','preview_snapshot.json','preview.html','bundle/manifest.json'))
