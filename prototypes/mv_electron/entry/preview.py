"""Self-contained 3D approval preview, sharing the diagnostic canvas renderer."""
import html
import json
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from .condition import identities, digest, check_equal
from ..gpu import p5b_core as core


def display_z(depth):
    return 30-depth


def geometry_json():
    from chatcarlo.preview import scene_to_json
    objects = [dict(name='横に無限の板を ±30 cm で切って表示', shape='box', material='water', center=[0, 0, 15], size_cm=[60, 60, 30]),
               dict(name='中心軸 2×2 cm', shape='box', material='scoring', color='#fbbf24', center=[0, 0, 15], size_cm=[2, 2, 30])]
    for depth in (5, 10, 20):
        for axis in (0, 1):
            size = [2, 2, 1]
            size[axis] = 18
            objects.append(dict(name=f'深さ {depth} cm / {"xy"[axis]} ±9 cm', shape='box', material='scoring', color='#f97316',
                                center=[0, 0, display_z(depth)], size_cm=size))
    raw = dict(geometry=objects, source=dict(position=[0, 0, 130], direction=[0, 0, -1],
                                           spectrum=[dict(energy_keV=10000, weight=1)],
                                           field=dict(shape='rect', size_cm=[10, 10], sid_cm=100)))
    data = scene_to_json(SimpleNamespace(raw=raw, warnings=[]))
    data['beam']['energy_label'] = 'Mohan 10 MV'
    return data


def snapshot(cond, mat):
    return dict(condition=cond, materials=mat, condition_hash=digest(cond), materials_hash=digest(mat), geometry=geometry_json())


def write(scene, directory, data=None):
    from chatcarlo.preview import _TEMPLATE, _SHARED_CANVAS_JS
    cond, mat, _ = identities(scene, data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snap = snapshot(cond, mat)
    core.write_json(directory/'condition.json', cond)
    core.write_json(directory/'materials.json', mat)
    core.write_json(directory/'preview_snapshot.json', snap)
    text = (_TEMPLATE.replace('__TITLE__', 'ChatCarlo — 10 MV 承認用プレビュー')
            .replace('__DATA__', json.dumps(snap['geometry'], ensure_ascii=False))
            .replace('__SHARED_CANVAS_JS__', _SHARED_CANVAS_JS))
    description = (f'条件ハッシュ: {snap["condition_hash"]}\n資材ハッシュ: {snap["materials_hash"]}\n'
                   f'照射野 10×10 cm² @ SSD 100 cm、水 30 cm（横に無限の板を ±30 cm で切って表示）\n'
                   f'採点: 中心軸 2×2 cm、深さ 5/10/20 cm の帯 ±9 cm\n'
                   f'履歴数 {cond["histories"]}、種 {cond["seed"]}、100 × {cond["batch_histories"]}、{cond["engine"]}')
    text = text.replace('<body>', '<body><pre style="position:relative;z-index:10;white-space:pre-wrap">'+html.escape(description)+'</pre>', 1)
    (directory/'preview.html').write_text(text)
    return snap


def approve(scene, directory, data=None):
    directory = Path(directory)
    cond, mat, _ = identities(scene, data)
    path = directory/'preview_snapshot.json'
    snap = json.loads(path.read_text())
    check_equal(snap, snapshot(cond, mat), 'preview_snapshot')
    text = (directory/'preview.html').read_text()
    # Hashes must be visible text, not only the embedded JSON/script.
    import re
    visible = re.sub(r'<script\b[^>]*>.*?</script>', '', text, flags=re.S | re.I)
    for value in (digest(cond), digest(mat)):
        if value not in visible:
            raise core.Stop('preview.html: visible hash missing')
    record = dict(condition_hash=digest(cond), materials_hash=digest(mat),
                  snapshot_sha256=core.file_sha(path), html_sha256=core.file_sha(directory/'preview.html'),
                  time=datetime.now(timezone.utc).isoformat())
    core.write_json(directory/'approval.json', record)
    return record


def require_approval(directory, cond, mat):
    directory = Path(directory)
    path = directory/'approval.json'
    if not path.is_file():
        raise core.Stop('approval.json: 使用者による3D表示の承認が必要です')
    record = json.loads(path.read_text())
    check_equal(record['condition_hash'], digest(cond), 'approval condition_hash')
    check_equal(record['materials_hash'], digest(mat), 'approval materials_hash')
    check_equal(record['snapshot_sha256'], core.file_sha(directory/'preview_snapshot.json'), 'snapshot SHA256')
    check_equal(record['html_sha256'], core.file_sha(directory/'preview.html'), 'HTML SHA256')
    return record
