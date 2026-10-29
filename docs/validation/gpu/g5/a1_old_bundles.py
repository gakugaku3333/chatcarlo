"""Build the old (19c281e) g2b/g4 bundles with the frozen old tree; run with cwd=old tree.

usage: cd ~/.cache/chatcarlo/g5/a1/old_full && PYTHONPATH=. python <repo>/docs/validation/gpu/g5/a1_old_bundles.py <repo> <configs> <out>
"""
import json,sys
from pathlib import Path
repo,configs,out=map(Path,sys.argv[1:4])
assert Path.cwd().name=='old_full' and str(Path.cwd()) in sys.path or '.' in sys.path
from chatcarlo.gpu.bundle import create,sha
from chatcarlo.scene import load_scene
from chatcarlo.detector import DetectorPlane
import chatcarlo
assert Path(chatcarlo.__file__).resolve().is_relative_to(Path.cwd().resolve()),'old tree not imported'
cfg=json.loads(configs.read_text())
for c in cfg['configurations']:
    if sha(repo/c['scene'])!=c['scene_file_sha256']:raise ValueError('scene SHA256 mismatch')
    scene=load_scene(str(repo/c['scene']))
    if not scene.ok:raise ValueError(scene.errors)
    create(scene,DetectorPlane(**c['detector']) if c['detector'] else None,out/c['id'],g2_tally=bool(c['grid']))
    print('old bundle',c['id'])
