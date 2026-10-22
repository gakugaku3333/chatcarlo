"""Regenerate independent NumPy batches BEFORE inspecting the twin gate-1 run."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from chatcarlo.gpu.bundle import sha
from chatcarlo.scene import load_scene
from chatcarlo.transport import run_transport

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCENES = {'water': ROOT/'examples/water_phantom_pdd_ocr.yaml',
          'lead': HERE/'scene_lead_slab.yaml'}


def generate(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, base in (('water', 220100), ('lead', 220200)):
        scene = load_scene(str(SCENES[name]))
        if not scene.ok:
            raise ValueError(scene.errors)
        batches = {'kerma': [], 'h10': []}
        started = time.perf_counter()
        for k in range(20):
            r = run_transport(scene, n_histories=200000, seed=base+k,
                              batch_size=200000, dose_grid=True, grid_resolution_cm=2)
            grid = r.grid
            batches['kerma'].append(grid.kerma_keV)
            batches['h10'].append(grid.h10_track_pSv_cm3)
            print(f'reference {name}: {k+1}/20', flush=True)
        path = out/f'ref_{name}_batches.npz'
        np.savez(path, **{k: np.asarray(v) for k, v in batches.items()},
                 origin=grid.origin_cm, shape=grid.shape, resolution=2,
                 seeds=np.arange(base, base+20), histories_per_batch=200000,
                 scene_sha256=sha(SCENES[name]))
        metadata[name] = dict(path=str(path), sha256=sha(path), seeds=list(range(base, base+20)),
                              wall_seconds=time.perf_counter()-started,
                              scene_sha256=sha(SCENES[name]))
    (out/'reference.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'data/gpu_g2')
    generate(ap.parse_args().out)
