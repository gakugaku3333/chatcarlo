"""Pre-registered G2 gates. A failed gate is recorded and never retried."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] if HERE.name == 'g2' else None


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # NumPy scalars (bool_, int64, float64) are written as plain JSON values.
    convert = lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=convert)+'\n')


def gate0(out):
    checks = {}
    for part in ('0a', '0b', '0c'):
        command = [sys.executable, '-m', 'pytest', 'tests/test_gpu_g2.py', '-q', '-k', 'gate'+part]
        t = time.perf_counter()
        p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        checks[part] = dict(status='pass' if p.returncode == 0 else 'fail',
                            command=command, exit_code=p.returncode,
                            stdout=p.stdout, stderr=p.stderr, wall_seconds=time.perf_counter()-t)
        write(Path(out)/'gate0.json', dict(status=checks[part]['status'], checks=checks))
        if p.returncode:
            return False
    return True


def stats(batches):
    mean = batches.mean(axis=0)
    sem = batches.std(axis=0, ddof=1)/np.sqrt(len(batches))
    hits = np.count_nonzero(batches, axis=0)
    r = np.divide(sem, mean, out=np.full_like(mean, np.inf), where=mean != 0)
    return mean, sem, hits, np.isfinite(r) & (r <= .10) & (hits >= len(batches)*.25)


def judge(reference, twin, regions):
    from scipy.stats import t
    rm, rs, rh, re = stats(reference)
    tm, ts, th, te = stats(twin)
    eligible = re & te
    n = int(eligible.sum())
    den = np.hypot(rs, ts)
    z = np.divide(tm-rm, den, out=np.zeros_like(tm), where=den > 0)
    z[(den == 0) & (tm != rm)] = np.inf
    missing = re & ((tm == 0) | ~np.isfinite(tm) | (th == 0))
    limit = float(t.isf(.005/n, 19)) if n else 0.
    relative = ((tm[eligible]-rm[eligible])/rm[eligible]) if n else np.array([])
    checks = dict(nonempty=n > 0, missing=int(missing.sum()) == 0,
                  outliers=bool(n and np.max(np.abs(z[eligible])) <= limit),
                  spread=bool(n and np.mean(np.abs(z[eligible]) > 3) <= .01))
    report = dict(n_eligible=n, missing_indices=np.argwhere(missing).tolist(),
                  max_abs_z=float(np.max(np.abs(z[eligible]))) if n else None,
                  t_limit=limit, fraction_abs_z_gt3=float(np.mean(np.abs(z[eligible]) > 3)) if n else None,
                  relative_difference_quantiles=np.quantile(relative, [.025, .5, .975]).tolist() if n else [],
                  mean_z2=float(np.mean(z[eligible]**2)) if n else None, regions={})
    for name, mask in {**regions, 'all': np.ones(rm.shape, bool)}.items():
        rb = reference[:, mask].sum(axis=1)
        tb = twin[:, mask].sum(axis=1)
        se = np.hypot(rb.std(ddof=1), tb.std(ddof=1))/np.sqrt(20)
        diff = tb.mean()-rb.mean()
        rz = float(diff/se) if se else (0. if diff == 0 else None)
        a = re & ~te & mask
        b = te & ~re & mask
        na, nb = int(a.sum()), int(b.sum())
        count = int((eligible & mask).sum())
        minimum = dict(water=1000, air=1000, bone=8, lead_layer=500).get(name, 0)
        checks[name+'_total'] = rz is not None and abs(rz) <= 3
        checks[name+'_minimum'] = count >= minimum
        checks[name+'_dropouts'] = abs(na-nb) <= 3*np.sqrt(na+nb)+5
        report['regions'][name] = dict(n_eligible=count, minimum=minimum, z=rz,
                                      reference_only=na, twin_only=nb,
                                      reference_only_indices=np.argwhere(a).tolist(),
                                      twin_only_indices=np.argwhere(b).tolist())
    report.update(checks=checks, status='pass' if all(checks.values()) else 'fail')
    return report


def gate1(out):
    from chatcarlo.geometry import Geometry
    from chatcarlo.tally import VoxelGrid
    from chatcarlo.transport import run_transport
    from chatcarlo.scene import load_scene
    from chatcarlo.gpu.bundle import sha
    from make_reference import generate, SCENES
    data = ROOT/'data/gpu_g2'
    # Unconditional regeneration, completed for BOTH scenes before the twin.
    refs = generate(data)
    report = dict(status='running', references=refs, scenes={},
                  interpretation='この標本の精度で不一致を検出したかを判定する。数 % 以内の一致の保証ではない。')
    write(Path(out)/'gate1.json', report)
    for name, base in (('water', 221100), ('lead', 221200)):
        scene = load_scene(str(SCENES[name]))
        with np.load(refs[name]['path']) as z:
            ref = {q: z[q] for q in ('kerma', 'h10')}
            grid = VoxelGrid(z['origin'], tuple(z['shape']), 2)
        geom = Geometry(scene.raw['geometry'])
        material = geom.material_at(grid.voxel_centers()).reshape(grid.shape)
        regions = {m: material == m for m in ('water', 'air') + (('bone',) if name == 'lead' else ())}
        if name == 'lead':
            zs = grid.origin_cm[2]+np.arange(grid.shape[2])*2
            layer = np.broadcast_to(((zs < 0) & (zs+2 > -.05))[None, None, :], grid.shape)
            regions = {m: mask & ~layer for m, mask in regions.items()}
            regions['lead_layer'] = layer
        twin = {'kerma': [], 'h10': []}
        started = time.perf_counter()
        for k in range(20):
            r = run_transport(scene, n_histories=200000, seed=base+k, batch_size=200000,
                              dose_grid=True, grid_resolution_cm=2, engine='gpu')
            twin['kerma'].append(r.grid.kerma_keV)
            twin['h10'].append(r.grid.h10_track_pSv_cm3)
            print(f'twin {name}: {k+1}/20', flush=True)
        elapsed = time.perf_counter()-started
        twin = {q: np.asarray(v) for q, v in twin.items()}
        path = data/f'twin_{name}_batches.npz'
        np.savez(path, **twin, seeds=np.arange(base, base+20))
        results = {q: judge(ref[q], twin[q], regions) for q in twin}
        report['scenes'][name] = dict(quantities=results, seeds=list(range(base, base+20)),
                                     source_seed='G1 SeedSequence derivation', wall_seconds=elapsed,
                                     twin_sha256=sha(path), scene_sha256=sha(SCENES[name]))
        report['status'] = 'pass' if all(v['status'] == 'pass' for v in results.values()) else 'fail'
        write(Path(out)/'gate1.json', report)
        if report['status'] != 'pass':
            return False
    return True


def prepare_gate2(out):
    from chatcarlo.gpu import bundle
    from chatcarlo.geometry import Geometry
    from chatcarlo.scene import load_scene
    from chatcarlo.tally import VoxelGrid
    from run_egs5_60kev import prepare
    prepare(Path(out)/'egs5')
    for name in ('lead', 'pdd'):
        target = Path(out)/name
        if name == 'lead':
            scene = load_scene(str(HERE/'scene_lead_slab.yaml'))
            bundle.create(scene, None, target, g2_tally=True)
            geom = Geometry(scene.raw['geometry'])
        else:
            import shutil
            shutil.copytree(Path(out)/'egs5/pdd', target)
            a, _, _, _, _ = bundle.load(target)
            class GeometryBounds:
                tally_bbox_min = a['bbox_min']
                tally_bbox_max = a['bbox_max']
            geom = GeometryBounds()
        grid = VoxelGrid.from_bbox(geom.tally_bbox_min, geom.tally_bbox_max, 2)
        a, _, _, settings, manifest = bundle.load(target)
        with np.load(target/'g2_tables.npz') as z:
            data = {k: z[k] for k in z.files}
        if name == 'lead':
            data.update(boxes=np.empty((0, 6)), water=np.array(settings['material_names'].index('water')))
        data.update(grid_origin=grid.origin_cm, grid_shape=grid.shape, grid_resolution=np.array(2.))
        np.savez(target/'g2_tables.npz', **data)
        manifest['files']['g2_tables.npz'] = bundle.sha(target/'g2_tables.npz')
        write(target/'manifest.json', manifest)
        bundle.verify(target)
        write(Path(out)/(name+'_job.json'), dict(runner='run_gates.py',
              runner_sha256=manifest['files']['run_gates.py'], gate=2))


def gate2(bundle_path, out):
    """VM-only paired comparison; no physics packages needed on the VM."""
    from chatcarlo.gpu import bundle, validation
    from chatcarlo.gpu.g2_core import Core
    from chatcarlo.gpu.source_host import sample_source_photons
    arrays, src, spec, settings, manifest = bundle.load(bundle_path)
    # Geometry/grid/box descriptions are supplied in the job bundle by the author.
    with np.load(Path(bundle_path)/'g2_tables.npz') as z:
        origin, shape, resolution = z['grid_origin'], z['grid_shape'], float(z['grid_resolution'])
        box_array = z['boxes']
        water = int(z['water'])
    class Grid:
        origin_cm = origin
        voxel_size_cm = resolution
    grid = Grid()
    grid.shape = tuple(shape)
    boxes = [(str(j), b[:3], b[3:]) for j, b in enumerate(box_array)]
    cpu = Core(arrays, 'cpu', grid=grid, boxes=boxes, water=water, count_additions=True)
    gpu = Core(arrays, 'cuda', grid=grid, boxes=boxes, water=water, count_additions=True)
    rng = np.random.default_rng(3201)
    values = {q: [[], [], []] for q in ('kerma', 'h10', 'box_sum', 'box_sum2')}
    integers = []
    timing = dict(cpu_s=0., gpu_s=0.)
    for b in range(10):
        o, d, e = sample_source_photons(src, 100000, rng, spec)
        pair = []
        for label, driver in (('cpu', cpu), ('gpu', gpu)):
            started = time.perf_counter()
            pair.append(driver.run(o, d, e, seed=3200, start=b*100000,
                                   fluorescence=settings['fluorescence'], record=True))
            timing[label+'_s'] += time.perf_counter()-started
        c, g = pair
        assert c['endings'].sum() == g['endings'].sum() == 100000
        ints = {}
        for q in ('grid_additions', 'box_additions'):
            ints[q] = np.flatnonzero(c[q] != g[q]).tolist()
        ints['voxel_hit'] = np.flatnonzero((c['kerma'] > 0) != (g['kerma'] > 0)).tolist()
        ints['diverged_histories'] = np.flatnonzero(np.any(c['i'] != g['i'], axis=1) |
                                                  np.any(c['d'] != g['d'], axis=1)).tolist()
        integers.append(ints)
        for q, (cs, gs, bounds) in values.items():
            count = 'grid_additions' if q in ('kerma', 'h10') else 'box_additions'
            cs.append(c[q]); gs.append(g[q])
            bounds.append((c[count]+g[count]+2*10+16)*2.**-52*(np.abs(c[q])+np.abs(g[q])))
        np.savez(Path(out)/f'paired_{b:02d}.npz', **{label+'_'+q: r[q] for label, r in zip(('cpu', 'gpu'), pair)
                                                  for q in ('kerma', 'h10', 'grid_additions', 'box_sum', 'box_sum2', 'box_additions')})
    reports = {q: validation.paired_judge(*v) for q, v in values.items()}
    report = dict(status='pass' if all(v['status'] == 'pass' for v in reports.values()) else 'fail',
                  quantities=reports, integer_differences=integers, bundle=manifest, timing=timing)
    write(Path(out)/'gate2.json', report)
    return report['status'] == 'pass'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', type=int, choices=(0, 1, 2))
    ap.add_argument('--prepare-gate2', action='store_true')
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--out', type=Path, default=HERE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare_gate2:
        prepare_gate2(args.out)
        return
    if args.gate is None:
        ap.error('--gate or --prepare-gate2 is required')
    ok = gate0(args.out) if args.gate == 0 else gate1(args.out) if args.gate == 1 else gate2(args.bundle, args.out)
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
