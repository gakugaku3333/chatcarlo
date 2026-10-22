"""G2 exact historical boxes; seeds/history counts are fixed before execution.

Prepare bundles on the host, execute each bundle on Colab, then compare on the
host using the unchanged historical comparison functions and saved EGS5 data.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] if HERE.name == 'g2' else None
OLD = ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07' if ROOT else None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare(out):
    from chatcarlo.scene import load_scene
    from chatcarlo.gpu import bundle
    import yaml
    old = module('g2_historical_runner', OLD/'run_chatcarlo.py')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for mode, depth, seed, n, batch in (('pdd', 20., 3101, 5000000, 100000),
                                       ('thin', .2, 3102, 500000, 25000),
                                       ('phantom', 20., 3103, 500000, 25000)):
        bins = old.old._build_bins() if mode == 'pdd' else [
            ('surface', np.array([-5., -5., 0.]), np.array([5., 5., .2]))]
        raw = dict(source=dict(type='xray_tube', spectrum=[dict(energy_keV=60., weight=1.)],
                               position=[0, 0, -.0001], direction=[0, 0, 1],
                               field=dict(shape='parallel', size_cm=[10, 10])),
                   geometry=[dict(name='water', shape='box', material='water',
                                  center=[0., 0., depth/2], size_cm=[30., 30., depth])],
                   physics=dict(fluorescence=True))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene.yaml'
            path.write_text(yaml.safe_dump(raw))
            scene = load_scene(str(path))
            if not scene.ok:
                raise ValueError(scene.errors)
            target = out/mode
            bundle.create(scene, None, target, bbox_margin_cm=.01, g2_tally=True)
        arrays, _, _, settings, _ = bundle.load(target)
        geom = old.geometry(depth)
        np.testing.assert_array_equal(arrays['bbox_min'], geom.bbox_min)
        np.testing.assert_array_equal(arrays['bbox_max'], geom.bbox_max)
        with np.load(target/'g2_tables.npz') as z:
            tables = {k: z[k] for k in z.files}
        tables.update(boxes=np.array([[*lo, *hi] for _, lo, hi in bins]),
                      box_names=np.array([name for name, _, _ in bins]),
                      water=np.array(settings['material_names'].index('water')),
                      seed=np.array(seed), n_histories=np.array(n), batch_size=np.array(batch),
                      depth=np.array(depth))
        np.savez(target/'g2_tables.npz', **tables)
        manifest = json.loads((target/'manifest.json').read_text())
        manifest['files']['g2_tables.npz'] = bundle.sha(target/'g2_tables.npz')
        (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        bundle.verify(target)


def execute(path, out):
    from chatcarlo.gpu.bundle import load
    from chatcarlo.gpu.g2_core import Core
    arrays, src, spec, settings, manifest = load(path)
    with np.load(Path(path)/'g2_tables.npz') as z:
        names = z['box_names'].tolist()
        boxes = [(name, row[:3], row[3:]) for name, row in zip(names, z['boxes'])]
        seed, n, batch, water = [int(z[k]) for k in ('seed', 'n_histories', 'batch_size', 'water')]
    driver = Core(arrays, 'cuda', boxes=boxes, water=water)
    sums, sums2 = np.zeros(len(boxes)), np.zeros(len(boxes))
    rng = np.random.default_rng(seed)
    diagnostics = dict(n_histories=n, seed=seed, source_seed=seed, batch_size=batch,
                       n_primary_transmitted=0, n_absorbed=0, n_escaped=0)
    started = time.perf_counter()
    for done in range(0, n, batch):
        count = min(batch, n-done)
        # Same draws/order as the historical _sample_parallel_beam.
        o = np.column_stack((rng.uniform(-5, 5, count), rng.uniform(-5, 5, count), np.full(count, -.0001)))
        d = np.tile([0., 0., 1.], (count, 1))
        e = np.full(count, 60.)
        r = driver.run(o, d, e, seed=seed, start=done, record=True, fluorescence=True)
        sums += r['box_sum']; sums2 += r['box_sum2']
        diagnostics['n_absorbed'] += int(r['endings'][0])
        diagnostics['n_escaped'] += int(r['endings'][1])
        diagnostics['n_primary_transmitted'] += int(((r['i'][:, 3] == 1) & (r['i'][:, 1] == 0)).sum())
        print(f'{Path(path).name}: {done+count}/{n}', flush=True)
    diagnostics['wall_seconds'] = time.perf_counter()-started
    rows = {}
    for j, (name, lo, hi) in enumerate(boxes):
        mass = float(np.prod(hi-lo))
        factor = 1.602176634e-13/mass  # historical KEV_PER_G_TO_GY, SI definition
        mean = sums[j]/n
        sem = np.sqrt(max(0., (sums2[j]-sums[j]**2/n)/(n-1))/n)
        rows[name] = dict(lo_cm=lo.tolist(), hi_cm=hi.tolist(), mass_g=mass,
                          sum_keV=float(sums[j]), sum2_keV2=float(sums2[j]),
                          mean_Gy_per_history=float(mean*factor), sem_Gy_per_history=float(sem*factor),
                          rel_err=float(sem/mean))
    result = dict(manifest=dict(bundle=manifest, seed=seed, source_seed=seed),
                  results=dict(diagnostics=diagnostics, bins=rows))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/(Path(path).name+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def compare(out):
    """Import the old comparison; override inputs/output only, never its math."""
    out = Path(out)
    comparator = module('g2_comparator', OLD/'compare.py')
    pdd = json.loads((out/'pdd.json').read_text())['results']
    thin = json.loads((out/'thin.json').read_text())['results']
    phantom = json.loads((out/'phantom.json').read_text())['results']
    a, b = phantom['bins']['surface'], thin['bins']['surface']
    ratio = a['mean_Gy_per_history']/b['mean_Gy_per_history']
    rel = float(np.hypot(a['rel_err'], b['rel_err']))
    bsf = dict(thin=thin, phantom=phantom, BSF_w=ratio, sem_BSF_w=ratio*rel, rel_sem_BSF_w=rel)
    (out/'bsf.json').write_text(json.dumps(dict(results=bsf), indent=2)+'\n')
    # Original functions retain OLD for EGS5 and published coefficients. Only
    # output paths and GPU inputs are redirected via a path adapter.
    class OutputPath:
        parents = OLD.parents
        def __fspath__(self):
            return str(OLD)
        def __truediv__(self, name):
            return out/name if name in ('comparison.csv', 'comparison.json', 'pdd_ocr_comparison.png') else OLD/name
    comparator.HERE = OutputPath()
    comparator.cc = lambda mode: pdd if mode == 'pdd' else bsf
    comparator.main()
    from chatcarlo.gpu.bundle import sha
    (out/'gpu_provenance.json').write_text(json.dumps(
        {name: sha(out/(name+'.json')) for name in ('pdd', 'thin', 'phantom', 'bsf')}, indent=2)+'\n')
    old_counts = json.loads((OLD/'comparison.json').read_text())['PDD_OCR']['counts']
    new = json.loads((out/'comparison.json').read_text())
    (out/'RESULTS.md').write_text(
        'NumPy (2026-10-07): '+json.dumps(old_counts)+'\n\nGPU: '+json.dumps(new['PDD_OCR']['counts'])+
        '\n\nBSF: '+json.dumps(new['BSF_w'])+
        '\n\nEGS5 Rayleigh fit warning: unquantified reference uncertainty.\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--compare', action='store_true')
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.prepare:
        prepare(args.out)
    elif args.compare:
        compare(args.out)
    else:
        execute(args.bundle, args.out)


if __name__ == '__main__':
    main()
