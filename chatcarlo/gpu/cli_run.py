"""CLI orchestration: fixed approved bundles on Colab, CPU twin locally."""
from pathlib import Path
import tempfile
import json
from . import condition, approval, bundle


def expected_run(scene, options):
    from ..scene import detector_from_scene
    plane, _ = detector_from_scene(scene)
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g4-condition-') as temp:
        manifest = bundle.create(scene, plane, temp)
        return condition.colab_condition(manifest, condition.job_for(options), scene, options['track_uncertainty'])


def run(scene, args):
    from ..scene import detector_from_scene
    from .aggregate import aggregate
    options = condition.resolve(args)
    plane, roi = detector_from_scene(scene)
    if options['device'] == 'cpu':
        from .engine import run as local_run
        result = local_run(scene, options['n_histories'], options['seed'], options['batch_size'], False, 1,
                           options['track_uncertainty'], plane, roi, 'cpu', options['source_seed'])
        cond = condition.local_condition(scene, options)
    else:
        from .colab_driver import launch, read_result, expected_identity
        cond = expected_run(scene, options)
        path = Path(args.run_dir) if args.run_dir else condition.default_run_dir(cond)
        approval.require(path, cond)
        job = json.loads((path/'job.json').read_text())
        _, _, _, settings, manifest = bundle.load(path/'bundle')
        results = path/'results'
        reused = set()
        for receipt in results.glob('*.receipt.json'):
            name = receipt.name[:-13]
            if name == 'source_reference.json': continue
            if read_result(results, name, expected_identity(manifest, job, name)) is not None:
                reused.add(name)
        gates = Path(__file__).resolve().parents[2]/'docs/validation/gpu/g1'
        launch(path/'bundle', job, results, gate1=gates/'gate1.json', gate2=gates/'gate2.json')
        batches, timing = approval.verify_batches(path/'bundle', job, results, reused)
        result = aggregate(batches, settings['material_names'], plane, roi, options['track_uncertainty'])
        result.gpu_metadata = dict(engine='gpu', gpu_device='cuda', seed=options['seed'], source_seed=options['source_seed'], batch_size=options['batch_size'], bundle=manifest, timing=timing)
    result.gpu_metadata.update(condition=cond, condition_sha256=condition.sha256(cond), verification='native inputs verified')
    return result


def save_detector(scene, args, result):
    from .outputs import save
    opts = condition.resolve(args)
    meta = getattr(result, 'gpu_metadata', {})
    cond = meta.get('condition', condition.local_condition(scene, opts))
    digest = condition.sha256(cond)
    meta = dict(meta, condition=cond, condition_sha256=digest, engine=opts['engine'], device=opts['device'])
    requested = getattr(args, 'detector_out', None)
    if opts['engine'] == 'gpu' and opts['device'] == 'colab':
        base = Path(args.run_dir) if args.run_dir else condition.default_run_dir(cond)
        out = base/'detector.npz'
    else:
        out = Path(requested) if requested else Path('chatcarlo_out')/digest[:12]/'detector.npz'
    save(result, out, meta)
    if requested is not None and Path(requested).with_suffix('.npz').resolve() != Path(out).with_suffix('.npz').resolve():
        import shutil
        destination = Path(requested).with_suffix('.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ('.npz', '.json', '.png'):
            shutil.copyfile(Path(out).with_suffix(suffix), destination.with_suffix(suffix))
        out = destination
    print(f'検出器画像: {Path(out).with_suffix(".npz")} (.json, .png)')
    return Path(out).with_suffix('.npz')
