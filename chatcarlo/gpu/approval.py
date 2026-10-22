"""Immutable preview generation, approval, and native-input verification."""
import json
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
import numpy as np
from . import bundle, condition


def _write(path, value):
    Path(path).write_text(condition.canonical(value)+'\n')


def prepare(scene, options, run_dir=None):
    from ..scene import detector_from_scene
    from ..preview import render_html, scene_to_json
    if not scene.ok: raise ValueError('; '.join(map(str, scene.errors)))
    plane, roi = detector_from_scene(scene)
    opts = condition.resolve(options)
    with tempfile.TemporaryDirectory(prefix='chatcarlo-g4-preview-') as temp:
        manifest = bundle.create(scene, plane, Path(temp)/'bundle', g2_tally=opts.get('dose_grid', False))
        job = condition.scene_job(scene, plane, opts)
        cond = condition.colab_condition(manifest, job, scene, opts['track_uncertainty'])
        target = Path(run_dir) if run_dir else condition.default_run_dir(cond)
        if (target/'approval.json').exists(): raise ValueError('approved run directory cannot be overwritten')
        # Check that the rendered scene and native bundle describe the same geometry.
        a, _, _, settings, _ = bundle.load(Path(temp)/'bundle')
        if not np.array_equal(a['centers'], np.array([g['center'] for g in scene.raw['geometry']])) or not np.array_equal(a['halves'], np.array([g['size_cm'] for g in scene.raw['geometry']])/2):
            raise ValueError('preview geometry mismatch')
        names = settings['material_names']
        if not np.array_equal(a['materials'], [names.index(g['material']) for g in scene.raw['geometry']]):
            raise ValueError('preview material mismatch')
        expected = np.array([]) if plane is None else np.r_[plane.center_cm, plane.normal, plane.u_axis, plane.v_axis, plane.size_cm, plane.shape]
        if not np.array_equal(a['detector'], expected): raise ValueError('preview detector mismatch')
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(temp)/'bundle', target/'bundle', dirs_exist_ok=True)
        _write(target/'job.json', job)
        _write(target/'preview_snapshot.json', dict(scene=scene.raw, rendered=scene_to_json(scene), condition=cond))
        text = f"condition SHA256: {condition.sha256(cond)}; histories: {job['n_histories']}; seed: {job['seed']}; source_seed: {job['source_seed']}; detector: {cond['detector']}"
        if 'grid' in job:
            g = job['grid']
            text += f"; grid: resolution={g['voxel_size_cm']} cm; origin_cm={g['origin_cm']}; shape={g['shape']}; voxels={int(np.prod(g['shape']))}"
        (target/'preview.html').write_text(render_html(scene, approval_text=text))
        files = {name:bundle.sha(target/name) for name in ('bundle/manifest.json', 'job.json', 'preview_snapshot.json', 'preview.html')}
        _write(target/'generation.json', dict(condition=cond, condition_sha256=condition.sha256(cond), files=files))
    return target


def _verify(run_dir):
    path = Path(run_dir)
    generation = json.loads((path/'generation.json').read_text())
    if condition.sha256(generation['condition']) != generation['condition_sha256']: raise ValueError('condition hash mismatch')
    for name in ('bundle/manifest.json', 'job.json', 'preview_snapshot.json', 'preview.html'):
        if bundle.sha(path/name) != generation['files'].get(name):
            label = 'HTML' if name == 'preview.html' else 'bundle' if name.startswith('bundle/') else name
            raise ValueError(label+' SHA256 mismatch')
    manifest = bundle.verify(path/'bundle')
    job = json.loads((path/'job.json').read_text())
    if job != generation['condition']['job'] or condition.colab_condition(manifest, job, type('Scene', (), {'raw': {'detector': generation['condition']['detector']}})(), generation['condition']['track_uncertainty']) != generation['condition']:
        raise ValueError('condition mismatch: generated bundle/job')
    return generation


def approve(run_dir):
    path = Path(run_dir)
    _verify(path)
    _write(path/'approval.json', dict(generation_sha256=bundle.sha(path/'generation.json'), approved_at=datetime.now(timezone.utc).isoformat()))


def require(run_dir, expected_condition):
    path = Path(run_dir)
    if not (path/'approval.json').exists(): raise ValueError('approval.json missing')
    approval = json.loads((path/'approval.json').read_text())
    if approval.get('generation_sha256') != bundle.sha(path/'generation.json'): raise ValueError('generation SHA256 mismatch')
    generation = _verify(path)
    if generation['condition'] != expected_condition: raise ValueError('condition mismatch')
    return generation


def check_observation(observed, expected, name):
    if not isinstance(observed, dict): raise ValueError('observation mismatch: '+name+' missing observed')
    for key in sorted(set(observed) | set(expected)):
        if key not in observed or key not in expected or observed[key] != expected[key]:
            raise ValueError('observation mismatch: '+name+' '+key)


def verify_batches(bundle_path, job, results, reused=()):
    from .core import expected_observation, health
    from .source_host import sample_source_photons
    from .colab_driver import read_result, expected_identity
    a, src, spec, settings, manifest = bundle.load(bundle_path)
    reference = None
    if job['device'] == 'cuda':
        name = 'source_reference.json'
        reference = read_result(results, name, expected_identity(manifest, job, name))
        if reference is None: raise ValueError('missing verified source_reference.json')
        names = {f'batch_{b:06d}.npz' for b, _ in enumerate(range(0, job['n_histories'], job['batch_size']))}
        if not isinstance(reference, dict) or set(reference) != names:
            raise ValueError('source_reference.json batch mismatch')
    rng = np.random.default_rng(job['source_seed'])
    batches = []
    timing = dict(new_batches=0, reused_batches=0, new_elapsed_s=0., reused_elapsed_s=0.)
    for b, start in enumerate(range(0, job['n_histories'], job['batch_size'])):
        n = min(job['batch_size'], job['n_histories']-start)
        name = f'batch_{b:06d}.npz'
        r = read_result(results, name, expected_identity(manifest, job, name))
        if r is None: raise ValueError('missing verified batch '+name)
        receipt = json.loads((Path(results)/(name+'.receipt.json')).read_text())
        o, d, e = sample_source_photons(src, n, rng, spec)
        if 'grid' in job:
            from .g2_core import expected_observation as g2_expected, grid_from, health as g2_health
            from .g2_tables import config
            descriptor = grid_from(job['grid'])
            expected = g2_expected(a, config(a, descriptor, [], -1), o, d, e, job['seed'], start, settings['fluorescence'], job['device'])
            g2_health(r, descriptor)
        else:
            expected = expected_observation(a, o, d, e, job['seed'], start, settings['fluorescence'], job['device'])
        if reference is not None:
            hashes = reference[name]
            if not isinstance(hashes, dict) or set(hashes) != {'origins', 'directions', 'energies'} or any(
                not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in hashes.values()
            ):
                raise ValueError('source_reference.json invalid hashes: '+name)
            # Origins/energies must also match the host replay. Only directions
            # use the independent replay in the same VM's rounding environment.
            check_observation({k: hashes[k] for k in ('origins', 'energies')},
                              {k: expected[f'{pos}:{k}'] for pos,k in ((15,'origins'),(17,'energies'))},
                              'source_reference.json '+name)
            expected['16:directions'] = hashes['directions']
        check_observation(receipt.get('observed'), expected, name)
        health(r, n, len(settings['material_names']), a['detector'])
        elapsed = receipt.get('elapsed_s')
        if not isinstance(elapsed, (int, float)) or not np.isfinite(elapsed) or elapsed < 0: raise ValueError('invalid elapsed_s: '+name)
        prefix = 'reused' if name in reused else 'new'
        timing[prefix+'_batches'] += 1
        timing[prefix+'_elapsed_s'] += elapsed
        batches.append((n, r))
    timing['elapsed_s'] = timing['new_elapsed_s'] + timing['reused_elapsed_s']
    return batches, timing
