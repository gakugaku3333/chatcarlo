"""Effective options and full condition identities (short hashes are paths only)."""
import hashlib
import json
from pathlib import Path
import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def resolve(options):
    get = lambda key, default=None: options.get(key, default) if isinstance(options, dict) else getattr(options, key, default)
    seed = get('seed')
    n = int(get('n_histories', 100000))
    batch = int(get('batch_size', 200000))
    engine = get('engine', 'numpy')
    device = get('gpu_device') or 'cpu'
    if engine == 'gpu':
        if seed is None: raise ValueError('G1 requires an explicit transport seed')
        if get('workers', 1) != 1: raise ValueError('G1 requires n_workers=1')
        if get('dose_grid', False): raise ValueError('G1 does not support voxel dose_grid tally')
        if get('kernel_chunks', 0) != 0 or get('kernel_max_segments_per_history', 16) != 16:
            raise ValueError('--engine gpu cannot use --kernel-*')
    elif get('gpu_device') is not None:
        raise ValueError('--gpu-device requires --engine gpu')
    if n <= 0 or batch <= 0: raise ValueError('n_histories and batch_size must be positive')
    source_seed = get('source_seed')
    if engine == 'gpu' and source_seed is None:
        source_seed = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])
    resolved = dict(engine=engine, device=device if engine == 'gpu' else 'cpu', n_histories=n,
                batch_size=batch, seed=seed, source_seed=source_seed,
                track_uncertainty=not get('no_uncertainty', False), workers=get('workers', 1))
    if get('dose_grid', False):
        resolved.update(dose_grid=True, grid_resolution_cm=get('resolution', 5.))
    if engine == 'kernel':
        resolved.update(kernel_chunks=get('kernel_chunks', 0), kernel_max_segments_per_history=get('kernel_max_segments_per_history', 16))
    return resolved


def job_for(options):
    return dict(mode='batches', device='cuda', **{k: options[k] for k in ('n_histories', 'batch_size', 'seed', 'source_seed')})


def colab_condition(manifest, job, scene, track_uncertainty):
    # Use the existing receipt's manifest identity verbatim.
    from .colab_driver import identity
    det = scene.raw.get('detector')
    return dict(format='chatcarlo-g4-v1', bundle_sha256=identity(manifest, job, 0, 0, 0)['bundle_sha256'],
                job=job, detector=det, roi=det.get('roi') if det else None, track_uncertainty=track_uncertainty)


def local_condition(scene, options):
    return dict(format='chatcarlo-g4-v1', scene_sha256=sha256(scene.raw), **options)


def default_run_dir(condition):
    return Path.home()/'.cache/chatcarlo/gpu_runs'/sha256(condition)[:12]
