"""Preserve what per-batch raw results carry beyond the retained summary NPZ, verify it,
and record the sources — before the raw files are retired.

This script never deletes or moves anything (`.claude/rules`: deletion is the user's step).
It follows docs/validation/gpu/water60_100M/compact_batches.py: read every batch through the
driver's receipt check, assert that the retained summary NPZ reproduces the raw sums exactly,
keep only what the summary does not already hold, and write a record with the source hashes,
the verification performed and the limitations.

Two targets (2026-11-08, user decision "raw_compaction.json を書いてから消す"):

* `ivr`   runs/ivr_room_box_gpu/colab/results, 20 batches x 5e6 of a 120x120x79 H*(10) room grid.
          dose.npz already holds the per-voxel batch sums (`kerma_keV`, `h10_track_pSv_cm3`),
          the per-voxel SEM and the hit counts, so no per-voxel array is copied here -- a second
          copy would duplicate ~36 MB into a public repository and invite cancellation when the
          SEM is checked back against it. Only the per-batch scalars are kept.
* `n1e8`  runs/xray_portrait_noise_20261024/n1e8/results, 500 batches x 2e5 of a 100x120 detector.
          Every one of the seven `images` channels sums exactly into detector.npz
          (`images[0..3]` = `category_fluence`, `[4]` = `photon_count`, `[5]` = `energy_sum_keV`,
          `[6]` = `energy_sum2_keV2`), verified here. detector.npz keeps only `total_rel_err`,
          one figure summed over categories, so the per-channel batch-to-batch second moment is
          NOT recoverable from it: that is kept (7 x 100 x 120, under 1 MB) together with the
          per-batch channel totals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TARGETS = {
    'ivr': dict(
        run=ROOT/'runs/ivr_room_box_gpu/colab',
        summary=ROOT/'runs/ivr_room_box_gpu/dose.npz',
        out=HERE/'ivr_room_box_compaction',
        batches=20, histories=100000000, batch_size=5000000,
        scalars=('deposits', 'endings', 'scatter_sum', 'n_fluorescence'),
        format='ivr-room-box-batch-compaction-v1',
    ),
    'n1e8': dict(
        run=ROOT/'runs/xray_portrait_noise_20261024/n1e8',
        summary=ROOT/'runs/xray_portrait_noise_20261024/n1e8/detector.npz',
        out=HERE/'request_2_entry/n1e8_compaction',
        batches=500, histories=100000000, batch_size=200000,
        scalars=('deposits', 'endings', 'scatter_sum', 'n_fluorescence'),
        format='xray-portrait-n1e8-batch-compaction-v1',
    ),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sources(spec):
    """Read every batch through the driver's identity+SHA256 receipt check."""
    from chatcarlo.gpu.colab_driver import expected_identity, read_result
    run = spec['run']
    job = json.loads((run/'job.json').read_text())
    manifest = json.loads((run/'bundle/manifest.json').read_text())
    if job['mode'] != 'batches': raise ValueError('expected a batches job')
    if job['n_histories'] != spec['histories'] or job['batch_size'] != spec['batch_size']:
        raise ValueError('job histories/batch_size differ from the recorded run')
    if job['n_histories'] // job['batch_size'] != spec['batches']:
        raise ValueError('batch count differs from the recorded run')
    rows = []
    for batch in range(spec['batches']):
        name = f'batch_{batch:06d}.npz'
        path = run/'results'/name
        raw = read_result(path.parent, name, expected_identity(manifest, job, name))
        if raw is None: raise FileNotFoundError(path)
        receipt_path = path.with_name(name+'.receipt.json')
        receipt = json.loads(receipt_path.read_text())
        if receipt['sha256'] != sha(path): raise ValueError('receipt SHA256 mismatch: '+name)
        rows.append((name, path, receipt, receipt_path, raw))
    return job, manifest, rows


def build_ivr(spec, job, rows):
    """Keep per-batch scalars only; assert dose.npz reproduces the raw sums exactly."""
    n = len(rows)
    tables = dict(histories=np.full(n, spec['batch_size'], dtype=np.int64),
                  starts=np.arange(n, dtype=np.int64)*spec['batch_size'])
    for name in spec['scalars']:
        sample = np.asarray(rows[0][4][name])
        tables[name] = np.empty((n,)+sample.shape, dtype=np.float64)
    for key in ('grid_kerma_total', 'grid_h10_total'):
        tables[key] = np.empty(n, dtype=np.float64)
    totals = None
    hit = None
    for i, (_, _, _, _, raw) in enumerate(rows):
        for name in spec['scalars']: tables[name][i] = raw[name]
        if totals is None:
            totals = {k: np.zeros_like(raw[k], dtype=np.float64) for k in ('kerma', 'h10')}
            hit = np.zeros(raw['kerma'].shape, dtype=np.int32)
        for k in totals:
            if not (np.isfinite(raw[k]).all() and np.all(raw[k] >= 0)):
                raise ValueError('non-finite or negative '+k)
            totals[k] += raw[k]
        hit += ((raw['kerma'] > 0) | (raw['h10'] > 0)).astype(np.int32)
        tables['grid_kerma_total'][i] = raw['kerma'].sum()
        tables['grid_h10_total'][i] = raw['h10'].sum()
    shape = tuple(job['grid']['shape'])
    size = job['grid']['voxel_size_cm']
    with np.load(spec['summary']) as full:
        np.testing.assert_array_equal(totals['kerma'].reshape(shape), full['kerma_keV'])
        np.testing.assert_array_equal(totals['h10'].reshape(shape), full['h10_track_pSv_cm3'])
        np.testing.assert_array_equal(hit.reshape(shape), full['n_batches_hit'])
        if int(full['n_batches']) != n: raise ValueError('dose.npz n_batches differs')
    checks = ('Every batch read through expected_identity/read_result (identity + SHA256) and its '
              'receipt SHA256 recomputed from the file; kerma/h10 finite and non-negative; the raw '
              'per-voxel sums and the hit counts reproduce dose.npz exactly (assert_array_equal); '
              'dose.npz n_batches == 20.')
    limits = (f'Per-voxel batch values and cross-voxel batch covariance are not kept: with 83% of the '
              f'{np.prod(shape)} voxels hit, a per-voxel table would be larger than the raw files and '
              f'would duplicate data. dose.npz retains the per-voxel batch sums (kerma_keV, '
              f'h10_track_pSv_cm3), mean, SEM, relative error and n_batches_hit for the full grid, so '
              f'per-voxel first and second moments stay recoverable; what is lost is the per-voxel '
              f'batch time series, hence any re-analysis over a subset of batches. Voxel size '
              f'{size} cm. No Drive copy.')
    return tables, checks, limits


def build_n1e8(spec, job, rows):
    """Keep per-channel batch sums of squares (not in detector.npz) and per-batch totals."""
    n = len(rows)
    tables = dict(histories=np.full(n, spec['batch_size'], dtype=np.int64),
                  starts=np.arange(n, dtype=np.int64)*spec['batch_size'])
    for name in spec['scalars']:
        sample = np.asarray(rows[0][4][name])
        tables[name] = np.empty((n,)+sample.shape, dtype=np.float64)
    im0 = rows[0][4]['images']; ca0 = rows[0][4]['category_count']
    images_sum = np.zeros_like(im0, dtype=np.float64)
    images_sum2 = np.zeros_like(im0, dtype=np.float64)
    category_sum = np.zeros_like(ca0, dtype=np.float64)
    category_sum2 = np.zeros_like(ca0, dtype=np.float64)
    hit = np.zeros(im0.shape[1:], dtype=np.int32)
    tables['images_batch_totals'] = np.empty((n, im0.shape[0]), dtype=np.float64)
    tables['category_batch_totals'] = np.empty((n, ca0.shape[0]), dtype=np.float64)
    for i, (_, _, _, _, raw) in enumerate(rows):
        for name in spec['scalars']: tables[name][i] = raw[name]
        im = np.asarray(raw['images'], dtype=np.float64)
        ca = np.asarray(raw['category_count'], dtype=np.float64)
        if not np.isfinite(im).all(): raise ValueError('non-finite images')
        images_sum += im; images_sum2 += im*im
        category_sum += ca; category_sum2 += ca*ca
        hit += (im.sum(0) > 0).astype(np.int32)
        tables['images_batch_totals'][i] = im.sum((1, 2))
        tables['category_batch_totals'][i] = ca.sum((1, 2))
    with np.load(spec['summary']) as det:
        # Every images channel must sum exactly into detector.npz; otherwise the raw files hold
        # a first moment nothing else keeps and must not be retired.
        for channel, key in ((0, None), (4, 'photon_count'), (5, 'energy_sum_keV'),
                             (6, 'energy_sum2_keV2')):
            if key is not None:
                np.testing.assert_array_equal(images_sum[channel], det[key])
        np.testing.assert_array_equal(images_sum[:4], det['category_fluence'])
        np.testing.assert_array_equal(hit, det['n_batches_hit'])
        if int(det['n_batches']) != n or int(det['n_histories']) != spec['histories']:
            raise ValueError('detector.npz n_batches/n_histories differ')
    tables['images_sum2'] = images_sum2
    tables['category_count_sum'] = category_sum
    tables['category_count_sum2'] = category_sum2
    checks = ('Every batch read through expected_identity/read_result (identity + SHA256) and its '
              'receipt SHA256 recomputed from the file; images finite; all seven images channels sum '
              'exactly into detector.npz (images[0..3] == category_fluence, [4] == photon_count, '
              '[5] == energy_sum_keV, [6] == energy_sum2_keV2, assert_array_equal) and the recomputed '
              'n_batches_hit matches; detector.npz n_batches == 500 and n_histories == 1e8.')
    limits = ('Per-pixel batch values are not kept, so cross-pixel batch covariance and any '
              're-analysis over a subset of batches are lost. What is kept instead: the per-channel '
              'batch sum of squares (images_sum2, category_count_sum2), which detector.npz does not '
              'hold -- it exports only total_rel_err, one figure summed over categories -- so the '
              'per-pixel batch SEM stays recoverable for every channel; plus the per-batch channel '
              'totals and the per-batch scalars. category_count per-pixel sums are kept here because '
              'detector.npz exports category_fluence (energy/area), not the counts. No Drive copy.')
    return tables, checks, limits


def build(key, spec):
    job, manifest, rows = sources(spec)
    tables, checks, limits = (build_ivr if key == 'ivr' else build_n1e8)(spec, job, rows)
    out = spec['out']; out.mkdir(parents=True, exist_ok=True)
    target = out/'batch_tables.npz'
    if target.exists(): raise FileExistsError(target)
    np.savez_compressed(target, **tables)
    with np.load(target) as stored:
        for name, value in tables.items(): np.testing.assert_array_equal(stored[name], value)
    receipts = {name: receipt for name, _, receipt, _, _ in rows}
    (out/'batch_receipts.json').write_text(json.dumps(receipts, sort_keys=True)+'\n')
    source_files = [dict(name=name, bytes=path.stat().st_size, sha256=receipt['sha256'],
                         receipt_sha256=sha(receipt_path))
                    for name, path, receipt, receipt_path, _ in rows]
    record = dict(
        format=spec['format'], job=job, bundle_manifest_sha256=sha(spec['run']/'bundle/manifest.json'),
        source_directory=str(spec['run']/'results'),
        source_files=source_files, source_bytes=sum(x['bytes'] for x in source_files),
        archive_bytes=target.stat().st_size, archive_sha256=sha(target),
        receipts_sha256=sha(out/'batch_receipts.json'),
        kept=sorted(tables), preserved_npz={str(spec['summary']): sha(spec['summary'])},
        verification=checks, limitations=limits, deletion_status='not deleted')
    (out/'raw_compaction.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ('format', 'source_bytes', 'archive_bytes',
                                             'deletion_status')}, indent=2))


def verify(key, spec):
    out = spec['out']
    record = json.loads((out/'raw_compaction.json').read_text())
    assert sha(out/'batch_tables.npz') == record['archive_sha256']
    assert sha(out/'batch_receipts.json') == record['receipts_sha256']
    for path, digest in record['preserved_npz'].items():
        assert sha(path) == digest, path
    with np.load(out/'batch_tables.npz') as table:
        assert sorted(table.files) == record['kept']
        assert int(table['histories'].sum()) == spec['histories']
        assert np.all(table['endings'].sum(1) == table['histories'])
        with np.load(spec['summary']) as summary:
            if key == 'ivr':
                np.testing.assert_allclose(table['grid_kerma_total'].sum(),
                                           summary['kerma_keV'].sum(), rtol=1e-12, atol=0)
                np.testing.assert_allclose(table['grid_h10_total'].sum(),
                                           summary['h10_track_pSv_cm3'].sum(), rtol=1e-12, atol=0)
            else:
                np.testing.assert_allclose(table['images_batch_totals'][:, :4].sum(0),
                                           summary['category_fluence'].sum((1, 2)),
                                           rtol=1e-12, atol=0)
                np.testing.assert_allclose(table['images_batch_totals'][:, 4].sum(),
                                           summary['photon_count'].sum(), rtol=1e-12, atol=0)
                assert np.all(table['images_sum2'] >= 0)
    print(f'PASS ({key}): compact table, receipts, retained-NPZ hash and totals verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('target', choices=sorted(TARGETS))
    args = parser.parse_args()
    chosen = TARGETS[args.target]
    if args.action == 'build': build(args.target, chosen)
    verify(args.target, chosen)
