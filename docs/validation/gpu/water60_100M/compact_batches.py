"""Preserve water-voxel batch tables and verify them before raw-file removal.

This script never deletes raw files. Full-grid means/SEM remain in dose.npz;
cross-voxel batch covariance is retained for water, for the additional 50M only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE.parents[3] / 'runs/diag_water_60kev_gpu/add50M_20261026'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(run, out):
    from chatcarlo.gpu.colab_driver import expected_identity, read_result
    job = json.loads((run / 'run/job.json').read_text())
    manifest = json.loads((run / 'run/bundle/manifest.json').read_text())
    assert job == json.loads((HERE / 'additional_job.json').read_text())
    assert manifest == json.loads((HERE / 'bundle_manifest.json').read_text())
    assert job['n_histories'] == 50000000 and job['batch_size'] == 500000
    shape = tuple(job['grid']['shape'])
    origin = np.array(job['grid']['origin_cm'])
    size = job['grid']['voxel_size_cm']
    xyz = [origin[i] + (np.arange(shape[i]) + .5) * size for i in range(3)]
    water = ((np.abs(xyz[0])[:, None, None] < 15)
             & (np.abs(xyz[1])[None, :, None] < 15)
             & (xyz[2][None, None, :] > 0) & (xyz[2][None, None, :] < 20))
    indices = np.flatnonzero(water)
    assert len(indices) == 18000
    tables = dict(water_flat_indices=indices, grid_shape=np.array(shape),
                  origin_cm=origin, voxel_size_cm=np.array(size),
                  histories=np.full(100, 500000, dtype=np.int64),
                  starts=np.arange(100, dtype=np.int64) * 500000)
    for name in ('kerma', 'h10'):
        tables[name] = np.empty((100, len(indices)), dtype=np.float64)
    for name, columns in (('deposits', 2), ('endings', 3)):
        tables[name] = np.empty((100, columns), dtype=np.float64)
    for name in ('scatter_sum', 'n_fluorescence', 'grid_kerma_total', 'grid_h10_total'):
        tables[name] = np.empty(100, dtype=np.float64)
    totals = {k: np.zeros(np.prod(shape), dtype=np.float64) for k in ('kerma', 'h10')}
    hit = np.zeros(np.prod(shape), dtype=np.int32)
    receipts, removed = {}, []
    for batch in range(100):
        name = f'batch_{batch:06d}.npz'
        path = run / 'run/results' / name
        key = expected_identity(manifest, job, name)
        raw = read_result(path.parent, name, key)
        assert raw is not None
        receipt_path = path.with_name(name + '.receipt.json')
        receipt = json.loads(receipt_path.read_text())
        receipts[name] = receipt
        removed.append(dict(name=name, bytes=path.stat().st_size, sha256=receipt['sha256'],
                            receipt_sha256=sha(receipt_path)))
        for quantity in ('kerma', 'h10'):
            assert raw[quantity].shape == (np.prod(shape),)
            assert np.isfinite(raw[quantity]).all() and np.all(raw[quantity] >= 0)
            totals[quantity] += raw[quantity]
            tables[quantity][batch] = raw[quantity][indices]
        hit += ((raw['kerma'] > 0) | (raw['h10'] > 0)).astype(np.int32)
        for name in ('deposits', 'endings', 'scatter_sum', 'n_fluorescence'):
            tables[name][batch] = raw[name]
        tables['grid_kerma_total'][batch] = raw['kerma'].sum()
        tables['grid_h10_total'][batch] = raw['h10'].sum()
        if (batch + 1) % 20 == 0:
            print(f'Checked source receipts and extracted {batch + 1}/100', flush=True)
    with np.load(run / 'dose.npz') as full:
        np.testing.assert_array_equal(totals['kerma'].reshape(shape), full['kerma_keV'])
        np.testing.assert_array_equal(totals['h10'].reshape(shape), full['h10_track_pSv_cm3'])
        np.testing.assert_array_equal(hit.reshape(shape), full['n_batches_hit'])
        for name, mean_key, sem_key, conversion in (
            ('kerma', 'dose_per_history_Gy', 'sem_dose_per_history_Gy', 1.602176634e-13 / size**3),
            ('h10', 'h10_per_history_pSv', 'sem_h10_per_history_pSv', 1 / size**3),
        ):
            values = tables[name] / 500000 * conversion
            np.testing.assert_allclose(values.mean(0), full[mean_key].ravel()[indices], rtol=1e-12, atol=0)
            np.testing.assert_allclose(values.std(0, ddof=1) / 10, full[sem_key].ravel()[indices], rtol=1e-8, atol=0)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'additional_water_batches.npz'
    if target.exists():
        raise FileExistsError(target)
    np.savez_compressed(target, **tables)
    with np.load(target) as stored:
        for key, value in tables.items():
            np.testing.assert_array_equal(stored[key], value)
    (out / 'additional_batch_receipts.json').write_text(json.dumps(receipts, sort_keys=True) + '\n')
    record = dict(format='water60-batch-compaction-v1', job=job,
                  material_names=['air', 'water'], units=dict(kerma='keV per batch in each water voxel',
                  h10='pSv cm^3 per batch in each water voxel', deposits='keV per batch'),
                  water_bounds_cm=[[-15, 15], [-15, 15], [0, 20]],
                  source_files=removed, source_bytes=sum(x['bytes'] for x in removed),
                  archive_bytes=target.stat().st_size, archive_sha256=sha(target),
                  receipts_sha256=sha(out / 'additional_batch_receipts.json'),
                  preserved_npz={str(p): sha(p) for p in (run.parent / 'dose.npz', run / 'dose.npz', run / 'dose_100M.npz')},
                  verification='Source receipt identity and SHA256; full-grid raw sums and hits exactly match additional dose.npz; water means and SEM match; extracted arrays round-trip exactly.',
                  limitations='Individual air-voxel batch values are discarded. Water covariance is available for additional 50M only; original 50M batch data were already absent. Full-grid mean/SEM/R/hits remain in the three retained dose NPZs.',
                  deletion_status='not deleted')
    (out / 'raw_compaction.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('source_bytes', 'archive_bytes', 'deletion_status')}, indent=2))


def verify(run, out):
    record = json.loads((out / 'raw_compaction.json').read_text())
    assert sha(out / 'additional_water_batches.npz') == record['archive_sha256']
    assert sha(out / 'additional_batch_receipts.json') == record['receipts_sha256']
    for path, digest in record['preserved_npz'].items():
        assert sha(path) == digest, path
    with np.load(out / 'additional_water_batches.npz') as table, np.load(run / 'dose.npz') as full:
        indices = table['water_flat_indices']
        assert table['kerma'].shape == table['h10'].shape == (100, 18000)
        assert table['histories'].sum() == 50000000
        assert np.all(table['endings'].sum(1) == table['histories'])
        np.testing.assert_allclose(table['kerma'].sum(0), full['kerma_keV'].ravel()[indices], rtol=1e-14, atol=0)
        np.testing.assert_allclose(table['h10'].sum(0), full['h10_track_pSv_cm3'].ravel()[indices], rtol=1e-14, atol=0)
    print('PASS: compact table, receipts, full-grid NPZ hashes and water totals verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.run, args.out)
    verify(args.run, args.out)
