"""Pool two independent equal-size, equal-batch diagnostic runs."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(os.environ.get('CHATCARLO_WATER60_RUN_DIR', str(Path(__file__).resolve().parents[4] / 'runs/diag_water_60kev_gpu/add50M_20261026')))
OLD = ROOT.parent / 'dose.npz'
NEW = ROOT / 'dose.npz'
OUTPUT = Path(os.environ.get('CHATCARLO_WATER60_OUTPUT_DIR', str(ROOT)))
OUTPUT.mkdir(parents=True, exist_ok=True)
OUT = OUTPUT / 'dose_100M.npz'


def pool(mean1, sem1, m1, mean2, sem2, m2):
    m = m1 + m2
    mean = (m1 * mean1 + m2 * mean2) / m
    ss = m1 * (m1 - 1) * sem1**2 + m2 * (m2 - 1) * sem2**2
    ss += m1 * m2 / m * (mean1 - mean2)**2
    sem = np.sqrt(ss / (m * (m - 1)))
    return mean, sem


def main():
    old_job = json.loads((ROOT.parent / 'run/job.json').read_text())
    new_job = json.loads((ROOT / 'run/job.json').read_text())
    assert old_job['grid'] == new_job['grid']
    assert old_job['batch_size'] == new_job['batch_size'] == 500000
    assert old_job['n_histories'] == new_job['n_histories'] == 50000000
    assert old_job['seed'] != new_job['seed']
    assert old_job['source_seed'] != new_job['source_seed']
    old_manifest = json.loads((ROOT.parent / 'run/bundle/manifest.json').read_text())
    new_manifest = json.loads((ROOT / 'run/bundle/manifest.json').read_text())
    assert old_manifest == new_manifest
    old_scene = json.loads((ROOT.parent / 'run/preview_snapshot.json').read_text())['scene']
    new_scene = json.loads((ROOT / 'run/preview_snapshot.json').read_text())['scene']
    assert old_scene == new_scene
    with np.load(OLD) as z:
        a = {k: z[k] for k in z.files}
    with np.load(NEW) as z:
        b = {k: z[k] for k in z.files}
    for group, job in ((a, old_job), (b, new_job)):
        np.testing.assert_array_equal(group['origin_cm'], job['grid']['origin_cm'])
        np.testing.assert_array_equal(group['shape'], job['grid']['shape'])
        assert float(group['voxel_size_cm']) == job['grid']['voxel_size_cm']
        shape = tuple(job['grid']['shape'])
        for key in ('dose_per_history_Gy', 'h10_per_history_pSv', 'kerma_keV',
                    'h10_track_pSv_cm3', 'sem_dose_per_history_Gy',
                    'sem_h10_per_history_pSv', 'n_batches_hit'):
            assert group[key].shape == shape, key
            assert np.isfinite(group[key]).all() and np.all(group[key] >= 0), key
        assert np.issubdtype(group['n_batches_hit'].dtype, np.integer)
        assert np.all(group['n_batches_hit'] <= int(group['n_batches']))
    out = {}
    for key in ('origin_cm', 'voxel_size_cm', 'shape'):
        np.testing.assert_array_equal(a[key], b[key])
        out[key] = a[key]
    m1, m2 = int(a['n_batches']), int(b['n_batches'])
    assert m1 == m2 == 100
    for mean_key, sem_key, r_key in (
        ('dose_per_history_Gy', 'sem_dose_per_history_Gy', 'rel_err_dose'),
        ('h10_per_history_pSv', 'sem_h10_per_history_pSv', 'rel_err_h10'),
    ):
        for group in (a, b):
            assert np.isfinite(group[mean_key]).all()
            assert np.isfinite(group[sem_key]).all()
            np.testing.assert_allclose(group[sem_key], group[r_key] * group[mean_key], rtol=1e-12)
        out[mean_key], out[sem_key] = pool(a[mean_key], a[sem_key], m1, b[mean_key], b[sem_key], m2)
        out[r_key] = np.divide(out[sem_key], out[mean_key], out=np.full_like(out[mean_key], np.nan), where=out[mean_key] != 0)
    for key in ('kerma_keV', 'h10_track_pSv_cm3', 'n_batches_hit'):
        out[key] = a[key] + b[key]
    out['n_batches'] = np.array(m1 + m2)
    out['n_histories'] = np.array(100000000)
    out['batch_size'] = np.array(500000)
    assert np.all(out['n_batches_hit'] <= m1 + m2)
    # Independent H*(10) conversion: track / voxel volume / incident histories.
    np.testing.assert_allclose(out['h10_per_history_pSv'], out['h10_track_pSv_cm3'] / float(out['voxel_size_cm'])**3 / 100000000, rtol=1e-12)
    np.savez(OUT, **out)
    provenance = dict(
        n_histories=100000000, n_batches=m1+m2, batch_size=500000,
        inputs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), job=j)
                for p,j in ((OLD,old_job),(NEW,new_job))],
        output=dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest()),
        uncertainty='Pooled sample variance of 200 equally sized batches, including between-group mean difference.',
        old_raw_batches_available=False,
    )
    (OUTPUT / 'merge_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ('n_histories','n_batches','batch_size')}, indent=2))


if __name__ == '__main__':
    main()
