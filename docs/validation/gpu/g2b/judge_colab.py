"""関門 C・C2 の判定（Claude、2026-10-23）。計画 6 節。使い方: judge_colab.py RUN_DIR OUT.json"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'docs/validation/gpu/g2')]
from run_gates import judge
from chatcarlo.geometry import Geometry
from chatcarlo.tally import VoxelGrid
from chatcarlo.scene import load_scene
from chatcarlo.gpu import approval, bundle

REF = ROOT/'data/gpu_g2/ref_lead_batches.npz'
REF_SHA = '9e6d3ab523f2101205b374a4188cc17135eb9b5b811029da959dc8ca8a44c84e'
SCENE = ROOT/'docs/validation/gpu/g2/scene_lead_slab.yaml'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(run, out):
    run = Path(run); job = json.loads((run/'job.json').read_text()); g = job['grid']
    shape = tuple(g['shape'])
    # 照合はホストの本番の関数で（受領票・観測すべて）
    batches, _ = approval.verify_batches(run/'bundle', job, run/'results', set())
    k = np.array([r['kerma'] for _, r in batches]).reshape(-1, *shape)
    h = np.array([r['h10'] for _, r in batches]).reshape(-1, *shape)
    n = [m for m, _ in batches]
    rep = dict(run=str(run), n_batches=len(batches))
    # (ii) 保存した npz と、累積の差分から独立に作り直した統計
    z = np.load(run/'dose.npz')
    ck = np.zeros(shape); ch = np.zeros(shape); s2k = np.zeros(shape); s2h = np.zeros(shape); hit = np.zeros(shape, np.int32)
    for i in range(len(batches)):
        pk, ph = ck.copy(), ch.copy(); ck += k[i]; ch += h[i]
        dk, dh = ck-pk, ch-ph; s2k += dk*dk/n[i]; s2h += dh*dh/n[i]; hit += ((dk != 0) | (dh != 0))
    N, M = sum(n), len(batches)
    from chatcarlo.tally import relative_error
    rep['ii'] = dict(kerma_bytes=z['kerma_keV'].tobytes() == ck.tobytes(),
                     h10_bytes=z['h10_track_pSv_cm3'].tobytes() == ch.tobytes(),
                     n_batches_hit=bool(np.array_equal(z['n_batches_hit'], hit)),
                     n_batches=int(z['n_batches']) == M,
                     origin=z['origin_cm'].tolist() == g['origin_cm'], shape=z['shape'].tolist() == list(shape),
                     voxel=float(z['voxel_size_cm']) == g['voxel_size_cm'])
    rk = relative_error(ck, s2k, M, N); rh = relative_error(ch, s2h, M, N)
    # R は保存側が線量（質量で割った値）から作るので、ボクセルごとの比は同じでも最後の桁が変わり得る。相対 1e-12 で比べる
    rep['ii']['rel_err_dose'] = bool(np.allclose(z['rel_err_dose'], rk, rtol=1e-12, atol=0, equal_nan=True))
    rep['ii']['rel_err_h10'] = bool(np.allclose(z['rel_err_h10'], rh, rtol=1e-12, atol=0, equal_nan=True))
    # (iii) G2a 関門 1 と同じ判定
    assert sha(REF) == REF_SHA, 'reference sha'
    ref = np.load(REF)
    assert ref['origin'].tolist() == g['origin_cm'] and ref['shape'].tolist() == list(shape) and float(ref['resolution']) == g['voxel_size_cm']
    assert int(ref['histories_per_batch']) == job['batch_size'] and M == 20
    assert str(ref['scene_sha256']) == sha(SCENE), 'scene sha'
    scene = load_scene(str(SCENE)); grid = VoxelGrid(np.array(g['origin_cm']), shape, 2)
    material = Geometry(scene.raw['geometry']).material_at(grid.voxel_centers()).reshape(shape)
    regions = {m: material == m for m in ('water', 'air', 'bone')}
    zs = grid.origin_cm[2]+np.arange(shape[2])*2
    layer = np.broadcast_to(((zs < 0) & (zs+2 > -.05))[None, None, :], shape)
    regions = {m: msk & ~layer for m, msk in regions.items()}; regions['lead_layer'] = layer
    rep['iii'] = {q: judge(ref[q], v, regions) for q, v in (('kerma', k), ('h10', h))}
    for q in rep['iii']:
        for r in rep['iii'][q]['regions'].values():
            r.pop('reference_only_indices'); r.pop('twin_only_indices')
    rep['status'] = 'pass' if all(rep['ii'].values()) and all(v['status'] == 'pass' for v in rep['iii'].values()) else 'fail'
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str)+'\n')
    print(rep['status'], json.dumps(rep['ii']), {q: v['checks'] for q, v in rep['iii'].items()})

if __name__ == '__main__':
    main(*sys.argv[1:])
