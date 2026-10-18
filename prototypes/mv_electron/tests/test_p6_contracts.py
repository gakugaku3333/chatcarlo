"""Seed/range, ledger, resume and deferred Colab contracts; no GPU runs."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p6, p6_core as h, p6_colab as colab, p5b_core as core


def synthetic(n=100000):
    r=h.empty(h.run_shapes());r['nspec'][0,0]=n
    return r


def test_p6_checkpoint_seed_range_settings_and_hash(tmp_path):
    ident={'fixture':'p6'};r=synthetic();path=tmp_path/'batch_002.npz'
    h.save_checkpoint(path,ident,'gate3',2,r)
    assert h.exact(h.read_checkpoint(path,ident,'gate3',2),r)['status']=='pass'
    meta=h.checkpoint_meta(ident,'production',99,synthetic(460000))
    assert meta['start']==45540000 and meta['end']==46000000 and meta['contract']['seed']==20291764
    for stage,b,identity in [('production',2,ident),('gate3',0,ident),('gate3',2,{'fixture':'changed'})]:
        with pytest.raises(h.Stop):h.read_checkpoint(path,identity,stage,b)
    damaged=core.read_npz(path);damaged['regions'][0,0]=1.
    core.save_npz(path,damaged)
    with pytest.raises(h.Stop):h.read_checkpoint(path,ident,'gate3',2)


@pytest.mark.parametrize('field',['region_ledger','regions','totals','counts'])
def test_p6_save_rejects_unhealthy_outputs(field,tmp_path):
    r=synthetic()
    if field=='region_ledger':r[field][3]=1
    elif field=='regions':r[field][0,420]=1.  # stale slab output, missing from the deposited ledger
    elif field=='totals':r[field][0]=np.inf
    else:r[field][5]=1
    assert h.health(r)['status']=='fail'
    with pytest.raises(h.Stop):h.save_checkpoint(tmp_path/'bad.npz',{},'gate3',0,r)
    assert not (tmp_path/'bad.npz').exists()


def test_p6_resume_preserves_and_rejects_mismatches(tmp_path):
    ident={'fixture':'p6'}
    assert h.resume(tmp_path,ident,'gate3')=={}
    r=synthetic();path=tmp_path/'batch_000.npz';h.save_checkpoint(path,ident,'gate3',0,r)
    assert h.resume(tmp_path,ident,'gate3')=={0:path}
    with pytest.raises(h.Stop):h.resume(tmp_path,ident,'production')
    with pytest.raises(h.Stop):h.save_checkpoint(path,ident,'gate3',0,r)
    (tmp_path/'p6_pending_interrupted.npz').write_bytes(b'retain this evidence')
    with pytest.raises(h.Stop):h.resume(tmp_path,ident,'gate3')
    assert path.exists() and (tmp_path/'p6_pending_interrupted.npz').exists()


def test_p6_reduction_retains_all_21_outputs_and_100_outer_batches():
    from prototypes.mv_electron.batch_kernels_p3b import KEYS
    r=synthetic();r['regions'][0,0]=2.;r['regions'][1,0]=3.
    reduced=h.reduced(r)
    assert set(reduced)==set(KEYS) and reduced['axis_coarse'][0]==5.
    stacked=h.stacked([r]*100,'gate3')
    from prototypes.mv_electron.analysis_p3b3 import dose
    assert dose(stacked)['axis_coarse'].shape==(100,120)
    assert stacked['N']==10000000 and stacked['seed']==h.SEED
    assert np.array_equal(stacked['starts'],np.arange(100)*100000)
    assert stacked['profile'].shape==(100,2,3,72)


def test_p6_gate3_checks_legacy_box_and_profiles():
    rows=[synthetic() for _ in range(100)]
    check=h.c4(rows,rows,1.)
    assert check['status']=='pass'
    assert set(check['quantities'])=={'legacy_deep','axis_blocks','profile_central'}
    bad=copy.deepcopy(rows)
    for r in bad:r['regions'][:,540:612]=1e12
    assert h.c4(rows,bad,1.)['status']=='fail'


def test_p6_atomic_merge_checks_every_region_bin():
    n=200;hist=h.empty(h.history_shapes(n));hist['calls'][:]=2
    aggregate=synthetic(n)
    hist['nspec_index'][:]=0
    additions=np.zeros((100,h.BINS),np.uint64)
    assert h.c3(hist,aggregate,region_additions=additions)['status']=='pass'
    aggregate['regions'][99,-1]=1e-30
    assert h.c3(hist,aggregate,region_additions=additions)['status']=='fail'


def test_p6_atomic_bound_adds_deposits_only_to_their_region_bin():
    n=200;hist=h.empty(h.history_shapes(n));hist['calls'][:]=2
    aggregate=synthetic(n);hist['nspec_index'][:]=0
    additions=np.zeros((100,h.BINS),np.uint64);additions[0,0]=1
    hist['regions'][0,:2]=1.;aggregate['regions'][0,:2]=1.
    eps=2**-52;aggregate['regions'][0,0]+=3*eps  # one deposit + two histories
    assert h.c3(hist,aggregate,region_additions=additions)['status']=='pass'
    aggregate['regions'][0,0]+=eps
    assert h.c3(hist,aggregate,region_additions=additions)['status']=='fail'
    aggregate['regions'][0,0]-=eps
    aggregate['regions'][0,1]+=3*eps
    assert h.c3(hist,aggregate,region_additions=additions)['status']=='fail'
    aggregate['regions'][0,1]=1.
    hist['batches'][0,0]=1.;aggregate['batches'][0,0]=1.+16*eps
    check=h.c3(hist,aggregate,region_additions=additions)
    assert check['status']=='fail' and check['region_exceeding']==0
    assert check['legacy']['exceeding']['batches']==1
    with pytest.raises(h.Stop):h.c3(hist,aggregate,region_additions=additions[:1])
    with pytest.raises(h.Stop):h.c3(hist,aggregate,region_additions=additions.astype(float))


def test_p6_c2_health_rejects_stale_refill_arrays():
    n=3*256;a=h.empty(h.history_shapes(n));a['calls'][:]=2
    b=copy.deepcopy(a);b['regions'][-1,420]=.01
    assert h.c2(a,a,1.,1.)['status']=='pass'
    assert h.c2(a,b,1.,1.)['status']=='fail'


def test_p6_colab_import_has_no_private_transport_dependencies():
    code="from prototypes.mv_electron.gpu import p6_colab; import sys; assert 'numba' not in sys.modules; assert 'cupy' not in sys.modules; assert 'prototypes.mv_electron.physics' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_p6_gpu_source_and_notebook_are_deferred():
    source=h.source()
    assert '#include "p6' not in source
    assert '#include "p5' not in source
    assert '__global__ void p6_kernel' in source and '__ballot_sync' in source
    notebook=json.loads((h.HERE/'p6_colab.ipynb').read_text())
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'])
    code=''.join(''.join(c['source']) for c in notebook['cells'])
    assert 'p6_colab.run_all' in code and 'analyse(' not in code
    assert 'ncu' not in (h.HERE/'p6_colab.py').read_text()


def test_p6_incomplete_gate_records_refused():
    for f in (colab.require_gate2,colab.require_gate3):
        with pytest.raises(h.Stop):f({'status':'pass','identity':{}},{})
