"""P7 layouts, spectra, analysis ranges, history-count rule, contracts and kernel diff."""
import ctypes as ct
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p7, p7_core as h
from prototypes.mv_electron import analysis_p7 as A, run_p3a


def test_p7_layouts_match_canonical_boxes():
    from prototypes.mv_electron.photon_transport_p3b import expand_regions
    for name,bins in (('p6',1836),('p7',3420)):
        L=h.layout(name);expected=expand_regions(L['regions'],30.,.25,.1);g,arrays=h.geometry(name)
        for key in ('lo','hi','origins','widths','axes','sizes'):
            assert np.array_equal(arrays[key].reshape(expected['region_'+key].shape),expected['region_'+key])
        assert L['bins']==bins and ct.sizeof(g)==1144


def test_p7_spectra():
    e,p,c=h.spectrum('mohan10');e0,p0,c0=run_p3a.load_spectrum()
    assert np.array_equal(e,e0) and np.array_equal(p,p0) and np.array_equal(c,c0)
    e,p,c=h.spectrum('mohan4')
    assert len(c)==16 and e[-1]==4. and c[-1]==1.


def test_p7_ranges_follow_plan_table():
    table=dict(A=([8,8,9],[(16,24),(17,25),(18,26)]),B=([33,35,38],[(63,71),(66,74),(72,80)]),
               C=([16,17,19],[(32,40),(33,41),(36,44)]))
    for name,(central,outside) in table.items():
        s=A.spec(h.condition(name)['field_half'])
        assert s['central']==central and s['outside']==outside


def test_p7_analysis_ranges_synthetic():
    assert p7.check_analysis_ranges()['status']=='pass'


def test_p7_history_count_rule():
    assert p7.check_sizes()['status']=='pass'


def test_p7_contracts(tmp_path):
    from prototypes.mv_electron.gpu.p5a import load_inputs
    assert p7.check_contracts(tmp_path,load_inputs(full=True))['status']=='pass'


def test_p7_kernel_diff_is_only_the_permitted_edits():
    assert p7.check_kernel_diff()['status']=='pass'


def test_p7_identity_excludes_launch_tools():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    ident=h.identity(h.condition_inputs(load_inputs(full=True),'C'),'C')
    assert not any(n.endswith(('.ipynb','colab.py','p7.py')) for n in ident['sources'])
    assert ident['params']['spectrum']=='mohan4' and ident['condition']=='C'
    with pytest.raises(h.Stop):h.identity(load_inputs(full=True),'C')  # 10 MV inputs cannot claim condition C
