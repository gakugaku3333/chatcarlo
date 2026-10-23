import json
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.entry import compare, bundle
from prototypes.mv_electron.gpu import p5b_core as core
from prototypes.mv_electron import analysis_p3b3 as stats


def energy(value=1.):
    return dict(**{k: np.full(s, value) for k, s in stats.SHAPES.items()}, N=np.array(2000), nbatch=np.array(100))


def test_entry_compare_converts_mev(monkeypatch):
    # Known 1 MeV per region / 20 histories; axis volume = 1 cm³.
    monkeypatch.setattr(stats, 'BOOTSTRAPS', 3)
    a, b = energy(2.), energy(1.)
    ref = stats.MEV_TO_GY/20
    result = compare.statistics(a, b, ref)
    assert result['comparison']['J1'][0]['delta'] == pytest.approx(100.)
    assert result['comparison']['J1'][0]['proto'] == pytest.approx(2*ref, rel=1e-12)
    assert result['abstract']['deep_Dref_percent']['RMS'] > 0
    assert len(result['comparison']['J1']+result['comparison']['J2']) == 8
    assert len(result['comparison']['J3']) == 3
    assert result['slab_ratios'][0]['ratio_minus_1_percent'] == pytest.approx(100.)


@pytest.fixture
def comparison(tmp_path):
    root = tmp_path
    core.save_npz(root/'egs5/main_egs/result.npz', energy())
    core.write_json(root/'egs5/main_egs/summary.json', {'status': 'synthetic'})
    core.write_json(root/'egs5/main_egs/config.json', {'expected': True})
    core.write_json(root/'pilot_lock_p3b3.json', {'N_main': 46000000, 'Dref': 1.})
    manifest = dict(Dref=1., comparison_inputs={n: core.file_sha(root/n) for n in bundle.COMPARISON})
    return root, manifest


@pytest.mark.parametrize('name', bundle.COMPARISON)
def test_entry_compare_input_byte_tamper(comparison, name):
    root, manifest = comparison
    p = root/name
    content = bytearray(p.read_bytes()); content[0] ^= 1; p.write_bytes(content)
    with pytest.raises(core.Stop, match='SHA256'):
        compare.verify_comparison(root, manifest, {}, expected_config={'expected': True})


def test_entry_compare_config_mismatch(comparison):
    root, manifest = comparison
    with pytest.raises(core.Stop, match='config.json'):
        compare.verify_comparison(root, manifest, {}, expected_config={'expected': False})


def test_entry_collect_missing(tmp_path):
    with pytest.raises(core.Stop, match='missing'):
        bundle.collect(tmp_path, {}, {})


def test_entry_stack_layout_and_shapes(monkeypatch):
    from prototypes.mv_electron.gpu import p7_core
    layouts = []
    def reduced(row, layout):
        layouts.append(layout)
        return {k: np.full(shape[1:], row) for k, shape in stats.SHAPES.items()}
    monkeypatch.setattr(p7_core, 'reduced', reduced)
    result = compare.stacked(list(range(100)), {'histories': 2000})
    assert layouts == ['p6']*100
    assert result['N'] == 2000 and result['nbatch'] == 100
    for key, shape in stats.SHAPES.items():
        assert result[key].shape == shape
        assert np.all(result[key][20] == 20)


def test_entry_ranges_overlap_rejected(tmp_path, monkeypatch):
    # The collector must derive coverage from the observations, not trust the metadata.
    monkeypatch.setattr(bundle, 'existing', lambda *args: set(range(100)))
    monkeypatch.setattr(bundle, 'read_batch', lambda *args: {})
    monkeypatch.setattr(bundle, 'read_json', lambda path: {'observed': {'start': 0, 'n': 20}})
    with pytest.raises(core.Stop, match='overlap/gap'):
        bundle.collect(tmp_path, {'histories': 2000}, {})


def test_entry_profiles_without_egs5(tmp_path, monkeypatch):
    """4 MV: PDD and OCR tables from the GPU batches alone; units follow stats.dose (MeV -> Gy per history)."""
    root = tmp_path
    (root/'condition.json').write_text('{"histories": 2000}'); (root/'materials.json').write_text('{}')
    monkeypatch.setattr(bundle, 'load', lambda d: ({}, {}))
    monkeypatch.setattr(bundle, 'read_json', lambda p: {'histories': 2000})
    monkeypatch.setattr(bundle, 'collect', lambda r, c, m: [None]*100)
    monkeypatch.setattr(compare, 'verify_sessions', lambda *a: None)
    monkeypatch.setattr(compare, 'stacked', lambda rows, cond: energy(2.))
    result = compare.profiles(root, root)
    assert result['status'].startswith('EGS5 との比較なし')
    assert len(result['pdd']['dose']) == 120 and len(result['ocr']['dose']) == 3 and len(result['ocr']['dose'][0]) == 36
    assert result['pdd']['dose'][0] == pytest.approx(2.*stats.MEV_TO_GY/(20*1.), rel=1e-12)
    assert result['ocr']['depth_cm'] == [5, 10, 20]
