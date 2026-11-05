"""S1 identity pairs and unchanged native photon transport, independently frozen."""
import re
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from chatcarlo.gpu import core, bundle, physics_sets as ps, contracts, therapy
from chatcarlo.scene import load_scene

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '02bab32af9083897cd255666d91ea7979c6f99af'


def test_specializations_and_quantities():
    assert set(ps.REGISTRY) == {'chatcarlo-photon-electron-v1'}
    local = ps.declaration(ps.NAME, specialization=ps.LOCAL)
    electron = ps.declaration(ps.NAME, specialization=ps.ELECTRON)
    assert local['tag'] != electron['tag']
    for spec, allowed, denied in ((ps.LOCAL, 'H*(10)', 'absorbed-dose'),
                                 (ps.ELECTRON, 'absorbed-dose', 'track-length-kerma')):
        ps.require_quantity(ps.NAME, allowed, specialization=spec)
        with pytest.raises(ValueError, match='unsupported'):
            ps.require_quantity(ps.NAME, denied, specialization=spec)
    for old in ('diagnostic-photon-kerma-v1', 'therapy-electron-v1'):
        with pytest.raises(ValueError, match='retired'):
            ps.declaration(old)
    with pytest.raises(ValueError, match='mismatch'):
        ps.check(local, electron)


def test_seals_bind_specialization(tmp_path):
    for spec in (ps.LOCAL, ps.ELECTRON):
        cond = dict(physics_set=ps.declaration(ps.NAME, specialization=spec))
        contracts.seal(tmp_path, cond, [])
        assert contracts.verify(tmp_path)['condition'] == cond
    old = dict(physics_set=dict(name='therapy-electron-v1', id=2, tag=1))
    with pytest.raises(ValueError, match='retired'):
        contracts.seal(tmp_path, old, [])


def frozen_headers(path):
    path.mkdir()
    names = subprocess.check_output(['git', 'ls-tree', '--name-only', BASELINE+':chatcarlo/gpu'], text=True).splitlines()
    for name in names:
        if Path(name).suffix in ('.h', '.cpp', '.cu'):
            (path/name).write_bytes(subprocess.check_output(['git', 'show', BASELINE+':chatcarlo/gpu/'+name]))
    return path


def test_diagnostic_native_frozen_and_legacy(tmp_path, monkeypatch):
    scene = load_scene(str(ROOT/'docs/validation/gpu/g1/scenes/s2.yaml'))
    bundle.create(scene, None, tmp_path/'bundle')
    arrays, _, spectrum, _, _ = bundle.load(tmp_path/'bundle')
    current = core.Core(arrays, build_dir=tmp_path/'current')
    legacy = core.Core(arrays, build_dir=tmp_path/'legacy', _test_legacy_physics=True)
    assert current.check_physics_set()['specialization'] == ps.LOCAL
    from chatcarlo.gpu.source_host import sample_source_photons
    source = sample_source_photons(scene.raw['source'], 2048, np.random.default_rng(9504000),
        spectrum=spectrum)
    expected = legacy.run(*source, seed=9504001, record=True)
    baseline = frozen_headers(tmp_path/'baseline')
    monkeypatch.setattr(core, 'HERE', baseline)
    monkeypatch.setattr(core, 'SOURCES', tuple(n for n in core.SOURCES if n!='photon_physics.h')+('g1_physics.h',))
    # The frozen driver knows the original single-photon aggregation layout.
    import types
    module=types.ModuleType('chatcarlo.gpu._a4u_frozen_core');module.__package__='chatcarlo.gpu';module.__file__=str(baseline/'core.py')
    code=subprocess.check_output(['git','show',BASELINE+':chatcarlo/gpu/core.py'],text=True)
    exec(compile(code,module.__file__,'exec'),module.__dict__)
    frozen=module.Core(arrays,build_dir=tmp_path/'old')
    for key,value in frozen.run(*source,seed=9504001,record=True).items():
        np.testing.assert_array_equal(value,expected[key])


def test_therapy_native_frozen_and_legacy(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT/'docs/validation/gpu/g5'))
    import a2_common
    _, materials, arrays = a2_common.inputs()
    arrays = a2_common.no_spectrum(arrays)
    current = therapy.CPU(tmp_path/'current', materials, arrays)
    legacy = therapy.CPU(tmp_path/'legacy', materials, arrays, _test_legacy_physics=True)
    expected = legacy.histories(32, 20301831, slots=3)
    assert current.observed['specialization'] == ps.ELECTRON
    for key, value in legacy.histories(32, 20301831, slots=3).items():
        np.testing.assert_array_equal(value, expected[key])
    baseline = frozen_headers(tmp_path/'baseline')
    # The current build driver fingerprints the new header as well; supply it
    # beside frozen headers, which do not include it.
    (baseline/'photon_physics.h').write_text((therapy.HERE/'photon_physics.h').read_text())
    monkeypatch.setattr(therapy, 'HERE', baseline)
    frozen = therapy.CPU(tmp_path/'old', materials, arrays)
    for key, value in frozen.histories(32, 20301831, slots=3).items():
        np.testing.assert_array_equal(value, expected[key])


@pytest.mark.parametrize('entry', ['therapy', 'seed'])
def test_cuda_expansion_contains_active_shared_sections(entry, tmp_path):
    from chatcarlo.gpu import seed_core
    source = therapy.cuda_source() if entry=='therapy' else seed_core.cuda_source()
    assert '#include "' not in source
    path = tmp_path/'expanded.cpp'
    path.write_text(source)
    active = subprocess.check_output(['clang++', '-E', '-P', '-x', 'c++', '-DP5_GPU', '-DP5_RNG_PHILOX', '-DT2_SEED='+str(int(entry=='seed')), str(path)], text=True)
    for name in ('p5_compton', 'p8g_coefficients', 'p8g_relax'):
        assert len(re.findall(r'(?m)^.*?\b(?:void|int) '+name+r'\(', active)) == 1


def test_seed_scene_new_identity_and_retired_rejection():
    import yaml
    from chatcarlo.gpu import seed_scene
    raw = yaml.safe_load((ROOT/'examples/therapy/prostate_seed_7field_g5_a2b.yaml').read_text())
    raw['physics']['set'] = ps.NAME
    assert seed_scene.validate(raw) == raw
    for old in ('diagnostic-photon-kerma-v1', 'therapy-electron-v1'):
        raw['physics']['set'] = old
        with pytest.raises(ValueError, match='physics.set'):
            seed_scene.validate(raw)
