"""A1a-3/4/5/8: source preservation, native identities and table upper limit."""
import ctypes
import difflib
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import numpy as np
import pytest
from chatcarlo.gpu import bundle,core,physics_sets
from chatcarlo.scene import load_scene

ROOT=Path(__file__).resolve().parents[1]
OLD_SHA='da3645b296c090ce719989326e4954dbb09792ed0a778cb619bd82996c30736a'


def test_expanded_source_insertions_and_fixture(tmp_path):
    old=(ROOT/'tests/fixtures/g5_a1_old_cuda_source.txt').read_text()
    assert hashlib.sha256(old.encode()).hexdigest()==OLD_SHA
    # Reconstruct the baseline entirely through git show, independent of the
    # working-tree headers used by cuda_source().
    names=subprocess.check_output(['git','ls-tree','--name-only','19c281e:chatcarlo/gpu'],text=True).splitlines()
    for name in names:
        if Path(name).suffix in ('.py','.h','.cpp','.cu'):
            (tmp_path/name).write_bytes(subprocess.check_output(['git','show','19c281e:chatcarlo/gpu/'+name]))
    spec=importlib.util.spec_from_file_location('a1_old_core',tmp_path/'core.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.cuda_source()==old
    new=core.cuda_source();assert '#include "' not in new
    assert 'g5_physics_set_kernel' in new
    # Explicit permitted line vocabulary; every other line must be an old line
    # in its original order. Includes repeated header guards after expansion.
    allowed={'#ifndef G5_BASE_H','#define G5_BASE_H','#endif','',
        '#ifndef G5_PHYSICS_SET_H','#define G5_PHYSICS_SET_H','#ifndef G5_SET_ID',
        '#define G5_SET_ID 1','#ifndef G5_SET_TAG','#define G5_SET_TAG 0ULL',
        'extern "C" __global__ void g5_physics_set_kernel(uint64_t *out) {if(blockIdx.x || threadIdx.x)return;out[0]=G5_SET_ID;out[1]=G5_SET_TAG;}'}
    changes=difflib.SequenceMatcher(None,old.splitlines(),new.splitlines(),autojunk=False)
    for tag,i,j,k,l in changes.get_opcodes():
        assert tag in ('equal','insert'),(tag,old.splitlines()[i:j],new.splitlines()[k:l])
        if tag=='insert':assert set(new.splitlines()[k:l])<=allowed


def test_source_closure():
    found=set()
    def walk(name):
        for included in re.findall(r'^#include "([^"]+)"',(core.HERE/name).read_text(),re.M):
            if included not in found:found.add(included);walk(included)
    walk('g1_cpu.cpp');walk('g1_history.h') # .cu's COMMON marker expands this root
    walk('g1_gpu.cu')
    assert found==set(core.SOURCES)-{'g1_cpu.cpp','g1_gpu.cu'}


@pytest.fixture
def arrays(tmp_path):
    bundle.create(load_scene(str(ROOT/'docs/validation/gpu/g1/scenes/s1.yaml')),None,tmp_path/'bundle')
    return bundle.load(tmp_path/'bundle')[0]


def test_native_identity_and_same_cache_directory(arrays,tmp_path,monkeypatch):
    normal=core.Core(arrays,build_dir=tmp_path)
    assert normal.physics_set==physics_sets.declaration()
    before=set(tmp_path.glob('*.so'))
    original=physics_sets.declaration
    monkeypatch.setattr(physics_sets,'declaration',lambda *a,**kw:dict(original(),tag=original()['tag']+1))
    changed=core.Core(arrays,build_dir=tmp_path)
    assert changed.lib._name!=normal.lib._name
    assert len(set(tmp_path.glob('*.so'))-before)==1
    # G2 calls the same translation unit's identity and histories.
    from chatcarlo.gpu.g2_tables import pack
    from chatcarlo.gpu.g2_core import Core
    aa=dict(arrays,**pack(['air','water']))
    # This test only needs to establish inherited identity verification; G2
    # arrays are exercised with the actual scene's names below in driver tests.
    with pytest.raises(ValueError,match='declaration mismatch'):
        Core(aa,build_dir=tmp_path,physics_declaration=normal.physics_set)


@pytest.mark.parametrize('g2',[False,True])
@pytest.mark.parametrize('macro,value',[('G5_SET_ID','99'),('G5_SET_TAG','0ULL')])
def test_wrong_compiled_identity_rejected(arrays,tmp_path,monkeypatch,macro,value,g2):
    real=subprocess.run
    def compile_bad(command,**kwargs):
        command=[('-D'+macro+'='+value) if v.startswith('-D'+macro+'=') else v for v in command]
        return real(command,**kwargs)
    monkeypatch.setattr(core.subprocess,'run',compile_bad)
    driver=core.Core
    if g2:
        from chatcarlo.gpu.g2_core import Core
        driver=Core
    with pytest.raises(ValueError,match='ID/tag mismatch'):driver(arrays,build_dir=tmp_path)


def test_unsupported_and_bundle_declaration(arrays):
    with pytest.raises(ValueError,match='not implemented'):core.Core(arrays,physics_set='therapy-electron-v1')
    with pytest.raises(ValueError,match='declaration mismatch'):
        core.Core(arrays,physics_declaration=dict(physics_sets.declaration(),id=2))


def test_energy_upper_limit(arrays):
    driver=core.Core(arrays)
    o=np.zeros((1,3));d=np.array([[0.,0.,1.]])
    driver.run(o,d,np.array([150.]),seed=42)
    from chatcarlo.spectrum import _default_spectrum
    energies,_=_default_spectrum(150,2.5,12.)
    driver.run(o,d,np.array([energies[-1]]),seed=42)
    with pytest.raises(ValueError,match='upper bound'):driver.run(o,d,np.array([150.0001]),seed=42)
    # Padding is excluded: replacing it by a huge energy cannot raise the cap.
    le=arrays['data'][arrays['offsets'][3]:arrays['offsets'][4]].reshape(driver.nm,driver.me,driver.ng)
    counts=arrays['data'][arrays['offsets'][5]:arrays['offsets'][6]].reshape(driver.nm,driver.me)
    for m in range(driver.nm):
        for j in range(driver.me):le[m,j,int(counts[m,j]):]=100.
    assert core.Core(arrays).max_log_energy==driver.max_log_energy
