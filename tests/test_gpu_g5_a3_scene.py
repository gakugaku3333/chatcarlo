"""D10 and entry routing without private inputs or cache access."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from chatcarlo.scene import validate_scene
from chatcarlo.gpu import therapy_scene, physics_sets
from chatcarlo import __main__ as cli

ROOT=Path(__file__).resolve().parents[1]


def dx(energy=60):
    return dict(source=dict(spectrum=[dict(energy_keV=energy,weight=1)],position=[0,0,-10],direction=[0,0,1],field=dict(shape='parallel',size_cm=[1,1])),geometry=[dict(shape='box',material='water',center=[0,0,5],size_cm=[10,10,10])])


def mv():
    return yaml.safe_load((ROOT/'examples/therapy/water_10mv_10x10.yaml').read_text())


@pytest.mark.parametrize('energies',[[151],[1000],[60,151],[151,60]])
def test_unverified_energy(energies):
    raw=dx();raw['source']['spectrum']=[dict(energy_keV=e,weight=1) for e in energies]
    s=validate_scene(raw)
    assert not s.ok and any('検証していない区間' in e.message for e in s.errors)


@pytest.mark.parametrize('kvp',[150,151])
def test_kvp(kvp):
    raw=dx();raw['source'].pop('spectrum');raw['source']['kvp']=kvp
    s=validate_scene(raw)
    assert s.ok==(kvp==150)
    if not s.ok:assert '検証していない区間' in str(s.errors)


@pytest.mark.parametrize('energy',[60,150])
@pytest.mark.parametrize('setting',[None,physics_sets.DIAGNOSTIC])
@pytest.mark.parametrize('fluor',[True,False])
def test_diagnostic_valid(energy,setting,fluor):
    raw=dx(energy);raw['physics']={'fluorescence':fluor}
    if setting:raw['physics']['set']=setting
    assert validate_scene(raw).ok


@pytest.mark.parametrize('setting',['therapy-electron-v1','unknown'])
def test_diagnostic_set_rejected(setting):
    raw=dx();raw['physics']={'set':setting}
    assert not validate_scene(raw).ok


@pytest.mark.parametrize('spectrum',['mohan10','mohan4'])
@pytest.mark.parametrize('setting',[None,'therapy-electron-v1'])
def test_therapy_valid(spectrum,setting):
    raw=mv();raw['source']['spectrum']=spectrum
    if setting:raw['physics']={'set':setting}
    assert therapy_scene.validate(raw)==raw


@pytest.mark.parametrize('damage',['spectrum','diagnostic','unknown','fluor','quantity'])
def test_therapy_refusals(damage):
    raw=mv()
    if damage=='spectrum':raw['source']['spectrum']='mohan6'
    elif damage=='quantity':raw['quantity']='H*(10)'
    elif damage=='fluor':raw['physics']={'fluorescence':False}
    else:raw['physics']={'set':physics_sets.DIAGNOSTIC if damage=='diagnostic' else 'unknown'}
    with pytest.raises(ValueError):therapy_scene.validate(raw)


def test_quantity_api():
    with pytest.raises(ValueError,match='unsupported'):physics_sets.require_quantity('therapy-electron-v1','H*(10)')


def test_no_prototype_runtime_and_help():
    code="""import sys
from chatcarlo.gpu.therapy_entry import main,geometry_json
geometry_json('mohan4')
assert not any(k=='prototypes' or k.startswith('prototypes.') for k in sys.modules)
main(['run','--help'])
"""
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,cwd=ROOT)
    assert p.returncode==0,p.stderr
    assert '--dose-out' not in p.stdout and '--quantity' not in p.stdout


def test_seed_ratio_dispatch(monkeypatch):
    from prototypes.mv_electron.entry import cli as old
    monkeypatch.setattr(old,'main',lambda argv:37)
    assert cli._mv_dispatch(['validate',str(ROOT/'examples/therapy/prostate_seed_7field.yaml')])==37


def test_product_cli_has_no_indirect_prototype_imports(tmp_path):
    test_module=ROOT/'tests/test_gpu_g5_a3_contracts.py'
    code='''import importlib.util,sys,pytest
from pathlib import Path
spec=importlib.util.spec_from_file_location('a3_contract_probe',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with pytest.MonkeyPatch.context() as patch:
    module.test_cli_modes_with_fake_vm(Path(sys.argv[2]),patch)
assert not any(k=='prototypes' or k.startswith('prototypes.') for k in sys.modules)
'''
    result=subprocess.run([sys.executable,'-c',code,str(test_module),str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
