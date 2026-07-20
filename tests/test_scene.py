"""scene.yamlバリデーション — source.spectrum（陽なスペクトル指定、単色ビーム等）関連。"""
import pytest

from chatcarlo.scene import validate_scene

_BASE = {
    "geometry": [
        {"name": "phantom", "shape": "box", "material": "water",
         "size_cm": [30, 30, 20], "center": [0, 0, 10]},
    ],
}


def _src(**overrides):
    src = {
        "position": [0, -100, 10],
        "direction": [0, 1, 0],
        "field": {"shape": "rect", "size_cm": [10, 10], "sid_cm": 100},
    }
    src.update(overrides)
    return {**_BASE, "source": src}


def test_kvp_still_required_without_spectrum():
    scene = validate_scene(_src())
    assert not scene.ok
    assert any("kvp" in e.path for e in scene.errors)


def test_spectrum_makes_kvp_optional():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}]))
    assert scene.ok, scene.errors


def test_spectrum_malformed_rejected():
    scene = validate_scene(_src(spectrum=[{"energy_keV": -1.0, "weight": 1.0}]))
    assert not scene.ok
    assert any(e.path == "source.spectrum" for e in scene.errors)


def test_spectrum_above_150kev_rejected():
    """診断X線領域の上限150 keV超は断面積テーブル範囲外で輸送実行時に
    エラーになるため、validate時にfail-fastで弾く
    （docs/plan_transport_speedup.md、輸送カーネル高速化でテーブル化した際に
    判明した回帰——以前はxraylib直呼びで上限が事実上無制限だった）。"""
    scene = validate_scene(_src(spectrum=[{"energy_keV": 200.0, "weight": 1.0}]))
    assert not scene.ok
    assert any(e.path == "source.spectrum" for e in scene.errors)


def test_spectrum_at_150kev_accepted():
    """境界値150.0 keVちょうどは許容される（断面積テーブル上限と一致、境界含む）。"""
    scene = validate_scene(_src(spectrum=[{"energy_keV": 150.0, "weight": 1.0}]))
    assert scene.ok, scene.errors


def test_kvp_above_150_rejected():
    scene = validate_scene(_src(kvp=180.0))
    assert not scene.ok
    assert any("kvp" in e.path for e in scene.errors)


def test_spectrum_with_mas_rejected():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}], mas=4.0))
    assert not scene.ok
    assert any("mas" in e.message for e in scene.errors)


def test_spectrum_with_heel_effect_rejected():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}],
                                 heel_effect=True, anode_direction=[1, 0, 0]))
    assert not scene.ok
    assert any("heel_effect" in e.message for e in scene.errors)


def test_spectrum_with_ctdi_rejected():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}],
                                 ctdi_vol_mGy=10.0, rotation={"isocenter": [0, 0, 10]}))
    assert not scene.ok
    assert any("ctdi_vol_mGy" in e.message for e in scene.errors)


def test_spectrum_with_kvp_rejected():
    """spectrum指定時にkvpも残っていると、輸送(spectrum優先)とpreview表示が
    食い違う温床になる（vive-auditor所見）。曖昧さを許さずエラーにする。"""
    scene = validate_scene(_src(kvp=80.0, spectrum=[{"energy_keV": 60.0, "weight": 1.0}]))
    assert not scene.ok
    assert any("kvp" in e.message for e in scene.errors)


def test_parallel_field_no_sid_required():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}],
                                 field={"shape": "parallel", "size_cm": [10, 10]}))
    assert scene.ok, scene.errors


def test_parallel_field_requires_size_cm():
    scene = validate_scene(_src(spectrum=[{"energy_keV": 60.0, "weight": 1.0}],
                                 field={"shape": "parallel"}))
    assert not scene.ok
    assert any("size_cm" in e.path for e in scene.errors)


def test_parallel_field_with_mas_rejected():
    scene = validate_scene(_src(kvp=100.0, mas=4.0,
                                 field={"shape": "parallel", "size_cm": [10, 10]}))
    assert not scene.ok
    assert any("mas" in e.message for e in scene.errors)


def test_parallel_field_with_heel_effect_rejected():
    scene = validate_scene(_src(kvp=100.0, heel_effect=True, anode_direction=[1, 0, 0],
                                 field={"shape": "parallel", "size_cm": [10, 10]}))
    assert not scene.ok
    assert any("heel_effect" in e.message for e in scene.errors)
