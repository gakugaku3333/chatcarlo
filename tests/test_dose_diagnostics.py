"""最大値ボクセルの診断（chatcarlo run --dose-grid の「最大」統計が
点線源近傍・空気中ボクセルという非物理的な位置に落ちていないかの検証）。

背景: chest_room.yamlで実際にH*(10)最大値を調べたところ、線源から
3〜19cmの空気中ボクセルに集中しており（1/r²則の点線源発散）、106桁の
pSv値になっていた。吸収線量の最大値も、患者体表のすぐ外側の低密度な
空気ボクセルに落ちており（カーマは同程度でも密度が小さいぶん線量が
増幅される）、どちらも「実在する位置の被ばく評価」には使えない値
だった（docs/lessons_learned.md参照）。
"""
from __future__ import annotations

import numpy as np
import pytest

from chatcarlo.geometry import Geometry
from chatcarlo.scene import validate_scene
from chatcarlo.tally import VoxelGrid
from chatcarlo.diagnostics import (background_medium_warning, batch_shortage_message,
                                    dose_map_Gy, grid_reliability_summary,
                                    max_voxel_position_cm, near_source_air_warning,
                                    unreliable_max_warning)
from chatcarlo.transport import run_transport

_BASE_SOURCE = {
    "kvp": 120, "position": [0, -180, 140], "direction": [0, 1, 0],
    "field": {"size_cm": [35, 43], "sid_cm": 180}, "filtration_mm_al": 2.5, "mas": 4.0,
}
_SLAB_GEOMETRY = [{
    "name": "slab", "shape": "box", "material": "water",
    "center": [0, 0, 140], "size_cm": [30, 30, 30],
}]


def test_max_voxel_position_matches_argmax():
    grid = VoxelGrid(origin_cm=np.array([0.0, 0.0, 0.0]), shape=(4, 4, 4), voxel_size_cm=2.0)
    data = np.zeros(grid.shape)
    data[1, 2, 3] = 99.0
    pos = max_voxel_position_cm(grid, data)
    expected = grid.origin_cm + (np.array([1, 2, 3]) + 0.5) * grid.voxel_size_cm
    assert np.allclose(pos, expected)


def test_background_medium_warning_fires_for_air():
    msg = background_medium_warning("air", background="air")
    assert msg is not None
    assert "空気" in msg


def test_background_medium_warning_silent_for_declared_material():
    assert background_medium_warning("water", background="air") is None


def test_near_source_air_warning_fires_when_closer_than_any_object():
    msg = near_source_air_warning("air", background="air",
                                   distance_from_source_cm=3.0,
                                   nearest_object_distance_cm=160.0)
    assert msg is not None
    assert "1/r" in msg or "非物理的" in msg


def test_near_source_air_warning_silent_when_material_not_background():
    assert near_source_air_warning("lead", background="air",
                                    distance_from_source_cm=3.0,
                                    nearest_object_distance_cm=160.0) is None


def test_near_source_air_warning_silent_when_farther_than_nearest_object():
    # 物体より線源から遠い位置は「実在し得ない近傍」ではないので警告しない
    assert near_source_air_warning("air", background="air",
                                    distance_from_source_cm=200.0,
                                    nearest_object_distance_cm=160.0) is None


def test_near_source_air_warning_silent_without_any_object():
    assert near_source_air_warning("air", background="air",
                                    distance_from_source_cm=3.0,
                                    nearest_object_distance_cm=None) is None


def test_h10_max_voxel_in_real_scene_triggers_near_source_warning():
    """実際のシーンでH*(10)最大値ボクセルを診断し、線源近傍の空気だと分かること。"""
    scene = validate_scene({"source": _BASE_SOURCE, "geometry": _SLAB_GEOMETRY})
    assert scene.ok
    geometry = Geometry(scene.raw["geometry"])
    result = run_transport(scene, n_histories=200_000, seed=3,
                            dose_grid=True, grid_resolution_cm=2.0)

    h10_per_history = result.grid.h10_map_pSv() / result.n_histories
    pos = max_voxel_position_cm(result.grid, h10_per_history)
    material = str(geometry.material_at(pos[None, :])[0])
    src_pos = scene.raw["source"]["position"]
    dist = float(np.linalg.norm(pos - np.asarray(src_pos, dtype=float)))
    nearest_obj = geometry.nearest_object_distance_cm(src_pos)

    warning = near_source_air_warning(material, geometry.background, dist, nearest_obj)
    assert warning is not None, (
        f"線源近傍の空気ボクセル(材料={material}, 距離={dist:.1f}cm, "
        f"最寄り物体={nearest_obj}cm)のはずが警告が出なかった"
    )


def test_dose_max_voxel_on_water_slab_surface_triggers_background_warning():
    """スラブ中心を狙った鉛筆ビームで、最大吸収線量が表面直前の空気ボクセルに
    落ちたとき警告が出ることの検証（解像度2cm）。

    経緯: 当初この境界アーティファクトは解像度1cmでも常に空気側に出ていた
    （2cm:1.4e-13, 1cm:1.5e-13, 0.5cm:6.4e-13, 0.2cm:3.5e-12 Gy/history と
    細かいほど成長）。tally.pyのサブステップを中点（決定的）から層化乱数点に
    変えた修正（表面ボクセル-2.7%の系統バイアス除去、監査で発見）以降は、
    1cmでは最大値が物理的に正しい水の表面ボクセルへ移る（空気側1.212e-13 vs
    水側1.246e-13の僅差、n=100k時点）ようになった。ただし解像度を細かくすると
    最大値が成長する現象自体は残っている（0.5cm:4.8e-13。これは系統バイアスでは
    なく、max_substepsクランプ由来の分散増大＋最大値の極値統計）ため、
    警告機構の存在意義は変わらない。ここでは空気側が確実に最大になる
    解像度2cmで警告発火を検証する。
    """
    src = {**_BASE_SOURCE, "position": [0, -20, 140], "field": {"size_cm": [2, 2], "sid_cm": 20}}
    scene = validate_scene({"source": src, "geometry": _SLAB_GEOMETRY})
    assert scene.ok
    geometry = Geometry(scene.raw["geometry"])
    result = run_transport(scene, n_histories=300_000, seed=4,
                            dose_grid=True, grid_resolution_cm=2.0)

    dose_per_history = dose_map_Gy(result.grid, geometry) / result.n_histories
    pos = max_voxel_position_cm(result.grid, dose_per_history)
    material = str(geometry.material_at(pos[None, :])[0])

    warning = background_medium_warning(material, geometry.background)
    assert warning is not None, (
        f"最大吸収線量ボクセルの材料={material}（想定: 表面直前のair）。"
        "境界効果が起きない環境になっているなら、このテストの前提を見直すこと。"
    )


def test_dose_max_voxel_inside_all_water_grid_does_not_trigger_warning():
    """観測グリッドを物体内部だけに限定すれば、最大値は必然的にその材料になる。

    run_transportの既定グリッドは物体+50cmマージンを覆うため常に空気ボクセルを
    含む。ここではtransport_photonsを直接呼び、水スラブ内部だけを覆う専用グリッドを
    敷くことで「境界効果を含まない領域の最大値」では警告が出ないことを確認する。
    """
    from chatcarlo.tally import VoxelGrid
    from chatcarlo.source import sample_source_photons
    from chatcarlo.transport import transport_photons

    src = {**_BASE_SOURCE, "position": [0, -20, 140], "field": {"size_cm": [2, 2], "sid_cm": 20}}
    scene = validate_scene({"source": src, "geometry": _SLAB_GEOMETRY})
    assert scene.ok
    geometry = Geometry(scene.raw["geometry"])

    # スラブは center=[0,0,140], size=[30,30,30] -> y方向 [-15,15]。
    # 表面から5cm内側だけを覆うグリッド（境界効果を含まない）。
    grid = VoxelGrid(origin_cm=np.array([-10.0, -10.0, 135.0]), shape=(20, 20, 10), voxel_size_cm=1.0)

    rng = np.random.default_rng(5)
    pos, dirv, energy = sample_source_photons(scene.raw["source"], 300_000, rng)
    transport_photons(pos, dirv, energy, geometry, rng, grid=grid)

    dose_per_history = dose_map_Gy(grid, geometry)
    max_pos = max_voxel_position_cm(grid, dose_per_history)
    material = str(geometry.material_at(max_pos[None, :])[0])

    assert material == "water"
    assert background_medium_warning(material, geometry.background) is None


# --- Phase 3 (docs/plan_statistical_uncertainty.md): 統計信頼性の警告関数 ---


def test_unreliable_max_warning_silent_when_r_low_and_hit_fraction_high():
    assert unreliable_max_warning(R=0.02, n_hit_batches=48, n_batches=50) is None


def test_unreliable_max_warning_fires_on_high_r():
    msg = unreliable_max_warning(R=0.25, n_hit_batches=48, n_batches=50)
    assert msg is not None
    assert "R=0.250" in msg


def test_unreliable_max_warning_fires_on_low_hit_fraction_even_if_r_looks_small():
    """設計判断7: 寄与バッチ数が少ないとRは楽観バイアスで小さく出る——
    R自体が小さくてもhit_fractionが閾値未満なら警告する（偽の安心を防ぐ）。"""
    msg = unreliable_max_warning(R=0.01, n_hit_batches=2, n_batches=50)
    assert msg is not None
    assert "2/50" in msg


def test_unreliable_max_warning_silent_when_n_batches_zero():
    assert unreliable_max_warning(R=float("nan"), n_hit_batches=0, n_batches=0) is None


def test_batch_shortage_message_none_when_m_sufficient():
    assert batch_shortage_message(n_batches=20, n_histories=4_000_000, batch_size=200_000) is None


def test_batch_shortage_message_computes_actionable_numbers():
    """既定設定（n=1e5, batch_size=200,000 -> M=1）で、-nと--batch-sizeの
    具体的な提案値が実際の設定値から計算されていること（固定文言ではない）。"""
    msg = batch_shortage_message(n_batches=1, n_histories=100_000, batch_size=200_000)
    assert msg is not None
    assert "M=1" in msg
    assert "4e+06" in msg or "4,000,000" in msg  # batch_size*20
    assert "5000" in msg  # ceil(n_histories/20)


def test_grid_reliability_summary_counts_low_r_and_unreached_fractions():
    R = np.array([0.01, 0.02, 0.5, np.nan])
    n_hit = np.array([10, 10, 10, 0])
    summary = grid_reliability_summary(R, n_hit, n_batches=10)
    # 非nanは3つ、うちR<0.10は2つ -> 2/3
    assert summary["frac_low_r"] == pytest.approx(2 / 3)
    # 全4ボクセル中1つがhit=0 -> 1/4
    assert summary["frac_unreached"] == pytest.approx(0.25)


def test_grid_reliability_summary_all_nan_gives_nan_frac_low_r():
    R = np.full(4, np.nan)
    n_hit = np.zeros(4, dtype=int)
    summary = grid_reliability_summary(R, n_hit, n_batches=0)
    assert np.isnan(summary["frac_low_r"])
    assert summary["frac_unreached"] == pytest.approx(1.0)
