"""chatcarlo/kernel.py（Phase B-1: per-history Numbaカーネル）のテスト。

B-1bは多材料・多元素（重元素の非等間隔格子含む）・K殻蛍光・prange並列化に
対応した本実装（B-1aの簡略化——water単色・背景真空・蛍光無効——は解消済み）。
B-2は`--dose-grid`相当のtrack-lengthタリー統合（カーネルは区間を吐き出すだけで、
線量換算・グリッド分配は既存の監査済み`tally.accumulate_track_length_multi`を
再利用、`run_batch_with_tally`/`run_dose_grid`）。

`transport_photons`参照実装との統計的クロスチェック（Phase Bの検証戦略layer 1、
事前登録済みの許容基準込み。B-2のグリッド合計カーマ/H*(10)クロスチェックも同様）は
`docs/speedup_baseline/kernel_crosscheck.py`・`kernel_dose_grid_crosscheck.py`に
分離してある（実行に数十秒かかるため通常のpytestには含めない）。
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import chatcarlo.kernel as kernel_mod
from chatcarlo.kernel import (_compute_tally_weights, bake_box_scene, bake_scene_materials,
                               run_batch, run_batch_with_tally, run_dose_grid,
                               run_water_slab_probe)
from chatcarlo.dose_coefficients import h_star_10_per_fluence
from chatcarlo.materials import density, linear_mu, material_groups, mu_en_rho
from chatcarlo.tally import VoxelGrid

SCENARIOS = {
    "water20kev": (1.5, 20.0),
    "water60_free": (10.0, 60.0),
    "water150kev": (10.0, 150.0),
}


def _legacy_tally_weights(tables, material_codes, energies):
    names = np.array(tables.material_names, dtype=object)[material_codes]
    mu_en_linear = np.zeros(len(energies))
    for name, mask in material_groups(names):
        mu_en_linear[mask] = mu_en_rho(name, energies[mask]) * density(name)
    return energies * mu_en_linear, h_star_10_per_fluence(energies)


def _weight_tables(material_names):
    return SimpleNamespace(material_names=list(material_names))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_compute_tally_weights_empty(dtype):
    tables = _weight_tables(["water", "air"])
    got = _compute_tally_weights(
        tables, np.array([], dtype=dtype), np.array([], dtype=np.float64))
    for weights in got:
        assert weights.shape == (0,)
        assert weights.dtype == np.float64


def test_compute_tally_weights_rejects_invalid_contract_inputs():
    tables = _weight_tables(["water", "air"])
    with pytest.raises(ValueError):
        _compute_tally_weights(tables, np.array([0]), np.array([60.0, 70.0]))
    with pytest.raises(ValueError):
        _compute_tally_weights(tables, np.array([[0]]), np.array([60.0]))
    with pytest.raises(ValueError):
        _compute_tally_weights(tables, np.array([0]), np.array([[60.0]]))
    for invalid in (np.array([0.0, 1.0]), np.array([True, False])):
        with pytest.raises(TypeError):
            _compute_tally_weights(tables, invalid, np.array([60.0, 70.0]))
    for invalid in (np.array([-1]), np.array([2])):
        with pytest.raises(ValueError):
            _compute_tally_weights(tables, invalid, np.array([60.0]))
    with pytest.raises(ValueError):
        _compute_tally_weights(
            _weight_tables([]), np.array([0]), np.array([60.0]))


@pytest.mark.parametrize(
    "names,codes,energies",
    [
        (["water"], [0, 0, 0], [60.0, 60.0, 60.0]),
        (["water"], [0, 0, 0], [20.0, 60.0, 120.0]),
        (["water", "air"], [0, 1, 0, 1, 0], [20.0, 30.0, 60.0, 90.0, 120.0]),
        (["water", "air", "lead"], [2, 0, 1, 2, 0], [30.0, 40.0, 60.0, 80.0, 120.0]),
    ],
)
def test_compute_tally_weights_bit_exact_with_legacy(names, codes, energies):
    tables = _weight_tables(names)
    material_codes = np.asarray(codes, dtype=np.int64)
    segment_energies = np.asarray(energies, dtype=np.float64)
    codes_before = material_codes.copy()
    energies_before = segment_energies.copy()
    expected = _legacy_tally_weights(tables, material_codes, segment_energies)
    got = _compute_tally_weights(tables, material_codes, segment_energies)
    assert all(np.array_equal(a, b) for a, b in zip(got, expected))
    assert all(a.shape == segment_energies.shape and a.dtype == np.float64 for a in got)
    assert np.array_equal(material_codes, codes_before)
    assert np.array_equal(segment_energies, energies_before)
    if len(set(codes)) == 1 and len(set(energies)) == 1:
        assert got[0].min() == got[0].max()
        assert got[1].min() == got[1].max()


def test_compute_tally_weights_uses_scene_local_name_order_and_int32():
    energies = np.array([60.0, 60.0])
    codes64 = np.array([0, 1], dtype=np.int64)
    water_air = _compute_tally_weights(
        _weight_tables(["water", "air"]), codes64, energies)
    air_water = _compute_tally_weights(
        _weight_tables(["air", "water"]), codes64, energies)
    int32 = _compute_tally_weights(
        _weight_tables(["water", "air"]), codes64.astype(np.int32), energies)
    assert np.array_equal(water_air[0], air_water[0][::-1])
    assert np.array_equal(water_air[1], air_water[1])
    assert all(np.array_equal(a, b) for a, b in zip(water_air, int32))


def test_multi_material_tally_grid_matches_legacy_weights(monkeypatch):
    tables = bake_scene_materials(["water", "lead", "air"])
    boxes = [
        {"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 20.0, 20.0),
         "material": "water"},
        {"center": (0.0, 0.0, 0.0), "size_cm": (0.02, 20.0, 20.0),
         "material": "lead"},
    ]
    geom = bake_box_scene(
        boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    lo = np.array([-5.01, -10.01, -10.01])
    grid_new = VoxelGrid.from_bbox(
        lo, np.array([5.01, 10.01, 10.01]), resolution_cm=2.0)
    grid_old = VoxelGrid.from_bbox(
        lo, np.array([5.01, 10.01, 10.01]), resolution_cm=2.0)
    arguments = (
        tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), 20_000)
    run_batch_with_tally(
        *arguments, seed=29, grid=grid_new, n_chunks=4, use_njit_dda=False)
    monkeypatch.setattr(kernel_mod, "_compute_tally_weights",
                        _legacy_tally_weights)
    run_batch_with_tally(
        *arguments, seed=29, grid=grid_old, n_chunks=4, use_njit_dda=False)
    assert np.array_equal(grid_new.kerma_keV, grid_old.kerma_keV)
    assert np.array_equal(
        grid_new.h10_track_pSv_cm3, grid_old.h10_track_pSv_cm3)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_uncollided_fraction_matches_beer_lambert(scenario):
    """一次透過率（無散乱透過率）がBeer-Lambert解析解と統計誤差内で一致すること
    （背景=air・蛍光有効という本番同等の条件で、B-1bのデフォルト設定のまま）。
    """
    thickness_cm, energy_kev = SCENARIOS[scenario]
    n = 300_000
    mu = float(linear_mu("water", np.array([energy_kev]))[0])
    expected = math.exp(-mu * thickness_cm)
    stderr = math.sqrt(expected * (1 - expected) / n)

    r = run_water_slab_probe(thickness_cm=thickness_cm, energy_kev=energy_kev,
                              n_histories=n, seed=11, warmup_histories=2000, n_chunks=4)
    assert abs(r.uncollided_frac - expected) < 5 * stderr


def test_energy_conservation_per_history_water():
    """吸収履歴はenergy_deposited==入射エネルギー、脱出履歴はenergy_deposited+
    final_energy==入射エネルギーが浮動小数点誤差の範囲で厳密に成り立つこと。
    """
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    n = 50_000
    r = run_batch(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=5, n_chunks=4)

    assert (r.absorbed | r.escaped).all()
    assert np.allclose(r.energy_deposited[r.absorbed], 60.0, atol=1e-9)
    assert np.allclose((r.energy_deposited + r.final_energy)[r.escaped], 60.0, atol=1e-9)


def test_energy_conservation_with_fluorescence_copper():
    """K殻蛍光が実際に発生する材料(銅)でもエネルギー保存が厳密に成り立つこと
    ——蛍光放出時は(e - e_line)だけをその場で計上し、光子は新エネルギーで
    輸送を継続するため、吸収/脱出いずれの終端でも収支が閉じる必要がある。
    """
    tables = bake_scene_materials(["copper", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (0.05, 20.0, 20.0), "material": "copper"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    n = 100_000
    r = run_batch(tables, geom, 100.0, (-0.035, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=7, n_chunks=4,
                  fluorescence_enabled=True)

    assert r.n_fluorescence.sum() > 0  # このシナリオでは蛍光が実際に起きることを確認
    assert (r.absorbed | r.escaped).all()
    assert np.allclose(r.energy_deposited[r.absorbed], 100.0, atol=1e-9)
    assert np.allclose((r.energy_deposited + r.final_energy)[r.escaped], 100.0, atol=1e-9)


def test_fluorescence_disabled_matches_full_local_absorption():
    """fluorescence_enabled=Falseでは、光電吸収イベントは常にその場で全量吸収
    される（emit常にFalse）——`transport_photons`のfluorescence_enabled=False
    経路と同じ意味論。
    """
    tables = bake_scene_materials(["copper", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (0.05, 20.0, 20.0), "material": "copper"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    n = 50_000
    r = run_batch(tables, geom, 100.0, (-0.035, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=7, n_chunks=4,
                  fluorescence_enabled=False)
    assert r.n_fluorescence.sum() == 0
    assert np.allclose(r.energy_deposited[r.absorbed], 100.0, atol=1e-9)


def test_non_uniform_grid_element_used_via_air_background():
    """空気中のAr(Z=18)は吸収端補強点により非等間隔格子（materials.pyの
    _uniform_log_stepがNoneを返す）——B-1bの`_element_index_frac`はこの
    fallback(二分探索)経路もbakeして使えることを、air背景を含むシーンが
    エラーなく完走することで確認する（bake_scene_materialsが非等間隔材料を
    拒否しないこと自体もここで検証——B-1aは拒否する設計だった）。
    """
    from chatcarlo.materials import _element_xs_tables
    assert _element_xs_tables(18)["uniform_step"] is None  # 前提の確認

    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    r = run_batch(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), 20_000, seed=3, n_chunks=2)
    assert (r.absorbed | r.escaped).all()


def test_chunk_reproducibility_same_seed_same_n_chunks():
    """同一(seed, n_chunks)なら結果がビット一致で再現すること
    （チャンク単位の決定的シード設計、docs/plan_chatcarlo_speedup_post_egs5.md
    B-0/B-1参照）。"""
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    args = (tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), 10_000)
    r1 = run_batch(*args, seed=123, n_chunks=4)
    r2 = run_batch(*args, seed=123, n_chunks=4)
    assert np.array_equal(r1.n_scatter, r2.n_scatter)
    assert np.array_equal(r1.absorbed, r2.absorbed)
    assert np.array_equal(r1.final_energy, r2.final_energy)


def test_chunk_count_changes_stream_but_not_statistics():
    """n_chunksを変えるとチャンク分割自体が変わるためビット一致はしない
    （既存の`--workers`と同じ制約、意図した設計）が、一次透過率は統計誤差内で
    同等であること。"""
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    n = 200_000
    r1 = run_batch(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=99, n_chunks=1)
    r4 = run_batch(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=99, n_chunks=4)
    assert not np.array_equal(r1.n_scatter, r4.n_scatter)

    p1 = np.sum(r1.escaped & (r1.n_scatter == 0)) / n
    p4 = np.sum(r4.escaped & (r4.n_scatter == 0)) / n
    stderr = math.sqrt(p1 * (1 - p1) / n)
    assert abs(p1 - p4) < 5 * stderr


def test_multi_box_last_wins_material_overlap():
    """複数box物体が重なる場合、リスト後方が優先されること
    （`geometry.Geometry.material_at`と同じ規則）——ここでは鉛の薄い箱を
    水の中に完全に埋め込み、鉛が透過率を大きく下げることで確認する。
    """
    tables = bake_scene_materials(["water", "lead", "air"])
    water_only = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    with_lead = water_only + [
        {"center": (0.0, 0.0, 0.0), "size_cm": (0.2, 100.0, 100.0), "material": "lead"}]

    n = 50_000
    geom_water = bake_box_scene(water_only, background="air", tables=tables, bbox_margin_cm=0.01)
    geom_lead = bake_box_scene(with_lead, background="air", tables=tables, bbox_margin_cm=0.01)
    r_water = run_batch(tables, geom_water, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=1, n_chunks=4)
    r_lead = run_batch(tables, geom_lead, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=1, n_chunks=4)

    p_water = np.sum(r_water.escaped & (r_water.n_scatter == 0)) / n
    p_lead = np.sum(r_lead.escaped & (r_lead.n_scatter == 0)) / n
    assert p_lead < p_water * 0.5  # 鉛2mmが埋め込まれた分、明確に透過率が下がる


def _water_scene():
    tables = bake_scene_materials(["water", "air"])
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (10.0, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background="air", tables=tables, bbox_margin_cm=0.01)
    return tables, geom


def _water_bbox():
    return (np.array([-5.01, -50.01, -50.01]), np.array([5.01, 50.01, 50.01]))


def test_tally_variant_matches_reference_variant():
    """`_transport_one_tally`（B-2、区間記録つき）は`_transport_one`（B-1b、
    タリーなし、三層検証済み）と輸送結果（区間を除く全戻り値）が同一seedで
    厳密に一致すること——タリー記録はRNGを消費しない副作用であるべき、という
    設計上の要件をコードに固定するテスト（2つの実装をコピーして作った以上、
    将来どちらかだけ変更されて物理ロジックがズレるのを検出する番犬テスト）。
    """
    tables, geom = _water_scene()
    lo, hi = _water_bbox()
    grid = VoxelGrid.from_bbox(lo, hi, resolution_cm=2.0)
    n = 20_000
    r_plain = run_batch(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=17, n_chunks=4)
    r_tally = run_batch_with_tally(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=17,
                                    grid=grid, n_chunks=4, max_segments_per_history=16)

    assert np.array_equal(r_plain.n_scatter, r_tally.n_scatter)
    assert np.array_equal(r_plain.absorbed, r_tally.absorbed)
    assert np.array_equal(r_plain.escaped, r_tally.escaped)
    assert np.array_equal(r_plain.final_energy, r_tally.final_energy)
    assert np.array_equal(r_plain.energy_deposited, r_tally.energy_deposited)
    assert np.array_equal(r_plain.n_fluorescence, r_tally.n_fluorescence)
    assert grid.kerma_keV.sum() > 0  # タリー自体も何かしら積算されていること


def test_dose_grid_overflow_raises_actionable_error():
    """区間バッファ容量(max_segments_per_history)が明らかに不足していれば、
    タリーを黙って欠落させずValueErrorになること。
    """
    tables, geom = _water_scene()
    lo, hi = _water_bbox()
    grid = VoxelGrid.from_bbox(lo, hi, resolution_cm=2.0)
    with pytest.raises(ValueError):
        run_batch_with_tally(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), 5_000, seed=1,
                              grid=grid, n_chunks=2, max_segments_per_history=1)


def test_dose_grid_kerma_order_of_magnitude_matches_collision_estimator():
    """track-length推定量(グリッド合計カーマ)とcollision推定量
    (`energy_deposited`合計)は異なる推定量だが、水（高Z材料ではない）では
    近い値になるはず——極端な桁違い(単位換算バグ等)を検出する粗いチェック。
    厳密な統計的一致は`docs/speedup_baseline/kernel_dose_grid_crosscheck.py`
    （参照実装とのクロスチェック）で担保する。
    """
    tables, geom = _water_scene()
    lo, hi = _water_bbox()
    grid = VoxelGrid.from_bbox(lo, hi, resolution_cm=2.0)
    n = 100_000
    r = run_batch_with_tally(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=3,
                              grid=grid, n_chunks=4, max_segments_per_history=16)
    total_kerma = grid.kerma_keV.sum()
    total_collision = r.energy_deposited.sum()
    assert total_kerma > 0
    assert 0.5 < total_kerma / total_collision < 2.0


def test_run_dose_grid_batches_without_crashing():
    """`run_dose_grid`のbatch_size分割ラッパーが正しい件数を返し、
    グリッドへタリーが積算されること。"""
    tables, geom = _water_scene()
    lo, hi = _water_bbox()
    grid = VoxelGrid.from_bbox(lo, hi, resolution_cm=2.0)
    n = 30_000
    r = run_dose_grid(tables, geom, 60.0, (-5.01, 0.0, 0.0), (1.0, 0.0, 0.0), n, seed=9, grid=grid,
                       batch_size=10_000, n_chunks=4, max_segments_per_history=16)
    assert len(r.absorbed) == n
    assert (r.absorbed | r.escaped).all()
    assert grid.kerma_keV.sum() > 0
