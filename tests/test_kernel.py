"""chatcarlo/kernel.py（Phase B-1a: per-historyスカラーカーネルのプローブ）のテスト。

B-1aの意図的な簡略化（water単色・box1個・背景真空・蛍光無効、kernel.pyの
モジュールdocstring参照）の範囲内での物理的正しさを検証する。`transport_photons`
参照実装との統計的クロスチェック（Phase Bの検証戦略layer 1）はB-1bで実施する。
"""
import math

import numpy as np
import pytest

from chatcarlo.kernel import bake_material_tables, run_water_slab_probe
from chatcarlo.materials import linear_mu

SCENARIOS = {
    "water20kev": (1.5, 20.0),
    "water60_free": (10.0, 60.0),
    "water150kev": (10.0, 150.0),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_uncollided_fraction_matches_beer_lambert(scenario):
    """一次透過率（無散乱透過率）がBeer-Lambert解析解と統計誤差内で一致すること。"""
    thickness_cm, energy_kev = SCENARIOS[scenario]
    n = 500_000
    mu = float(linear_mu("water", np.array([energy_kev]))[0])
    expected = math.exp(-mu * thickness_cm)
    stderr = math.sqrt(expected * (1 - expected) / n)

    r = run_water_slab_probe(thickness_cm=thickness_cm, energy_kev=energy_kev,
                              n_histories=n, seed=11, warmup_histories=2000)
    assert abs(r.uncollided_frac - expected) < 5 * stderr


def test_energy_conservation_per_history():
    """吸収履歴はenergy_deposited==入射エネルギー、脱出履歴はenergy_deposited+
    final_energy==入射エネルギーが浮動小数点誤差の範囲で厳密に成り立つこと
    （コンプトン損失の逐次積算・光電吸収の全量計上にバグがないことの直接証拠）。
    """
    from chatcarlo.kernel import _run_batch_scalar

    tables = bake_material_tables("water")
    hx, hy, hz = 5.0, 50.0, 50.0
    margin = 0.01
    whx, why, whz = hx + margin, hy + margin, hz + margin
    energy0 = 60.0
    args = (energy0, -hx - margin, 0.0, 0.0, 1.0, 0.0, 0.0, hx, hy, hz, whx, why, whz,
            tables.n_elem, tables.zs, tables.fracs, tables.log_e_grid, tables.step,
            tables.density_g_cm3, tables.photo_tab, tables.compt_tab, tables.rayl_tab,
            tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a)
    n = 50_000
    n_scatter, absorbed, escaped, final_energy, energy_deposited = _run_batch_scalar(n, 5, *args)

    assert (absorbed | escaped).all()
    assert np.allclose(energy_deposited[absorbed], energy0, atol=1e-9)
    assert np.allclose((energy_deposited + final_energy)[escaped], energy0, atol=1e-9)


def test_bake_material_tables_rejects_non_uniform_grid_element():
    """B-1aは元素間でエネルギー格子が共有される軽元素材料のみ対応
    （重元素を含む材料はStep 1と同じ理由で非対応）——leadはこの前提を満たさない
    ため明示的にValueErrorになることを確認する。
    """
    with pytest.raises(ValueError):
        bake_material_tables("lead")
