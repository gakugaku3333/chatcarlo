"""Phase 1 kernel CLI adapter contracts."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from chatcarlo.kernel import bake_box_scene, bake_scene_materials, run_batch_origins
from chatcarlo.scene import validate_scene
from chatcarlo.source import sample_source_photons
from chatcarlo.transport import (_kernel_material_names, _run_kernel_batches,
                                 run_transport)


def _scene(*, fluorescence=True):
    return validate_scene({
        "source": {
            "type": "xray_tube",
            "spectrum": [{"energy_keV": 60.0, "weight": 1.0}],
            "position": [0, 0, 0], "direction": [1, 0, 0],
            "field": {"shape": "parallel", "size_cm": [4, 4]},
        },
        "geometry": [{"name": "water", "shape": "box", "material": "water",
                      "center": [5, 0, 0], "size_cm": [10, 20, 20]}],
        "physics": {"fluorescence": fluorescence},
    })


def test_kernel_rejects_before_source_sampling(monkeypatch):
    scene = _scene()
    scene.raw["source"]["field"]["shape"] = "rect"
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must not sample")

    monkeypatch.setattr("chatcarlo.transport.sample_source_photons", forbidden)
    with pytest.raises(ValueError, match="parallel"):
        run_transport(scene, n_histories=10, engine="kernel", track_uncertainty=False)
    assert not called


def test_kernel_material_order_includes_background():
    scene = _scene()
    from chatcarlo.geometry import Geometry
    assert _kernel_material_names(scene, Geometry(scene.raw["geometry"])) == ["water", "air"]


def test_origin_kernel_is_reproducible_and_chunk_safe():
    tables = bake_scene_materials(["water", "air"])
    geom = bake_box_scene([{"center": (5, 0, 0), "size_cm": (10, 20, 20), "material": "water"}],
                          "air", tables)
    origins = np.zeros((200, 3))
    args = (tables, geom, 60.0, origins, (1.0, 0.0, 0.0), 19)
    a = run_batch_origins(*args, n_chunks=4)
    b = run_batch_origins(*args, n_chunks=4)
    assert np.array_equal(a.energy_deposited_by_material, b.energy_deposited_by_material)
    assert np.array_equal(a.n_scatter, b.n_scatter)


@pytest.mark.parametrize("fluorescence", [True, False])
def test_kernel_dispatch_runs_with_fluorescence_settings(fluorescence):
    result = run_transport(_scene(fluorescence=fluorescence), n_histories=100,
                           seed=4, batch_size=50, engine="kernel", kernel_chunks=1,
                           track_uncertainty=False)
    assert result.n_batches == 0
    assert result.energy_deposited_sem_MeV == {}


def _lead_scene(*, source_position=(0, 0, 0), fluorescence=True):
    """Overlapping boxes exercise material order, fluorescence, and field origins."""
    scene = _scene(fluorescence=fluorescence)
    scene.raw["source"]["position"] = list(source_position)
    scene.raw["source"]["spectrum"][0]["energy_keV"] = 100.0
    scene.raw["geometry"] = [
        {"name": "water", "shape": "box", "material": "water",
         "center": [5, 0, 0], "size_cm": [10, 20, 20]},
        {"name": "lead_overlap", "shape": "box", "material": "lead",
         "center": [6, 0, 0], "size_cm": [2, 20, 20]},
    ]
    return validate_scene(scene.raw)


# 6シード/4σ判定はn_historiesが小さいと検出力が壊滅的に落ちる。当初値 n=2,000 では
# 検出できる最小バイアスが水8.9%・鉛5.9%しかなかったため n=50,000 に引き上げた
# （検出限界0.7〜1.2%、このファイル全体で約20秒）。
#
# **このテストが検出できる/できないバグの範囲**（Claudeのミューテーション検証で
# 実測、2026-08-01。この定数を変えるときは「4σを形式的に満たすこと」ではなく
# 「狙ったバイアスを実際に検出できること」を同じ方法で再確認すること）:
# - 検出できる（配線バグの類）: per-chunk集約を`sum(axis=0)`から先頭チャンクのみに
#   壊す変異、kernel経路だけfluorescence_enabledを握り潰す変異——いずれも即座に失敗する。
# - **検出できない**: コンプトン沈着を1%過小にする変異は素通りする。この変異の
#   総沈着エネルギーへの実効果量は水で0.83%（鉛は光電優勢のためほぼ0%）で、
#   n=50,000の検出限界1.2%をわずかに下回るため。捕捉には n=1e6 規模（4σ≈0.29%）が
#   必要で、単体テストの実行時間では現実的でない。**この粒度の系統差を確認したい
#   ときは、このテストではなくEGS5相互検証（docs/egs5_crosscheck/）で見ること。**
_CROSSCHECK_HISTORIES = 50_000


def _assert_six_seed_agreement(numpy_values, kernel_values):
    """B-2 acceptance rule: independent six seeds, combined 4 sigma."""
    a = np.asarray(numpy_values, dtype=float)
    b = np.asarray(kernel_values, dtype=float)
    combined_sem = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    # A genuinely zero-variance quantity must be exactly equal.
    assert abs(a.mean() - b.mean()) <= 4 * combined_sem + 1e-12


@pytest.mark.parametrize("fluorescence", [True, False])
def test_kernel_numpy_six_seed_crosscheck_energy_grid_and_fluorescence(fluorescence):
    """Full adapter check with high-Z fluorescence and dose-grid tallies."""
    scene = _lead_scene(fluorescence=fluorescence)
    n = _CROSSCHECK_HISTORIES
    numpy_runs = [run_transport(scene, n_histories=n, seed=s, batch_size=n // 3,
                                dose_grid=True, grid_resolution_cm=4, engine="numpy",
                                track_uncertainty=False) for s in range(11, 17)]
    kernel_runs = [run_transport(scene, n_histories=n, seed=s, batch_size=n // 3,
                                 dose_grid=True, grid_resolution_cm=4, engine="kernel",
                                 kernel_chunks=1, track_uncertainty=False) for s in range(11, 17)]
    for material in ("water", "lead", "air"):
        _assert_six_seed_agreement(
            [r.energy_deposited_MeV.get(material, 0.0) for r in numpy_runs],
            [r.energy_deposited_MeV.get(material, 0.0) for r in kernel_runs])
    for attr in ("kerma_keV", "h10_track_pSv_cm3"):
        _assert_six_seed_agreement(
            [getattr(r.grid, attr).sum() for r in numpy_runs],
            [getattr(r.grid, attr).sum() for r in kernel_runs])
    _assert_six_seed_agreement([r.n_fluorescence for r in numpy_runs],
                                [r.n_fluorescence for r in kernel_runs])


def test_kernel_source_adapter_matches_direct_sample_with_same_rng(monkeypatch):
    """The adapter delegates its entrance-plane sampling to source.py unchanged."""
    scene = _scene()
    captured = []
    direct_rng = np.random.default_rng(1234)
    expected = sample_source_photons(scene.raw["source"], 31, direct_rng)

    def capture_source(src, n, rng):
        got = sample_source_photons(src, n, rng)
        captured.append(got)
        return got

    class FakeResult:
        n_scatter = np.zeros(31, dtype=np.int64)
        absorbed = np.zeros(31, dtype=bool)
        escaped = np.ones(31, dtype=bool)
        n_fluorescence = np.zeros(31, dtype=np.int64)
        energy_deposited_by_material = np.zeros(2)

    monkeypatch.setattr("chatcarlo.transport.sample_source_photons", capture_source)
    monkeypatch.setattr("chatcarlo.kernel.run_batch_origins", lambda *a, **k: FakeResult())
    # Seed 1234 is deliberately not used here: derive the adapter's documented source child.
    top_seed = 1234
    source_child = np.random.SeedSequence(top_seed).spawn(1)[0].spawn(2)[0]
    expected = sample_source_photons(scene.raw["source"], 31, np.random.default_rng(source_child))
    _run_kernel_batches(scene, 31, top_seed, 100, None, 1, True, 16)
    assert len(captured) == 1
    for got, want in zip(captured[0], expected):
        assert np.array_equal(got, want)


def test_kernel_rng_spawn_tree_and_source_consumption_are_independent(monkeypatch):
    scene = _scene()

    def seeds_after_source_consumption(extra_draws):
        seen = []

        def source(src, n, rng):
            rng.random(extra_draws)
            return (np.zeros((n, 3)), np.tile([[1., 0., 0.]], (n, 1)), np.full(n, 60.))

        def kernel(*args, **kwargs):
            seen.append(args[5])  # kernel integer seed
            n = len(args[3])
            return SimpleNamespace(n_scatter=np.zeros(n, dtype=np.int64),
                                   absorbed=np.zeros(n, dtype=bool), escaped=np.ones(n, dtype=bool),
                                   n_fluorescence=np.zeros(n, dtype=np.int64),
                                   energy_deposited_by_material=np.zeros(2))

        monkeypatch.setattr("chatcarlo.transport.sample_source_photons", source)
        monkeypatch.setattr("chatcarlo.kernel.run_batch_origins", kernel)
        _run_kernel_batches(scene, 9, 31415, 4, None, 1, True, 16)
        return seen

    actual = seeds_after_source_consumption(0)
    assert actual == seeds_after_source_consumption(97)
    expected = [int(batch.spawn(2)[1].generate_state(1)[0])
                for batch in np.random.SeedSequence(31415).spawn(3)]
    assert actual == expected


def test_kernel_chunks_statistically_agree_and_are_bit_reproducible():
    scene = _lead_scene()
    n = _CROSSCHECK_HISTORIES
    one, four = [], []
    for seed in range(31, 37):
        one.append(run_transport(scene, n_histories=n, seed=seed, batch_size=n,
                                 engine="kernel", kernel_chunks=1, track_uncertainty=False))
        four.append(run_transport(scene, n_histories=n, seed=seed, batch_size=n,
                                  engine="kernel", kernel_chunks=4, track_uncertainty=False))
    for material in ("water", "lead", "air"):
        _assert_six_seed_agreement([r.energy_deposited_MeV.get(material, 0.) for r in one],
                                    [r.energy_deposited_MeV.get(material, 0.) for r in four])
    a = run_transport(scene, n_histories=1_500, seed=101, batch_size=700,
                      engine="kernel", kernel_chunks=4, track_uncertainty=False)
    b = run_transport(scene, n_histories=1_500, seed=101, batch_size=700,
                      engine="kernel", kernel_chunks=4, track_uncertainty=False)
    assert a.energy_deposited_MeV == b.energy_deposited_MeV
    assert a.n_fluorescence == b.n_fluorescence


def test_kernel_boundary_at_box_entrance_has_no_systematic_cross_engine_bias():
    scene = _lead_scene(source_position=(0, 0, 0), fluorescence=False)
    # The first water face is x=0, exactly the parallel field's source plane.
    n = _CROSSCHECK_HISTORIES
    numpy_values, kernel_values = [], []
    for seed in range(51, 57):
        numpy_values.append(run_transport(scene, n_histories=n, seed=seed, batch_size=n // 3,
                                          engine="numpy", track_uncertainty=False))
        kernel_values.append(run_transport(scene, n_histories=n, seed=seed, batch_size=n // 3,
                                           engine="kernel", kernel_chunks=1,
                                           track_uncertainty=False))
    _assert_six_seed_agreement([sum(r.energy_deposited_MeV.values()) for r in numpy_values],
                                [sum(r.energy_deposited_MeV.values()) for r in kernel_values])


def test_numpy_engine_default_path_is_bit_identical_to_explicit_numpy():
    scene = _lead_scene()
    implicit = run_transport(scene, n_histories=1_000, seed=73, batch_size=400,
                             track_uncertainty=False)
    explicit = run_transport(scene, n_histories=1_000, seed=73, batch_size=400,
                             engine="numpy", track_uncertainty=False)
    assert implicit == explicit


def _cli_args(**overrides):
    values = dict(scene="unused.yaml", workers=1, engine="kernel", kernel_chunks=0,
                  kernel_max_segments_per_history=16, no_uncertainty=True, n_histories=10,
                  batch_size=200_000, dose_grid=False, resolution=5.0, seed=1,
                  dose_out=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_cli_kernel_workers_zero_is_rejected_after_auto_expansion(monkeypatch, capsys):
    from chatcarlo import __main__ as cli
    monkeypatch.setattr("chatcarlo.scene.load_scene", lambda _: _scene())
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert cli.cmd_run(_cli_args(workers=0)) == 1
    assert "workers=1" in capsys.readouterr().err


def test_cli_kernel_segment_overflow_names_adjustment_flag(monkeypatch, capsys):
    from chatcarlo import __main__ as cli
    monkeypatch.setattr("chatcarlo.scene.load_scene", lambda _: _scene())

    def overflow(*args, **kwargs):
        raise ValueError("タリー用の区間バッファが不足しました")

    monkeypatch.setattr("chatcarlo.transport.run_transport", overflow)
    assert cli.cmd_run(_cli_args(dose_grid=True)) == 1
    assert "--kernel-max-segments-per-history" in capsys.readouterr().err


def test_cli_kernel_last_batch_effective_chunks(monkeypatch, capsys):
    from chatcarlo import __main__ as cli
    monkeypatch.setattr("chatcarlo.scene.load_scene", lambda _: _scene())
    monkeypatch.setattr("chatcarlo.transport.run_transport", lambda *a, **k: SimpleNamespace(
        n_histories=250_000, fraction_absorbed=0., fraction_escaped=1., mean_scatter_events=0.,
        n_fluorescence=0, n_batches=0, energy_deposited_MeV={}, n_photons_real=None))
    assert cli.cmd_run(_cli_args(n_histories=250_000, batch_size=200_000, kernel_chunks=100_000)) == 0
    assert "最終バッチ実効値=50000" in capsys.readouterr().out
