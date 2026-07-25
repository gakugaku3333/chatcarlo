"""CLI出力層（Phase 3: docs/plan_statistical_uncertainty.md）のスモークテスト。

`chatcarlo run --dose-grid`がRと寄与バッチ数を並記すること、`--no-uncertainty`で
それらが消えること、`.npz`に統計キーが入る/入らないこと、`chatcarlo plot
--quantity relerr-dose`が実際に.npzを読んで図を出すことを、cmd_run/cmd_plotを
直接呼び出すことで確認する（サブプロセスより高速で、標準出力・戻り値の両方を
直接検証できる）。
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from chatcarlo.__main__ import cmd_plot, cmd_run

_SCENE_DICT = {
    "source": {"kvp": 100, "position": [0, -50, 0], "direction": [0, 1, 0],
               "field": {"size_cm": [30, 30], "sid_cm": 100}},
    "geometry": [
        {"name": "target", "shape": "box", "material": "water",
         "center": [0, 0, 0], "size_cm": [20, 20, 20]},
    ],
}


def _write_scene(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(_SCENE_DICT), encoding="utf-8")
    return str(path)


def _run_args(scene_path, tmp_path, *, n_histories, batch_size, no_uncertainty,
              dose_grid=True, seed=1):
    return argparse.Namespace(
        scene=scene_path, n_histories=n_histories, seed=seed,
        dose_grid=dose_grid, resolution=5.0, dose_out=str(tmp_path / "dose.npz"),
        batch_size=batch_size, no_uncertainty=no_uncertainty, workers=1,
    )


def test_cmd_run_reports_r_and_batches_when_batches_sufficient(tmp_path, capsys):
    scene_path = _write_scene(tmp_path)
    # n_histories=20000, batch_size=2000 -> M=10（>=2、Rが出る条件）。
    args = _run_args(scene_path, tmp_path, n_histories=20000.0, batch_size=2000,
                      no_uncertainty=False)
    rc = cmd_run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "最大吸収線量 [Gy/history]:" in out
    assert "(相対誤差 R=" in out
    assert "寄与バッチ" in out
    assert "グリッド統計:" in out
    # 材料別吸収エネルギーにもSEMが付く
    assert "± " in out and "相対" in out

    with np.load(tmp_path / "dose.npz") as npz:
        for key in ("rel_err_dose", "rel_err_h10", "sem_dose_per_history_Gy",
                    "sem_h10_per_history_pSv", "n_batches", "n_batches_hit"):
            assert key in npz.files, f"{key} が.npzに無い"
        assert int(npz["n_batches"]) == 10


def test_cmd_run_no_uncertainty_omits_stats_and_npz_keys(tmp_path, capsys):
    scene_path = _write_scene(tmp_path)
    args = _run_args(scene_path, tmp_path, n_histories=20000.0, batch_size=2000,
                      no_uncertainty=True)
    rc = cmd_run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "--no-uncertainty により無効化されています" in out
    assert "(相対誤差 R=" not in out
    assert "グリッド統計:" not in out

    with np.load(tmp_path / "dose.npz") as npz:
        for key in ("rel_err_dose", "rel_err_h10", "sem_dose_per_history_Gy",
                    "sem_h10_per_history_pSv", "n_batches", "n_batches_hit"):
            assert key not in npz.files, f"--no-uncertainty指定時に{key}が.npzに書かれている"


def test_cmd_run_batch_shortage_message_is_actionable(tmp_path, capsys):
    """既定に近い設定（M=1）では、Rの代わりに実行可能な対処メッセージが出る。"""
    scene_path = _write_scene(tmp_path)
    args = _run_args(scene_path, tmp_path, n_histories=5000.0, batch_size=200_000,
                      no_uncertainty=False, dose_grid=False)
    rc = cmd_run(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "バッチ数が不足しています" in out
    assert "M=1" in out
    # 実際の設定値から計算された提案値が入っている（固定文言でないことの確認）
    assert "batch_size=200,000" in out
    assert "(相対誤差 R=" not in out


def test_cmd_plot_relerr_dose_from_cli_dose_out(tmp_path, capsys):
    """run --dose-grid --dose-out の.npzを、そのままplot --quantity relerr-doseに渡せること。"""
    scene_path = _write_scene(tmp_path)
    run_args = _run_args(scene_path, tmp_path, n_histories=20000.0, batch_size=2000,
                          no_uncertainty=False)
    assert cmd_run(run_args) == 0
    capsys.readouterr()

    out_png = tmp_path / "relerr.png"
    plot_args = argparse.Namespace(npz=str(tmp_path / "dose.npz"), out=str(out_png),
                                    scene=None, quantity="relerr-dose", axis=None, pos=None)
    rc = cmd_plot(plot_args)
    assert rc == 0
    assert out_png.exists()
    assert out_png.stat().st_size > 0
