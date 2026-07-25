"""線量マップ断面図（chatcarlo/plotting.py）のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from chatcarlo.plotting import extent_cm, max_voxel_index, plot_dose_npz


def _write_npz(path, dose, origin_cm=(0.0, 0.0, 0.0), voxel_size_cm=1.0):
    shape = np.array(dose.shape)
    np.savez(path, dose_per_history_Gy=dose,
              h10_per_history_pSv=dose * 1000.0,
              origin_cm=np.array(origin_cm, dtype=float),
              voxel_size_cm=voxel_size_cm, shape=shape)


def test_max_voxel_index_matches_argmax_unravel():
    """断面選択ロジック: max_voxel_indexはargmax→unravel_indexと一致する純関数。"""
    rng = np.random.default_rng(0)
    data = rng.random((4, 5, 6))
    expected = np.unravel_index(np.argmax(data), data.shape)
    assert max_voxel_index(data) == expected


def test_extent_cm_matches_origin_and_voxel_size():
    """extent計算: origin/voxel_sizeから求めた物理座標範囲の数値照合。"""
    origin = np.array([-10.0, -20.0, 5.0])
    shape = (4, 5, 6)  # nx, ny, nz
    voxel_size = 2.0
    # axis="z"の断面はx,y方向の範囲になる
    ext = extent_cm(origin, voxel_size, shape, axis="z")
    assert ext == [origin[0], origin[0] + shape[0] * voxel_size,
                    origin[1], origin[1] + shape[1] * voxel_size]
    # axis="x"の断面はy,z方向の範囲になる
    ext_x = extent_cm(origin, voxel_size, shape, axis="x")
    assert ext_x == [origin[1], origin[1] + shape[1] * voxel_size,
                      origin[2], origin[2] + shape[2] * voxel_size]


def test_plot_dose_npz_writes_nonempty_png(tmp_path):
    """8x8x8で1ボクセルだけ非ゼロのnpzからPNGが生成され、非ゼロサイズであること。"""
    dose = np.zeros((8, 8, 8))
    dose[3, 4, 5] = 1.23e-6
    npz_path = tmp_path / "dose.npz"
    _write_npz(npz_path, dose)

    out_path = tmp_path / "maps.png"
    ok = plot_dose_npz(str(npz_path), str(out_path))
    assert ok is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_dose_npz_single_axis(tmp_path):
    """--axis相当（1断面のみ）でもPNGが生成されること。"""
    dose = np.zeros((6, 6, 6))
    dose[2, 3, 4] = 5e-7
    npz_path = tmp_path / "dose.npz"
    _write_npz(npz_path, dose)

    out_path = tmp_path / "maps_axis.png"
    ok = plot_dose_npz(str(npz_path), str(out_path), axis="z", pos_cm=4.5)
    assert ok is True
    assert out_path.stat().st_size > 0


def test_plot_dose_npz_all_zero_returns_false_without_crashing(tmp_path):
    """全ゼロデータでは例外にならず、Falseを返してファイルも書き出さない。"""
    dose = np.zeros((5, 5, 5))
    npz_path = tmp_path / "dose_zero.npz"
    _write_npz(npz_path, dose)

    out_path = tmp_path / "maps_zero.png"
    ok = plot_dose_npz(str(npz_path), str(out_path))
    assert ok is False
    assert not out_path.exists()


def test_plot_dose_npz_prefers_calibrated_keys(tmp_path):
    """dose_Gyキーがあれば per_history より優先して使う（校正済みラベルになる）。"""
    dose_per_history = np.zeros((4, 4, 4))
    dose_per_history[1, 1, 1] = 1e-8
    dose_abs = dose_per_history * 1e6  # 校正済み絶対値（適当な倍率）

    np.savez(tmp_path / "dose_cal.npz",
              dose_per_history_Gy=dose_per_history,
              h10_per_history_pSv=dose_per_history * 1000.0,
              dose_Gy=dose_abs,
              h10_pSv=dose_abs * 1000.0,
              origin_cm=np.zeros(3), voxel_size_cm=1.0, shape=np.array(dose_per_history.shape))

    out_path = tmp_path / "maps_cal.png"
    ok = plot_dose_npz(str(tmp_path / "dose_cal.npz"), str(out_path))
    assert ok is True
    assert out_path.stat().st_size > 0


# --- Phase 3 (docs/plan_statistical_uncertainty.md): relerr-dose / relerr-h10 ---


def _write_npz_with_uncertainty(path, dose, rel_err_dose, rel_err_h10, n_batches_hit,
                                 origin_cm=(0.0, 0.0, 0.0), voxel_size_cm=1.0):
    shape = np.array(dose.shape)
    np.savez(path, dose_per_history_Gy=dose,
              h10_per_history_pSv=dose * 1000.0,
              rel_err_dose=rel_err_dose, rel_err_h10=rel_err_h10,
              n_batches_hit=n_batches_hit, n_batches=np.array(20),
              origin_cm=np.array(origin_cm, dtype=float),
              voxel_size_cm=voxel_size_cm, shape=shape)


def test_plot_relerr_dose_writes_png(tmp_path):
    dose = np.zeros((6, 6, 6))
    dose[2, 3, 4] = 1e-8
    rel_err_dose = np.full((6, 6, 6), np.nan)
    rel_err_dose[2, 3, 4] = 0.05
    rel_err_h10 = rel_err_dose.copy()
    n_hit = np.zeros((6, 6, 6), dtype=np.int32)
    n_hit[2, 3, 4] = 20
    npz_path = tmp_path / "dose.npz"
    _write_npz_with_uncertainty(npz_path, dose, rel_err_dose, rel_err_h10, n_hit)

    out_path = tmp_path / "relerr.png"
    ok = plot_dose_npz(str(npz_path), str(out_path), quantity="relerr-dose")
    assert ok is True
    assert out_path.stat().st_size > 0


def test_plot_relerr_h10_writes_png(tmp_path):
    dose = np.zeros((6, 6, 6))
    dose[1, 1, 1] = 1e-8
    rel_err_dose = np.full((6, 6, 6), np.nan)
    rel_err_h10 = rel_err_dose.copy()
    rel_err_h10[1, 1, 1] = 0.12
    n_hit = np.zeros((6, 6, 6), dtype=np.int32)
    n_hit[1, 1, 1] = 20
    npz_path = tmp_path / "dose.npz"
    _write_npz_with_uncertainty(npz_path, dose, rel_err_dose, rel_err_h10, n_hit)

    out_path = tmp_path / "relerr_h10.png"
    ok = plot_dose_npz(str(npz_path), str(out_path), quantity="relerr-h10")
    assert ok is True
    assert out_path.stat().st_size > 0


def test_plot_relerr_all_nan_returns_false(tmp_path):
    """統計未到達（R全nan）ボクセルのみのグリッドではクラッシュせずFalseを返す。"""
    dose = np.zeros((5, 5, 5))
    rel_err = np.full((5, 5, 5), np.nan)
    n_hit = np.zeros((5, 5, 5), dtype=np.int32)
    npz_path = tmp_path / "dose_allnan.npz"
    _write_npz_with_uncertainty(npz_path, dose, rel_err, rel_err, n_hit)

    out_path = tmp_path / "relerr_allnan.png"
    ok = plot_dose_npz(str(npz_path), str(out_path), quantity="relerr-dose")
    assert ok is False
    assert not out_path.exists()


def test_plot_relerr_missing_key_raises_friendly_error(tmp_path):
    """--no-uncertaintyで生成した（統計キーの無い）旧形式.npzへの親切なエラー。"""
    dose = np.zeros((4, 4, 4))
    dose[1, 1, 1] = 1e-8
    npz_path = tmp_path / "dose_no_uncertainty.npz"
    _write_npz(npz_path, dose)  # rel_err_dose等を含まない既存フォーマット

    with pytest.raises(ValueError, match="rel_err_dose"):
        plot_dose_npz(str(npz_path), str(tmp_path / "out.png"), quantity="relerr-dose")


def test_plot_relerr_missing_n_batches_hit_raises_friendly_error(tmp_path):
    """rel_err_doseキーはあるがn_batches_hitが無い（想定外に壊れた）npzへのエラー。"""
    dose = np.zeros((4, 4, 4))
    dose[1, 1, 1] = 1e-8
    rel_err = np.full((4, 4, 4), np.nan)
    rel_err[1, 1, 1] = 0.03
    npz_path = tmp_path / "dose_partial.npz"
    np.savez(npz_path, dose_per_history_Gy=dose, h10_per_history_pSv=dose * 1000.0,
              rel_err_dose=rel_err, rel_err_h10=rel_err,
              origin_cm=np.zeros(3), voxel_size_cm=1.0, shape=np.array(dose.shape))

    with pytest.raises(ValueError, match="n_batches_hit"):
        plot_dose_npz(str(npz_path), str(tmp_path / "out.png"), quantity="relerr-dose")
