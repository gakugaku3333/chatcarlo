"""線量マップの断面図 — chatcarlo run --dose-out の .npz を可視化する。

輸送・タリーとは独立な後処理。「結果を視覚的に確認する」という
human-in-the-loopワークフローの3つ目の関門（vive-checkスキル）向け。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_LABELS = {"x": ("Y", "Z"), "y": ("X", "Z"), "z": ("X", "Y")}
_DEFAULT_SLICE_ORDER = ("z", "y", "x")


_RELERR_KEYS = {"relerr-dose": "rel_err_dose", "relerr-h10": "rel_err_h10"}
_RELERR_REF_QUANTITY = {"relerr-dose": "dose", "relerr-h10": "h10"}


def select_quantity(npz, quantity: str):
    """.npzから量(dose|h10|relerr-dose|relerr-h10)の配列と単位ラベルを選ぶ。

    dose/h10は校正済みキーがあれば優先する。relerr-*は統計不確かさの相対誤差R
    マップ（docs/plan_statistical_uncertainty.md Phase 3）——`chatcarlo run
    --dose-grid --dose-out`が`--no-uncertainty`無しで書き出した.npzにのみ入っている。

    ラベルは英語で書く（matplotlib PNG出力はCJKフォントが無い環境だと文字化け
    するため。既存のcmd_xsも同じ理由で図中ラベルは英語 — CLIメッセージ側は
    日本語のまま）。
    """
    if quantity == "dose":
        if "dose_Gy" in npz.files:
            return npz["dose_Gy"], "Gy (calibrated)"
        return npz["dose_per_history_Gy"], "Gy/history"
    if quantity == "h10":
        if "h10_pSv" in npz.files:
            return npz["h10_pSv"], "pSv (calibrated)"
        return npz["h10_per_history_pSv"], "pSv/history"
    if quantity in _RELERR_KEYS:
        key = _RELERR_KEYS[quantity]
        if key not in npz.files:
            raise ValueError(
                f"{key!r} が.npzにありません（quantity={quantity!r}）。"
                "統計不確かさキーが無い.npzです — 旧バージョンで生成したか、"
                "'chatcarlo run --dose-grid --dose-out ... --no-uncertainty' で"
                "生成した.npzの可能性があります。--no-uncertaintyを外して"
                "再生成してください。")
        return npz[key], "relative error R (dimensionless)"
    raise ValueError(f"未知のquantity: {quantity!r}（'dose'/'h10'/'relerr-dose'/'relerr-h10'）")


def max_voxel_index(data: np.ndarray) -> tuple[int, int, int]:
    """dataの最大値を持つボクセルの(ix,iy,iz)。既定3断面の中心に使う純関数。"""
    return np.unravel_index(int(np.argmax(data)), data.shape)


def extent_cm(origin_cm: np.ndarray, voxel_size_cm: float, shape, axis: str) -> list[float]:
    """axisを法線とする断面の物理座標範囲 [left,right,bottom,top]（imshowのextent形式）。"""
    ax = _AXIS_INDEX[axis]
    u, v = [k for k in range(3) if k != ax]
    lo = np.asarray(origin_cm, dtype=float)
    hi = lo + np.asarray(shape, dtype=float) * voxel_size_cm
    return [lo[u], hi[u], lo[v], hi[v]]


def _plane_at(data: np.ndarray, axis: str, index: int) -> np.ndarray:
    ax = _AXIS_INDEX[axis]
    sl = [slice(None)] * 3
    sl[ax] = index
    # data は (nx,ny,nz)。imshowは(row,col)=(縦軸,横軸)で描くため転置する。
    return data[tuple(sl)].T


def _draw_slice(ax_plt, data, axis, index, origin_cm, voxel_size_cm, shape,
                 unit_label, vmin, vmax, geometry=None):
    plane = _plane_at(data, axis, index)
    masked = np.ma.masked_where(plane <= 0, plane)
    cmap = matplotlib.colormaps["inferno"].with_extremes(bad="#1e1e2e")
    im = ax_plt.imshow(masked, origin="lower", cmap=cmap,
                        norm=LogNorm(vmin=vmin, vmax=vmax),
                        extent=extent_cm(origin_cm, voxel_size_cm, shape, axis),
                        aspect="equal")
    pos_cm = origin_cm[_AXIS_INDEX[axis]] + (index + 0.5) * voxel_size_cm
    ax_plt.set_title(f"{axis} = {pos_cm:.1f} cm")
    xl, yl = _AXIS_LABELS[axis]
    ax_plt.set_xlabel(f"{xl} [cm]")
    ax_plt.set_ylabel(f"{yl} [cm]")
    plt.colorbar(im, ax=ax_plt, label=unit_label)

    if geometry is not None:
        _overlay_geometry_contour(ax_plt, geometry, axis, pos_cm, origin_cm, voxel_size_cm, shape)


_RELERR_CLIP_MAX = 0.5


def _draw_relerr_slice(ax_plt, data, n_hit, axis, index, origin_cm, voxel_size_cm, shape,
                        geometry=None):
    """相対誤差Rの断面図。線形0〜0.5クリップ・inferno以外の配色（cividis）。

    寄与バッチ0（統計未到達）のボクセルは「Rが小さく出る」のではなく「そもそも
    推定不能」なので、Rの値ではなくグレーで別マスクする
    （docs/plan_statistical_uncertainty.md 設計判断7の運用強制）。
    """
    plane = _plane_at(data, axis, index)
    hit_plane = _plane_at(n_hit, axis, index)
    clipped = np.clip(plane, 0.0, _RELERR_CLIP_MAX)
    masked = np.ma.masked_where(np.isnan(plane) | (hit_plane == 0), clipped)
    cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#555555")
    im = ax_plt.imshow(masked, origin="lower", cmap=cmap, vmin=0.0, vmax=_RELERR_CLIP_MAX,
                        extent=extent_cm(origin_cm, voxel_size_cm, shape, axis),
                        aspect="equal")
    pos_cm = origin_cm[_AXIS_INDEX[axis]] + (index + 0.5) * voxel_size_cm
    ax_plt.set_title(f"{axis} = {pos_cm:.1f} cm")
    xl, yl = _AXIS_LABELS[axis]
    ax_plt.set_xlabel(f"{xl} [cm]")
    ax_plt.set_ylabel(f"{yl} [cm]")
    plt.colorbar(im, ax=ax_plt,
                 label=f"relative error R (clipped at {_RELERR_CLIP_MAX:g}; grey = unreached)")

    if geometry is not None:
        _overlay_geometry_contour(ax_plt, geometry, axis, pos_cm, origin_cm, voxel_size_cm, shape)


def _overlay_geometry_contour(ax_plt, geometry, axis, pos_cm, origin_cm, voxel_size_cm, shape):
    """断面グリッドの各ピクセル中心の材料を整数IDに変換し、境界を輪郭線で描く。"""
    ax_i = _AXIS_INDEX[axis]
    u, v = [k for k in range(3) if k != ax_i]
    nu, nv = shape[u], shape[v]
    us = origin_cm[u] + (np.arange(nu) + 0.5) * voxel_size_cm
    vs = origin_cm[v] + (np.arange(nv) + 0.5) * voxel_size_cm
    uu, vv = np.meshgrid(us, vs, indexing="xy")  # shape (nv, nu)

    points = np.zeros((uu.size, 3))
    points[:, ax_i] = pos_cm
    points[:, u] = uu.ravel()
    points[:, v] = vv.ravel()
    mat = geometry.material_at(points)
    _, mat_ids = np.unique(mat, return_inverse=True)
    mat_ids = mat_ids.reshape(uu.shape).astype(float)

    if mat_ids.max() <= 0:
        return  # 断面内が単一材料のみなら輪郭は引けない
    levels = np.arange(mat_ids.max()) + 0.5
    ax_plt.contour(us, vs, mat_ids, levels=levels, colors="white", linewidths=0.7, alpha=0.8)


def plot_dose_npz(npz_path: str, out_path: str, quantity: str = "dose",
                   axis: str | None = None, pos_cm: float | None = None,
                   scene_path: str | None = None) -> bool:
    """線量/H*(10)/相対誤差マップをPNGに書き出す。

    axis未指定時は最大値ボクセルを通る3断面（既定）。axis指定時はその1断面のみ
    （posも指定すればその位置、未指定なら最大値ボクセルの位置）。
    quantityが'relerr-dose'/'relerr-h10'の場合は対応する量（dose/h10）の
    最大値ボクセルを中心に断面を取る（Rそのものの最大値は「まだ大きな寄与を
    引いていない」ノイズであることが多く、中心として無意味なため）。
    全ボクセルが無効（dose/h10は全ゼロ、relerr-*は全nan）の場合は何も書き出さず
    Falseを返す（クラッシュさせない）。
    """
    is_relerr = quantity in _RELERR_KEYS
    n_hit = None
    with np.load(npz_path) as npz:
        data, unit_label = select_quantity(npz, quantity)
        origin_cm = np.asarray(npz["origin_cm"], dtype=float)
        voxel_size_cm = float(npz["voxel_size_cm"])
        shape = tuple(int(s) for s in npz["shape"])
        if is_relerr:
            if "n_batches_hit" not in npz.files:
                raise ValueError(
                    "'n_batches_hit' が.npzにありません。統計未到達ボクセルの"
                    "マスクに必須のキーです — --no-uncertainty無しで再生成してください。")
            n_hit = np.asarray(npz["n_batches_hit"])
            ref_data, _ = select_quantity(npz, _RELERR_REF_QUANTITY[quantity])
            center_idx = dict(zip(("x", "y", "z"), max_voxel_index(ref_data)))

    if is_relerr:
        if not np.any(~np.isnan(data)):
            return False
    else:
        if not np.any(data > 0):
            return False
        vmax = float(data.max())
        vmin = max(float(data[data > 0].min()), vmax * 1e-6)
        center_idx = dict(zip(("x", "y", "z"), max_voxel_index(data)))

    geometry = None
    if scene_path:
        from .geometry import Geometry
        from .scene import load_scene
        scene = load_scene(scene_path)
        geometry = Geometry(scene.raw["geometry"])

    def _draw(ax_plt, ax_name, idx):
        if is_relerr:
            _draw_relerr_slice(ax_plt, data, n_hit, ax_name, idx, origin_cm, voxel_size_cm,
                                shape, geometry)
        else:
            _draw_slice(ax_plt, data, ax_name, idx, origin_cm, voxel_size_cm, shape,
                        unit_label, vmin, vmax, geometry)

    if axis is not None:
        idx = center_idx[axis]
        if pos_cm is not None:
            ax_i = _AXIS_INDEX[axis]
            idx = int(np.clip(round((pos_cm - origin_cm[ax_i]) / voxel_size_cm - 0.5),
                               0, shape[ax_i] - 1))
        fig, ax_plt = plt.subplots(figsize=(6, 5))
        _draw(ax_plt, axis, idx)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax_plt, ax_name in zip(axes, _DEFAULT_SLICE_ORDER):
            _draw(ax_plt, ax_name, center_idx[ax_name])

    fig.suptitle(f"ChatCarlo — {unit_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True
