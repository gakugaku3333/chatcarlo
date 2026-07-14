"""光子輸送カーネル — 解析面トラッキング（Woodcock不要）。

各光子は「次に消費すべき光学的厚み τ = -ln(ξ)」を持ち、材料境界を
跨ぐたびに τ から μ·Δs を差し引いていく。τ が現在の区間内で尽きたら
そこが実際の相互作用点になる。均質な区間ごとに μ が一定なので
Woodcock delta-trackingの仮想衝突は不要（空気の広い空間で無駄がない）。

相互作用種別は光電/コンプトン/レイリーの3種（診断領域で対生成は無視）。
- 光電: 光子消滅、エネルギー全量をその場で局所吸収（電子飛程を無視する
  カーマ近似。README/[[lessons_learned]]参照）
- コンプトン: Klein-Nishina微分断面積からKahn型棄却法でε=E'/Eと散乱角を抽出
- レイリー: 弾性散乱、エネルギー変化なし。角度分布は原子形状因子F(Z,q)込みの
  微分断面積 dσ/dΩ ∝ (1+cos²θ)·F(Z,q)² から棄却法で抽出する（xraylib.FF_Rayl,
  EPDLベース）。化合物・混合物では質量分率×元素別レイリー断面積で
  構成元素をまず抽選してからその元素のF(Z,q)を使う

スペクトルはSpekPyで生成する（タングステン陽極、カサレイ物理モデルの
SpekPy既定値。陽極角は scene.source.anode_angle_deg、既定12度）。
SpekPy未インストール環境ではKramers則＋Al濾過減弱の粗い近似にフォールバック
する。explicit な scene.source.spectrum（[{energy_keV, weight}, ...]）が
あればどちらより優先する。
"""
from __future__ import annotations

import functools
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np

from .dose_coefficients import h_star_10_per_fluence
from .geometry import Geometry
from .materials import (density, linear_mu, mu_en_rho, mu_rho_parts,
                         rayleigh_element_weights, rayleigh_form_factor_table)
from .tally import VoxelGrid, accumulate_track_length

_MEC2_KEV = 511.0
_HC_KEV_ANGSTROM = 12.3984193  # xraylib.MomentTransfと同じ定数（hc）

try:
    import spekpy as _spekpy
    _HAS_SPEKPY = True
except ImportError:
    _spekpy = None
    _HAS_SPEKPY = False


@functools.lru_cache(maxsize=32)
def _spekpy_spectrum(kvp: float, filtration_mm_al: float, anode_angle_deg: float):
    s = _spekpy.Spek(kvp=kvp, th=anode_angle_deg)
    s.filter("Al", filtration_mm_al)
    e_mid, phi = s.get_spectrum(edges=False)
    w = np.clip(np.asarray(phi, dtype=float), 0.0, None)
    return np.asarray(e_mid, dtype=float), w / w.sum()


def _kramers_fallback_spectrum(kvp: float, filtration_mm_al: float, n_bins: int = 60):
    e = np.linspace(5.0, kvp, n_bins + 1)
    e_mid = 0.5 * (e[:-1] + e[1:])
    raw = np.clip(e_mid * (kvp - e_mid), 0, None)  # Kramers則（未濾過、特性X線なし）
    mu_al = linear_mu("aluminum", e_mid)
    atten = np.exp(-mu_al * filtration_mm_al / 10.0)
    w = raw * atten
    return e_mid, w / w.sum()


def _default_spectrum(kvp: float, filtration_mm_al: float, anode_angle_deg: float = 12.0):
    if _HAS_SPEKPY:
        return _spekpy_spectrum(float(kvp), float(filtration_mm_al), float(anode_angle_deg))
    _warnings.warn("spekpy が見つからないためKramers則の粗い近似スペクトルを使用します。"
                    "`pip install spekpy` を推奨します。", stacklevel=2)
    return _kramers_fallback_spectrum(kvp, filtration_mm_al)


def sample_spectrum(src: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    spec = src.get("spectrum")
    if spec:
        e = np.array([s["energy_keV"] for s in spec], dtype=float)
        w = np.array([s["weight"] for s in spec], dtype=float)
        w = w / w.sum()
    else:
        e, w = _default_spectrum(src["kvp"], src.get("filtration_mm_al", 2.5),
                                  src.get("anode_angle_deg", 12.0))
    return e[rng.choice(len(e), size=n, p=w)]


_ROTATION_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
                   "z": np.array([0.0, 0.0, 1.0])}


def _rotate_batch(v: np.ndarray, axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """固定ベクトルvを、軸axis周りに角度angles[i]だけ回転した結果をi行目に返す（Rodrigues）。"""
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    return (v[None, :] * cos_a + np.cross(axis, v)[None, :] * sin_a
            + axis[None, :] * (axis @ v) * (1.0 - cos_a))


def cone_half_angle_rad(fld: dict) -> float:
    """cone照射野の半頂角[rad]。開口はSID面での直径diameter_cmで指定する。"""
    import math
    return math.atan((fld["diameter_cm"] / 2.0) / fld["sid_cm"])


_HEEL_N_BINS = 15


@functools.lru_cache(maxsize=8)
def _heel_spectra(kvp: float, filtration_mm_al: float, anode_angle_deg: float,
                   sid: float, span_cm: float, n_bins: int = _HEEL_N_BINS):
    """ヒール軸ビンごとの(中心座標, スペクトル列, 相対フルエンス, 絶対フルエンス/mAs)。

    座標sはSID面上のヒール軸方向オフセット[cm]で、s>0が陽極側。
    SpekPyの軸外計算（座標系はx<0が陽極側 — 実測確認済み）で各ビンの
    スペクトルとフルエンスを求める。陽極カットオフ（take-off角が0以下に
    なる領域）ではフルエンス0になり、その旨警告する。
    """
    if not _HAS_SPEKPY:
        raise RuntimeError("ヒール効果の計算にはspekpyが必要です"
                            "（`.venv/bin/pip install spekpy`）")
    centers = (np.arange(n_bins) + 0.5) / n_bins * 2.0 * span_cm - span_cm
    spectra = []
    flu_abs = []
    for s in centers:
        sp = _spekpy.Spek(kvp=kvp, th=anode_angle_deg, z=sid, x=-s)  # s>0(陽極側)→SpekPy -x
        sp.filter("Al", filtration_mm_al)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # カットオフ域のゼロ放出警告は下でまとめて出す
            e_mid, phi = sp.get_spectrum(edges=False)
            w = np.clip(np.asarray(phi, dtype=float), 0.0, None)
            tot = float(w.sum())
            flu = float(sp.get_flu()) if tot > 0 else 0.0
        spectra.append((np.asarray(e_mid, dtype=float),
                        w / tot if tot > 0 else w))
        flu_abs.append(flu)
    flu_abs = np.array(flu_abs)
    if flu_abs.max() <= 0:
        raise ValueError("ヒール効果計算: 全ビンでX線放出がゼロです。"
                          "陽極角と照射野の組み合わせを確認してください")
    if (flu_abs <= 0).any():
        _warnings.warn(
            "照射野の陽極側端が陽極カットオフ（take-off角0度）を超えており、"
            "その領域の放出はゼロです。実機でも照射できない配置なので、"
            "照射野サイズまたは陽極角を見直してください。", stacklevel=2)
    return centers, tuple(spectra), flu_abs / flu_abs.max(), flu_abs


def _heel_axis_coeffs(src: dict, d: np.ndarray, u: np.ndarray, v: np.ndarray):
    """anode_direction（世界座標）をビーム直交面に射影し、(u,v)基底の係数で返す。

    係数(h_u, h_v)は局所基底に対して定義されるため、rotation使用時も
    管球と一緒に回るヒール軸（実機の陽極-陰極軸は管球に固定）を正しく表す。
    """
    a = np.asarray(src["anode_direction"], dtype=float)
    a = a - (a @ d) * d
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        raise ValueError("source.anode_direction がビーム方向と平行です"
                          "（陽極-陰極軸はビーム中心軸に直交する成分が必要）")
    a = a / norm
    return float(a @ u), float(a @ v)


def sample_source_photons(src: dict, n: int, rng: np.random.Generator):
    """点線源から照射野への発散ビームで光子(位置,方向,エネルギー)を生成。

    照射野は2種類（source.field.shape）:
    - rect（既定）: SID面の矩形上に一様に目標点を抽選（コリメータ開口の
      幾何を優先する近似。立体角あたりの光子数は視野端でcos³θ分だけ過剰に
      なるが、一般撮影のSID・視野では数%）
    - cone: 中心軸周りの円錐（半頂角=atan(diameter_cm/2/sid_cm)）内で
      方向を立体角一様に抽選。実管球の等方放出に対応し、
      CBCT・広角視野で物理的により正しい

    source.heel_effect: true（+ anode_direction）でヒール効果を適用する:
    SpekPyの軸外スペクトルをヒール軸に沿ってビン化し、陽極側ほど光子数が
    少なく（棄却サンプリング）・スペクトルが硬くなる（ビン別エネルギー抽選）。
    rect/cone・rotation・ヘリカルすべてと合成可。anode_direction は
    局所基底の係数に変換されるため、rotation時は管球と一緒に回る。

    source.rotation が指定されている場合は、CTガントリー回転を光子ごとの
    角度抽選で表現する: 焦点位置・ビーム方向をisocenter周りに抽選角だけ回転させる。
    角度は既定で連続一様（実機CTの連続曝射に対応）、n_angles指定時のみ
    離散一様（トモシンセシス等の離散投影モダリティ用）。

    rotation.scan_length_cm 指定時はヘリカル撮影の位相平均近似:
    回転角と独立に、回転軸方向の一様分布で焦点を平行移動する。
    螺旋の開始位相が体に対してランダムであることを位相平均すると
    角度とテーブル位置は厳密に独立になるため、これは「位相平均された
    ヘリカル軌道」の統計的に厳密な表現である（スキャン端の半回転以内と
    over-rangingを除く）。
    """
    pos = np.asarray(src["position"], dtype=float)
    d = np.asarray(src["direction"], dtype=float)
    fld = src["field"]
    sid = fld["sid_cm"]
    if abs(d[2]) < 0.999:
        u = np.array([-d[1], d[0], 0.0])
    else:
        u = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)

    # 焦点位置と局所基底（d,u,v）を決める。回転時は光子ごとの(n,3)配列、
    # 静止時は(3,)のままにしてブロードキャストで共通処理する。
    rot = src.get("rotation")
    if rot is not None:
        iso = np.asarray(rot["isocenter"], dtype=float)
        axis = _ROTATION_AXES[rot.get("axis", "z")]
        n_angles = rot.get("n_angles")
        if n_angles:
            angles = 2.0 * np.pi * rng.integers(0, int(n_angles), size=n) / int(n_angles)
        else:
            angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
        pos_a = iso[None, :] + _rotate_batch(pos - iso, axis, angles)
        d_a = _rotate_batch(d, axis, angles)
        u_a = _rotate_batch(u, axis, angles)
        v_a = _rotate_batch(v, axis, angles)
        scan = float(rot.get("scan_length_cm") or 0.0)
        if scan > 0.0:
            shift = rng.uniform(-scan / 2.0, scan / 2.0, size=n)
            pos_a = pos_a + axis[None, :] * shift[:, None]
        origins = pos_a
    else:
        pos_a, d_a, u_a, v_a = pos, d, u, v
        origins = np.tile(pos, (n, 1))

    shape = fld.get("shape", "rect")

    # ヒール効果: 陽極-陰極軸に沿った強度低下（棄却サンプリング）と
    # 線質硬化（ビンごとのスペクトルからエネルギー抽選）を適用する
    heel = bool(src.get("heel_effect"))
    if heel:
        h_u, h_v = _heel_axis_coeffs(src, d, u, v)
        if shape == "cone":
            span = fld["diameter_cm"] / 2.0
        else:
            w0, h0 = fld["size_cm"]
            span = abs(h_u) * w0 / 2.0 + abs(h_v) * h0 / 2.0
        centers, spectra, rel_flu, _ = _heel_spectra(
            float(src["kvp"]), float(src.get("filtration_mm_al", 2.5)),
            float(src.get("anode_angle_deg", 12.0)), float(sid), float(span))

        def _accept(draw):
            """draw(m) -> (ヒール座標s, ペイロード列...) を棄却法でn個集める。"""
            kept = None
            got = 0
            while got < n:
                m = max(int((n - got) * 1.8), 1024)
                s, *payload = draw(m)
                p = np.interp(s, centers, rel_flu)
                keep = rng.random(m) < p
                cols = [s[keep]] + [c[keep] for c in payload]
                kept = cols if kept is None else [np.concatenate([a, b])
                                                   for a, b in zip(kept, cols)]
                got = len(kept[0])
            return [c[:n] for c in kept]

    if shape == "cone":
        cos_half = np.cos(cone_half_angle_rad(fld))

        def _draw_cone(m):
            # 円錐キャップ内の立体角一様抽選: cosθ ~ U[cos(半頂角), 1]、方位角一様
            c = rng.uniform(cos_half, 1.0, m)
            ph = rng.uniform(0.0, 2.0 * np.pi, m)
            tan_t = np.sqrt(1.0 - c ** 2) / c
            s = sid * tan_t * (np.cos(ph) * h_u + np.sin(ph) * h_v) if heel else None
            return s, c, ph

        if heel:
            s_heel, cos_t, phi = _accept(_draw_cone)
        else:
            _, cos_t, phi = _draw_cone(n)
        sin_t = np.sqrt(1.0 - cos_t ** 2)
        dirs = (cos_t[:, None] * d_a
                + (sin_t * np.cos(phi))[:, None] * u_a
                + (sin_t * np.sin(phi))[:, None] * v_a)
    else:
        w, h = fld["size_cm"]

        def _draw_rect(m):
            a = rng.uniform(-w / 2, w / 2, m)
            b = rng.uniform(-h / 2, h / 2, m)
            s = a * h_u + b * h_v if heel else None
            return s, a, b

        if heel:
            s_heel, su, sv = _accept(_draw_rect)
        else:
            _, su, sv = _draw_rect(n)
        target = pos_a + d_a * sid + su[:, None] * u_a + sv[:, None] * v_a
        dirs = target - origins
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    if heel:
        # エネルギーは光子が属すヒールビンのスペクトルから抽選（陽極側ほど硬い）
        bin_w = 2.0 * span / len(centers)
        bins = np.clip(((s_heel + span) / bin_w).astype(int), 0, len(centers) - 1)
        # 陽極カットオフ境界の補間で受理された光子がゼロ放出ビンに落ちることが
        # あるため、非ゼロフルエンスのビン範囲にクランプする
        nz = np.where(rel_flu > 0)[0]
        bins = np.clip(bins, nz.min(), nz.max())
        energies = np.empty(n)
        for b in np.unique(bins):
            m = bins == b
            e_b, w_b = spectra[b]
            energies[m] = e_b[rng.choice(len(e_b), size=int(m.sum()), p=w_b)]
    else:
        energies = sample_spectrum(src, n, rng)
    return origins, dirs, energies


def _linear_mu_batch(materials: np.ndarray, energies: np.ndarray) -> np.ndarray:
    mu = np.zeros(len(materials))
    for name in set(materials.tolist()):
        m = materials == name
        mu[m] = linear_mu(name, energies[m])
    return mu


def _mu_en_linear_batch(materials: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """μen = (μen/ρ)·ρ [1/cm] — カーマtrack-length estimator用の線減弱係数。"""
    mu_en = np.zeros(len(materials))
    for name in set(materials.tolist()):
        m = materials == name
        mu_en[m] = mu_en_rho(name, energies[m]) * density(name)
    return mu_en


def _sample_klein_nishina(e_keV: np.ndarray, rng: np.random.Generator):
    """Kahn型棄却法。ε=E'/E ∈ [1/(1+2α),1] を KN微分断面積に従って抽出。

    g(ε) = 1/ε + ε - sin²θ(ε) は ε=ε_min（後方散乱）で最大値
    M = 1/ε_min + ε_min を取るため、それを一様提案の包絡線に使う。
    """
    alpha = e_keV / _MEC2_KEV
    eps_min = 1.0 / (1.0 + 2.0 * alpha)
    envelope = 1.0 / eps_min + eps_min
    n = len(e_keV)
    eps = np.empty(n)
    cos_theta = np.empty(n)
    pending = np.arange(n)
    while len(pending) > 0:
        a_p, emin_p, m_p = alpha[pending], eps_min[pending], envelope[pending]
        xi1 = rng.random(len(pending))
        xi2 = rng.random(len(pending))
        eps_p = emin_p + xi1 * (1.0 - emin_p)
        cos_p = 1.0 - (1.0 / eps_p - 1.0) / a_p
        sin2_p = 1.0 - cos_p ** 2
        g = 1.0 / eps_p + eps_p - sin2_p
        accept = xi2 * m_p <= g
        acc = pending[accept]
        eps[acc] = eps_p[accept]
        cos_theta[acc] = cos_p[accept]
        pending = pending[~accept]
    return eps, cos_theta


def _sample_rayleigh_element(materials: np.ndarray, energies: np.ndarray,
                              rng: np.random.Generator) -> np.ndarray:
    """化合物・混合物の中で、レイリー相互作用がどの構成元素で起きたかを抽選する。

    質量分率×元素別レイリー断面積で規格化した重みに従う（materials.py参照）。
    """
    z_chosen = np.empty(len(materials), dtype=int)
    for name in set(materials.tolist()):
        m = materials == name
        zs, w = rayleigh_element_weights(name, energies[m])  # w: (n_elem, sum(m))
        cumw = np.cumsum(w, axis=0)
        r = rng.random(int(np.sum(m)))
        idx = np.clip(np.sum(r[None, :] > cumw, axis=0), 0, len(zs) - 1)
        z_chosen[m] = zs[idx]
    return z_chosen


def _sample_rayleigh_cos_theta(z_array: np.ndarray, e_array: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """原子形状因子込みの微分断面積 (1+cos²θ)·F(Z,q)² を棄却法で抽出。

    q = E·sin(θ/2)/hc [Å⁻¹]（xraylib.MomentTransfと同じ定義）。F(Z,q)はq=0で
    Zを取り単調減少するため、g(cosθ)=(1+cos²θ)F(Z,q)²の最大値は前方散乱
    (θ=0, q=0)での 2Z² となり、これを棄却法の包絡線に使う。
    """
    n = len(z_array)
    cos_theta = np.empty(n)
    pending = np.arange(n)
    while len(pending) > 0:
        zp = z_array[pending]
        ep = e_array[pending]
        c = rng.uniform(-1.0, 1.0, len(pending))
        theta = np.arccos(c)
        q = ep * np.sin(theta / 2.0) / _HC_KEV_ANGSTROM

        f = np.empty(len(pending))
        for z in set(zp.tolist()):
            m = zp == z
            q_grid, f_grid = rayleigh_form_factor_table(int(z))
            f[m] = np.interp(q[m], q_grid, f_grid)

        g = (1.0 + c ** 2) * f ** 2
        envelope = 2.0 * zp.astype(float) ** 2
        xi2 = rng.random(len(pending))
        accept = xi2 * envelope <= g
        acc = pending[accept]
        cos_theta[acc] = c[accept]
        pending = pending[~accept]
    return cos_theta


def _scatter_direction(d: np.ndarray, cos_theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = d.shape[0]
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, None))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    up = np.where((np.abs(d[:, 2]) < 0.999)[:, None],
                  np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    u = np.cross(up, d)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(d, u)
    new_dir = ((sin_theta * np.cos(phi))[:, None] * u
               + (sin_theta * np.sin(phi))[:, None] * v
               + cos_theta[:, None] * d)
    return new_dir / np.linalg.norm(new_dir, axis=1, keepdims=True)


def _deposit(energy_deposited: dict, mat_arr: np.ndarray, e_arr: np.ndarray) -> None:
    for name in set(mat_arr.tolist()):
        m = mat_arr == name
        energy_deposited[name] = energy_deposited.get(name, 0.0) + float(np.sum(e_arr[m]))


@dataclass
class BatchResult:
    n_scatter: np.ndarray       # (N,) int — 相互作用回数（吸収前含む）
    absorbed: np.ndarray        # (N,) bool — 光電吸収で消滅したか
    escaped: np.ndarray         # (N,) bool — 相互作用なしで世界境界を脱出したか
    final_energy: np.ndarray    # (N,) keV
    energy_deposited: dict = field(default_factory=dict)  # 材料名 -> keV


@dataclass
class TrajectoryRecorder:
    """軌跡記録（小history可視化用）。ループ1周ごとに飛行区間を追記する。

    starts/ends/energies/events/photon_ids はそれぞれ「1反復ぶんの配列」の
    リストとして貯め、trajectories_to_json() で光子ごとのポリラインにまとめる。
    event は区間の終端で起きたことを表す文字列:
      "boundary"（材料境界を通過して継続）, "photoelectric", "compton",
      "rayleigh", "escape"
    """
    starts: list = field(default_factory=list)
    ends: list = field(default_factory=list)
    energies: list = field(default_factory=list)
    events: list = field(default_factory=list)
    photon_ids: list = field(default_factory=list)

    def record(self, photon_id: np.ndarray, start: np.ndarray, end: np.ndarray,
               energy_keV: np.ndarray, event: np.ndarray) -> None:
        self.photon_ids.append(np.asarray(photon_id))
        self.starts.append(np.asarray(start))
        self.ends.append(np.asarray(end))
        self.energies.append(np.asarray(energy_keV, dtype=float))
        self.events.append(np.asarray(event, dtype=object))


def trajectories_to_json(recorder: TrajectoryRecorder) -> list[dict]:
    """TrajectoryRecorderの飛行区間データを光子ごとのポリラインにまとめる。

    区間はrecorderへの追記順（=輸送ループの反復順）であり、同一photon_idの
    区間は反復ごとに高々1つしか記録されないため、そのまま連結すれば
    時系列順のポリラインになる。
    """
    if not recorder.photon_ids:
        return []
    photon_ids = np.concatenate(recorder.photon_ids)
    starts = np.concatenate(recorder.starts)
    ends = np.concatenate(recorder.ends)
    energies = np.concatenate(recorder.energies)
    events = np.concatenate(recorder.events)

    by_photon: dict[int, dict] = {}
    order: list[int] = []
    for i in range(len(photon_ids)):
        pid = int(photon_ids[i])
        traj = by_photon.get(pid)
        if traj is None:
            traj = {"points": [starts[i].tolist()], "energies": [], "events": []}
            by_photon[pid] = traj
            order.append(pid)
        traj["points"].append(ends[i].tolist())
        traj["energies"].append(float(energies[i]))
        traj["events"].append(str(events[i]))

    return [by_photon[pid] for pid in order]


def transport_photons(pos: np.ndarray, dirv: np.ndarray, energy: np.ndarray,
                       geometry: Geometry, rng: np.random.Generator,
                       grid: VoxelGrid | None = None,
                       recorder: TrajectoryRecorder | None = None) -> BatchResult:
    """光源サンプリングとは独立な輸送カーネル本体（テストで直接叩ける）。

    pos/dirv/energy は呼び出し側の配列を破壊的に更新する。
    grid を渡すと、各飛行区間ごとにカーマのtrack-length estimatorを
    ボクセルグリッドへ積算する（vivemonte/tally.py参照）。
    recorder を渡すと、各飛行区間を可視化用に記録する（既定Noneで無効、
    乱数を一切消費しないため同一seedでの輸送結果に影響しない）。
    """
    n = pos.shape[0]
    alive = np.ones(n, dtype=bool)
    tau = -np.log(rng.random(n))
    n_scatter = np.zeros(n, dtype=int)
    absorbed = np.zeros(n, dtype=bool)
    escaped = np.zeros(n, dtype=bool)
    energy_deposited: dict = {}

    while np.any(alive):
        idx = np.where(alive)[0]
        o, d, e = pos[idx], dirv[idx], energy[idx]
        mat = geometry.material_at(o)
        mu = _linear_mu_batch(mat, e)
        t_boundary, escape = geometry.next_boundary(o, d)
        mu_safe = np.where(mu > 0, mu, 1e-30)
        tau_to_boundary = mu * t_boundary
        will_interact = tau[idx] < tau_to_boundary

        ds = np.where(will_interact, tau[idx] / mu_safe, t_boundary)
        ends = o + d * ds[:, None]

        if grid is not None:
            mu_en_linear = _mu_en_linear_batch(mat, e)
            accumulate_track_length(grid.kerma_keV, grid, o, d, ds, e * mu_en_linear)
            accumulate_track_length(grid.h10_track_pSv_cm3, grid, o, d, ds, h_star_10_per_fluence(e))

        pos[idx] = ends

        noninteract = ~will_interact
        gidx = idx[noninteract]
        tau[gidx] -= tau_to_boundary[noninteract]
        pos[gidx] += dirv[gidx] * 1e-6
        esc_now = idx[noninteract & escape]
        alive[esc_now] = False
        escaped[esc_now] = True

        interact = will_interact
        iidx = idx[interact]
        if len(iidx) > 0:
            mat_i = mat[interact]
            e_i = e[interact]
            r_type = rng.random(len(iidx))
            p_photo = np.zeros(len(iidx))
            p_compt = np.zeros(len(iidx))
            for name in set(mat_i.tolist()):
                m = mat_i == name
                parts = mu_rho_parts(name, e_i[m])
                tot = parts["photoelectric"] + parts["compton"] + parts["rayleigh"]
                tot = np.where(tot > 0, tot, 1.0)
                p_photo[m] = parts["photoelectric"] / tot
                p_compt[m] = parts["compton"] / tot

            is_photo = r_type < p_photo
            is_compt = (~is_photo) & (r_type < p_photo + p_compt)
            is_rayl = (~is_photo) & (~is_compt)

            photo_idx = iidx[is_photo]
            if len(photo_idx) > 0:
                _deposit(energy_deposited, mat_i[is_photo], e_i[is_photo])
                alive[photo_idx] = False
                absorbed[photo_idx] = True
                n_scatter[photo_idx] += 1

            compt_idx = iidx[is_compt]
            if len(compt_idx) > 0:
                e_c = e_i[is_compt]
                eps, cos_theta = _sample_klein_nishina(e_c, rng)
                e_new = e_c * eps
                _deposit(energy_deposited, mat_i[is_compt], e_c - e_new)
                dirv[compt_idx] = _scatter_direction(dirv[compt_idx], cos_theta, rng)
                energy[compt_idx] = e_new
                tau[compt_idx] = -np.log(rng.random(len(compt_idx)))
                n_scatter[compt_idx] += 1

            rayl_idx = iidx[is_rayl]
            if len(rayl_idx) > 0:
                z_r = _sample_rayleigh_element(mat_i[is_rayl], e_i[is_rayl], rng)
                cos_theta = _sample_rayleigh_cos_theta(z_r, e_i[is_rayl], rng)
                dirv[rayl_idx] = _scatter_direction(dirv[rayl_idx], cos_theta, rng)
                tau[rayl_idx] = -np.log(rng.random(len(rayl_idx)))
                n_scatter[rayl_idx] += 1

        if recorder is not None:
            event = np.full(len(idx), "boundary", dtype=object)
            event[noninteract & escape] = "escape"
            if len(iidx) > 0:
                interact_positions = np.where(interact)[0]
                event[interact_positions[is_photo]] = "photoelectric"
                event[interact_positions[is_compt]] = "compton"
                event[interact_positions[is_rayl]] = "rayleigh"
            recorder.record(idx, o, ends, e, event)

    return BatchResult(n_scatter=n_scatter, absorbed=absorbed, escaped=escaped,
                        final_energy=energy, energy_deposited=energy_deposited)


@dataclass
class TransportResult:
    n_histories: int
    energy_deposited_MeV: dict
    fraction_absorbed: float
    fraction_escaped: float
    mean_scatter_events: float
    grid: VoxelGrid | None = None
    # 絶対値換算係数（per-history値×これ=実線量）。mas指定時は照射野を通過する
    # 実光子数、ctdi_vol_mGy指定時はCTDIファントム校正による実効光子数。
    n_photons_real: float | None = None


def photon_count_through_field(src: dict) -> float:
    """指定されたmAsで実際に照射野を通過する光子数（フルエンス×照射野面積）。

    SpekPyの絶対フルエンス計算（get_flu、focus-to-detector距離z=SIDでの
    photons/cm²）を使う。フィールド内のフルエンス分布は中心軸上の値で
    一様と近似する（斜入射による濾過路長の増加等は無視、教育・研究用の
    一次近似）。SpekPy未インストール環境では校正できない
    （Kramers近似フォールバックは絶対規格化を持たないため）。
    """
    if not _HAS_SPEKPY:
        raise RuntimeError("光子数校正にはspekpyが必要です（`.venv/bin/pip install spekpy`）")
    mas = src.get("mas")
    if mas is None:
        raise ValueError("source.mas が指定されていません（光子数校正には管電流時間積[mAs]が必要）")
    fld = src["field"]
    sid = fld["sid_cm"]
    if fld.get("shape", "rect") == "cone":
        area = np.pi * (fld["diameter_cm"] / 2.0) ** 2
    else:
        w, h = fld["size_cm"]
        area = w * h

    if src.get("heel_effect"):
        # ヒール適用時は中心軸値ではなく照射野の面平均フルエンスを使う
        d = np.asarray(src["direction"], dtype=float)
        if abs(d[2]) < 0.999:
            u = np.array([-d[1], d[0], 0.0])
        else:
            u = np.array([1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)
        h_u, h_v = _heel_axis_coeffs(src, d, u, v)
        if fld.get("shape", "rect") == "cone":
            span = fld["diameter_cm"] / 2.0
            g = np.linspace(-span, span, 101)
            gu, gv = np.meshgrid(g, g)
            inside = gu ** 2 + gv ** 2 <= span ** 2
            s_grid = (gu * h_u + gv * h_v)[inside]
        else:
            w0, h0 = fld["size_cm"]
            span = abs(h_u) * w0 / 2.0 + abs(h_v) * h0 / 2.0
            gu, gv = np.meshgrid(np.linspace(-w0 / 2, w0 / 2, 101),
                                  np.linspace(-h0 / 2, h0 / 2, 101))
            s_grid = gu * h_u + gv * h_v
        centers, _, _, flu_abs = _heel_spectra(
            float(src["kvp"]), float(src.get("filtration_mm_al", 2.5)),
            float(src.get("anode_angle_deg", 12.0)), float(sid), float(span))
        flu_mean_per_mas = float(np.interp(s_grid.ravel(), centers, flu_abs).mean())
        return flu_mean_per_mas * mas * area

    s = _spekpy.Spek(kvp=src["kvp"], th=src.get("anode_angle_deg", 12.0), z=sid, mas=mas)
    s.filter("Al", src.get("filtration_mm_al", 2.5))
    fluence_per_cm2 = s.get_flu()
    return fluence_per_cm2 * area


def dose_map_Gy(grid: VoxelGrid, geometry: Geometry) -> np.ndarray:
    """ボクセル中心の材料を判定し、その密度でカーマ→吸収線量[Gy]に換算する。

    グリッドはタリー専用であり材料を保持しないため、密度は出力時に
    ジオメトリーへ問い合わせて求める（ボクセル解像度が粗い場合、
    境界付近のボクセルは中心点1点で代表材料を決める近似になる）。
    """
    centers = grid.voxel_centers()
    mat = geometry.material_at(centers)
    density_flat = np.array([density(m) for m in mat])
    return grid.dose_map_Gy(density_flat.reshape(grid.shape))


def max_voxel_position_cm(grid: VoxelGrid, data: np.ndarray) -> np.ndarray:
    """dataの最大値を持つボクセルの中心座標[cm]。"""
    idx = np.unravel_index(int(np.argmax(data)), data.shape)
    return grid.origin_cm + (np.asarray(idx, dtype=float) + 0.5) * grid.voxel_size_cm


def background_medium_warning(material: str, background: str) -> str | None:
    """吸収線量の最大値ボクセルが背景（既定air）かどうかを判定する。

    吸収線量 = カーマ/密度 は媒質固有の量（同じカーマでも密度が違えば
    値が変わる）。空気は密度が非常に小さいため、材料境界のすぐ外側の
    空気ボクセルはカーマが同程度でも線量が大きく増幅されて見えることがある
    （[[lessons_learned]]参照）。この値は患者・検出器等、実体のある位置の
    被ばく評価には使えない。
    """
    if material != background:
        return None
    return ("最大値は空気中ボクセル（材料=air）です。吸収線量は媒質固有の量のため、"
            "この値は患者・検出器等の実体がある位置の被ばく評価には使えません。")


def near_source_air_warning(material: str, background: str, distance_from_source_cm: float,
                             nearest_object_distance_cm: float | None) -> str | None:
    """H*(10)最大値が点線源モデルの1/r²発散による非物理的な値かどうかを判定する。

    材料が背景（空気）かつ、シーン内のどの物体よりも線源に近い位置にある場合のみ
    警告する。「シーン内に実在する物体よりも線源に近い」は、その位置に人や検出器が
    存在し得ないことの明確な根拠になる（実際のX線管は housing/コリメータで
    覆われているが、viveMonteの点線源モデルはそれを持たない）。
    """
    if material != background:
        return None
    if nearest_object_distance_cm is None or distance_from_source_cm >= nearest_object_distance_cm:
        return None
    return (f"最大値は線源から{distance_from_source_cm:.1f}cmの空気中ボクセルで、"
            f"シーン内のどの物体（最寄り{nearest_object_distance_cm:.1f}cm）よりも"
            "線源に近い位置です。点線源モデルの1/r²発散による非物理的な値であり、"
            "実在する位置の被ばく評価には使えません。評価したい位置（患者表面・"
            "操作者位置等）には直接細かいグリッドを敷いて計算してください。")


def run_transport(scene, n_histories: int = 100_000, seed: int | None = None,
                   batch_size: int = 200_000, dose_grid: bool = False,
                   grid_resolution_cm: float = 5.0) -> TransportResult:
    rng = np.random.default_rng(seed)
    src = scene.raw["source"]
    geometry = Geometry(scene.raw["geometry"])
    grid = VoxelGrid.from_bbox(geometry.bbox_min, geometry.bbox_max, grid_resolution_cm) if dose_grid else None

    energy_deposited: dict = {}
    n_absorbed = 0
    n_escaped = 0
    scatter_sum = 0
    remaining = n_histories
    while remaining > 0:
        n = min(batch_size, remaining)
        remaining -= n
        pos, dirv, energy = sample_source_photons(src, n, rng)
        result = transport_photons(pos, dirv, energy, geometry, rng, grid=grid)
        for name, e_keV in result.energy_deposited.items():
            energy_deposited[name] = energy_deposited.get(name, 0.0) + e_keV
        n_absorbed += int(np.sum(result.absorbed))
        n_escaped += int(np.sum(result.escaped))
        scatter_sum += int(np.sum(result.n_scatter))

    # 絶対線量校正: CTDIvol基準（CT向け、実測に装置特性が折り込まれ汎用性が高い）
    # が指定されていればそちらを優先。なければmAs+SpekPyフルエンス基準。
    if src.get("ctdi_vol_mGy") is not None:
        from .ctdi import effective_histories_from_ctdi
        n_photons_real = effective_histories_from_ctdi(src, seed=seed)
    elif src.get("mas") is not None:
        n_photons_real = photon_count_through_field(src)
    else:
        n_photons_real = None

    return TransportResult(
        n_histories=n_histories,
        energy_deposited_MeV={k: v / 1000.0 for k, v in energy_deposited.items()},
        fraction_absorbed=n_absorbed / n_histories,
        fraction_escaped=n_escaped / n_histories,
        mean_scatter_events=scatter_sum / n_histories,
        grid=grid,
        n_photons_real=n_photons_real,
    )
