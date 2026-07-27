"""Phase B-1a: per-historyスカラー輸送カーネルの実現性プローブ（Numba njit）。

docs/plan_chatcarlo_speedup_post_egs5.md のPhase B設計に基づく。**これはB-1の
最終実装ではなくスループット計測専用のプローブ**——アドバイザーレビューの助言に
従い、まず「per-historyスカラーループをコンパイルすれば1.5M histories/sに届く
のか」だけを最小コストで確認してから、物理的完全性（蛍光・多材料・多形状）を
順に積み増す方針にした。

**このプローブの意図的な簡略化（B-1bで解消する）**:
- 材料は`water`単色（box 1個）のみ。box外の背景は真空として扱う（μ=0で
  相互作用させない）——本番シーン（`water60_free`等）はbackground="air"だが、
  境界マージンは0.01cmとごく薄く寄与が無視できる領域であり、スループット
  プローブとしては許容する（統計的クロスチェックはB-1bで背景を正しく
  扱ってから行う）。
- K殻蛍光は無効固定（`fluorescence_enabled=False`相当）。xraylibの
  `CS_Photo_Partial`呼び出しがnjit内で使えない（B-0で確認済み）ため、
  テーブル化が必要——このプローブでは後回しにした。
- 元素間でエネルギー格子が同一であることを要求する（`bake_material_tables`が
  検証）。water(H,O)は両元素とも吸収端が格子下限1 keV未満のため成立するが、
  重元素を含む材料は非対応（Step 1と同じ制約、docs/plan_chatcarlo_speedup_post_egs5.md
  Step 1参照）。
- 乱数はプローブ全体で単一の`np.random.seed()`のみ（historyをまたいでMT19937
  ストリームを継続する、単一スレッド前提）。チャンク単位の決定的シード
  （B-0で確定した設計）はprange並列化を実装するB-1bで導入する。

物理サンプリングのロジックは`chatcarlo/physics.py`のベクトル化実装
（`sample_compton_bound`・`sample_rayleigh_cos_theta`・`transport_photons`の
主ループ）をスカラー・njit向けに書き直したもので、アルゴリズムは同一
（Kahn型棄却法+S(Z,q)追加棄却、2段階レイリー逆変換+角度棄却、解析面
トラッキング）。乱数消費順序・アルゴリズム自体はレガシーMT19937を使う都合上
ベクトル化参照実装（PCG64）とビット一致しない——Phase Bはそもそもビット一致を
要求せず統計的クロスチェックで検証する設計（計画書「Phase Bの検証戦略」）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .materials import (_element_xs_tables, density, element_composition,
                         incoherent_sq_table, rayleigh_cumulative_table)

_MEC2_KEV = 511.0
_HC_KEV_ANGSTROM = 12.3984193
_BOUNDARY_EPS = 1e-6
_BOX_EPS = 1e-12
_NUDGE = 1e-6


@dataclass
class MaterialTables:
    """water専用に焼いた元素別断面積テーブル（B-1aは単一材料のみ対応）。"""
    n_elem: int
    zs: np.ndarray            # (n_elem,) int64
    fracs: np.ndarray         # (n_elem,) float64 質量分率
    log_e_grid: np.ndarray    # (n_grid,) 元素間で共有（要検証）
    step: float               # 対数等間隔ステップ幅
    photo_tab: np.ndarray     # (n_elem, n_grid) log(光電cs[cm2/g])
    compt_tab: np.ndarray     # (n_elem, n_grid) log(コンプトンcs[cm2/g])
    rayl_tab: np.ndarray      # (n_elem, n_grid) log(レイリーcs[cm2/g])
    incoh_q: np.ndarray       # (n_elem, n_incoh) S(Z,q)テーブルのq格子
    incoh_s: np.ndarray       # (n_elem, n_incoh) S(Z,q)
    rayl_x: np.ndarray        # (n_elem, n_rayl) レイリー累積テーブルのx=q^2格子
    rayl_a: np.ndarray        # (n_elem, n_rayl) 累積A(x)
    density_g_cm3: float


def bake_material_tables(material: str) -> MaterialTables:
    """材料の元素別断面積テーブルをカーネル用の固定長ndarrayに焼く。

    B-1aは軽元素（対数等間隔格子・かつ元素間でグリッドが完全一致）のみ対応
    （water=H,Oで検証済み、chatcarlo/materials.pyの`_uniform_log_step`/
    `_element_energy_grid_kev`参照）。この前提が崩れる材料（重元素を含む・
    格子が元素間で不一致）は明示的にValueErrorで弾く——プローブの結果を
    間違った前提のまま拡大解釈しないため。
    """
    comp = element_composition(material)
    n_elem = len(comp)
    zs = np.array([z for z, _ in comp], dtype=np.int64)
    fracs = np.array([f for _, f in comp], dtype=np.float64)

    base_log_e = None
    step = None
    photo_rows, compt_rows, rayl_rows = [], [], []
    incoh_q_rows, incoh_s_rows = [], []
    rayl_x_rows, rayl_a_rows = [], []
    for z in zs.tolist():
        t = _element_xs_tables(z)
        if t["uniform_step"] is None:
            raise ValueError(
                f"材料'{material}'の元素Z={z}は非等間隔格子——B-1aプローブは非対応"
                "（軽元素のみ対応、docs/plan_chatcarlo_speedup_post_egs5.md Step 1参照）")
        if base_log_e is None:
            base_log_e = t["log_e"]
            step = t["uniform_step"]
        elif t["log_e"].shape != base_log_e.shape or not np.allclose(t["log_e"], base_log_e):
            raise ValueError(
                f"材料'{material}'は元素間でエネルギー格子が一致しない——B-1aプローブは非対応")
        photo_rows.append(t["photo"])
        compt_rows.append(t["compt"])
        rayl_rows.append(t["rayl"])
        q_grid, s_grid = incoherent_sq_table(int(z))
        incoh_q_rows.append(q_grid)
        incoh_s_rows.append(s_grid)
        x_grid, a_grid = rayleigh_cumulative_table(int(z))
        rayl_x_rows.append(x_grid)
        rayl_a_rows.append(a_grid)

    return MaterialTables(
        n_elem=n_elem, zs=zs, fracs=fracs,
        log_e_grid=base_log_e, step=float(step),
        photo_tab=np.stack(photo_rows), compt_tab=np.stack(compt_rows),
        rayl_tab=np.stack(rayl_rows),
        incoh_q=np.stack(incoh_q_rows), incoh_s=np.stack(incoh_s_rows),
        rayl_x=np.stack(rayl_x_rows), rayl_a=np.stack(rayl_a_rows),
        density_g_cm3=density(material),
    )


@njit(cache=True)
def _lerp_lookup(x_grid, y_grid, x):
    """1本の昇順格子上での二分探索＋線形補間（境界はクランプ）。

    Compton用S(Z,q)テーブル・Rayleigh用累積テーブル（順変換・逆変換とも）で
    共有する唯一のルックアップ実装（アドバイザー助言: 手書きバリアントを
    増やすとそこに符号ミスが混入するため、1本にまとめて使い回す）。
    """
    n = x_grid.shape[0]
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[n - 1]:
        return y_grid[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    y0 = y_grid[lo]
    y1 = y_grid[hi]
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


@njit(cache=True)
def _arith_index_frac(log_e_grid, step, log_eq):
    """chatcarlo.materials._element_interp_index_frac のスカラー・±1補正版。

    searchsorted+clipと厳密に同じインデックスを返すことがベクトル化版で
    フルグリッド突き合わせテスト済み（tests/test_materials.py）——同じ±1
    補正ロジックをそのままスカラーに移植する。
    """
    hi = log_e_grid.shape[0] - 1
    idx = int((log_eq - log_e_grid[0]) / step) + 1
    if idx < 1:
        idx = 1
    if idx > hi:
        idx = hi
    if log_eq <= log_e_grid[idx - 1]:
        idx -= 1
        if idx < 1:
            idx = 1
    if log_e_grid[idx] < log_eq:
        idx += 1
        if idx > hi:
            idx = hi
    x0 = log_e_grid[idx - 1]
    x1 = log_e_grid[idx]
    frac = (log_eq - x0) / (x1 - x0)
    return idx, frac


@njit(cache=True)
def _mu_and_parts_scalar(e, n_elem, fracs, log_e_grid, step, photo_tab, compt_tab, rayl_tab,
                          density_g_cm3, photo_e, compt_e, rayl_e):
    """透過光子1個の(μ[1/cm], 光電/コンプトン/レイリー質量減弱係数和[cm2/g])。

    photo_e/compt_e/rayl_e は呼び出し側が確保した長さn_elemの作業配列
    （後段の元素抽選で再利用するため、ここで書き込んで返す）。
    """
    log_eq = math.log(e)
    idx, frac = _arith_index_frac(log_e_grid, step, log_eq)
    tot_photo = 0.0
    tot_compt = 0.0
    tot_rayl = 0.0
    for i in range(n_elem):
        p = math.exp(photo_tab[i, idx - 1] + frac * (photo_tab[i, idx] - photo_tab[i, idx - 1]))
        c = math.exp(compt_tab[i, idx - 1] + frac * (compt_tab[i, idx] - compt_tab[i, idx - 1]))
        r = math.exp(rayl_tab[i, idx - 1] + frac * (rayl_tab[i, idx] - rayl_tab[i, idx - 1]))
        photo_e[i] = p
        compt_e[i] = c
        rayl_e[i] = r
        tot_photo += fracs[i] * p
        tot_compt += fracs[i] * c
        tot_rayl += fracs[i] * r
    mu = (tot_photo + tot_compt + tot_rayl) * density_g_cm3
    return mu, tot_photo, tot_compt, tot_rayl


@njit(cache=True)
def _select_element(n_elem, fracs, cs_e, total):
    """元素別断面積(fracs[i]*cs_e[i])に比例する重みで元素indexを1個抽選する。

    `physics.sample_compton_element`/`sample_rayleigh_element`と同じ選択則
    （質量分率×断面積で規格化した累積分布からの逆変換）のスカラー版。
    """
    target = np.random.random() * total
    cum = 0.0
    for i in range(n_elem):
        cum += fracs[i] * cs_e[i]
        if target <= cum:
            return i
    return n_elem - 1


@njit(cache=True)
def _intersect_box_scalar(ox, oy, oz, dx, dy, dz, hx, hy, hz):
    """原点中心・半径(hx,hy,hz)の軸平行boxとレイの交差（entry,exit,hit）。

    chatcarlo.geometry._intersect_box のスカラー版（3軸のスラブ交差判定）。
    """
    t_enter = -np.inf
    t_exit = np.inf
    miss = False

    if abs(dx) < _BOX_EPS:
        if ox < -hx or ox > hx:
            miss = True
    else:
        ta = (-hx - ox) / dx
        tb = (hx - ox) / dx
        t_enter = max(t_enter, min(ta, tb))
        t_exit = min(t_exit, max(ta, tb))

    if abs(dy) < _BOX_EPS:
        if oy < -hy or oy > hy:
            miss = True
    else:
        ta = (-hy - oy) / dy
        tb = (hy - oy) / dy
        t_enter = max(t_enter, min(ta, tb))
        t_exit = min(t_exit, max(ta, tb))

    if abs(dz) < _BOX_EPS:
        if oz < -hz or oz > hz:
            miss = True
    else:
        ta = (-hz - oz) / dz
        tb = (hz - oz) / dz
        t_enter = max(t_enter, min(ta, tb))
        t_exit = min(t_exit, max(ta, tb))

    hit = (not miss) and (t_enter <= t_exit)
    return t_enter, t_exit, hit


@njit(cache=True)
def _next_boundary_scalar(ox, oy, oz, dx, dy, dz, hx, hy, hz, whx, why, whz):
    """geometry.Geometry.next_boundary のスカラー版（単一box専用）。

    戻り値: (次の境界までの距離t, 世界脱出かどうか)。box自体の境界（材料が
    変わる点）と、box+marginで作る世界境界の両方を判定し、世界境界の方が
    近ければ脱出とする（Geometry.next_boundaryと同じ規則）。
    """
    t_enter, t_exit, hit = _intersect_box_scalar(ox, oy, oz, dx, dy, dz, hx, hy, hz)
    t_obj = np.inf
    if hit:
        if t_enter > _BOUNDARY_EPS:
            t_obj = t_enter
        if t_exit > _BOUNDARY_EPS and t_exit < t_obj:
            t_obj = t_exit

    _, t_exit_w, hit_w = _intersect_box_scalar(ox, oy, oz, dx, dy, dz, whx, why, whz)
    t_exit_world = t_exit_w if hit_w else np.inf

    escape = t_exit_world <= t_obj
    t = min(t_obj, t_exit_world)
    return t, escape


@njit(cache=True)
def _scatter_direction_scalar(dx, dy, dz, cos_theta):
    """physics.scatter_direction のスカラー版。方位角は一様抽選。"""
    sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    phi = np.random.random() * 2.0 * math.pi
    if abs(dz) < 0.999:
        upx, upy, upz = 0.0, 0.0, 1.0
    else:
        upx, upy, upz = 1.0, 0.0, 0.0
    # u = up x d
    ux = upy * dz - upz * dy
    uy = upz * dx - upx * dz
    uz = upx * dy - upy * dx
    un = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= un
    uy /= un
    uz /= un
    # v = d x u
    vx = dy * uz - dz * uy
    vy = dz * ux - dx * uz
    vz = dx * uy - dy * ux
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    nx = sin_theta * cphi * ux + sin_theta * sphi * vx + cos_theta * dx
    ny = sin_theta * cphi * uy + sin_theta * sphi * vy + cos_theta * dy
    nz = sin_theta * cphi * uz + sin_theta * sphi * vz + cos_theta * dz
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx / nn, ny / nn, nz / nn


@njit(cache=True)
def _sample_compton_bound_scalar(e_kev, n_elem, fracs, zs, incoh_q, incoh_s, compt_e_work):
    """physics.sample_compton_bound のスカラー版（Kahn型棄却＋S(Z,q)/Z追加棄却）。

    元素選択は呼び出し側で既に求めたcompt_e_work（各元素の断面積）を使う
    （`_mu_and_parts_scalar`が同じエネルギーで既に計算済みの値を再利用し、
    断面積テーブルへの再アクセスを避ける——ベクトル化参照実装と同じ考え方）。
    S(Z,q)/Zの割り算は選ばれた元素の実際の原子番号`zs[elem_i]`を使う
    （割り算対象を1.0に固定するのは物理的に誤り——旧稿のバグをレビューで訂正）。
    """
    tot_compt = 0.0
    for i in range(n_elem):
        tot_compt += fracs[i] * compt_e_work[i]
    elem_i = _select_element(n_elem, fracs, compt_e_work, tot_compt)
    z_val = float(zs[elem_i])
    alpha = e_kev / _MEC2_KEV
    eps_min = 1.0 / (1.0 + 2.0 * alpha)
    envelope = 1.0 / eps_min + eps_min
    while True:
        xi1 = np.random.random()
        xi2 = np.random.random()
        eps_p = eps_min + xi1 * (1.0 - eps_min)
        cos_p = 1.0 - (1.0 / eps_p - 1.0) / alpha
        sin2_p = 1.0 - cos_p * cos_p
        g = 1.0 / eps_p + eps_p - sin2_p
        if xi2 * envelope > g:
            continue
        cc = cos_p
        if cc > 1.0:
            cc = 1.0
        if cc < -1.0:
            cc = -1.0
        theta_p = math.acos(cc)
        q_p = e_kev * math.sin(theta_p / 2.0) / _HC_KEV_ANGSTROM
        s_over_z = _lerp_lookup(incoh_q[elem_i], incoh_s[elem_i], q_p) / z_val
        xi3 = np.random.random()
        if xi3 <= s_over_z:
            return eps_p, cos_p, elem_i


@njit(cache=True)
def _sample_rayleigh_cos_theta_scalar(e_kev, n_elem, fracs, rayl_x, rayl_a, rayl_e_work):
    """physics.sample_rayleigh_cos_theta のスカラー版（2段階逆変換＋角度棄却）。

    元素選択は`_mu_and_parts_scalar`が同じエネルギーで既に求めたrayl_e_workを
    再利用する（`_sample_compton_bound_scalar`と同じ理由）。
    """
    tot_rayl = 0.0
    for i in range(n_elem):
        tot_rayl += fracs[i] * rayl_e_work[i]
    elem_i = _select_element(n_elem, fracs, rayl_e_work, tot_rayl)
    x_max = (e_kev / _HC_KEV_ANGSTROM) ** 2
    while True:
        a_cut = _lerp_lookup(rayl_x[elem_i], rayl_a[elem_i], x_max)
        xi1 = np.random.random()
        x_val = _lerp_lookup(rayl_a[elem_i], rayl_x[elem_i], xi1 * a_cut)
        if x_val > x_max:
            x_val = x_max
        cos_c = 1.0 - 2.0 * x_val / x_max
        if cos_c > 1.0:
            cos_c = 1.0
        if cos_c < -1.0:
            cos_c = -1.0
        xi2 = np.random.random()
        if xi2 <= (1.0 + cos_c * cos_c) / 2.0:
            return cos_c


@njit(cache=True)
def _transport_one(energy0_kev, ox, oy, oz, dx, dy, dz,
                    hx, hy, hz, whx, why, whz,
                    n_elem, zs, fracs, log_e_grid, step, density_g_cm3,
                    photo_tab, compt_tab, rayl_tab, incoh_q, incoh_s, rayl_x, rayl_a):
    """1光子をtransport_photons(transport.py)の主ループと同一アルゴリズムで
    吸収/脱出まで追跡する（box 1個・材料water固定・box外は真空・蛍光無効の
    B-1aプローブ版）。

    戻り値: (n_scatter, absorbed, escaped, final_energy_kev, energy_deposited_kev)。
    """
    x, y, z = ox, oy, oz
    e = energy0_kev
    tau = -math.log(np.random.random())
    n_scatter = 0
    energy_deposited = 0.0
    photo_e = np.empty(n_elem)
    compt_e = np.empty(n_elem)
    rayl_e = np.empty(n_elem)

    while True:
        inside = (abs(x) <= hx) and (abs(y) <= hy) and (abs(z) <= hz)
        if inside:
            mu, tot_photo, tot_compt, tot_rayl = _mu_and_parts_scalar(
                e, n_elem, fracs, log_e_grid, step, photo_tab, compt_tab, rayl_tab,
                density_g_cm3, photo_e, compt_e, rayl_e)
        else:
            mu = 0.0
            tot_photo = tot_compt = tot_rayl = 0.0

        t_boundary, escape = _next_boundary_scalar(x, y, z, dx, dy, dz, hx, hy, hz, whx, why, whz)
        mu_safe = mu if mu > 0.0 else 1e-30
        tau_to_boundary = mu * t_boundary
        will_interact = tau < tau_to_boundary

        if will_interact:
            ds = tau / mu_safe
        else:
            ds = t_boundary
        x += dx * ds
        y += dy * ds
        z += dz * ds

        if not will_interact:
            tau -= tau_to_boundary
            x += dx * _NUDGE
            y += dy * _NUDGE
            z += dz * _NUDGE
            if escape:
                return n_scatter, False, True, e, energy_deposited
            continue

        n_scatter += 1
        r_type = np.random.random()
        tot = tot_photo + tot_compt + tot_rayl
        p_photo = tot_photo / tot
        p_compt = tot_compt / tot

        if r_type < p_photo:
            energy_deposited += e
            return n_scatter, True, False, e, energy_deposited

        if r_type < p_photo + p_compt:
            eps_c, cos_c, _elem = _sample_compton_bound_scalar(
                e, n_elem, fracs, zs, incoh_q, incoh_s, compt_e)
            e_new = e * eps_c
            energy_deposited += e - e_new
            dx, dy, dz = _scatter_direction_scalar(dx, dy, dz, cos_c)
            e = e_new
            tau = -math.log(np.random.random())
        else:
            cos_c = _sample_rayleigh_cos_theta_scalar(e, n_elem, fracs, rayl_x, rayl_a, rayl_e)
            dx, dy, dz = _scatter_direction_scalar(dx, dy, dz, cos_c)
            tau = -math.log(np.random.random())
        # レイリー・コンプトン散乱後は光子が生き残るのでreturnせずループを継続
        # （n_scatter・energy_depositedは次回の境界/相互作用判定に持ち越す）。


@njit(cache=True)
def _run_batch_scalar(n_histories, base_seed, energy0_kev, ox, oy, oz, dx, dy, dz,
                       hx, hy, hz, whx, why, whz,
                       n_elem, zs, fracs, log_e_grid, step, density_g_cm3,
                       photo_tab, compt_tab, rayl_tab, incoh_q, incoh_s, rayl_x, rayl_a):
    """single-threadでn_histories回`_transport_one`を回す（B-1aは並列化前）。

    乱数は`np.random.seed(base_seed)`をバッチ先頭で1回だけ呼び、以降は
    MT19937ストリームをhistoryをまたいで継続する——ベクトル化参照実装が
    1本のGeneratorをバッチ全体で使い回すのと同じ発想（`transport_photons`の
    `rng`引数）。チャンク単位の決定的シード割り当て（B-0で確定した設計、
    prange並列化用）はこのプローブでは未実装——B-1bでprange対応時に導入する。
    """
    np.random.seed(base_seed)
    n_scatter = np.zeros(n_histories, dtype=np.int64)
    absorbed = np.zeros(n_histories, dtype=np.bool_)
    escaped = np.zeros(n_histories, dtype=np.bool_)
    final_energy = np.zeros(n_histories, dtype=np.float64)
    energy_deposited = np.zeros(n_histories, dtype=np.float64)
    for i in range(n_histories):
        ns, ab, es, fe, ed = _transport_one(
            energy0_kev, ox, oy, oz, dx, dy, dz, hx, hy, hz, whx, why, whz,
            n_elem, zs, fracs, log_e_grid, step, density_g_cm3,
            photo_tab, compt_tab, rayl_tab, incoh_q, incoh_s, rayl_x, rayl_a)
        n_scatter[i] = ns
        absorbed[i] = ab
        escaped[i] = es
        final_energy[i] = fe
        energy_deposited[i] = ed
    return n_scatter, absorbed, escaped, final_energy, energy_deposited


@dataclass
class ProbeResult:
    n_histories: int
    wall_s: float
    histories_per_s: float
    uncollided_frac: float
    fraction_absorbed: float
    fraction_escaped: float
    mean_scatter_events: float
    energy_deposited_keV: float


def run_water_slab_probe(thickness_cm: float, energy_kev: float, n_histories: int,
                          seed: int = 1, warmup_histories: int = 1000) -> ProbeResult:
    """water60_free等のスラブシナリオ（box 1個・鉛筆ビーム垂直入射）を
    `_run_batch_scalar`カーネルで実行し、スループット[histories/s]を計測する。

    B-1a限定の簡略化（モジュールdocstring参照: 背景真空・蛍光無効・単一材料）
    を明示した上で使うこと——`transport_photons`との統計的クロスチェックは
    まだ行っていない。`warmup_histories`は初回JIT compileを計測対象から
    除外するための空撃ち（B-0実測: cache=True初回コンパイルは約0.3秒、
    計測対象に含めると especially小n・大nどちらでもスループットの解釈を誤る）。
    """
    tables = bake_material_tables("water")
    margin = 0.01
    hx, hy, hz = thickness_cm / 2.0, 50.0, 50.0
    whx, why, whz = hx + margin, hy + margin, hz + margin
    ox, oy, oz = -hx - margin, 0.0, 0.0
    ddx, ddy, ddz = 1.0, 0.0, 0.0

    common_args = (energy_kev, ox, oy, oz, ddx, ddy, ddz, hx, hy, hz, whx, why, whz,
                   tables.n_elem, tables.zs, tables.fracs, tables.log_e_grid, tables.step,
                   tables.density_g_cm3, tables.photo_tab, tables.compt_tab, tables.rayl_tab,
                   tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a)

    # JITコンパイル（+ファイルキャッシュ書き込み）を計測対象から除外するための空撃ち。
    _run_batch_scalar(warmup_histories, 0, *common_args)

    import time
    t0 = time.perf_counter()
    n_scatter, absorbed, escaped, final_energy, energy_deposited = _run_batch_scalar(
        n_histories, seed, *common_args)
    wall_s = time.perf_counter() - t0

    uncollided = float(np.sum(escaped & (n_scatter == 0))) / n_histories
    return ProbeResult(
        n_histories=n_histories,
        wall_s=wall_s,
        histories_per_s=n_histories / wall_s,
        uncollided_frac=uncollided,
        fraction_absorbed=float(np.sum(absorbed)) / n_histories,
        fraction_escaped=float(np.sum(escaped)) / n_histories,
        mean_scatter_events=float(np.sum(n_scatter)) / n_histories,
        energy_deposited_keV=float(np.sum(energy_deposited)),
    )
