"""Phase B-1: per-history輸送カーネル（Numba njit、tallyなしパス）。

docs/plan_chatcarlo_speedup_post_egs5.md のPhase B設計に基づく。B-1a（
`chatcarlo/kernel.py`の初版、コミット履歴参照）で「per-historyスカラー
ループをコンパイルすれば1.5M histories/sの目標に届くのか」だけを
最小コスト（water単色・box1個・背景真空・蛍光無効）で確認し、premise確認は
合格だった。本バージョン（B-1b）はその簡略化を全て解消する:

1. **多材料対応**: 材料は`chatcarlo.materials.element_composition`が返す
   任意の元素構成（重元素含む）を、材料コード配列でN個まで焼き込む。
   元素ごとのエネルギー格子は材料間は元より**元素間でも共有を仮定しない**
   （B-1aの「water(H,O)は偶然grid一致」という前提をここでは置かない——
   airのAr(Z=18)は吸収端補強点により2031点の非等間隔格子になり、H/N/O
   (2000点・等間隔)と混在する。`_element_index_frac`が対数等間隔なら算術
   ±1補正、非等間隔ならsearchsorted相当の二分探索にフォールバックする、
   元素単位の判定——`materials._element_interp_index_frac`と同じ設計）。
2. **多形状（box複数）対応**: シーンはbox物体を複数持て、`geometry.Geometry`
   と同じ「リスト後方が優先」規則・世界境界脱出規則で解析面トラッキングする
   （cylinder/sphereはB-1c、計画書参照、未対応）。
3. **K殻蛍光**: `physics.sample_fluorescence`のロジックを、xraylibの
   `CS_Photo_Partial`呼び出し（njit内で不可、B-0で確認済み）をエネルギー格子上に
   事前テーブル化したK殻イオン化分率で置き換えて再現する。
4. **prange並列化＋チャンク単位の決定的シード**: B-0で確定した設計
   （`SeedSequence.spawn`でチャンクごとの整数シードをnjitの外で生成し、
   各prange反復の先頭で`np.random.seed(derived_seed)`する）を実装。
   同一(seed, n_chunks)なら再現するが、n_chunksを変えると別ストリーム分割に
   なるため統計的にのみ同等（`--workers`と同じ制約、意図した設計）。

**乱数はレガシーMT19937（njit内で使える唯一のAPI）、ベクトル化参照実装
（`transport.transport_photons`、PCG64）とはビット一致しない**——Phase Bは
そもそもビット一致を要求せず統計的クロスチェックで検証する設計
（計画書「Phase Bの検証戦略」）。統計的クロスチェック本体は
`docs/speedup_baseline/kernel_crosscheck.py`（層1）。
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
import xraylib
from numba import njit, prange

from .dose_coefficients import h_star_10_per_fluence
from .materials import (_element_energy_grid_kev, _element_xs_tables, density,
                         element_composition, fluorescence_k_data,
                         incoherent_sq_table, mu_en_rho,
                         rayleigh_cumulative_table)
from . import tally, tally_njit
from .tally import VoxelGrid

_MEC2_KEV = 511.0
_HC_KEV_ANGSTROM = 12.3984193
_FLUOR_CUTOFF_KEV = 5.0  # physics.py の _FLUOR_CUTOFF_KEV と同値
_BOUNDARY_EPS = 1e-6
_BOX_EPS = 1e-12
_NUDGE = 1e-6
_MAX_LINES = 8  # physics.sample_fluorescence が抽選する8線（KL2..KP_LINE）と同数
_N_INCOH = 2000  # materials.incoherent_sq_table の既定n（元素によらず一定）
_N_RAYL = 2000   # materials.rayleigh_cumulative_table の既定n（同上）


@functools.lru_cache(maxsize=None)
def _k_shell_table_for_element(z: int):
    """元素zのK殻蛍光データ＋K殻イオン化分率テーブルを、この元素の断面積格子
    （`_element_energy_grid_kev(z)`）上に焼く。

    physics.sample_fluorescence が光電吸収イベントごとにxraylibへライブで
    問い合わせている`CS_Photo_Partial(Z,K,E)/CS_Photo(Z,E)`比を、njit内から
    呼べる形にするため事前計算する（B-0で確認済み: xraylib呼び出しはnjit内で
    失敗する）。lru_cacheで元素ごとに1回だけ計算する（吸収端未満の格子点は
    ゼロのまま——`sample_fluorescence`と同じくE<=edgeでは蛍光を出さない）。
    """
    edge_keV, omega_k, line_energies, line_probs = fluorescence_k_data(z)
    grid = _element_energy_grid_kev(z)
    k_frac = np.zeros_like(grid)
    if omega_k > 0.0 and line_energies.size > 0 and line_energies.max() >= _FLUOR_CUTOFF_KEV:
        for i, ek in enumerate(grid):
            if ek <= edge_keV:
                continue
            try:
                cs_photo = xraylib.CS_Photo(z, float(ek))
                cs_k = xraylib.CS_Photo_Partial(z, xraylib.K_SHELL, float(ek))
            except ValueError:
                continue
            if cs_photo > 0.0:
                k_frac[i] = cs_k / cs_photo
    return edge_keV, omega_k, line_energies, line_probs, k_frac


def _bake_single_material(name: str) -> dict:
    """材料1個分の元素別データを、パディング前のリスト形式で集める。"""
    comp = element_composition(name)
    zs, fracs = [], []
    log_e, photo, compt, rayl = [], [], [], []
    step, n_grid = [], []
    incoh_q, incoh_s, rayl_x, rayl_a = [], [], [], []
    k_edge, k_omega, k_line_e, k_line_p, n_lines, k_frac = [], [], [], [], [], []
    for z, f in comp:
        t = _element_xs_tables(z)
        zs.append(z)
        fracs.append(f)
        log_e.append(t["log_e"])
        photo.append(t["photo"])
        compt.append(t["compt"])
        rayl.append(t["rayl"])
        step.append(t["uniform_step"] if t["uniform_step"] is not None else -1.0)
        n_grid.append(len(t["log_e"]))
        q_grid, s_grid = incoherent_sq_table(z)
        incoh_q.append(q_grid)
        incoh_s.append(s_grid)
        x_grid, a_grid = rayleigh_cumulative_table(z)
        rayl_x.append(x_grid)
        rayl_a.append(a_grid)
        edge, omega, line_e, line_p, kf = _k_shell_table_for_element(z)
        k_edge.append(edge)
        k_omega.append(omega)
        k_line_e.append(line_e)
        k_line_p.append(line_p)
        n_lines.append(len(line_e))
        k_frac.append(kf)
    return dict(n_elem=len(comp), zs=zs, fracs=fracs, log_e=log_e, photo=photo, compt=compt,
                rayl=rayl, step=step, n_grid=n_grid, incoh_q=incoh_q, incoh_s=incoh_s,
                rayl_x=rayl_x, rayl_a=rayl_a, k_edge=k_edge, k_omega=k_omega,
                k_line_e=k_line_e, k_line_p=k_line_p, n_lines=n_lines, k_frac=k_frac,
                density=density(name))


@dataclass
class SceneMaterialTables:
    """N材料分の元素別データを固定形状ndarrayへパディングして焼いたもの。

    材料コードはmaterial_namesのindex（0起点）。全ての(材料,元素)配列は
    (n_materials, max_elem, ...)形状にパディングされ、実際の要素数は
    n_elem[材料]・n_grid[材料,元素]で境界を示す——パディング部分は
    ルックアップのクリップ範囲外なので参照されない。
    """
    material_names: list[str]
    n_elem: np.ndarray        # (n_mat,) int64
    zs: np.ndarray            # (n_mat, max_elem) int64
    fracs: np.ndarray         # (n_mat, max_elem) float64
    log_e: np.ndarray         # (n_mat, max_elem, max_grid) float64
    step: np.ndarray          # (n_mat, max_elem) float64（-1.0なら非等間隔=bisect）
    n_grid: np.ndarray        # (n_mat, max_elem) int64
    photo: np.ndarray         # (n_mat, max_elem, max_grid) float64 log(cs)
    compt: np.ndarray         # 同上
    rayl: np.ndarray          # 同上
    incoh_q: np.ndarray       # (n_mat, max_elem, 2000)
    incoh_s: np.ndarray       # (n_mat, max_elem, 2000)
    rayl_x: np.ndarray        # (n_mat, max_elem, 2000)
    rayl_a: np.ndarray        # (n_mat, max_elem, 2000)
    k_edge: np.ndarray        # (n_mat, max_elem)
    k_omega: np.ndarray       # (n_mat, max_elem)
    k_line_e: np.ndarray      # (n_mat, max_elem, 8)
    k_line_p: np.ndarray      # (n_mat, max_elem, 8)
    n_lines: np.ndarray       # (n_mat, max_elem) int64
    k_frac: np.ndarray        # (n_mat, max_elem, max_grid)
    density_g_cm3: np.ndarray  # (n_mat,)

    def code(self, name: str) -> int:
        return self.material_names.index(name)


def bake_scene_materials(material_names: list[str]) -> SceneMaterialTables:
    """材料名のリストをカーネル用の固定長ndarrayに焼く（重元素・非等間隔格子・
    元素混在も対応、B-1aの「軽元素かつ元素間共有格子」制約を解消）。
    """
    raw = [_bake_single_material(n) for n in material_names]
    n_materials = len(raw)
    max_elem = max(r["n_elem"] for r in raw)
    max_grid = max(len(g) for r in raw for g in r["log_e"])

    n_elem = np.zeros(n_materials, dtype=np.int64)
    zs = np.zeros((n_materials, max_elem), dtype=np.int64)
    fracs = np.zeros((n_materials, max_elem))
    log_e = np.zeros((n_materials, max_elem, max_grid))
    step = np.full((n_materials, max_elem), -1.0)
    n_grid = np.zeros((n_materials, max_elem), dtype=np.int64)
    photo = np.zeros((n_materials, max_elem, max_grid))
    compt = np.zeros((n_materials, max_elem, max_grid))
    rayl = np.zeros((n_materials, max_elem, max_grid))
    incoh_q = np.zeros((n_materials, max_elem, _N_INCOH))
    incoh_s = np.zeros((n_materials, max_elem, _N_INCOH))
    rayl_x = np.zeros((n_materials, max_elem, _N_RAYL))
    rayl_a = np.zeros((n_materials, max_elem, _N_RAYL))
    k_edge = np.zeros((n_materials, max_elem))
    k_omega = np.zeros((n_materials, max_elem))
    k_line_e = np.zeros((n_materials, max_elem, _MAX_LINES))
    k_line_p = np.zeros((n_materials, max_elem, _MAX_LINES))
    n_lines = np.zeros((n_materials, max_elem), dtype=np.int64)
    k_frac = np.zeros((n_materials, max_elem, max_grid))
    density_g_cm3 = np.zeros(n_materials)

    for mi, r in enumerate(raw):
        n_elem[mi] = r["n_elem"]
        density_g_cm3[mi] = r["density"]
        for ei in range(r["n_elem"]):
            g = r["n_grid"][ei]
            zs[mi, ei] = r["zs"][ei]
            fracs[mi, ei] = r["fracs"][ei]
            log_e[mi, ei, :g] = r["log_e"][ei]
            if g < max_grid:
                log_e[mi, ei, g:] = r["log_e"][ei][-1]
            step[mi, ei] = r["step"][ei]
            n_grid[mi, ei] = g
            photo[mi, ei, :g] = r["photo"][ei]
            compt[mi, ei, :g] = r["compt"][ei]
            rayl[mi, ei, :g] = r["rayl"][ei]
            incoh_q[mi, ei, :] = r["incoh_q"][ei]
            incoh_s[mi, ei, :] = r["incoh_s"][ei]
            rayl_x[mi, ei, :] = r["rayl_x"][ei]
            rayl_a[mi, ei, :] = r["rayl_a"][ei]
            k_edge[mi, ei] = r["k_edge"][ei]
            k_omega[mi, ei] = r["k_omega"][ei]
            nl = r["n_lines"][ei]
            n_lines[mi, ei] = nl
            k_line_e[mi, ei, :nl] = r["k_line_e"][ei]
            k_line_p[mi, ei, :nl] = r["k_line_p"][ei]
            k_frac[mi, ei, :g] = r["k_frac"][ei]

    return SceneMaterialTables(
        material_names=list(material_names), n_elem=n_elem, zs=zs, fracs=fracs,
        log_e=log_e, step=step, n_grid=n_grid, photo=photo, compt=compt, rayl=rayl,
        incoh_q=incoh_q, incoh_s=incoh_s, rayl_x=rayl_x, rayl_a=rayl_a,
        k_edge=k_edge, k_omega=k_omega, k_line_e=k_line_e, k_line_p=k_line_p,
        n_lines=n_lines, k_frac=k_frac, density_g_cm3=density_g_cm3,
    )


@dataclass
class SceneGeometry:
    """box物体のリスト（リスト後方優先）＋背景材料＋世界境界（Geometry.bbox_min/max
    と同じ既定margin加算）を、カーネル用の固定形状ndarrayに焼いたもの。
    """
    n_boxes: int
    box_center: np.ndarray     # (n_boxes, 3)
    box_half: np.ndarray       # (n_boxes, 3)
    box_material: np.ndarray   # (n_boxes,) int64 材料コード
    background_material: int
    world_center: np.ndarray   # (3,)
    world_half: np.ndarray     # (3,)


def bake_box_scene(boxes: list[dict], background: str, tables: SceneMaterialTables,
                    bbox_margin_cm: float = 50.0) -> SceneGeometry:
    """`geometry.Geometry`と同じ規則（box物体のみ、cylinder/sphereはB-1c未対応）
    でシーンをカーネル用に焼く。boxes: [{"center":(x,y,z),"size_cm":(sx,sy,sz),
    "material":name}, ...]。
    """
    n = len(boxes)
    centers = np.array([b["center"] for b in boxes], dtype=np.float64)
    halves = np.array([np.asarray(b["size_cm"], dtype=np.float64) / 2.0 for b in boxes])
    mat_codes = np.array([tables.code(b["material"]) for b in boxes], dtype=np.int64)
    los = centers - halves
    his = centers + halves
    bbox_min = los.min(axis=0) - bbox_margin_cm
    bbox_max = his.max(axis=0) + bbox_margin_cm
    world_center = (bbox_min + bbox_max) / 2.0
    world_half = (bbox_max - bbox_min) / 2.0
    return SceneGeometry(
        n_boxes=n, box_center=centers, box_half=halves, box_material=mat_codes,
        background_material=tables.code(background), world_center=world_center,
        world_half=world_half,
    )


@njit(cache=True)
def _lerp_lookup(x_grid, y_grid, x):
    """1本の昇順格子上での二分探索＋線形補間（境界はクランプ）。

    Compton用S(Z,q)テーブル・Rayleigh用累積テーブル（順変換・逆変換とも）で
    共有する唯一のルックアップ実装（手書きバリアントを増やすとそこに符号
    ミスが混入するため、1本にまとめて使い回す）。
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
def _element_index_frac(log_e_row, step, n_grid_actual, log_eq):
    """元素1個分のエネルギー格子上での区分線形補間インデックス・重み。

    `materials._element_interp_index_frac`のスカラー版: 格子が対数等間隔
    （step>0）なら算術±1補正、非等間隔（step<0、重元素の吸収端補強格子）なら
    searchsorted+clip相当の二分探索（`materials._element_interp_index_frac`の
    フォールバック経路と同一の選択則）。`n_grid_actual`はパディング前の実際の
    格子長——パディング領域を探索範囲に含めないための境界（bake_scene_materials
    参照）。
    """
    hi = n_grid_actual - 1
    if step > 0.0:
        idx = int((log_eq - log_e_row[0]) / step) + 1
        if idx < 1:
            idx = 1
        if idx > hi:
            idx = hi
        if log_eq <= log_e_row[idx - 1]:
            idx -= 1
            if idx < 1:
                idx = 1
        if log_e_row[idx] < log_eq:
            idx += 1
            if idx > hi:
                idx = hi
    else:
        lo_i = 0
        hi_i = n_grid_actual
        while lo_i < hi_i:
            mid = (lo_i + hi_i) // 2
            if log_e_row[mid] < log_eq:
                lo_i = mid + 1
            else:
                hi_i = mid
        idx = lo_i
        if idx < 1:
            idx = 1
        if idx > hi:
            idx = hi
    x0 = log_e_row[idx - 1]
    x1 = log_e_row[idx]
    frac = (log_eq - x0) / (x1 - x0)
    return idx, frac


@njit(cache=True)
def _select_element(n_elem, fracs, cs_e, total):
    """元素別断面積(fracs[i]*cs_e[i])に比例する重みで元素indexを1個抽選する。

    `physics.sample_compton_element`/`sample_rayleigh_element`/
    `sample_photo_element`と同じ選択則（質量分率×断面積で規格化した累積分布
    からの逆変換）のスカラー版。
    """
    target = np.random.random() * total
    cum = 0.0
    for i in range(n_elem):
        cum += fracs[i] * cs_e[i]
        if target <= cum:
            return i
    return n_elem - 1


@njit(cache=True)
def _mu_and_parts_scalar(e, mat_idx, n_elem_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr,
                          photo_arr, compt_arr, rayl_arr, density_arr,
                          photo_e, compt_e, rayl_e, idx_e, frac_e):
    """材料mat_idxの透過光子1個の(μ[1/cm], 光電/コンプトン/レイリー質量減弱係数和[cm2/g])。

    元素ごとに独立したエネルギー格子ルックアップを行う（B-1aの「元素間で
    格子共有」という前提をここでは置かない——air中のAr(Z=18)はH/N/O等と
    格子が異なる、上のモジュールdocstring参照）。photo_e/compt_e/rayl_e/
    idx_e/frac_eは呼び出し側が確保した長さmax_elemの作業配列（後段の元素
    抽選・K殻蛍光サンプリングで同じ(idx,frac)を再利用するため書き込んで返す
    ——二重導出を避ける、B-1aレビュー指摘と同じ理由）。
    """
    n_elem = n_elem_arr[mat_idx]
    log_eq = math.log(e)
    tot_photo = 0.0
    tot_compt = 0.0
    tot_rayl = 0.0
    for i in range(n_elem):
        idx, frac = _element_index_frac(log_e_arr[mat_idx, i], step_arr[mat_idx, i],
                                         n_grid_arr[mat_idx, i], log_eq)
        idx_e[i] = idx
        frac_e[i] = frac
        p = math.exp(photo_arr[mat_idx, i, idx - 1]
                      + frac * (photo_arr[mat_idx, i, idx] - photo_arr[mat_idx, i, idx - 1]))
        c = math.exp(compt_arr[mat_idx, i, idx - 1]
                      + frac * (compt_arr[mat_idx, i, idx] - compt_arr[mat_idx, i, idx - 1]))
        r = math.exp(rayl_arr[mat_idx, i, idx - 1]
                      + frac * (rayl_arr[mat_idx, i, idx] - rayl_arr[mat_idx, i, idx - 1]))
        photo_e[i] = p
        compt_e[i] = c
        rayl_e[i] = r
        fr = fracs_arr[mat_idx, i]
        tot_photo += fr * p
        tot_compt += fr * c
        tot_rayl += fr * r
    mu = (tot_photo + tot_compt + tot_rayl) * density_arr[mat_idx]
    return mu, tot_photo, tot_compt, tot_rayl


@njit(cache=True)
def _sample_fluorescence_scalar(mat_idx, elem_i, e_kev, idx, frac,
                                 k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr,
                                 k_line_p_arr, n_lines_arr):
    """physics.sample_fluorescence のスカラー版（K殻蛍光の放出可否・線エネルギー）。

    idx/fracは光電相互作用元素選択のために`_mu_and_parts_scalar`が既に求めた
    この元素のエネルギー格子インデックスをそのまま使う（同じ理由で二重導出
    しない）。K殻イオン化分率`k_frac_arr`は`_k_shell_table_for_element`が
    xraylib.CS_Photo_Partial/CS_Photo比を事前に焼いたテーブル
    （njit内でxraylibを直接呼べないため、B-0で確認済みの制約）。
    """
    edge = k_edge_arr[mat_idx, elem_i]
    if e_kev <= edge:
        return False, 0.0
    omega = k_omega_arr[mat_idx, elem_i]
    if omega <= 0.0:
        return False, 0.0
    kf0 = k_frac_arr[mat_idx, elem_i, idx - 1]
    kf1 = k_frac_arr[mat_idx, elem_i, idx]
    k_frac = kf0 + frac * (kf1 - kf0)
    if np.random.random() >= k_frac:
        return False, 0.0
    if np.random.random() >= omega:
        return False, 0.0
    n = n_lines_arr[mat_idx, elem_i]
    r = np.random.random()
    cum = 0.0
    chosen = n - 1
    for j in range(n):
        cum += k_line_p_arr[mat_idx, elem_i, j]
        if r <= cum:
            chosen = j
            break
    e_line = k_line_e_arr[mat_idx, elem_i, chosen]
    if e_line < _FLUOR_CUTOFF_KEV:
        return False, 0.0
    return True, e_line


@njit(cache=True)
def _sample_compton_bound_scalar(mat_idx, e_kev, n_elem, fracs_arr, zs_arr, incoh_q_arr,
                                  incoh_s_arr, compt_e_work, tot_compt):
    """physics.sample_compton_bound のスカラー版（Kahn型棄却＋S(Z,q)/Z追加棄却）。

    元素選択は呼び出し側で既に求めたcompt_e_work・tot_compt（`_mu_and_parts_scalar`
    が同じエネルギーで既に計算済みの値）をそのまま使う——独立再計算による
    二重導出を避ける（B-1aレビューで訂正した設計をそのまま踏襲）。
    """
    elem_i = _select_element(n_elem, fracs_arr[mat_idx], compt_e_work, tot_compt)
    z_val = float(zs_arr[mat_idx, elem_i])
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
        s_over_z = _lerp_lookup(incoh_q_arr[mat_idx, elem_i], incoh_s_arr[mat_idx, elem_i], q_p) / z_val
        xi3 = np.random.random()
        if xi3 <= s_over_z:
            return eps_p, cos_p, elem_i


@njit(cache=True)
def _sample_rayleigh_cos_theta_scalar(mat_idx, e_kev, n_elem, fracs_arr, rayl_x_arr, rayl_a_arr,
                                       rayl_e_work, tot_rayl):
    """physics.sample_rayleigh_cos_theta のスカラー版（2段階逆変換＋角度棄却）。

    元素選択は`_mu_and_parts_scalar`が同じエネルギーで既に求めたrayl_e_work・
    tot_rayl を再利用する（`_sample_compton_bound_scalar`と同じ理由）。
    """
    elem_i = _select_element(n_elem, fracs_arr[mat_idx], rayl_e_work, tot_rayl)
    x_max = (e_kev / _HC_KEV_ANGSTROM) ** 2
    while True:
        a_cut = _lerp_lookup(rayl_x_arr[mat_idx, elem_i], rayl_a_arr[mat_idx, elem_i], x_max)
        xi1 = np.random.random()
        x_val = _lerp_lookup(rayl_a_arr[mat_idx, elem_i], rayl_x_arr[mat_idx, elem_i], xi1 * a_cut)
        if x_val > x_max:
            x_val = x_max
        cos_c = 1.0 - 2.0 * x_val / x_max
        if cos_c > 1.0:
            cos_c = 1.0
        if cos_c < -1.0:
            cos_c = -1.0
        xi2 = np.random.random()
        if xi2 <= (1.0 + cos_c * cos_c) / 2.0:
            return cos_c, elem_i


@njit(cache=True)
def _scatter_direction_scalar(dx, dy, dz, cos_theta):
    """physics.scatter_direction のスカラー版。方位角は一様抽選。"""
    sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    phi = np.random.random() * 2.0 * math.pi
    if abs(dz) < 0.999:
        upx, upy, upz = 0.0, 0.0, 1.0
    else:
        upx, upy, upz = 1.0, 0.0, 0.0
    ux = upy * dz - upz * dy
    uy = upz * dx - upx * dz
    uz = upx * dy - upy * dx
    un = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux /= un
    uy /= un
    uz /= un
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
def _isotropic_direction_scalar():
    """physics.isotropic_direction のスカラー版（蛍光X線の放出方向）。"""
    cos_theta = np.random.random() * 2.0 - 1.0
    sin_theta = math.sqrt(max(1.0 - cos_theta * cos_theta, 0.0))
    phi = np.random.random() * 2.0 * math.pi
    return sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta


@njit(cache=True)
def _intersect_box_scalar(ox, oy, oz, dx, dy, dz, hx, hy, hz):
    """box中心を原点とした相対座標(ox,oy,oz)でのbox交差（entry,exit,hit）。

    `geometry._intersect_box`のスカラー版（3軸のスラブ交差判定）。
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
def _material_at_scalar(x, y, z, n_boxes, box_center, box_half, box_material, background_code):
    """`geometry.Geometry.material_at`のスカラー版（リスト後方優先、既定background）。"""
    mat = background_code
    for bi in range(n_boxes):
        cx = box_center[bi, 0]
        cy = box_center[bi, 1]
        cz = box_center[bi, 2]
        hx = box_half[bi, 0]
        hy = box_half[bi, 1]
        hz = box_half[bi, 2]
        if abs(x - cx) <= hx and abs(y - cy) <= hy and abs(z - cz) <= hz:
            mat = box_material[bi]
    return mat


@njit(cache=True)
def _next_boundary_scalar(x, y, z, dx, dy, dz, n_boxes, box_center, box_half,
                           world_center, world_half):
    """`geometry.Geometry.next_boundary`のスカラー版（box複数対応）。"""
    t_obj = np.inf
    for bi in range(n_boxes):
        cx = box_center[bi, 0]
        cy = box_center[bi, 1]
        cz = box_center[bi, 2]
        hx = box_half[bi, 0]
        hy = box_half[bi, 1]
        hz = box_half[bi, 2]
        t_enter, t_exit, hit = _intersect_box_scalar(x - cx, y - cy, z - cz, dx, dy, dz, hx, hy, hz)
        if hit:
            if t_enter > _BOUNDARY_EPS and t_enter < t_obj:
                t_obj = t_enter
            if t_exit > _BOUNDARY_EPS and t_exit < t_obj:
                t_obj = t_exit

    _, t_exit_w, hit_w = _intersect_box_scalar(
        x - world_center[0], y - world_center[1], z - world_center[2],
        dx, dy, dz, world_half[0], world_half[1], world_half[2])
    t_exit_world = t_exit_w if hit_w else np.inf

    escape = t_exit_world <= t_obj
    t = min(t_obj, t_exit_world)
    return t, escape


@njit(cache=True)
def _transport_one(energy0_kev, ox, oy, oz, dx, dy, dz,
                    n_boxes, box_center, box_half, box_material, background_material,
                    world_center, world_half,
                    n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                    photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                    max_elem, fluorescence_enabled):
    """1光子を`transport_photons`(transport.py)の主ループと同一アルゴリズムで
    吸収/脱出まで追跡する。box複数・材料複数・K殻蛍光対応（B-1b）。

    戻り値: (n_scatter, absorbed, escaped, final_energy_kev, energy_deposited_kev,
    n_fluorescence)。
    """
    x, y, z = ox, oy, oz
    dirx, diry, dirz = dx, dy, dz
    e = energy0_kev
    tau = -math.log(np.random.random())
    n_scatter = 0
    n_fluorescence = 0
    energy_deposited = 0.0
    photo_e = np.empty(max_elem)
    compt_e = np.empty(max_elem)
    rayl_e = np.empty(max_elem)
    idx_e = np.empty(max_elem, dtype=np.int64)
    frac_e = np.empty(max_elem)

    while True:
        mat_idx = _material_at_scalar(x, y, z, n_boxes, box_center, box_half,
                                       box_material, background_material)
        mu, tot_photo, tot_compt, tot_rayl = _mu_and_parts_scalar(
            e, mat_idx, n_elem_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr,
            photo_arr, compt_arr, rayl_arr, density_arr,
            photo_e, compt_e, rayl_e, idx_e, frac_e)

        t_boundary, escape = _next_boundary_scalar(x, y, z, dirx, diry, dirz,
                                                    n_boxes, box_center, box_half,
                                                    world_center, world_half)
        mu_safe = mu if mu > 0.0 else 1e-30
        tau_to_boundary = mu * t_boundary
        will_interact = tau < tau_to_boundary

        if will_interact:
            ds = tau / mu_safe
        else:
            ds = t_boundary
        x += dirx * ds
        y += diry * ds
        z += dirz * ds

        if not will_interact:
            tau -= tau_to_boundary
            x += dirx * _NUDGE
            y += diry * _NUDGE
            z += dirz * _NUDGE
            if escape:
                return n_scatter, False, True, e, energy_deposited, n_fluorescence
            continue

        n_scatter += 1
        r_type = np.random.random()
        tot = tot_photo + tot_compt + tot_rayl
        p_photo = tot_photo / tot
        p_compt = tot_compt / tot

        if r_type < p_photo:
            n_elem = n_elem_arr[mat_idx]
            elem_i = _select_element(n_elem, fracs_arr[mat_idx], photo_e, tot_photo)
            emit = False
            e_line = 0.0
            if fluorescence_enabled:
                emit, e_line = _sample_fluorescence_scalar(
                    mat_idx, elem_i, e, idx_e[elem_i], frac_e[elem_i],
                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr)
            if emit:
                energy_deposited += e - e_line
                n_fluorescence += 1
                dirx, diry, dirz = _isotropic_direction_scalar()
                e = e_line
                tau = -math.log(np.random.random())
            else:
                energy_deposited += e
                return n_scatter, True, False, e, energy_deposited, n_fluorescence
        elif r_type < p_photo + p_compt:
            eps_c, cos_c, _elem = _sample_compton_bound_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, zs_arr, incoh_q_arr, incoh_s_arr,
                compt_e, tot_compt)
            e_new = e * eps_c
            energy_deposited += e - e_new
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            e = e_new
            tau = -math.log(np.random.random())
        else:
            cos_c, _elem = _sample_rayleigh_cos_theta_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, rayl_x_arr, rayl_a_arr, rayl_e, tot_rayl)
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            tau = -math.log(np.random.random())
        # 光電(蛍光放出時)・コンプトン・レイリーいずれも光子が生き残るので
        # returnせずループを継続する。


@njit(cache=True)
def _transport_one_tally(energy0_kev, ox, oy, oz, dx, dy, dz,
                          n_boxes, box_center, box_half, box_material, background_material,
                          world_center, world_half,
                          n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                          photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                          k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                          max_elem, fluorescence_enabled,
                          seg_o, seg_d, seg_ds, seg_e, seg_mat, seg_start, seg_capacity):
    """`_transport_one`と物理的に同一のアルゴリズムだが、飛行区間(o, d, ds, e, mat)を
    B-2のタリー統合用に呼び出し側バッファへ書き出す（B-2設計(b): カーネルは区間を
    吐き出すだけで、線量換算・グリッド分配は既存の監査済み`tally.accumulate_track_length_multi`
    に委ねる——計画書「B-2: タリー統合」参照）。

    区間の記録はRNGを一切消費しない副作用であり物理には影響しない（`transport_photons`の
    dose_grid引数と同じ性質）。`_transport_one`をそのまま再利用せずここに複製したのは、
    タリーなし経路（B-1bで三層検証済み）を一切変更しないため——両者の物理ロジックが
    将来ズレないことは`tests/test_kernel.py`の`test_tally_variant_matches_reference_variant`
    で同一seed・同一シナリオの輸送結果（区間を除く全戻り値）が完全一致することを検証する。

    seg_o/seg_d/seg_ds/seg_e/seg_matはこのチャンク専用の作業配列（呼び出し側が
    `_run_batch_scalar_tally`でチャンクごとに切り出して渡す、他チャンクと共有しない
    ためprangeでのデータ競合が原理的に起きない設計）。seg_startは呼び出し時点での
    書き込み位置、戻り値のseg_countはこの履歴で書き込んだ後の位置。容量
    （seg_capacity、`max_segments_per_history`から呼び出し側が見積もる）を超えたら
    それ以降の区間は記録せず（輸送自体は継続する）overflow=Trueを返す——黙って
    タリーを欠落させるのではなく、呼び出し側で検知してエラーにするための明示的な
    信号（`docs/lessons_learned.md`の「黙って一部を捨てるな」系の教訓と同じ設計判断）。
    """
    x, y, z = ox, oy, oz
    dirx, diry, dirz = dx, dy, dz
    e = energy0_kev
    tau = -math.log(np.random.random())
    n_scatter = 0
    n_fluorescence = 0
    energy_deposited = 0.0
    photo_e = np.empty(max_elem)
    compt_e = np.empty(max_elem)
    rayl_e = np.empty(max_elem)
    idx_e = np.empty(max_elem, dtype=np.int64)
    frac_e = np.empty(max_elem)
    seg_idx = seg_start
    overflow = False

    while True:
        mat_idx = _material_at_scalar(x, y, z, n_boxes, box_center, box_half,
                                       box_material, background_material)
        mu, tot_photo, tot_compt, tot_rayl = _mu_and_parts_scalar(
            e, mat_idx, n_elem_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr,
            photo_arr, compt_arr, rayl_arr, density_arr,
            photo_e, compt_e, rayl_e, idx_e, frac_e)

        t_boundary, escape = _next_boundary_scalar(x, y, z, dirx, diry, dirz,
                                                    n_boxes, box_center, box_half,
                                                    world_center, world_half)
        mu_safe = mu if mu > 0.0 else 1e-30
        tau_to_boundary = mu * t_boundary
        will_interact = tau < tau_to_boundary

        if will_interact:
            ds = tau / mu_safe
        else:
            ds = t_boundary

        if ds > 0.0:
            if seg_idx < seg_capacity:
                seg_o[seg_idx, 0] = x
                seg_o[seg_idx, 1] = y
                seg_o[seg_idx, 2] = z
                seg_d[seg_idx, 0] = dirx
                seg_d[seg_idx, 1] = diry
                seg_d[seg_idx, 2] = dirz
                seg_ds[seg_idx] = ds
                seg_e[seg_idx] = e
                seg_mat[seg_idx] = mat_idx
                seg_idx += 1
            else:
                overflow = True

        x += dirx * ds
        y += diry * ds
        z += dirz * ds

        if not will_interact:
            tau -= tau_to_boundary
            x += dirx * _NUDGE
            y += diry * _NUDGE
            z += dirz * _NUDGE
            if escape:
                return (n_scatter, False, True, e, energy_deposited, n_fluorescence,
                        seg_idx, overflow)
            continue

        n_scatter += 1
        r_type = np.random.random()
        tot = tot_photo + tot_compt + tot_rayl
        p_photo = tot_photo / tot
        p_compt = tot_compt / tot

        if r_type < p_photo:
            n_elem = n_elem_arr[mat_idx]
            elem_i = _select_element(n_elem, fracs_arr[mat_idx], photo_e, tot_photo)
            emit = False
            e_line = 0.0
            if fluorescence_enabled:
                emit, e_line = _sample_fluorescence_scalar(
                    mat_idx, elem_i, e, idx_e[elem_i], frac_e[elem_i],
                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr)
            if emit:
                energy_deposited += e - e_line
                n_fluorescence += 1
                dirx, diry, dirz = _isotropic_direction_scalar()
                e = e_line
                tau = -math.log(np.random.random())
            else:
                energy_deposited += e
                return (n_scatter, True, False, e, energy_deposited, n_fluorescence,
                        seg_idx, overflow)
        elif r_type < p_photo + p_compt:
            eps_c, cos_c, _elem = _sample_compton_bound_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, zs_arr, incoh_q_arr, incoh_s_arr,
                compt_e, tot_compt)
            e_new = e * eps_c
            energy_deposited += e - e_new
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            e = e_new
            tau = -math.log(np.random.random())
        else:
            cos_c, _elem = _sample_rayleigh_cos_theta_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, rayl_x_arr, rayl_a_arr, rayl_e, tot_rayl)
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            tau = -math.log(np.random.random())
        # 光電(蛍光放出時)・コンプトン・レイリーいずれも光子が生き残るので
        # returnせずループを継続する。


@njit(cache=True, parallel=True)
def _run_batch_scalar(n_histories, n_chunks, chunk_seeds, chunk_offsets, chunk_counts,
                       energy0_kev, ox, oy, oz, dx, dy, dz,
                       n_boxes, box_center, box_half, box_material, background_material,
                       world_center, world_half,
                       n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                       photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                       k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                       max_elem, fluorescence_enabled):
    """`_transport_one`をn_histories回、n_chunks個のチャンクにprange分割して回す。

    乱数再現性設計（B-0で確定）: `SeedSequence.spawn`で生成した決定的な
    整数シード(chunk_seeds)を、各チャンク（=各prangeスレッド反復）の先頭で
    `np.random.seed()`する。同一(seed, n_chunks)の組では再現するが、n_chunksを
    変えるとチャンク分割自体が変わるため他のn_chunksとはビット一致しない
    （既存`--workers`と同じ制約、計画書参照）。n_chunks=1でシングルスレッド
    実行になる（parallel=True自体のスレッド起動オーバーヘッドはprange範囲が
    1個のときも僅かに残るため、B-1aのシングルスレッド専用ループとの比較は
    別途行う）。
    """
    n_scatter = np.zeros(n_histories, dtype=np.int64)
    absorbed = np.zeros(n_histories, dtype=np.bool_)
    escaped = np.zeros(n_histories, dtype=np.bool_)
    final_energy = np.zeros(n_histories, dtype=np.float64)
    energy_deposited = np.zeros(n_histories, dtype=np.float64)
    n_fluorescence = np.zeros(n_histories, dtype=np.int64)

    for c in prange(n_chunks):
        np.random.seed(chunk_seeds[c])
        start = chunk_offsets[c]
        cnt = chunk_counts[c]
        for k in range(cnt):
            i = start + k
            ns, ab, es, fe, ed, nf = _transport_one(
                energy0_kev, ox, oy, oz, dx, dy, dz,
                n_boxes, box_center, box_half, box_material, background_material,
                world_center, world_half,
                n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                max_elem, fluorescence_enabled)
            n_scatter[i] = ns
            absorbed[i] = ab
            escaped[i] = es
            final_energy[i] = fe
            energy_deposited[i] = ed
            n_fluorescence[i] = nf
    return n_scatter, absorbed, escaped, final_energy, energy_deposited, n_fluorescence


@njit(cache=True, parallel=True)
def _run_batch_scalar_tally(n_histories, n_chunks, chunk_seeds, chunk_offsets, chunk_counts,
                             energy0_kev, ox, oy, oz, dx, dy, dz,
                             n_boxes, box_center, box_half, box_material, background_material,
                             world_center, world_half,
                             n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                             photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                             k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                             max_elem, fluorescence_enabled,
                             seg_o, seg_d, seg_ds, seg_e, seg_mat, seg_capacity_per_chunk):
    """`_run_batch_scalar`のB-2版: 各チャンクが自分専用の区間バッファ
    `seg_o[c]`/`seg_d[c]`/`seg_ds[c]`/`seg_e[c]`/`seg_mat[c]`（形状
    (n_chunks, seg_capacity_per_chunk, ...)）だけに書き込むため、prangeの
    複数チャンクが同時に走ってもデータ競合が起きない（共有カウンタへの
    アトミック加算が不要な設計——B-1bの乱数チャンク分割と同じ「チャンクは
    互いに完全独立」という原則をタリー用バッファにも適用した）。

    戻り値に各チャンクの実際の書き込み件数(seg_count_per_chunk)とオーバーフロー
    フラグ(seg_overflow_per_chunk)を含む——呼び出し側(`run_batch_with_tally`)が
    これを見て有効な区間だけを`accumulate_track_length_multi`に渡し、
    オーバーフローがあれば明示的にエラーにする。
    """
    n_scatter = np.zeros(n_histories, dtype=np.int64)
    absorbed = np.zeros(n_histories, dtype=np.bool_)
    escaped = np.zeros(n_histories, dtype=np.bool_)
    final_energy = np.zeros(n_histories, dtype=np.float64)
    energy_deposited = np.zeros(n_histories, dtype=np.float64)
    n_fluorescence = np.zeros(n_histories, dtype=np.int64)
    seg_count_per_chunk = np.zeros(n_chunks, dtype=np.int64)
    seg_overflow_per_chunk = np.zeros(n_chunks, dtype=np.bool_)

    for c in prange(n_chunks):
        np.random.seed(chunk_seeds[c])
        start = chunk_offsets[c]
        cnt = chunk_counts[c]
        seg_idx = 0
        overflow_c = False
        for k in range(cnt):
            i = start + k
            (ns, ab, es, fe, ed, nf, seg_idx, ov) = _transport_one_tally(
                energy0_kev, ox, oy, oz, dx, dy, dz,
                n_boxes, box_center, box_half, box_material, background_material,
                world_center, world_half,
                n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                max_elem, fluorescence_enabled,
                seg_o[c], seg_d[c], seg_ds[c], seg_e[c], seg_mat[c], seg_idx, seg_capacity_per_chunk)
            n_scatter[i] = ns
            absorbed[i] = ab
            escaped[i] = es
            final_energy[i] = fe
            energy_deposited[i] = ed
            n_fluorescence[i] = nf
            overflow_c = overflow_c or ov
        seg_count_per_chunk[c] = seg_idx
        seg_overflow_per_chunk[c] = overflow_c
    return (n_scatter, absorbed, escaped, final_energy, energy_deposited, n_fluorescence,
            seg_count_per_chunk, seg_overflow_per_chunk)


@njit(cache=True)
def _transport_one_origins(energy0_kev, origins, history_index, dx, dy, dz,
                           n_boxes, box_center, box_half, box_material, background_material,
                           world_center, world_half,
                           n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                           photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                           k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                           max_elem, fluorescence_enabled, material_energy):
    """CLI adapter variant of `_transport_one` for per-history origins.

    `material_energy` belongs exclusively to the enclosing prange chunk.
    """
    x, y, z = origins[history_index, 0], origins[history_index, 1], origins[history_index, 2]
    dirx, diry, dirz, e = dx, dy, dz, energy0_kev
    tau = -math.log(np.random.random())
    n_scatter = 0
    n_fluorescence = 0
    photo_e = np.empty(max_elem)
    compt_e = np.empty(max_elem)
    rayl_e = np.empty(max_elem)
    idx_e = np.empty(max_elem, dtype=np.int64)
    frac_e = np.empty(max_elem)
    while True:
        mat_idx = _material_at_scalar(x, y, z, n_boxes, box_center, box_half, box_material, background_material)
        mu, tot_photo, tot_compt, tot_rayl = _mu_and_parts_scalar(
            e, mat_idx, n_elem_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr,
            photo_arr, compt_arr, rayl_arr, density_arr, photo_e, compt_e, rayl_e, idx_e, frac_e)
        t_boundary, escape = _next_boundary_scalar(x, y, z, dirx, diry, dirz, n_boxes, box_center, box_half, world_center, world_half)
        mu_safe = mu if mu > 0.0 else 1e-30
        tau_to_boundary = mu * t_boundary
        will_interact = tau < tau_to_boundary
        ds = tau / mu_safe if will_interact else t_boundary
        x += dirx * ds
        y += diry * ds
        z += dirz * ds
        if not will_interact:
            tau -= tau_to_boundary
            x += dirx * _NUDGE
            y += diry * _NUDGE
            z += dirz * _NUDGE
            if escape:
                return n_scatter, False, True, e, n_fluorescence
            continue
        n_scatter += 1
        r_type = np.random.random()
        tot = tot_photo + tot_compt + tot_rayl
        p_photo = tot_photo / tot
        p_compt = tot_compt / tot
        if r_type < p_photo:
            n_elem = n_elem_arr[mat_idx]
            elem_i = _select_element(n_elem, fracs_arr[mat_idx], photo_e, tot_photo)
            emit = False
            e_line = 0.0
            if fluorescence_enabled:
                emit, e_line = _sample_fluorescence_scalar(
                    mat_idx, elem_i, e, idx_e[elem_i], frac_e[elem_i],
                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr)
            if emit:
                material_energy[mat_idx] += e - e_line
                n_fluorescence += 1
                dirx, diry, dirz = _isotropic_direction_scalar()
                e = e_line
                tau = -math.log(np.random.random())
            else:
                material_energy[mat_idx] += e
                return n_scatter, True, False, e, n_fluorescence
        elif r_type < p_photo + p_compt:
            eps_c, cos_c, _elem = _sample_compton_bound_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, zs_arr, incoh_q_arr, incoh_s_arr, compt_e, tot_compt)
            e_new = e * eps_c
            material_energy[mat_idx] += e - e_new
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            e = e_new
            tau = -math.log(np.random.random())
        else:
            cos_c, _elem = _sample_rayleigh_cos_theta_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, rayl_x_arr, rayl_a_arr, rayl_e, tot_rayl)
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            tau = -math.log(np.random.random())


@njit(cache=True, parallel=True)
def _run_batch_scalar_origins(n_histories, n_chunks, chunk_seeds, chunk_offsets, chunk_counts,
                              energy0_kev, origins, dx, dy, dz,
                              n_boxes, box_center, box_half, box_material, background_material,
                              world_center, world_half,
                              n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                              photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                              k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                              max_elem, fluorescence_enabled, material_energy_per_chunk):
    n_scatter = np.zeros(n_histories, dtype=np.int64)
    absorbed = np.zeros(n_histories, dtype=np.bool_)
    escaped = np.zeros(n_histories, dtype=np.bool_)
    final_energy = np.zeros(n_histories, dtype=np.float64)
    n_fluorescence = np.zeros(n_histories, dtype=np.int64)
    for c in prange(n_chunks):
        np.random.seed(chunk_seeds[c])
        for k in range(chunk_counts[c]):
            i = chunk_offsets[c] + k
            ns, ab, es, fe, nf = _transport_one_origins(
                energy0_kev, origins, i, dx, dy, dz, n_boxes, box_center, box_half, box_material, background_material,
                world_center, world_half, n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                max_elem, fluorescence_enabled, material_energy_per_chunk[c])
            n_scatter[i], absorbed[i], escaped[i], final_energy[i], n_fluorescence[i] = ns, ab, es, fe, nf
    return n_scatter, absorbed, escaped, final_energy, n_fluorescence


@dataclass
class KernelOriginBatchResult:
    n_scatter: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    final_energy: np.ndarray
    energy_deposited_by_material: np.ndarray
    n_fluorescence: np.ndarray


def run_batch_origins(tables: SceneMaterialTables, geom: SceneGeometry, energy0_kev: float,
                      origins: np.ndarray, direction: tuple[float, float, float], seed: int,
                      n_chunks: int = 1, fluorescence_enabled: bool = True) -> KernelOriginBatchResult:
    """CLI-only origin-array adapter; leaves the established public batch APIs untouched."""
    n_histories = len(origins)
    chunk_seeds, chunk_offsets, chunk_counts = _chunk_plan(n_histories, n_chunks, seed)
    per_chunk = np.zeros((len(chunk_seeds), len(tables.material_names)), dtype=np.float64)
    values = _run_batch_scalar_origins(
        n_histories, len(chunk_seeds), chunk_seeds, chunk_offsets, chunk_counts,
        energy0_kev, np.asarray(origins, dtype=np.float64), direction[0], direction[1], direction[2],
        geom.n_boxes, geom.box_center, geom.box_half, geom.box_material, geom.background_material,
        geom.world_center, geom.world_half,
        tables.n_elem, tables.zs, tables.fracs, tables.log_e, tables.step, tables.n_grid, tables.density_g_cm3,
        tables.photo, tables.compt, tables.rayl, tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a,
        tables.k_edge, tables.k_omega, tables.k_frac, tables.k_line_e, tables.k_line_p, tables.n_lines,
        tables.zs.shape[1], fluorescence_enabled, per_chunk)
    return KernelOriginBatchResult(*values[:4], per_chunk.sum(axis=0), values[4])


@njit(cache=True)
def _transport_one_tally_origins(energy0_kev, origins, history_index, dx, dy, dz,
                                 n_boxes, box_center, box_half, box_material, background_material,
                                 world_center, world_half,
                                 n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                                 photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                                 k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                                 max_elem, fluorescence_enabled, material_energy,
                                 seg_o, seg_d, seg_ds, seg_e, seg_mat, seg_start, seg_capacity):
    x, y, z = origins[history_index, 0], origins[history_index, 1], origins[history_index, 2]
    dirx, diry, dirz, e = dx, dy, dz, energy0_kev
    tau = -math.log(np.random.random())
    n_scatter = 0
    n_fluorescence = 0
    photo_e = np.empty(max_elem); compt_e = np.empty(max_elem); rayl_e = np.empty(max_elem)
    idx_e = np.empty(max_elem, dtype=np.int64); frac_e = np.empty(max_elem)
    seg_idx = seg_start
    overflow = False
    while True:
        mat_idx = _material_at_scalar(x, y, z, n_boxes, box_center, box_half, box_material, background_material)
        mu, tot_photo, tot_compt, tot_rayl = _mu_and_parts_scalar(
            e, mat_idx, n_elem_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr,
            photo_arr, compt_arr, rayl_arr, density_arr, photo_e, compt_e, rayl_e, idx_e, frac_e)
        t_boundary, escape = _next_boundary_scalar(x, y, z, dirx, diry, dirz, n_boxes, box_center, box_half, world_center, world_half)
        mu_safe = mu if mu > 0.0 else 1e-30
        tau_to_boundary = mu * t_boundary
        will_interact = tau < tau_to_boundary
        ds = tau / mu_safe if will_interact else t_boundary
        if ds > 0.0:
            if seg_idx < seg_capacity:
                seg_o[seg_idx, 0], seg_o[seg_idx, 1], seg_o[seg_idx, 2] = x, y, z
                seg_d[seg_idx, 0], seg_d[seg_idx, 1], seg_d[seg_idx, 2] = dirx, diry, dirz
                seg_ds[seg_idx], seg_e[seg_idx], seg_mat[seg_idx] = ds, e, mat_idx
                seg_idx += 1
            else:
                overflow = True
        x += dirx * ds; y += diry * ds; z += dirz * ds
        if not will_interact:
            tau -= tau_to_boundary
            x += dirx * _NUDGE; y += diry * _NUDGE; z += dirz * _NUDGE
            if escape:
                return n_scatter, False, True, e, n_fluorescence, seg_idx, overflow
            continue
        n_scatter += 1
        r_type = np.random.random()
        tot = tot_photo + tot_compt + tot_rayl
        p_photo, p_compt = tot_photo / tot, tot_compt / tot
        if r_type < p_photo:
            elem_i = _select_element(n_elem_arr[mat_idx], fracs_arr[mat_idx], photo_e, tot_photo)
            emit = False; e_line = 0.0
            if fluorescence_enabled:
                emit, e_line = _sample_fluorescence_scalar(
                    mat_idx, elem_i, e, idx_e[elem_i], frac_e[elem_i],
                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr)
            if emit:
                material_energy[mat_idx] += e - e_line
                n_fluorescence += 1
                dirx, diry, dirz = _isotropic_direction_scalar()
                e = e_line; tau = -math.log(np.random.random())
            else:
                material_energy[mat_idx] += e
                return n_scatter, True, False, e, n_fluorescence, seg_idx, overflow
        elif r_type < p_photo + p_compt:
            eps_c, cos_c, _elem = _sample_compton_bound_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, zs_arr, incoh_q_arr, incoh_s_arr, compt_e, tot_compt)
            e_new = e * eps_c
            material_energy[mat_idx] += e - e_new
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            e = e_new; tau = -math.log(np.random.random())
        else:
            cos_c, _elem = _sample_rayleigh_cos_theta_scalar(
                mat_idx, e, n_elem_arr[mat_idx], fracs_arr, rayl_x_arr, rayl_a_arr, rayl_e, tot_rayl)
            dirx, diry, dirz = _scatter_direction_scalar(dirx, diry, dirz, cos_c)
            tau = -math.log(np.random.random())


@njit(cache=True, parallel=True)
def _run_batch_scalar_tally_origins(n_histories, n_chunks, chunk_seeds, chunk_offsets, chunk_counts,
                                    energy0_kev, origins, dx, dy, dz,
                                    n_boxes, box_center, box_half, box_material, background_material,
                                    world_center, world_half,
                                    n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                                    photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                                    k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                                    max_elem, fluorescence_enabled, material_energy_per_chunk,
                                    seg_o, seg_d, seg_ds, seg_e, seg_mat, seg_capacity_per_chunk):
    n_scatter = np.zeros(n_histories, dtype=np.int64); absorbed = np.zeros(n_histories, dtype=np.bool_)
    escaped = np.zeros(n_histories, dtype=np.bool_); final_energy = np.zeros(n_histories, dtype=np.float64)
    n_fluorescence = np.zeros(n_histories, dtype=np.int64)
    seg_count_per_chunk = np.zeros(n_chunks, dtype=np.int64); seg_overflow_per_chunk = np.zeros(n_chunks, dtype=np.bool_)
    for c in prange(n_chunks):
        np.random.seed(chunk_seeds[c]); seg_idx = 0; overflow_c = False
        for k in range(chunk_counts[c]):
            i = chunk_offsets[c] + k
            ns, ab, es, fe, nf, seg_idx, ov = _transport_one_tally_origins(
                energy0_kev, origins, i, dx, dy, dz, n_boxes, box_center, box_half, box_material, background_material,
                world_center, world_half, n_elem_arr, zs_arr, fracs_arr, log_e_arr, step_arr, n_grid_arr, density_arr,
                photo_arr, compt_arr, rayl_arr, incoh_q_arr, incoh_s_arr, rayl_x_arr, rayl_a_arr,
                k_edge_arr, k_omega_arr, k_frac_arr, k_line_e_arr, k_line_p_arr, n_lines_arr,
                max_elem, fluorescence_enabled, material_energy_per_chunk[c],
                seg_o[c], seg_d[c], seg_ds[c], seg_e[c], seg_mat[c], seg_idx, seg_capacity_per_chunk)
            n_scatter[i], absorbed[i], escaped[i], final_energy[i], n_fluorescence[i] = ns, ab, es, fe, nf
            overflow_c = overflow_c or ov
        seg_count_per_chunk[c], seg_overflow_per_chunk[c] = seg_idx, overflow_c
    return n_scatter, absorbed, escaped, final_energy, n_fluorescence, seg_count_per_chunk, seg_overflow_per_chunk


def run_batch_with_tally_origins(tables: SceneMaterialTables, geom: SceneGeometry, energy0_kev: float,
                                 origins: np.ndarray, direction: tuple[float, float, float], seed: int,
                                 grid: VoxelGrid, n_chunks: int = 1, fluorescence_enabled: bool = True,
                                 max_segments_per_history: int = 16) -> KernelOriginBatchResult:
    n_histories = len(origins)
    chunk_seeds, chunk_offsets, chunk_counts = _chunk_plan(n_histories, n_chunks, seed)
    n_chunks_actual = len(chunk_seeds)
    capacity = int(chunk_counts.max()) * max_segments_per_history
    seg_o = np.zeros((n_chunks_actual, capacity, 3)); seg_d = np.zeros((n_chunks_actual, capacity, 3))
    seg_ds = np.zeros((n_chunks_actual, capacity)); seg_e = np.zeros((n_chunks_actual, capacity))
    seg_mat = np.zeros((n_chunks_actual, capacity), dtype=np.int64)
    per_chunk = np.zeros((n_chunks_actual, len(tables.material_names)))
    values = _run_batch_scalar_tally_origins(
        n_histories, n_chunks_actual, chunk_seeds, chunk_offsets, chunk_counts, energy0_kev,
        np.asarray(origins, dtype=np.float64), direction[0], direction[1], direction[2],
        geom.n_boxes, geom.box_center, geom.box_half, geom.box_material, geom.background_material, geom.world_center, geom.world_half,
        tables.n_elem, tables.zs, tables.fracs, tables.log_e, tables.step, tables.n_grid, tables.density_g_cm3,
        tables.photo, tables.compt, tables.rayl, tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a,
        tables.k_edge, tables.k_omega, tables.k_frac, tables.k_line_e, tables.k_line_p, tables.n_lines,
        tables.zs.shape[1], fluorescence_enabled, per_chunk, seg_o, seg_d, seg_ds, seg_e, seg_mat, capacity)
    n_scatter, absorbed, escaped, final_energy, n_fluorescence, seg_count, overflow = values
    if np.any(overflow):
        raise ValueError("タリー用の区間バッファが不足しました")
    parts = [int(seg_count[c]) for c in range(n_chunks_actual)]
    if sum(parts):
        o = np.concatenate([seg_o[c, :parts[c]] for c in range(n_chunks_actual) if parts[c]])
        d = np.concatenate([seg_d[c, :parts[c]] for c in range(n_chunks_actual) if parts[c]])
        ds = np.concatenate([seg_ds[c, :parts[c]] for c in range(n_chunks_actual) if parts[c]])
        e = np.concatenate([seg_e[c, :parts[c]] for c in range(n_chunks_actual) if parts[c]])
        m = np.concatenate([seg_mat[c, :parts[c]] for c in range(n_chunks_actual) if parts[c]])
        kerma_weights, h10_weights = _compute_tally_weights(tables, m, e)
        tally_njit.accumulate_track_length_multi_njit(((grid.kerma_keV, kerma_weights), (grid.h10_track_pSv_cm3, h10_weights)), grid, o, d, ds)
    return KernelOriginBatchResult(n_scatter, absorbed, escaped, final_energy, per_chunk.sum(axis=0), n_fluorescence)


def _chunk_plan(n_histories: int, n_chunks: int, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n_historiesをn_chunks個にできるだけ均等分割し、チャンクごとの決定的な
    整数シードを`SeedSequence.spawn`で生成する（njitの外、B-0で確定した設計）。
    """
    n_chunks = max(1, min(n_chunks, n_histories))
    counts = np.full(n_chunks, n_histories // n_chunks, dtype=np.int64)
    counts[: n_histories % n_chunks] += 1
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    children = np.random.SeedSequence(seed).spawn(n_chunks)
    seeds = np.array([int(c.generate_state(1)[0]) for c in children], dtype=np.int64)
    return seeds, offsets, counts


@dataclass
class KernelBatchResult:
    n_scatter: np.ndarray
    absorbed: np.ndarray
    escaped: np.ndarray
    final_energy: np.ndarray
    energy_deposited: np.ndarray
    n_fluorescence: np.ndarray


def run_batch(tables: SceneMaterialTables, geom: SceneGeometry, energy0_kev: float,
              origin: tuple[float, float, float], direction: tuple[float, float, float],
              n_histories: int, seed: int, n_chunks: int = 1,
              fluorescence_enabled: bool = True) -> KernelBatchResult:
    """カーネル本体を1バッチ実行する低レベルAPI（tallyなし）。`--dose-grid`相当の
    タリー込み実行は`run_batch_with_tally`を使う（B-2、下記）。"""
    max_elem = tables.zs.shape[1]
    chunk_seeds, chunk_offsets, chunk_counts = _chunk_plan(n_histories, n_chunks, seed)
    (n_scatter, absorbed, escaped, final_energy, energy_deposited, n_fluorescence) = _run_batch_scalar(
        n_histories, len(chunk_seeds), chunk_seeds, chunk_offsets, chunk_counts,
        energy0_kev, origin[0], origin[1], origin[2], direction[0], direction[1], direction[2],
        geom.n_boxes, geom.box_center, geom.box_half, geom.box_material, geom.background_material,
        geom.world_center, geom.world_half,
        tables.n_elem, tables.zs, tables.fracs, tables.log_e, tables.step, tables.n_grid,
        tables.density_g_cm3, tables.photo, tables.compt, tables.rayl,
        tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a,
        tables.k_edge, tables.k_omega, tables.k_frac, tables.k_line_e, tables.k_line_p, tables.n_lines,
        max_elem, fluorescence_enabled)
    return KernelBatchResult(n_scatter=n_scatter, absorbed=absorbed, escaped=escaped,
                              final_energy=final_energy, energy_deposited=energy_deposited,
                              n_fluorescence=n_fluorescence)


def _compute_tally_weights(
        tables: SceneMaterialTables,
        seg_mat_all: np.ndarray,
        seg_e_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """シーンローカル材料コードと区間エネルギーからタリー重みを返す。"""
    if seg_mat_all.ndim != 1 or seg_e_all.ndim != 1:
        raise ValueError("seg_mat_all and seg_e_all must be one-dimensional")
    if seg_mat_all.dtype.kind not in "iu":
        raise TypeError("seg_mat_all must have an integer dtype")
    if len(seg_mat_all) != len(seg_e_all):
        raise ValueError("seg_mat_all and seg_e_all must have equal lengths")

    mu_en_linear = np.empty(len(seg_e_all), dtype=np.float64)
    h10_weights = np.empty(len(seg_e_all), dtype=np.float64)
    if len(seg_mat_all) == 0:
        return mu_en_linear, h10_weights

    n_materials = len(tables.material_names)
    if np.any(seg_mat_all < 0) or np.any(seg_mat_all >= n_materials):
        raise ValueError("seg_mat_all contains an out-of-range material code")

    codes = np.unique(seg_mat_all)
    codes_by_name = sorted(
        (int(code) for code in codes),
        key=lambda code: tables.material_names[code],
    )
    for code in codes_by_name:
        mask = seg_mat_all == code
        name = tables.material_names[code]
        mu_en_linear[mask] = (
            mu_en_rho(name, seg_e_all[mask]) * density(name))
    kerma_weights = seg_e_all * mu_en_linear
    h10_weights[:] = h_star_10_per_fluence(seg_e_all)
    return kerma_weights, h10_weights


def run_batch_with_tally(tables: SceneMaterialTables, geom: SceneGeometry, energy0_kev: float,
                          origin: tuple[float, float, float], direction: tuple[float, float, float],
                          n_histories: int, seed: int, grid: VoxelGrid, n_chunks: int = 1,
                          fluorescence_enabled: bool = True,
                          max_segments_per_history: int = 16,
                          use_njit_dda: bool = True) -> KernelBatchResult:
    """B-2: `--dose-grid`相当のtrack-lengthタリー込みでカーネルを1バッチ実行する。

    設計(b)（計画書「B-2: タリー統合」で選定）: カーネルは飛行区間
    (o, d, ds, e, mat)をチャンクごとの専用バッファへ吐き出すだけで、線量
    換算・グリッドへの空間分配は既定で
    `chatcarlo.tally_njit.accumulate_track_length_multi_njit`に委ね、
    `use_njit_dda=False`では監査済みnumpy参照実装
    `chatcarlo.tally.accumulate_track_length_multi`へ切り戻す。
    カーネル内でDDAを実行する設計(a)より検証コストが低い
    ——線量計算のロジック自体は既存の監査・テスト済みコードを1行も変えずに
    再利用しており、新規に検証が必要なのは「カーネルが正しい区間を吐き出す
    こと」だけになる（`tests/test_kernel.py`の
    `test_tally_variant_matches_reference_variant`が保証）。

    `max_segments_per_history`はチャンクごとの区間バッファ容量を
    `max(chunk_counts) * max_segments_per_history`で見積もるための係数
    ——診断領域での典型的な散乱回数（mean_scatter_events、数〜十数回程度）
    に対して十分な安全マージンを既定値16に取っている。超過した場合は
    タリーを黙って欠落させず`ValueError`にする（このバッチの寄与は一切
    積算しない）——欠落を検知せず一部だけ積算すると線量が系統的に過小評価
    される、検出困難なバグになるため。
    """
    max_elem = tables.zs.shape[1]
    chunk_seeds, chunk_offsets, chunk_counts = _chunk_plan(n_histories, n_chunks, seed)
    n_chunks_actual = len(chunk_seeds)
    seg_capacity_per_chunk = int(chunk_counts.max()) * max_segments_per_history

    seg_o = np.zeros((n_chunks_actual, seg_capacity_per_chunk, 3))
    seg_d = np.zeros((n_chunks_actual, seg_capacity_per_chunk, 3))
    seg_ds = np.zeros((n_chunks_actual, seg_capacity_per_chunk))
    seg_e = np.zeros((n_chunks_actual, seg_capacity_per_chunk))
    seg_mat = np.zeros((n_chunks_actual, seg_capacity_per_chunk), dtype=np.int64)

    (n_scatter, absorbed, escaped, final_energy, energy_deposited, n_fluorescence,
     seg_count_per_chunk, seg_overflow_per_chunk) = _run_batch_scalar_tally(
        n_histories, n_chunks_actual, chunk_seeds, chunk_offsets, chunk_counts,
        energy0_kev, origin[0], origin[1], origin[2], direction[0], direction[1], direction[2],
        geom.n_boxes, geom.box_center, geom.box_half, geom.box_material, geom.background_material,
        geom.world_center, geom.world_half,
        tables.n_elem, tables.zs, tables.fracs, tables.log_e, tables.step, tables.n_grid,
        tables.density_g_cm3, tables.photo, tables.compt, tables.rayl,
        tables.incoh_q, tables.incoh_s, tables.rayl_x, tables.rayl_a,
        tables.k_edge, tables.k_omega, tables.k_frac, tables.k_line_e, tables.k_line_p, tables.n_lines,
        max_elem, fluorescence_enabled,
        seg_o, seg_d, seg_ds, seg_e, seg_mat, seg_capacity_per_chunk)

    if np.any(seg_overflow_per_chunk):
        raise ValueError(
            f"タリー用の区間バッファが不足しました(max_segments_per_history="
            f"{max_segments_per_history}、チャンクあたり容量{seg_capacity_per_chunk})。"
            "max_segments_per_historyを増やすか、n_histories/n_chunksを小さくして"
            "再実行してください。このバッチのタリー寄与は積算していません。")

    seg_o_parts, seg_d_parts, seg_ds_parts, seg_e_parts, seg_mat_parts = [], [], [], [], []
    for c in range(n_chunks_actual):
        cnt = int(seg_count_per_chunk[c])
        if cnt == 0:
            continue
        seg_o_parts.append(seg_o[c, :cnt])
        seg_d_parts.append(seg_d[c, :cnt])
        seg_ds_parts.append(seg_ds[c, :cnt])
        seg_e_parts.append(seg_e[c, :cnt])
        seg_mat_parts.append(seg_mat[c, :cnt])

    if seg_o_parts:
        seg_o_all = np.concatenate(seg_o_parts)
        seg_d_all = np.concatenate(seg_d_parts)
        seg_ds_all = np.concatenate(seg_ds_parts)
        seg_e_all = np.concatenate(seg_e_parts)
        seg_mat_all = np.concatenate(seg_mat_parts)

        kerma_weights, h10_weights = _compute_tally_weights(
            tables, seg_mat_all, seg_e_all)

        accumulator = (tally_njit.accumulate_track_length_multi_njit
                       if use_njit_dda else tally.accumulate_track_length_multi)
        accumulator(
            ((grid.kerma_keV, kerma_weights),
             (grid.h10_track_pSv_cm3, h10_weights)),
            grid, seg_o_all, seg_d_all, seg_ds_all)

    return KernelBatchResult(n_scatter=n_scatter, absorbed=absorbed, escaped=escaped,
                              final_energy=final_energy, energy_deposited=energy_deposited,
                              n_fluorescence=n_fluorescence)


def run_dose_grid(tables: SceneMaterialTables, geom: SceneGeometry, energy0_kev: float,
                   origin: tuple[float, float, float], direction: tuple[float, float, float],
                   n_histories: int, seed: int, grid: VoxelGrid, batch_size: int = 200_000,
                   n_chunks: int = 1, fluorescence_enabled: bool = True,
                   max_segments_per_history: int = 16,
                   use_njit_dda: bool = True) -> KernelBatchResult:
    """`run_batch_with_tally`をbatch_size単位で繰り返し呼ぶ高レベルAPI。

    `transport._run_batches`と同じ理由（区間バッファのピークメモリを
    `n_histories`ではなく`batch_size`規模に抑える）でバッチ分割する
    ——計画書のB-2節が指摘する「区間バッファのメモリを食う」という懸念への
    直接の対応。バッチごとの乱数シードは`SeedSequence.spawn`で決定的に導出する
    （B-0/B-1と同じ階層的シード設計）。
    """
    n_batches = math.ceil(n_histories / batch_size)
    batch_seed_seqs = np.random.SeedSequence(seed).spawn(n_batches)

    n_scatter_parts, absorbed_parts, escaped_parts = [], [], []
    final_energy_parts, energy_deposited_parts, n_fluorescence_parts = [], [], []
    remaining = n_histories
    for b in range(n_batches):
        n = min(batch_size, remaining)
        remaining -= n
        batch_seed = int(batch_seed_seqs[b].generate_state(1)[0])
        r = run_batch_with_tally(tables, geom, energy0_kev, origin, direction, n, batch_seed, grid,
                                  n_chunks=n_chunks, fluorescence_enabled=fluorescence_enabled,
                                  max_segments_per_history=max_segments_per_history,
                                  use_njit_dda=use_njit_dda)
        n_scatter_parts.append(r.n_scatter)
        absorbed_parts.append(r.absorbed)
        escaped_parts.append(r.escaped)
        final_energy_parts.append(r.final_energy)
        energy_deposited_parts.append(r.energy_deposited)
        n_fluorescence_parts.append(r.n_fluorescence)

    return KernelBatchResult(
        n_scatter=np.concatenate(n_scatter_parts), absorbed=np.concatenate(absorbed_parts),
        escaped=np.concatenate(escaped_parts), final_energy=np.concatenate(final_energy_parts),
        energy_deposited=np.concatenate(energy_deposited_parts),
        n_fluorescence=np.concatenate(n_fluorescence_parts))


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
    n_fluorescence: int


def run_water_slab_probe(thickness_cm: float, energy_kev: float, n_histories: int,
                          seed: int = 1, warmup_histories: int = 1000, n_chunks: int = 1,
                          fluorescence_enabled: bool = True,
                          background: str = "air") -> ProbeResult:
    """water60_free等のスラブシナリオ（box1個・鉛筆ビーム垂直入射）をB-1b
    カーネルで実行し、スループット[histories/s]を計測する。

    B-1aと異なり、既定で背景は本番と同じ"air"・K殻蛍光も既定で有効
    （`transport_photons`のデフォルト`fluorescence_enabled=True`と同じ）——
    `docs/speedup_baseline/kernel_crosscheck.py`の統計的クロスチェックと
    同じ条件で速度も測れるようにするため。
    """
    tables = bake_scene_materials(["water", background])
    margin = 0.01
    hx, hy, hz = thickness_cm / 2.0, 50.0, 50.0
    boxes = [{"center": (0.0, 0.0, 0.0), "size_cm": (thickness_cm, 100.0, 100.0), "material": "water"}]
    geom = bake_box_scene(boxes, background=background, tables=tables, bbox_margin_cm=margin)
    origin = (-hx - margin, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0)

    # JITコンパイル（+ファイルキャッシュ書き込み）を計測対象から除外するための空撃ち。
    run_batch(tables, geom, energy_kev, origin, direction, warmup_histories, seed=0,
              n_chunks=n_chunks, fluorescence_enabled=fluorescence_enabled)

    import time
    t0 = time.perf_counter()
    r = run_batch(tables, geom, energy_kev, origin, direction, n_histories, seed=seed,
                  n_chunks=n_chunks, fluorescence_enabled=fluorescence_enabled)
    wall_s = time.perf_counter() - t0

    uncollided = float(np.sum(r.escaped & (r.n_scatter == 0))) / n_histories
    return ProbeResult(
        n_histories=n_histories,
        wall_s=wall_s,
        histories_per_s=n_histories / wall_s,
        uncollided_frac=uncollided,
        fraction_absorbed=float(np.sum(r.absorbed)) / n_histories,
        fraction_escaped=float(np.sum(r.escaped)) / n_histories,
        mean_scatter_events=float(np.sum(r.n_scatter)) / n_histories,
        energy_deposited_keV=float(np.sum(r.energy_deposited)),
        n_fluorescence=int(np.sum(r.n_fluorescence)),
    )
