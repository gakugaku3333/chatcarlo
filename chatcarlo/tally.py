"""ボクセル吸収線量・周辺線量当量タリー — 輸送ジオメトリーとは独立なグリッド。

輸送は解析面トラッキング（chatcarlo/geometry.py, chatcarlo/transport.py）
で行い、スコアリングだけをこのモジュールの一様グリッドに刻む。

track-length estimator（trackごとの飛程積分）で2種類の量を積算する:
  カーマ:    K += E * (μen/ρ) * ρ * dl                      [keV]
  H*(10):   H += (h*(10)/Φ)(E) * dl、後でボクセル体積で正規化   [pSv]
どちらも区間内はエネルギー・材料とも一定なので積分自体に離散化誤差はない。
「その区間がどのボクセルに何cm分入っているか」の空間分配は、区間と各軸の
ボクセル境界面との交差点を解析的に列挙する厳密グリッド走査（3D DDA、
docs/egs5_crosscheck/run_chatcarlo_bsf60.py等の`_segment_box_overlap_cm`
（単一直方体との厳密重なり長）を、グリッド全体を横切る一般の区間へ拡張したもの）
で行う。乱数を一切使わず、各ボクセルへの重なり長は浮動小数点誤差を除いて厳密。
以前は区間をサブステップに分割し層化乱数点で分配するモンテカルロ方式だったが
（不偏だが空間分配自体に分散があり、`max_substeps`クランプも必要だった）、
解析的重なり長方式に置き換えて空間分配の分散をゼロにした
（docs/plan_statistical_uncertainty.md 候補3、docs/lessons_learned.md参照）。
電子飛程を無視するカーマ近似のため、カーマ＝吸収線量とみなす
（README/[[lessons_learned]]の設計判断と同じ割り切り）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MEV_TO_JOULE = 1.602176634e-13
_G_TO_KG = 1e-3


@dataclass
class VoxelGrid:
    origin_cm: np.ndarray          # (3,) グリッド原点（最小コーナー）
    shape: tuple                   # (nx, ny, nz)
    voxel_size_cm: float
    track_uncertainty: bool = False  # Trueならバッチ統計（Q・寄与バッチ数）も積算する
    kerma_keV: np.ndarray = field(init=False)      # (nx,ny,nz) 積算カーマ [keV]
    h10_track_pSv_cm3: np.ndarray = field(init=False)  # (nx,ny,nz) H*(10)飛程積分 [pSv・cm³]（体積正規化前）
    kerma_sum2: np.ndarray = field(init=False, default=None)  # (nx,ny,nz) Σ(バッチ寄与²/n_b) — カーマ
    h10_sum2: np.ndarray = field(init=False, default=None)    # 同、H*(10)飛程積分
    n_batches_hit: np.ndarray = field(init=False, default=None)  # (nx,ny,nz) int32 — 寄与のあったバッチ数
    n_batches: int = field(init=False, default=0)
    n_histories: int = field(init=False, default=0)
    _kerma_prev: np.ndarray = field(init=False, default=None, repr=False)
    _h10_prev: np.ndarray = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.kerma_keV = np.zeros(self.shape, dtype=float)
        self.h10_track_pSv_cm3 = np.zeros(self.shape, dtype=float)
        if self.track_uncertainty:
            self.kerma_sum2 = np.zeros(self.shape, dtype=float)
            self.h10_sum2 = np.zeros(self.shape, dtype=float)
            self.n_batches_hit = np.zeros(self.shape, dtype=np.int32)
            # _kerma_prev/_h10_prevは最初のend_batchで遅延確保する（end_batch参照）

    @classmethod
    def from_bbox(cls, bbox_min: np.ndarray, bbox_max: np.ndarray, resolution_cm: float,
                  track_uncertainty: bool = False) -> "VoxelGrid":
        extent = bbox_max - bbox_min
        shape = tuple(max(1, int(np.ceil(x / resolution_cm))) for x in extent)
        return cls(origin_cm=np.asarray(bbox_min, dtype=float), shape=shape, voxel_size_cm=resolution_cm,
                    track_uncertainty=track_uncertainty)

    def end_batch(self, n_histories_in_batch: int) -> None:
        """バッチ（=輸送のバッチ、history境界と一致）終了時に呼ぶ。

        track_uncertainty=Falseなら何もしない（既存呼び出し元に非侵襲）。
        total（kerma_keV/h10_track_pSv_cm3）自体は一切書き換えない
        （スナップショット差分方式）。バッチ寄与 delta = total - 直前スナップショット
        を読むだけなので、totalの浮動小数点加算順序は統計ON/OFFで完全に同一のまま
        —— これが「統計機構の有無でtotalがビット一致する」という絶対制約の根拠。
        Σ(delta²/n_b) をQとして積算する（不偏推定器の導出はdocs/
        plan_statistical_uncertainty.md 設計判断2参照）。呼ばないと統計だけが
        出ない（totalは常に正しいまま）。
        """
        if not self.track_uncertainty:
            return
        if n_histories_in_batch <= 0:
            return
        if self._kerma_prev is None:
            # スナップショットは最初のend_batchまで確保しない。並列実行では親
            # プロセスのグリッドは集約先でありスコアリングを行わない（end_batchを
            # 呼ばない）ため、ここを__post_init__で確保すると細解像度で
            # ボクセルあたり16バイトが完全な死蔵になる。
            self._kerma_prev = np.zeros(self.shape, dtype=float)
            self._h10_prev = np.zeros(self.shape, dtype=float)
        delta_kerma = self.kerma_keV - self._kerma_prev
        delta_h10 = self.h10_track_pSv_cm3 - self._h10_prev
        self.kerma_sum2 += delta_kerma ** 2 / n_histories_in_batch
        self.h10_sum2 += delta_h10 ** 2 / n_histories_in_batch
        # μen/ρ・h*(10)/Φはいずれの材料・エネルギーでも正なので、ある区間が
        # 触れるボクセル集合はkerma・h10で本来同一（設計判断6）。ORで両者を拾う。
        self.n_batches_hit += ((delta_kerma != 0) | (delta_h10 != 0)).astype(np.int32)
        # copy()ではなくcopytoで既存バッファを再利用する（バッチごとにグリッド
        # 2枚分を確保・解放し直すのを避ける。細解像度では1枚790MB級になる）。
        np.copyto(self._kerma_prev, self.kerma_keV)
        np.copyto(self._h10_prev, self.h10_track_pSv_cm3)
        self.n_batches += 1
        self.n_histories += n_histories_in_batch

    def kerma_relative_error(self) -> np.ndarray:
        return relative_error(self.kerma_keV, self.kerma_sum2, self.n_batches, self.n_histories)

    def h10_relative_error(self) -> np.ndarray:
        return relative_error(self.h10_track_pSv_cm3, self.h10_sum2, self.n_batches, self.n_histories)

    def voxel_index(self, points: np.ndarray):
        """点(N,3) -> ボクセル添字(N,3)とグリッド内かどうか(N,)。"""
        idx = np.floor((points - self.origin_cm) / self.voxel_size_cm).astype(int)
        shape_arr = np.array(self.shape)
        valid = np.all((idx >= 0) & (idx < shape_arr), axis=1)
        return idx, valid

    def voxel_centers(self) -> np.ndarray:
        """全ボクセル中心座標 (nx*ny*nz, 3)。"""
        nx, ny, nz = self.shape
        xs = self.origin_cm[0] + (np.arange(nx) + 0.5) * self.voxel_size_cm
        ys = self.origin_cm[1] + (np.arange(ny) + 0.5) * self.voxel_size_cm
        zs = self.origin_cm[2] + (np.arange(nz) + 0.5) * self.voxel_size_cm
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    def voxel_volume_cm3(self) -> float:
        return self.voxel_size_cm ** 3

    def total_kerma_MeV(self) -> float:
        return float(self.kerma_keV.sum()) / 1000.0

    def dose_map_Gy(self, density_g_cm3: np.ndarray) -> np.ndarray:
        """材料密度マップ(shapeと同じ)からボクセルごとの吸収線量[Gy]を計算。"""
        mass_kg = density_g_cm3 * self.voxel_volume_cm3() * _G_TO_KG
        energy_J = self.kerma_keV * 1e-3 * _MEV_TO_JOULE
        return np.divide(energy_J, mass_kg, out=np.zeros_like(energy_J), where=mass_kg > 0)

    def h10_map_pSv(self) -> np.ndarray:
        """ボクセルごとの周辺線量当量H*(10) [pSv]（飛程積分をボクセル体積で正規化）。"""
        return self.h10_track_pSv_cm3 / self.voxel_volume_cm3()


def _batch_variance(T, Q, n_batches: int, n_histories: int):
    """バッチ統計(T,Q,M,N)からhistoryあたり寄与の分散σ̂²を不偏推定する。

    σ̂² = (Q - T²/N) / (M-1)。T=ΣS_b（バッチ寄与和）、Q=ΣS_b²/n_b。
    E[Q]=Mσ²+Nμ², E[T²/N]=σ²+Nμ² なので差は(M-1)σ²——n_bがバッチごとに
    不揃いでも厳密に不偏（docs/plan_statistical_uncertainty.md 設計判断2）。
    M<2（バッチ数不足）はnan。丸め誤差でσ̂²が微小負になるケースは0にクランプする
    （真の分散は非負なので、丸めによる符号だけの負値を意味のある負の分散として
    扱わない）。T/Qがスカラーでもndarrayでも同じ式で動く。
    """
    T = np.asarray(T, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if n_batches < 2 or n_histories <= 0:
        return np.full_like(T, np.nan)
    var = (Q - T ** 2 / n_histories) / (n_batches - 1)
    return np.clip(var, 0.0, None)


def standard_error(T, Q, n_batches: int, n_histories: int):
    """バッチ統計から平均値(T/N)の標準誤差(SEM)を求める。M<2はnan。"""
    var = _batch_variance(T, Q, n_batches, n_histories)
    if n_histories <= 0:
        return np.full_like(var, np.nan)
    return np.sqrt(var / n_histories)


def relative_error(T, Q, n_batches: int, n_histories: int):
    """バッチ統計から相対誤差 R = SEM/(T/N) を求める。

    寄与が真にゼロ（T==0）のボクセル・材料はnan（相対誤差は定義されない）。
    M<2（バッチ数不足で推定不可）もnan——「意味のある小さな値」と「推定不能」を
    混同しないため、0や無限大ではなくnanを返す。
    """
    T = np.asarray(T, dtype=float)
    sem = standard_error(T, Q, n_batches, n_histories)
    if n_histories <= 0:
        return np.full_like(T, np.nan)
    mean = T / n_histories
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(mean != 0, sem / mean, np.nan)
    return r


def combine_moments(moments_a: tuple, moments_b: tuple) -> tuple:
    """並列ワーカー間で(T,Q,n_batches,n_histories)を合成する純関数。

    加算だけなので交換律・結合律を満たす不偏推定量の性質は保たれるが、
    浮動小数点の丸めは加算順序に依存する——並列集約では既存のワーカー番号順
    （docs/plan_phase3_parallel.md）をそのまま踏襲し、呼び出し側で順序を固定すること。
    T/Qはスカラーでもndarrayでも動く。
    """
    t_a, q_a, m_a, n_a = moments_a
    t_b, q_b, m_b, n_b = moments_b
    return (t_a + t_b, q_a + q_b, m_a + m_b, n_a + n_b)


@dataclass
class ScalarMoments:
    """材料別吸収エネルギー等、辞書形式のスカラー量に対するバッチ統計。

    VoxelGridと同じ不偏推定器（history境界=バッチ境界）が使える。あるバッチに
    現れない材料はS_b=0として扱う——それ以前のバッチで真に寄与ゼロだったので
    0で正しく、新規に現れた材料も以降のバッチ分だけ正しく積算される。
    """
    sums: dict = field(default_factory=dict)     # 材料名 -> T = ΣS_b
    sum2s: dict = field(default_factory=dict)     # 材料名 -> Q = ΣS_b²/n_b
    n_batches: int = 0
    n_histories: int = 0

    def add_batch(self, batch_values: dict, n_histories_in_batch: int) -> None:
        """batch_values: 材料名 -> このバッチでの寄与和 S_b。"""
        if n_histories_in_batch <= 0:
            return
        for name in set(self.sums) | set(batch_values):
            s_b = batch_values.get(name, 0.0)
            self.sums[name] = self.sums.get(name, 0.0) + s_b
            self.sum2s[name] = self.sum2s.get(name, 0.0) + s_b ** 2 / n_histories_in_batch
        self.n_batches += 1
        self.n_histories += n_histories_in_batch

    def relative_errors(self) -> dict:
        return {name: float(relative_error(self.sums[name], self.sum2s[name],
                                            self.n_batches, self.n_histories))
                for name in self.sums}

    def standard_errors(self) -> dict:
        return {name: float(standard_error(self.sums[name], self.sum2s[name],
                                            self.n_batches, self.n_histories))
                for name in self.sums}

    def as_moments(self) -> dict:
        """材料名 -> (T,Q,n_batches,n_histories) の辞書。combine_moments用。"""
        return {name: (self.sums[name], self.sum2s[name], self.n_batches, self.n_histories)
                for name in self.sums}

    def merge_from(self, other_moments: dict, other_n_batches: int, other_n_histories: int) -> None:
        """並列ワーカー等、他プロセスで独立に積算した`as_moments()`の出力を合成する。

        材料ごとのT・Qは単純加算する（combine_momentsと同じ加算則）。M・Nは
        ワーカー全体の値であって材料ごとの値ではないため、材料ループの外で
        一度だけ加算する。自分に無い材料キー・相手に無い材料キーは(0,0)として
        扱う——その材料が一度も現れなかった全バッチは真に寄与ゼロだったので
        0が正しい（`add_batch`が単一プロセス内で未出現材料をS_b=0とする
        仕様の並列版）。
        """
        for name in set(self.sums) | set(other_moments):
            t_other, q_other = other_moments.get(name, (0.0, 0.0))[:2]
            self.sums[name] = self.sums.get(name, 0.0) + t_other
            self.sum2s[name] = self.sum2s.get(name, 0.0) + q_other
        self.n_batches += other_n_batches
        self.n_histories += other_n_histories


_EPS_PLANE = 1e-7  # ボクセル境界面インデックス(実数)を整数面とみなす許容誤差（単位: 面間隔=1）


def _segment_grid_traversal(grid: VoxelGrid, origin: np.ndarray, direction: np.ndarray,
                             length_cm: np.ndarray):
    """区間群とグリッドの厳密な交差分解: 戻り値は (seg_id, voxel_idx(nx,ny,nz), overlap_cm)。

    各区間をグリッドのボクセル境界面との交点で分割し、生じる各部分区間について
    「どのボクセルか（部分区間の中点で判定）」と「厳密な重なり長」を返す。
    区間の始点・終点をまたぐ境界面はもちろん、区間の一部だけがグリッド内にある
    場合も先にAABBとの交差区間[t_enter,t_exit]へ厳密にクリップしてから走査するので、
    グリッド外の部分は最初から計算に含まれない。

    ラグド配列（区間ごとに交差数が異なる）を`np.repeat`/累積和で構築し、
    「最長区間に合わせて全区間を密にパディングする」方式を避けている
    （疎な解像度でも1区間が数百〜数千回交差しうるため、密パディングだと
    区間数×最大交差数でメモリが吹き飛ぶ。docs/plan_statistical_uncertainty.md
    候補3の設計上の注意点）。
    """
    n = origin.shape[0]
    h = grid.voxel_size_cm
    lo = grid.origin_cm
    shape = np.asarray(grid.shape)
    hi = lo + shape * h

    t_enter = np.zeros(n)
    t_exit = length_cm.astype(float).copy()
    for k in range(3):
        dk = direction[:, k]
        ok = origin[:, k]
        parallel = np.abs(dk) < 1e-12
        dk_safe = np.where(parallel, 1.0, dk)
        ta = (lo[k] - ok) / dk_safe
        tb = (hi[k] - ok) / dk_safe
        tmin_k = np.where(parallel, -np.inf, np.minimum(ta, tb))
        tmax_k = np.where(parallel, np.inf, np.maximum(ta, tb))
        outside_slab = parallel & ((ok < lo[k]) | (ok > hi[k]))
        tmin_k = np.where(outside_slab, np.inf, tmin_k)
        tmax_k = np.where(outside_slab, -np.inf, tmax_k)
        t_enter = np.maximum(t_enter, tmin_k)
        t_exit = np.minimum(t_exit, tmax_k)

    active = np.where(t_exit > t_enter)[0]
    if len(active) == 0:
        return (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int64), np.empty(0))

    o = origin[active]
    d = direction[active]
    t_enter = t_enter[active]
    t_exit = t_exit[active]
    m = len(active)

    # 各軸ごとに、区間内(t_enter,t_exit)にある「内部」境界面のインデックス範囲を求める。
    # 境界面mはx = lo[k] + m*hにある。区間の端がちょうど境界面上に乗る場合
    # （区間の始点/終点自体を規定した軸）はEPS_PLANEで除外し、重複ゼロ長区間を作らない。
    counts = np.zeros((3, m), dtype=np.int64)
    m_lo_all = np.zeros((3, m), dtype=np.int64)
    d_safe_all = np.zeros((3, m))
    for k in range(3):
        dk = d[:, k]
        parallel = np.abs(dk) < 1e-12
        dk_safe = np.where(parallel, 1.0, dk)
        x_enter = o[:, k] + t_enter * dk
        x_exit = o[:, k] + t_exit * dk
        p_enter = (x_enter - lo[k]) / h
        p_exit = (x_exit - lo[k]) / h
        p_small = np.minimum(p_enter, p_exit)
        p_big = np.maximum(p_enter, p_exit)
        m_lo_k = np.ceil(p_small + _EPS_PLANE).astype(np.int64)
        m_hi_k = np.floor(p_big - _EPS_PLANE).astype(np.int64)
        cnt_k = np.clip(m_hi_k - m_lo_k + 1, 0, None)
        cnt_k = np.where(parallel, 0, cnt_k)
        counts[k] = cnt_k
        m_lo_all[k] = m_lo_k
        d_safe_all[k] = dk_safe

    t_parts = [t_enter, t_exit]
    seg_parts = [np.arange(m), np.arange(m)]
    for k in range(3):
        cnt_k = counts[k]
        total_k = int(cnt_k.sum())
        if total_k == 0:
            continue
        seg_id_k = np.repeat(np.arange(m), cnt_k)
        starts_k = np.cumsum(cnt_k) - cnt_k
        offset_k = np.arange(total_k) - np.repeat(starts_k, cnt_k)
        m_idx_k = m_lo_all[k][seg_id_k] + offset_k
        t_k = (lo[k] + m_idx_k * h - o[seg_id_k, k]) / d_safe_all[k][seg_id_k]
        t_parts.append(t_k)
        seg_parts.append(seg_id_k)

    all_t = np.concatenate(t_parts)
    all_seg = np.concatenate(seg_parts)
    order = np.lexsort((all_t, all_seg))  # 主キー: セグメント（lexsortの最終引数）、副キー: t
    sorted_t = all_t[order]
    sorted_seg = all_seg[order]

    same_seg = sorted_seg[1:] == sorted_seg[:-1]
    overlap = np.where(same_seg, sorted_t[1:] - sorted_t[:-1], 0.0)
    keep = overlap > 0
    if not np.any(keep):
        return (np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.int64), np.empty(0))

    seg_for_interval = sorted_seg[:-1][keep]
    mid_t = ((sorted_t[:-1] + sorted_t[1:]) / 2.0)[keep]
    overlap = overlap[keep]

    points = o[seg_for_interval] + d[seg_for_interval] * mid_t[:, None]
    idx, in_grid = grid.voxel_index(points)
    idx, overlap, seg_for_interval = idx[in_grid], overlap[in_grid], seg_for_interval[in_grid]

    return (active[seg_for_interval], idx, overlap)


def accumulate_track_length(target: np.ndarray, grid: VoxelGrid, origin: np.ndarray,
                             direction: np.ndarray, length_cm: np.ndarray,
                             weight_per_cm: np.ndarray) -> None:
    """区間(origin, direction, length_cm)ごとに weight_per_cm * dl を target グリッドへ積算する。

    target は grid.shape と同じ形の任意の量（カーマ・H*(10)飛程積分など）で、
    weight_per_cm は区間内で一定（材料・エネルギーとも不変の前提）とする。
    各区間とグリッドのボクセル境界面との交点を解析的に求め（`_segment_grid_traversal`）、
    各ボクセルへ weight_per_cm * (厳密な重なり長) を加算する——乱数を一切使わない
    決定論的な処理で、空間分配自体の分散はゼロ（docs/plan_statistical_uncertainty.md
    候補3。旧実装はサブステップ+層化乱数点によるモンテカルロ分配で、不偏だが
    空間分配に伴う分散があり`max_substeps`クランプも必要だった。単一直方体との
    厳密重なり長を計算する`_segment_box_overlap_cm`（EGS5相互検証スクリプト、
    vive-auditor監査済み）を、グリッド全体を貫く一般の区間へ一般化したもの）。
    """
    n = origin.shape[0]
    if n == 0:
        return
    seg_id, idx, overlap = _segment_grid_traversal(grid, origin, direction, length_cm)
    if len(seg_id) == 0:
        return
    weight_flat = weight_per_cm[seg_id] * overlap
    flat_idx = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.shape)
    np.add.at(target.reshape(-1), flat_idx, weight_flat)
