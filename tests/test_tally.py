"""ボクセル線量タリーのテスト。

1. accumulate_track_length単体: 区間全体のエネルギーが（どのボクセルに
   割り振られようと）過不足なく積算されることを厳密に検証。
2. 統合テスト: 同一の輸送シミュレーション中で、相互作用点ごとに集計する
   衝突推定量（transport_photons.energy_deposited、既存の物理検証済み）と、
   飛程積分によるカーマtrack-length estimator（グリッド積算）が、
   同じ物理量（全カーマ）を異なる手法で推定していることを利用し、
   統計誤差の範囲で一致することを確認する（独立した交差検証）。
"""
from __future__ import annotations

import numpy as np

from chatcarlo.dose_coefficients import h_star_10_per_fluence
from chatcarlo.geometry import Geometry
from chatcarlo.tally import VoxelGrid, accumulate_track_length, _argsort_within_segment
from chatcarlo.transport import transport_photons


def _lexsort_within_segment_reference(all_t: np.ndarray, all_seg: np.ndarray) -> np.ndarray:
    """`_argsort_within_segment`（単一キーargsort、docs/plan_tally_speedup.md Step 3）
    が置き換えた旧方式（np.lexsort）そのものをテスト用オラクルとして再現する。

    本番コード（chatcarlo/tally.py）には存在しない——このファイル固有の
    検証専用ヘルパー。
    """
    return np.lexsort((all_t, all_seg))


def test_accumulate_exact_energy_conservation():
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]), resolution_cm=1.0)
    n = 500
    origin = np.tile(np.array([0.5, 0.5, 0.0]), (n, 1))
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    length_cm = np.full(n, 7.3)          # ボクセル境界をまたぐ長さ
    weight = np.full(n, 12.0)            # keV/cm

    accumulate_track_length(grid.kerma_keV, grid, origin, direction, length_cm, weight)

    expected_total = np.sum(length_cm * weight)
    assert np.isclose(grid.kerma_keV.sum(), expected_total, rtol=1e-9)


def test_accumulate_partial_out_of_grid():
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]), resolution_cm=1.0)
    n = 200
    # グリッド外(x=-5)からグリッドを突き抜けて外(x=15)へ抜ける長い区間
    origin = np.tile(np.array([-5.0, 5.0, 5.0]), (n, 1))
    direction = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    length_cm = np.full(n, 20.0)
    weight = np.full(n, 5.0)

    accumulate_track_length(grid.kerma_keV, grid, origin, direction, length_cm, weight)

    total_in_grid = np.sum(length_cm * weight) * (10.0 / 20.0)  # グリッド内は全長の半分
    assert grid.kerma_keV.sum() > 0
    assert grid.kerma_keV.sum() < np.sum(length_cm * weight)
    assert np.isclose(grid.kerma_keV.sum(), total_in_grid, rtol=1e-9)  # 解析的重なり長方式なので厳密一致


def test_grid_kerma_matches_collision_estimator():
    """同一の輸送で、track-length estimator(グリッド)とcollision estimator
    (相互作用点ごとの直接集計)が同じ全カーマを別手法で推定し、統計誤差内で一致する。"""
    material, thickness, energy_keV, n = "water", 15.0, 60.0, 300_000
    geom = Geometry([{
        "name": "slab", "shape": "box", "material": material,
        "center": [0.0, 0.0, 0.0], "size_cm": [thickness, 60.0, 60.0],
    }])
    grid = VoxelGrid.from_bbox(geom.bbox_min, geom.bbox_max, resolution_cm=3.0)

    rng = np.random.default_rng(7)
    pos = np.tile(np.array([-thickness / 2 - 5.0, 0.0, 0.0]), (n, 1))
    dirv = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    energy = np.full(n, energy_keV)

    result = transport_photons(pos, dirv, energy, geom, rng, grid=grid)

    collision_total_keV = sum(result.energy_deposited.values())
    tracklength_total_keV = grid.kerma_keV.sum()

    rel_diff = abs(tracklength_total_keV - collision_total_keV) / collision_total_keV
    assert rel_diff < 0.05


def test_h10_accumulate_exact_track_length():
    """H*(10)飛程積分も、カーマと同様にどのボクセルに割り振られようと
    区間全体の値 Σ coeff*dl を過不足なく積算する（体積正規化前の生値で検証）。"""
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]), resolution_cm=1.0)
    n = 500
    origin = np.tile(np.array([0.5, 0.5, 0.0]), (n, 1))
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    length_cm = np.full(n, 7.3)
    coeff = h_star_10_per_fluence(60.0)[0]  # pSv・cm²
    weight = np.full(n, coeff)

    accumulate_track_length(grid.h10_track_pSv_cm3, grid, origin, direction, length_cm, weight)

    expected_total = np.sum(length_cm * weight)  # pSv・cm³
    assert np.isclose(grid.h10_track_pSv_cm3.sum(), expected_total, rtol=1e-9)


def test_h10_map_normalizes_by_voxel_volume():
    """単一ボクセル内で全長Lを直進する光子束 -> H*(10) = N * h*(10)/Φ(E) * L / V の解析解と一致。"""
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), resolution_cm=5.0)
    n = 1000
    length_cm = 3.0
    energy_keV = 80.0
    origin = np.tile(np.array([2.5, 2.5, 1.0]), (n, 1))
    direction = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    coeff = h_star_10_per_fluence(energy_keV)[0]
    weight = np.full(n, coeff)

    accumulate_track_length(grid.h10_track_pSv_cm3, grid, origin, direction,
                             np.full(n, length_cm), weight)

    expected_pSv = n * coeff * length_cm / grid.voxel_volume_cm3()
    assert np.isclose(grid.h10_map_pSv().sum(), expected_pSv, rtol=1e-9)


def test_boundary_start_surface_voxel_unbiased():
    """区間の始点がボクセル境界ちょうどに揃うケース（parallel照射野で全光子が
    ファントム前面から出発する状況）の回帰テスト。

    旧・旧々実装の経緯: サブステップ中点の決定的サンプリングでは量子化誤差の
    位相が全区間で同期し、表面ボクセル層が約-3%系統的に過小評価されていた
    （vive-auditor監査で発見）。層化乱数点方式（不偏だが分散あり）で修正した後、
    解析的重なり長方式（本テスト、乱数を使わない厳密計算）に置き換えたことで、
    表面ボクセルへの割り当ては統計誤差すら持たず厳密な期待値 Σ min(L_i, 1) に
    浮動小数点誤差の範囲で一致する。"""
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([20.0, 10.0, 10.0]), resolution_cm=1.0)
    rng = np.random.default_rng(11)
    n = 200_000
    mfp_cm = 4.86  # 60 keV水の平均自由行程相当
    length_cm = rng.exponential(mfp_cm, n).clip(max=19.9)
    origin = np.tile(np.array([0.0, 5.5, 5.5]), (n, 1))  # 全区間がx=0境界ちょうどから出発
    direction = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    weight = np.ones(n)

    accumulate_track_length(grid.kerma_keV, grid, origin, direction, length_cm, weight)

    surface = grid.kerma_keV[0, 5, 5]              # x=[0,1)の表面ボクセル
    exact = np.minimum(length_cm, 1.0).sum()       # 厳密な重なり長の合計
    assert np.isclose(surface, exact, rtol=1e-9)   # 解析的重なり長方式なので厳密一致（旧実装は-2.7%ずれた）


def _reference_substep_accumulate(target, grid, origin, direction, length_cm, weight_per_cm,
                                   rng, substep_cm):
    """旧実装（サブステップ+層化乱数点によるモンテカルロ空間分配）の再実装。

    本体（chatcarlo/tally.py）は解析的重なり長方式に置き換え済み。このヘルパーは
    本体から独立に再実装した収束オラクル専用で、substep_cmを十分細かくすると
    厳密解（新実装の出力）へ収束するはず——という形で新実装を検証する
    （test_matches_fine_substep_reference_voxel_by_voxel参照）。
    """
    n = origin.shape[0]
    nsub = np.clip(np.ceil(length_cm / substep_cm).astype(int), 1, 200_000)
    max_n = int(nsub.max())
    j = np.arange(max_n)
    frac = (j[None, :] + rng.random((n, max_n))) / nsub[:, None]
    valid = j[None, :] < nsub[:, None]
    points = (origin[:, None, :] + direction[:, None, :]
              * (length_cm[:, None] * frac)[:, :, None])
    sub_weight = weight_per_cm * (length_cm / nsub)
    points_flat = points.reshape(-1, 3)
    weight_flat = np.broadcast_to(sub_weight[:, None], (n, max_n)).reshape(-1)
    valid_flat = valid.reshape(-1)
    idx, in_grid = grid.voxel_index(points_flat)
    keep = valid_flat & in_grid
    idx, weight_flat = idx[keep], weight_flat[keep]
    if len(idx) == 0:
        return
    flat_idx = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), grid.shape)
    np.add.at(target.reshape(-1), flat_idx, weight_flat)


def test_matches_fine_substep_reference_voxel_by_voxel():
    """新しい解析的重なり長方式(乱数不使用)が、旧サブステップ方式をsubstep_cm→0に
    細かくした極限（=厳密解への収束先）とボクセル単位で一致することを、独立に
    再実装した旧アルゴリズムをオラクルとして確認する。任意方向・グリッドに
    部分的にしかかからない区間も混ぜる（境界クリップの検証を兼ねる）。"""
    grid = VoxelGrid.from_bbox(np.array([0.0, 0.0, 0.0]), np.array([8.0, 8.0, 8.0]), resolution_cm=1.0)
    rng = np.random.default_rng(0)
    n = 3000
    origin = rng.uniform(-2.0, 10.0, size=(n, 3))
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length_cm = rng.uniform(0.5, 15.0, size=n)
    weight = rng.uniform(1.0, 5.0, size=n)

    exact = np.zeros(grid.shape)
    accumulate_track_length(exact, grid, origin, direction, length_cm, weight)

    substep_cm = grid.voxel_size_cm / 2000.0
    reference = np.zeros(grid.shape)
    _reference_substep_accumulate(reference, grid, origin, direction, length_cm, weight,
                                   np.random.default_rng(1), substep_cm=substep_cm)

    # サブステップ長が有限な限り旧実装にも残差誤差があるので、サブステップ1個分の
    # 重み程度を安全係数込みで許容する（厳密一致ではなく収束確認）。
    hit = (exact > 0) | (reference > 0)
    tol = weight.max() * substep_cm * 5
    assert np.max(np.abs(exact[hit] - reference[hit])) < tol
    assert np.isclose(exact.sum(), reference.sum(), rtol=1e-3)


def _overlap_per_segment_via_same_seg_logic(order, all_t, all_seg, n_seg):
    """`_segment_grid_traversal`本体と全く同じsame_seg/overlap計算を再現し、
    セグメントごとの重なり長の合計を返す（テスト用）。orderが正しく
    「セグメント昇順→セグメント内t昇順」になっていない限り、
    same_seg判定がセグメント境界を誤検出し、一部のセグメントで
    合計重なり長がspanより小さくなる（＝実害のある破損）。
    """
    sorted_t = all_t[order]
    sorted_seg = all_seg[order]
    same_seg = sorted_seg[1:] == sorted_seg[:-1]
    overlap = np.where(same_seg, sorted_t[1:] - sorted_t[:-1], 0.0)
    seg_for_interval = sorted_seg[:-1]
    totals = np.zeros(n_seg)
    np.add.at(totals, seg_for_interval, overlap)
    return totals


def test_argsort_within_segment_handles_adjacent_segment_boundary_collision():
    """docs/plan_tally_speedup.md Step 3: セグメントiの終点(frac=1.0)とセグメント
    i+1の始点(frac=0.0)がちょうど同じt値を持つとき、正規化キーseg+fracを
    0.5倍せずそのまま使うと両者のキーが数値的に衝突し(i+1.0 == (i+1)+0.0)、
    argsortの並び順が不定になってセグメント境界そのものが壊れる——
    `_segment_grid_traversal_accumulate`本体が行うsame_seg判定（隣接要素の
    セグメント一致で重なり長を計算する）が誤動作し、衝突した2セグメントの
    重なり長が本来のspanより小さく（最悪ゼロに）計算されてしまう。

    この破損は「各セグメントに属する値だけを取り出して昇順か見る」検証では
    検出できない（all_seg[order]によるラベル再割り当てはどんな並び順に対しても
    自明に成り立つため）。実際の破損箇所であるsame_seg/overlap計算をこのテストでも
    再現し、重なり長の合計がspanと一致するかで検証する。

    衝突ペアを1組（セグメント2個）だけにすると、numpyのargsortのタイの崩し方が
    たまたま正しい順序を再現してしまい、0.5倍を外すミューテーションを検出でき
    ないことが分かった（ミューテーション試験で確認）。隣接する20セグメントを
    鎖状に全て衝突させることで、タイの崩れ方が偶然一致し続ける確率を下げ、
    確実に検出できるようにしている（同じくミューテーション試験で確認済み）。
    """
    n_seg = 20
    # seg[i]のt_exitとseg[i+1]のt_enterが鎖状にちょうど一致する連続衝突ケース。
    t_enter_c = np.arange(n_seg, dtype=float) * 10.0
    t_exit_c = t_enter_c + 10.0
    all_t = np.empty(2 * n_seg)
    all_seg = np.empty(2 * n_seg, dtype=np.int64)
    all_t[0::2] = t_enter_c
    all_t[1::2] = t_exit_c
    all_seg[0::2] = np.arange(n_seg)
    all_seg[1::2] = np.arange(n_seg)

    order = _argsort_within_segment(all_t, all_seg, t_enter_c, t_exit_c)
    totals = _overlap_per_segment_via_same_seg_logic(order, all_t, all_seg, n_seg)

    np.testing.assert_allclose(totals, t_exit_c - t_enter_c)


def test_argsort_within_segment_matches_lexsort_reference_fuzz():
    """新しい単一キーargsort方式が、旧lexsort方式と同じセグメント境界検出
    （`_segment_grid_traversal`本体のsame_seg/overlap計算）を与えることを
    ランダムデータで確認する（docs/plan_tally_speedup.md Step 3）。
    セグメント数・1セグメントあたりの交点数を無作為化し、境界collisionが
    起きやすいよう一部セグメントの終点/始点を意図的に一致させたケースも混ぜる。
    各セグメントの重なり長合計が真のspanと一致するかで、境界破損の有無を検証する
    （前段の失敗から学んだ通り、ラベル再割り当てベースの検証では破損を検出できない
    ため、実際の下流ロジックを再現して検証する）。
    """
    rng = np.random.default_rng(42)
    n_seg = 500

    t_enter_c = np.sort(rng.uniform(0.0, 1000.0, size=n_seg))
    span = rng.uniform(0.5, 20.0, size=n_seg)

    # 意図的にseg[i]のt_exitとseg[i+1]のt_enterを一致させるケースを混ぜる
    # （t_exitはt_enter確定後にspanを足して求めるので、この時点で調整する）。
    collide_idx = rng.choice(n_seg - 1, size=n_seg // 10, replace=False)
    t_enter_c[collide_idx + 1] = t_enter_c[collide_idx] + span[collide_idx]

    t_exit_c = t_enter_c + span
    true_span = t_exit_c - t_enter_c

    all_t_parts = []
    all_seg_parts = []
    for i in range(n_seg):
        n_pts = rng.integers(2, 30)
        pts = rng.uniform(t_enter_c[i], t_exit_c[i], size=n_pts - 2)
        pts = np.concatenate(([t_enter_c[i]], pts, [t_exit_c[i]]))
        all_t_parts.append(pts)
        all_seg_parts.append(np.full(n_pts, i))
    all_t = np.concatenate(all_t_parts)
    all_seg = np.concatenate(all_seg_parts)

    order_new = _argsort_within_segment(all_t, all_seg, t_enter_c, t_exit_c)
    order_ref = _lexsort_within_segment_reference(all_t, all_seg)

    totals_new = _overlap_per_segment_via_same_seg_logic(order_new, all_t, all_seg, n_seg)
    totals_ref = _overlap_per_segment_via_same_seg_logic(order_ref, all_t, all_seg, n_seg)

    np.testing.assert_allclose(totals_new, true_span, rtol=1e-12)
    np.testing.assert_allclose(totals_ref, true_span, rtol=1e-12)


def test_run_transport_dose_grid_h10_finite_and_nonnegative():
    """実シーンでの統合テスト: H*(10)グリッドがクラッシュせず有限・非負の値を返す。"""
    from chatcarlo.scene import validate_scene
    from chatcarlo.transport import run_transport

    raw = {
        "source": {"kvp": 100, "position": [0, -50, 0], "direction": [0, 1, 0],
                    "field": {"size_cm": [30, 30], "sid_cm": 100}},
        "geometry": [
            {"name": "target", "shape": "box", "material": "water",
             "center": [0, 0, 0], "size_cm": [20, 20, 20]},
        ],
    }
    scene = validate_scene(raw)
    assert scene.ok
    result = run_transport(scene, n_histories=20_000, seed=1, dose_grid=True, grid_resolution_cm=5.0)
    h10 = result.grid.h10_map_pSv()
    assert np.all(np.isfinite(h10))
    assert np.all(h10 >= 0)
    assert h10.sum() > 0
