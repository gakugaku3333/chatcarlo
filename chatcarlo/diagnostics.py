"""線量グリッドの後処理と「最大値が非物理的な位置に落ちていないか」の診断。

`chatcarlo run --dose-grid` の最大吸収線量・最大H*(10)は、背景（空気）ボクセルや
点線源の1/r²発散近傍に落ちることがある（docs/lessons_learned.md参照）。
CLIはここにある判定関数で該当ケースを検出して警告を出す。
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .materials import density
from .tally import VoxelGrid


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
    覆われているが、ChatCarloの点線源モデルはそれを持たない）。
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


_UNRELIABLE_R_THRESHOLD = 0.10
_UNRELIABLE_HIT_FRACTION_THRESHOLD = 0.25


def unreliable_max_warning(R: float, n_hit_batches: int, n_batches: int) -> str | None:
    """最大値ボクセルの統計信頼性が低い（R統計の必要条件を満たさない）かどうかを判定する。

    docs/plan_statistical_uncertainty.md 設計判断7の運用強制:
    Rは必要条件であって十分条件ではない——寄与バッチ数が少ないボクセルは
    寄与分布がゼロ膨張・強い歪みを持ち、「まだ大きな寄与を引いていないだけ」の
    状態でRが小さめに出て偽の安心を与える（MCNPでも既知の落とし穴）。
    したがってR自体に加え、寄与バッチ数の割合も独立にチェックする。

    R が nan（バッチ数不足でそもそも推定不能）の場合は、この関数の呼び出し側
    （`n_batches>=2`のときだけ呼ぶ想定）で別途「バッチ数不足」メッセージを
    出す設計のため、ここでは扱わない（R>閾値の判定はnanで常にFalseになるため
    無害だが、意図的にここでは追加のnanメッセージは出さない）。
    """
    if n_batches <= 0:
        return None
    hit_fraction = n_hit_batches / n_batches
    r_bad = R > _UNRELIABLE_R_THRESHOLD
    hit_bad = hit_fraction < _UNRELIABLE_HIT_FRACTION_THRESHOLD
    if not r_bad and not hit_bad:
        return None
    reasons = []
    if r_bad:
        reasons.append(f"相対誤差R={R:.3f}が信頼できる目安（R<{_UNRELIABLE_R_THRESHOLD:g}）を超えている")
    if hit_bad:
        reasons.append(f"寄与バッチ数が{n_hit_batches}/{n_batches}（{hit_fraction * 100:.0f}%）と少ない")
    return ("最大値ボクセルの統計は信用できません（" + "、".join(reasons) + "）。"
            "特に寄与バッチ数が少ないボクセルは、まだ大きな寄与を引いていないだけで"
            "Rが小さめに出る楽観バイアスを持ちます（Rは必要条件であって十分条件では"
            "ありません）。histories数を増やすか、寄与バッチ数マップで確認してください。")


def batch_shortage_message(n_batches: int, n_histories: int, batch_size: int) -> str | None:
    """バッチ数不足（M<2）で統計誤差が推定不能なことを、実行可能な対処とともに示す。

    M=1（既定のn_histories=1e5・batch_size=200,000ではこれが既定の帰結）では
    分散推定に使える自由度がなく、Rは全てnanになる。「nanが出ている」ことだけ
    伝えても対処に繋がらないため、実際の設定値から-nまたは--batch-sizeの
    具体的な変更量を計算して埋め込む（固定文言にしない、設計判断3参照）。
    """
    if n_batches >= 2:
        return None
    n_suggest = batch_size * 20
    batch_size_suggest = max(1, -(-n_histories // 20))  # ceil(n_histories/20)
    return (f"バッチ数が不足しています（M={n_batches}、必要なのはM>=2、実用上はM>=20）。"
            f"現在の設定: n_histories={n_histories:g} / batch_size={batch_size:,} → "
            f"M=ceil(n_histories/batch_size)={n_batches}。"
            f"対処: -n を{n_suggest:g}以上（M>=20）にするか、"
            f"--batch-size を{batch_size_suggest:g}以下にしてください。")


def grid_reliability_summary(R: np.ndarray, n_batches_hit: np.ndarray, n_batches: int) -> dict:
    """グリッド全体の統計信頼性を2つの割合に要約する。

    - frac_low_r: R算出可能（非nan）なボクセルのうち R<0.10 の割合。
      算出可能なボクセルが一つもなければnan。
    - frac_unreached: 寄与バッチが一度もないボクセルの割合（全ボクセル数が分母、
      Rの算出可否に関わらず定義できる——寄与ゼロは「タリーが一度も触れていない」
      という事実であり、統計量ではないため常に求まる）。
    """
    total = R.size
    frac_unreached = float(np.sum(n_batches_hit == 0)) / total if total else float("nan")
    valid = ~np.isnan(R)
    n_valid = int(np.sum(valid))
    frac_low_r = float(np.sum(R[valid] < _UNRELIABLE_R_THRESHOLD)) / n_valid if n_valid else float("nan")
    return {"frac_low_r": frac_low_r, "frac_unreached": frac_unreached}
