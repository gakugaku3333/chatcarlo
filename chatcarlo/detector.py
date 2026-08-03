"""Ideal, terminal planar detector tallying for scatter studies."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tally import _batch_variance, accumulate_moment_and_snapshot, relative_error

CATEGORY_NAMES = ("primary", "single_scatter", "multiple_scatter", "fluorescence_secondary")
CAT_PRIMARY, CAT_SINGLE, CAT_MULTIPLE, CAT_FLUOR = 0, 1, 2, 3


@dataclass
class DetectorPlane:
    center_cm: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    size_cm: tuple
    shape: tuple
    v_axis: np.ndarray = field(init=False)

    def __post_init__(self):
        self.center_cm = np.asarray(self.center_cm, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        self.u_axis = np.asarray(self.u_axis, dtype=float)
        if self.center_cm.shape != (3,) or self.normal.shape != (3,) or self.u_axis.shape != (3,):
            raise ValueError("検出器のcenter_cm/normal/u_axisは長さ3で指定してください")
        if not np.isclose(np.linalg.norm(self.normal), 1.0) or not np.isclose(np.linalg.norm(self.u_axis), 1.0):
            raise ValueError("検出器のnormal/u_axisは単位ベクトルで指定してください")
        if not np.isclose(float(self.normal @ self.u_axis), 0.0):
            raise ValueError("検出器のnormalとu_axisは直交している必要があります")
        if len(self.size_cm) != 2 or any(x <= 0 for x in self.size_cm) or len(self.shape) != 2 or any(int(x) <= 0 for x in self.shape):
            raise ValueError("検出器のsize_cm/shapeは正の2要素で指定してください")
        self.size_cm = tuple(float(x) for x in self.size_cm)
        self.shape = tuple(int(x) for x in self.shape)
        self.v_axis = np.cross(self.normal, self.u_axis)

    @property
    def pixel_area_cm2(self) -> float:
        return self.size_cm[0] * self.size_cm[1] / (self.shape[0] * self.shape[1])

    def corners_cm(self) -> np.ndarray:
        u = self.u_axis * self.size_cm[0] / 2
        v = self.v_axis * self.size_cm[1] / 2
        return np.stack((self.center_cm - u - v, self.center_cm - u + v,
                         self.center_cm + u - v, self.center_cm + u + v))

    def intersect_segments(self, o, d, ds):
        o, d, ds = np.asarray(o, float), np.asarray(d, float), np.asarray(ds, float)
        denom = d @ self.normal
        t = np.full(len(o), np.inf)
        nonparallel = denom != 0
        t[nonparallel] = ((self.center_cm - o[nonparallel]) @ self.normal) / denom[nonparallel]
        p = np.zeros_like(o)
        p[nonparallel] = o[nonparallel] + d[nonparallel] * t[nonparallel, None]
        rel = p - self.center_cm
        uu, vv = rel @ self.u_axis, rel @ self.v_axis
        du, dv = self.size_cm[0] / self.shape[0], self.size_cm[1] / self.shape[1]
        iu = np.floor((uu + self.size_cm[0] / 2) / du).astype(np.int64)
        iv = np.floor((vv + self.size_cm[1] / 2) / dv).astype(np.int64)
        accept = (nonparallel & (denom < 0) & (t >= 0) & (t <= ds) & (iu >= 0) & (iu < self.shape[0]) & (iv >= 0) & (iv < self.shape[1]))
        t[~accept] = np.inf
        iu[~accept] = -1
        iv[~accept] = -1
        return t, iu, iv, accept


def classify(n_compton_rayleigh: np.ndarray, had_fluorescence: np.ndarray) -> np.ndarray:
    return np.where(had_fluorescence, CAT_FLUOR,
                    np.where(n_compton_rayleigh == 0, CAT_PRIMARY,
                             np.where(n_compton_rayleigh == 1, CAT_SINGLE, CAT_MULTIPLE))).astype(np.intp)


def rebin_area_preserving(fluence_image: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(fluence_image)
    if factor <= 0 or a.shape[-2] % factor or a.shape[-1] % factor:
        raise ValueError("factorは両辺を割り切る正の値で指定してください")
    return a.reshape(*a.shape[:-2], a.shape[-2] // factor, factor, a.shape[-1] // factor, factor).mean(axis=(-3, -1))


def rebin_counts(count_image: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(count_image)
    if factor <= 0 or a.shape[-2] % factor or a.shape[-1] % factor:
        raise ValueError("factorは両辺を割り切る正の値で指定してください")
    return a.reshape(*a.shape[:-2], a.shape[-2] // factor, factor, a.shape[-1] // factor, factor).sum(axis=(-3, -1))


@dataclass
class DetectorTally:
    plane: DetectorPlane
    track_uncertainty: bool = False
    roi: tuple | None = None
    category_fluence: np.ndarray = field(init=False)
    photon_count: np.ndarray = field(init=False)
    energy_sum_keV: np.ndarray = field(init=False)
    energy_sum2_keV2: np.ndarray = field(init=False)
    category_sum2: np.ndarray | None = field(init=False, default=None)
    total_sum2: np.ndarray | None = field(init=False, default=None)
    n_batches_hit: np.ndarray | None = field(init=False, default=None)
    roi_P: float = field(init=False, default=0.0)
    roi_S: float = field(init=False, default=0.0)
    roi_QP: float | None = field(init=False, default=None)
    roi_QS: float | None = field(init=False, default=None)
    roi_CPS: float | None = field(init=False, default=None)
    n_batches: int = field(init=False, default=0)
    n_histories: int = field(init=False, default=0)
    _category_prev: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        shape = self.plane.shape
        if self.roi is not None:
            (a, b), (c, d) = self.roi
            if not (0 <= a < b <= shape[0] and 0 <= c < d <= shape[1]):
                raise ValueError("検出器ROIが範囲外です")
        self.category_fluence = np.zeros((4, *shape))
        self.photon_count = np.zeros(shape)
        self.energy_sum_keV = np.zeros(shape)
        self.energy_sum2_keV2 = np.zeros(shape)
        if self.track_uncertainty:
            self.category_sum2 = np.zeros((4, *shape))
            self.total_sum2 = np.zeros(shape)
            self.n_batches_hit = np.zeros(shape, dtype=np.int32)
            self.roi_QP = self.roi_QS = self.roi_CPS = 0.0

    def total_fluence(self): return self.category_fluence.sum(axis=0)
    def category_relative_error(self):
        if not self.track_uncertainty: raise ValueError("MC統計は無効です")
        return relative_error(self.category_fluence, self.category_sum2, self.n_batches, self.n_histories)
    def total_relative_error(self):
        if not self.track_uncertainty: raise ValueError("MC統計は無効です")
        return relative_error(self.total_fluence(), self.total_sum2, self.n_batches, self.n_histories)
    def _roi_values(self):
        if self.roi is None: raise ValueError("STPRにはROIが必要です")
        (a, b), (c, d) = self.roi
        p = float(self.category_fluence[CAT_PRIMARY, a:b, c:d].sum())
        s = float(self.category_fluence[1:, a:b, c:d].sum())
        return p, s
    def stpr(self):
        p, s = self._roi_values()
        if p == 0: raise ValueError("ROI primaryがゼロです")
        return s / p
    def stpr_sem(self):
        """Return delta-method SEM of S/P using history-normalized batch moments."""
        if not self.track_uncertainty: raise ValueError("MC統計は無効です")
        p, s = self._roi_values()
        if p == 0: raise ValueError("ROI primaryがゼロです")
        vp = float(_batch_variance(p, self.roi_QP, self.n_batches, self.n_histories)) / self.n_histories
        vs = float(_batch_variance(s, self.roi_QS, self.n_batches, self.n_histories)) / self.n_histories
        cov = (self.roi_CPS - p * s / self.n_histories) / (self.n_batches - 1) / self.n_histories if self.n_batches >= 2 else np.nan
        ratio = s / p
        return float(np.sqrt(np.clip(ratio**2 * (vs / (s / self.n_histories)**2 + vp / (p / self.n_histories)**2 - 2 * cov / ((s / self.n_histories) * (p / self.n_histories))), 0, None)))
    def end_batch(self, n_histories_in_batch: int):
        if not self.track_uncertainty or n_histories_in_batch <= 0: return
        if self._category_prev is None: self._category_prev = np.zeros_like(self.category_fluence)
        delta = np.empty_like(self.category_fluence)
        for cat in range(4): delta[cat] = accumulate_moment_and_snapshot(self.category_fluence[cat], self._category_prev[cat], self.category_sum2[cat], n_histories_in_batch)
        total_delta = delta.sum(axis=0)
        self.total_sum2 += total_delta**2 / n_histories_in_batch
        self.n_batches_hit += np.any(delta != 0, axis=0).astype(np.int32)
        if self.roi is not None:
            (a,b),(c,d) = self.roi; pb = float(delta[0,a:b,c:d].sum()); sb = float(delta[1:,a:b,c:d].sum())
            self.roi_P += pb; self.roi_S += sb; self.roi_QP += pb**2/n_histories_in_batch; self.roi_QS += sb**2/n_histories_in_batch; self.roi_CPS += pb*sb/n_histories_in_batch
        self.n_batches += 1; self.n_histories += n_histories_in_batch
    def merge_from(self, other):
        if self.track_uncertainty != other.track_uncertainty or self.roi != other.roi or self.plane.shape != other.plane.shape or not (np.array_equal(self.plane.center_cm, other.plane.center_cm) and np.array_equal(self.plane.normal, other.plane.normal) and np.array_equal(self.plane.u_axis, other.plane.u_axis) and self.plane.size_cm == other.plane.size_cm): raise ValueError("互換性のない検出器タリーです")
        for name in ("category_fluence", "photon_count", "energy_sum_keV", "energy_sum2_keV2"): setattr(self, name, getattr(self, name) + getattr(other, name))
        self.roi_P += other.roi_P; self.roi_S += other.roi_S; self.n_batches += other.n_batches; self.n_histories += other.n_histories
        if self.track_uncertainty:
            self.category_sum2 += other.category_sum2; self.total_sum2 += other.total_sum2; self.n_batches_hit += other.n_batches_hit; self.roi_QP += other.roi_QP; self.roi_QS += other.roi_QS; self.roi_CPS += other.roi_CPS
