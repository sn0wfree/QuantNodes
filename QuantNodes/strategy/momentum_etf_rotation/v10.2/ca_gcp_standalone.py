"""
CA-GCP: Cross-Asset Graph Conformal Prediction
==============================================

Reference
---------
Parker, J. & Zhang, Y. (2026).
"Graph-Based Uncertainty-Aware Financial Forecasting via
 Cross-Asset Conformal Prediction."
Computer Life, 14(3), 21-29.

BibTeX:
    @article{parker2026cagcp,
        title   = {Graph-Based Uncertainty-Aware Financial Forecasting
                   via Cross-Asset Conformal Prediction},
        author  = {Parker, J. and Zhang, Y.},
        journal = {Computer Life},
        volume  = {14},
        number  = {3},
        pages   = {21--29},
        year    = {2026}
    }

Model Overview
--------------
CA-GCP is a model-agnostic post-hoc calibration layer that produces
distribution-free prediction intervals for asset returns, leveraging
cross-asset correlation structure via a KNN graph.

Problem: Given a point forecast r̂ for asset i at time t, construct an
interval [r̂ - hw, r̂ + hw] such that P(actual ∈ [lower, upper]) ≥ 1 - α,
*without* distributional assumptions, by borrowing nonconformity scores
from correlated peer assets.

Core Mechanism (3 steps):
    1. Graph construction (Sec. 4.1):
       Build KNN graph A on training window using either correlation
       (default) or sector membership. Nodes = assets, edges = neighbors.

    2. Cross-asset score pooling (Sec. 4.2):
       For target asset i at time t, pool nonconformity scores
       (|r - r̂| / σ) from neighbors N(i), weighted by:
           w_jk = corr(i, j)^sharpness_p * exp(-Δt_days / recency_tau)

    3. Weighted conformal quantile (Sec. 4.3):
       threshold = WeightedQuantile(pooled_scores, weights, level=1-α)
       hw = threshold * σ̂; modulated by systemic stress (Sec. 4.4):
           hw_adj = hw * exp(η * stress_t)
       where stress_t = sigmoid(z(cross_dispersion, anomaly_frac)).

Theoretical Guarantee (Sec. 4.5):
    Marginal coverage ≥ 1 - α under exchangeability, with finite-sample
    adjustment via TV distance bound on neighbor distributions.

Single-file implementation: core pipeline, validators, sector clustering,
and risk filter. Dependencies: numpy, pandas only.

Key Parameters (see CAGCPConfig for full list):
    k (int, default 8)         — KNN neighbors per asset
    sharpness_p (float, 1.0)   — exponent on correlation weight
    recency_tau (float, 60.0)  — half-life of time decay (days)
    sensitivity_eta (float,0.5)— stress modulation strength
    alpha (float, 0.05)        — target miscoverage rate

Usage:
    # As library
    from ca_gcp_standalone import CAGCPipeline, CAGCPConfig
    pipe = CAGCPipeline(CAGCPConfig(k=6, sensitivity_eta=0.5))
    pipe.fit(returns_full)              # Config 驱动自动拆分
    intervals = pipe.predict_fast(pipe._calib, pipe._test)

    # As script
    python ca_gcp_standalone.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# SECTION 1: CORE — Graph construction
# ============================================================================


def build_knn_graph(
    returns_train: pd.DataFrame,
    k: int = 8,
    method: str = "correlation",
    sectors: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[int, list[int]], list[str]]:
    """Build correlation KNN graph on training window (Sec. 4.1).

    Returns:
        A_norm: (N, N) degree-normalized adjacency.
        neighbors: dict mapping target index -> list of source indices (incl. self).
        codes: list of asset codes aligned with A_norm rows/cols.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    codes = list(returns_train.columns)
    n = len(codes)
    corr = returns_train.corr().fillna(0.0).values

    if method == "correlation":
        scores = corr.copy()
    elif method == "sector":
        if sectors is None:
            raise ValueError("sectors dict required for method='sector'")
        scores = np.zeros((n, n))
        for i, ci in enumerate(codes):
            for j, cj in enumerate(codes):
                if sectors.get(ci) == sectors.get(cj):
                    scores[i, j] = 1.0
        np.fill_diagonal(scores, 1.0)
    elif method == "random":
        scores = rng.uniform(0, 1, size=(n, n))
        scores = (scores + scores.T) / 2
        np.fill_diagonal(scores, 1.0)
    else:
        raise ValueError(f"unknown method={method}")

    np.fill_diagonal(scores, -np.inf)

    neighbors: dict[int, list[int]] = {}
    for i in range(n):
        top_idx = np.argsort(-scores[i])[:k]
        top_idx = top_idx[np.isfinite(scores[i, top_idx])]
        neighbors[i] = sorted([i, *top_idx.tolist()])

    adjacency = np.zeros((n, n), dtype=float)
    for i, nbrs in neighbors.items():
        for j in nbrs:
            adjacency[i, j] = 1.0
    adjacency = (adjacency + adjacency.T) / 2
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 1.0)

    deg = adjacency.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt

    return A_norm, neighbors, codes


# ============================================================================
# SECTION 2: CORE — Volatility estimation
# ============================================================================


def estimate_volatility(
    returns: pd.DataFrame,
    ewma_span: int = 60,
    realized_window: int = 20,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Per-asset volatility estimate (Sec. 4.3, Eq. 4)."""
    ewma_var = returns.pow(2).ewm(span=ewma_span, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)
    realized_vol = returns.rolling(window=realized_window, min_periods=5).std()
    blend = 0.5 * ewma_vol + 0.5 * realized_vol
    sigma = np.sqrt(np.maximum(blend.pow(2), 0.0))
    sigma = sigma.bfill().fillna(0.0)
    sigma = sigma + epsilon
    return sigma


# ============================================================================
# SECTION 3: CORE — Systemic stress modulator
# ============================================================================

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


def compute_systemic_stress(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    threshold_sigma: float = 1.5,
) -> pd.Series:
    """S_t = sigmoid(a + b * dispersion_t + c * anomalous_frac_t) (Sec. 4.5)."""
    aligned_returns, aligned_vol = returns.align(volatility, join="inner", axis=1)
    cross_dispersion = aligned_returns.std(axis=1)
    anomaly_count = (aligned_returns.abs() > (threshold_sigma * aligned_vol)).sum(axis=1)
    anomaly_frac = anomaly_count / aligned_returns.shape[1]

    a, b, c = -4.0, 1.5, 4.0
    z = a + b * cross_dispersion / (cross_dispersion.std() + 1e-8) + c * (
        anomaly_frac - anomaly_frac.mean()
    ) / (anomaly_frac.std() + 1e-8)
    s = pd.Series(_SIGMOID(z.values), index=returns.index)
    return s.fillna(0.0)


def apply_modulator(
    half_width: pd.DataFrame,
    stress: pd.Series,
    eta: float = 0.5,
) -> pd.DataFrame:
    """Multiply half-width by exp(eta * S)."""
    stress_aligned = stress.reindex(half_width.index).fillna(0.0)
    factor = np.exp(eta * stress_aligned.values)
    return half_width.multiply(factor, axis=0)


# ============================================================================
# SECTION 4: CORE — Weighted quantile
# ============================================================================


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float = 0.95,
    pseudo_count_inf: bool = True,
) -> float:
    """Weighted quantile with pseudo-count at infinity (Sec. 4.4, Eq. 5-6)."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.size == 0:
        return np.inf if pseudo_count_inf else 0.0

    weights = np.maximum(weights, 0.0)
    if pseudo_count_inf:
        scores = np.concatenate([scores, [np.inf]])
        weights = np.concatenate([weights, [1e-3]])

    total = weights.sum()
    if total <= 0:
        return np.inf if pseudo_count_inf else 0.0
    weights = weights / total

    order = np.argsort(scores)
    scores_sorted = scores[order]
    weights_sorted = weights[order]
    cum = np.cumsum(weights_sorted)

    idx = np.searchsorted(cum, level, side="left")
    if idx >= len(scores_sorted):
        idx = len(scores_sorted) - 1

    val = float(scores_sorted[idx])
    if np.isinf(val) and pseudo_count_inf:
        return float(scores_sorted[idx - 1]) if idx > 0 else 0.0
    return val


class PrecomputedWeightedQuantile:
    """Fast weighted quantile for batch queries via precomputed cache."""

    def __init__(self, scores: np.ndarray, weights: np.ndarray,
                 pseudo_count_inf: bool = True):
        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        if scores.size == 0:
            self._sorted = np.array([np.inf])
            self._cum = np.array([1.0])
            self._pseudo = pseudo_count_inf
            return
        if pseudo_count_inf:
            scores = np.concatenate([scores, [np.inf]])
            weights = np.concatenate([weights, [1e-3]])
        total = weights.sum()
        if total <= 0:
            self._sorted = np.array([np.inf])
            self._cum = np.array([1.0])
            self._pseudo = pseudo_count_inf
            return
        norm_w = weights / total
        order = np.argsort(scores, kind="mergesort")
        self._sorted = scores[order]
        self._cum = np.cumsum(norm_w[order])
        self._pseudo = pseudo_count_inf

    def query(self, level: float) -> float:
        if len(self._cum) == 0:
            return np.inf if self._pseudo else 0.0
        idx = int(np.searchsorted(self._cum, level, side="left"))
        if idx >= len(self._sorted):
            idx = len(self._sorted) - 1
        val = float(self._sorted[idx])
        if np.isinf(val) and self._pseudo and idx > 0:
            return float(self._sorted[idx - 1])
        return val


def _loss_pareto(metrics: dict, w_bps: float) -> float:
    """Default: 10*extreme - 5*pa_std - w_bps/1000."""
    if np.isnan(metrics.get("extreme", np.nan)):
        return -1e9
    return 10.0 * metrics["extreme"] - 5.0 * metrics["pa_std"] - w_bps / 1000.0


def _loss_coverage(metrics: dict, w_bps: float) -> float:
    """Pure coverage: marginal + extreme (no width penalty)."""
    if np.isnan(metrics.get("extreme", np.nan)):
        return -1e9
    return metrics["marginal"] + metrics["extreme"]


def _loss_sharpness(metrics: dict, w_bps: float) -> float:
    """Pure sharpness: -width (narrower intervals win)."""
    return -w_bps


_LOSS_REGISTRY = {
    "pareto": _loss_pareto,
    "coverage": _loss_coverage,
    "sharpness": _loss_sharpness,
}


def resolve_loss_fn(loss_fn):
    """Resolve loss_fn from string name or callable to a callable."""
    if callable(loss_fn):
        return loss_fn
    if isinstance(loss_fn, str) and loss_fn in _LOSS_REGISTRY:
        return _LOSS_REGISTRY[loss_fn]
    raise ValueError(
        f"Unknown loss_fn={loss_fn!r}; choose from {list(_LOSS_REGISTRY)} or pass a callable"
    )


def _filter_to_target(result: dict, target_codes: list[str] | None) -> dict:
    """Slice per-asset DataFrames to target_codes subset. stress untouched."""
    if not target_codes:
        return result
    keys_to_slice = ("lower", "upper", "half_width", "thresholds")
    sliced = {}
    for k, v in result.items():
        if k in keys_to_slice and isinstance(v, pd.DataFrame):
            cols = [c for c in target_codes if c in v.columns]
            sliced[k] = v[cols] if cols else v.iloc[:, :0]
        else:
            sliced[k] = v
    return sliced


# ============================================================================
# SECTION 5: CORE — CAGCPConfig + CAGCPipeline
# ============================================================================


@dataclass
class CAGCPConfig:
    """Configuration for CA-GCP pipeline."""
    alpha: float = 0.05
    k: int = 8
    sharpness_p: float = 1.0
    recency_tau: float = 60.0
    sensitivity_eta: float = 0.5
    graph_method: str = "correlation"
    ewma_span: int = 60
    realized_window: int = 20
    pseudo_count_inf: bool = True

    sectors: dict[str, str] | None = None
    target_codes: list[str] | None = None

    train_ratio: float = 0.60
    calib_ratio: float = 0.25
    train_end: str | "pd.Timestamp" | None = None
    calib_end: str | "pd.Timestamp" | None = None

    loss_fn: str | object = "pareto"


class CAGCPipeline:
    """Model-agnostic calibration layer producing prediction intervals."""

    def __init__(self, config: CAGCPConfig | None = None):
        self.config = config or CAGCPConfig()
        self.codes: list[str] = []
        self.neighbors: dict[int, list[int]] = {}
        self.A_norm: np.ndarray | None = None
        self.corr_matrix: np.ndarray | None = None
        self.train_sigma: pd.DataFrame | None = None
        self._train: pd.DataFrame | None = None
        self._calib: pd.DataFrame | None = None
        self._test: pd.DataFrame | None = None
        self._loss_fn = resolve_loss_fn(self.config.loss_fn)

    def _split_data(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split returns into train/calib/test per Config."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError("returns must have a DatetimeIndex for date-based splitting")
        if self.config.train_end is not None:
            train_end = pd.Timestamp(self.config.train_end)
            train = returns.loc[:train_end].iloc[:-1]
            if self.config.calib_end is not None:
                calib_end = pd.Timestamp(self.config.calib_end)
                calib = returns.loc[train_end:calib_end]
                if calib_end < returns.index[-1]:
                    test = returns.loc[calib_end:].iloc[1:]
                else:
                    test = returns.iloc[0:0]
            else:
                calib = returns.loc[train_end:]
                test = returns.iloc[0:0]
        else:
            n = len(returns)
            t = max(1, int(n * self.config.train_ratio))
            c = max(1, int(n * self.config.calib_ratio))
            train = returns.iloc[:t]
            calib = returns.iloc[t:t + c]
            test = returns.iloc[t + c:]
        return train, calib, test

    def fit(self, returns_full: pd.DataFrame) -> "CAGCPipeline":
        self._train, self._calib, self._test = self._split_data(returns_full)
        returns_train = self._train
        self.A_norm, self.neighbors, self.codes = build_knn_graph(
            returns_train, k=self.config.k,
            method=self.config.graph_method,
            sectors=self.config.sectors,
        )
        self.corr_matrix = returns_train.corr().fillna(0.0).loc[self.codes, self.codes].values
        self.train_sigma = estimate_volatility(
            returns_train, ewma_span=self.config.ewma_span,
            realized_window=self.config.realized_window,
        )
        return self

    def _pool_scores(self, scores_calib, target_idx, target_day, calib_days):
        pool_scores, pool_weights = [], []
        for src_idx in self.neighbors[target_idx]:
            corr_v = max(float(self.corr_matrix[target_idx, src_idx]), 0.0)
            if corr_v <= 0:
                corr_v = 1e-6
            corr_w = corr_v ** self.config.sharpness_p
            for t in calib_days:
                val = scores_calib.at[t, self.codes[src_idx]]
                if pd.isna(val):
                    continue
                delta_days = (target_day - t).days
                time_w = np.exp(-delta_days / self.config.recency_tau)
                pool_scores.append(float(val))
                pool_weights.append(corr_w * time_w)
        return np.array(pool_scores), np.array(pool_weights)

    def predict(self, returns_calib, returns_test, point_forecasts=None, volatility=None):
        """Predict intervals (slow version, for correctness verification)."""
        if point_forecasts is None:
            point_forecasts = pd.DataFrame(0.0, index=returns_test.index, columns=self.codes)
        point_forecasts = point_forecasts.reindex(index=returns_test.index, columns=self.codes).fillna(0.0)
        point_forecasts_calib = pd.DataFrame(0.0, index=returns_calib.index, columns=self.codes)

        if volatility is None:
            full_returns = pd.concat([returns_calib, returns_test], axis=0)
            volatility = estimate_volatility(full_returns, ewma_span=self.config.ewma_span,
                                             realized_window=self.config.realized_window)
        volatility = volatility.reindex(columns=self.codes)
        calib_sigma = volatility.reindex(index=returns_calib.index)
        test_sigma = volatility.reindex(index=returns_test.index)
        calib_resid = (returns_calib - point_forecasts_calib).abs()
        scores_calib = calib_resid / calib_sigma
        calib_days = list(returns_calib.index)

        lower = pd.DataFrame(np.nan, index=returns_test.index, columns=self.codes)
        upper = pd.DataFrame(np.nan, index=returns_test.index, columns=self.codes)
        half_width = pd.DataFrame(np.nan, index=returns_test.index, columns=self.codes)
        thresholds = pd.DataFrame(np.nan, index=returns_test.index, columns=self.codes)

        for v_idx, code in enumerate(self.codes):
            for t_idx, t in enumerate(returns_test.index):
                pool_s, pool_w = self._pool_scores(scores_calib, v_idx, t, calib_days)
                q = weighted_quantile(pool_s, pool_w, level=1.0 - self.config.alpha,
                                      pseudo_count_inf=self.config.pseudo_count_inf)
                hw = q * test_sigma.at[t, code]
                fcst = point_forecasts.at[t, code]
                lower.at[t, code] = fcst - hw
                upper.at[t, code] = fcst + hw
                half_width.at[t, code] = hw
                thresholds.at[t, code] = q

        stress = compute_systemic_stress(returns_test, test_sigma)
        half_width_adj = apply_modulator(half_width, stress, eta=self.config.sensitivity_eta)
        lower = point_forecasts - half_width_adj
        upper = point_forecasts + half_width_adj
        result = {"lower": lower, "upper": upper, "half_width": half_width_adj,
                  "thresholds": thresholds, "stress": stress}
        return _filter_to_target(result, self.config.target_codes)

    def predict_fast(self, returns_calib, returns_test, point_forecasts=None, volatility=None):
        """Vectorized prediction (~10x faster)."""
        if point_forecasts is None:
            point_forecasts = pd.DataFrame(0.0, index=returns_test.index, columns=self.codes)
        point_forecasts = point_forecasts.reindex(index=returns_test.index, columns=self.codes).fillna(0.0)
        point_forecasts_calib = pd.DataFrame(0.0, index=returns_calib.index, columns=self.codes)

        if volatility is None:
            full_returns = pd.concat([returns_calib, returns_test], axis=0)
            volatility = estimate_volatility(full_returns, ewma_span=self.config.ewma_span,
                                             realized_window=self.config.realized_window)
        volatility = volatility.reindex(columns=self.codes)
        calib_sigma = volatility.reindex(index=returns_calib.index)
        test_sigma = volatility.reindex(index=returns_test.index)
        calib_resid = (returns_calib - point_forecasts_calib).abs()
        scores_calib_arr = (calib_resid / calib_sigma).values
        n_codes = len(self.codes)
        n_test = len(returns_test.index)
        n_calib = len(returns_calib.index)

        if n_test == 0 or n_calib == 0:
            empty = pd.DataFrame(np.nan, index=returns_test.index, columns=self.codes)
            return {"lower": empty.copy(), "upper": empty.copy(), "half_width": empty.copy(),
                    "thresholds": empty.copy(), "stress": pd.Series(dtype=float)}

        calib_pos = pd.DatetimeIndex(returns_calib.index).asi8
        test_pos = pd.DatetimeIndex(returns_test.index).asi8
        thresholds_arr = np.full((n_test, n_codes), np.nan)

        for v_idx in range(n_codes):
            valid_mask = ~np.isnan(scores_calib_arr[:, v_idx])
            if not valid_mask.any():
                continue
            valid_calib_pos = calib_pos[valid_mask]
            for t_idx in range(n_test):
                if test_pos[t_idx] < valid_calib_pos[0]:
                    continue
                offsets = (test_pos[t_idx] - valid_calib_pos) / 86_400_000_000_000
                time_w = np.exp(-offsets / self.config.recency_tau) if self.config.recency_tau > 0 else np.ones_like(offsets)
                nbr_idx = self.neighbors[v_idx]
                corr_row = np.maximum(self.corr_matrix[v_idx, nbr_idx], 0.0)
                corr_row = np.where(corr_row > 0, corr_row, 1e-6)
                corr_w = corr_row ** self.config.sharpness_p
                pool, weights = [], []
                for k_i, src_idx in enumerate(nbr_idx):
                    src_valid = ~np.isnan(scores_calib_arr[valid_mask, src_idx])
                    if not src_valid.any():
                        continue
                    pool.append(scores_calib_arr[valid_mask, src_idx][src_valid])
                    weights.append(corr_w[k_i] * time_w[src_valid])
                if not pool:
                    continue
                ps = np.concatenate(pool)
                pw = np.concatenate(weights)
                c2 = PrecomputedWeightedQuantile(ps, pw, pseudo_count_inf=self.config.pseudo_count_inf)
                thresholds_arr[t_idx, v_idx] = c2.query(1.0 - self.config.alpha)

        thresholds = pd.DataFrame(thresholds_arr, index=returns_test.index, columns=self.codes)
        half_width_arr = thresholds_arr * test_sigma.values
        fcst_arr = point_forecasts.values
        half_width = pd.DataFrame(half_width_arr, index=returns_test.index, columns=self.codes)
        lower = pd.DataFrame(fcst_arr - half_width_arr, index=returns_test.index, columns=self.codes)
        upper = pd.DataFrame(fcst_arr + half_width_arr, index=returns_test.index, columns=self.codes)

        stress = compute_systemic_stress(returns_test, test_sigma)
        half_width_adj = apply_modulator(half_width, stress, eta=self.config.sensitivity_eta)
        lower = point_forecasts - half_width_adj
        upper = point_forecasts + half_width_adj
        result = {"lower": lower, "upper": upper, "half_width": half_width_adj,
                  "thresholds": thresholds, "stress": stress}
        return _filter_to_target(result, self.config.target_codes)

    def get_diagnostics(self) -> dict:
        return {"codes": self.codes, "neighbors": self.neighbors,
                "A_norm": self.A_norm, "corr_matrix": self.corr_matrix}


# ============================================================================
# SECTION 6: VALIDATORS — Coverage, Width, Early Warning
# ============================================================================


def _is_covered(actual, lower, upper):
    common_cols = actual.columns.intersection(lower.columns).intersection(upper.columns)
    common_idx = actual.index.intersection(lower.index).intersection(upper.index)
    return ((actual.loc[common_idx, common_cols] >= lower.loc[common_idx, common_cols]) &
            (actual.loc[common_idx, common_cols] <= upper.loc[common_idx, common_cols])).astype(float)


def compute_coverage_metrics(actual, lower, upper, extreme_vol=None):
    """Marginal + per-asset + worst-decile + extreme-day coverage."""
    covered = _is_covered(actual, lower, upper)
    per_asset = covered.mean(axis=0)
    marginal = float(covered.values.mean())
    pa_std = float(per_asset.std())
    sorted_pa = np.sort(per_asset.values)
    worst10 = float(sorted_pa[:max(1, len(sorted_pa) // 10)].mean())
    worst_min = float(per_asset.min())
    extr = float("nan")
    if extreme_vol is not None:
        aligned_mask = extreme_vol.reindex(covered.index)
        if aligned_mask.notna().any():
            extr = float(covered.loc[aligned_mask.fillna(False).astype(bool)].values.mean())
    return {"marginal": marginal, "pa_std": pa_std, "worst10": worst10,
            "min": worst_min, "extreme": extr}


def width_bps(half_width):
    """Mean interval width in basis points."""
    return float((2.0 * half_width).mean().mean() * 1e4)


def width_timeseries(half_width):
    return (2.0 * half_width).mean(axis=1)


def width_volatility_correlation(half_width, realized_vol):
    w = width_timeseries(half_width)
    aligned = pd.concat([w, realized_vol], axis=1).dropna()
    aligned.columns = ["width", "realized_vol"]
    return float(aligned["width"].corr(aligned["realized_vol"])) if len(aligned) >= 2 else 0.0


def width_stability(half_width, window=60):
    w = width_timeseries(half_width)
    rolling = w.rolling(window, min_periods=10).std() / w.rolling(window, min_periods=10).mean()
    return float(rolling.dropna().mean())


def detect_warnings(stress, half_width, width_z_thresh=2.0, stress_thresh=0.6, mode="and"):
    wts = width_timeseries(half_width)
    width_z = (wts - wts.rolling(60, min_periods=10).mean()) / wts.rolling(60, min_periods=10).std().replace(0, np.nan)
    width_alert = width_z > width_z_thresh
    stress_alert = stress > stress_thresh
    fired = (width_alert.fillna(False) & stress_alert.fillna(False)).astype(int) if mode == "and" \
        else (width_alert.fillna(False) | stress_alert.fillna(False)).astype(int)
    return pd.DataFrame({"width_z": width_z, "stress": stress, "fired": fired})


def evaluate_against_events(fired, events, horizon=5, drawdown_thresh=0.03, returns_for_eval=None):
    rows = []
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        post = returns_for_eval.loc[ev_date:ev_date + pd.Timedelta(days=horizon * 2)] if returns_for_eval is not None else None
        realized_dd = float(((1 + post).prod() - 1).min()) if post is not None and len(post) >= horizon else float("nan")
        prior_window = fired.loc[ev_date - pd.Timedelta(days=20):ev_date]
        fired_idx = prior_window.index[prior_window.fillna(0) == 1]
        lead_days = int((ev_date - fired_idx[-1]).days) if len(fired_idx) else None
        rows.append({"event": ev["name"], "date": ev["date"], "lead_days": lead_days,
                     "realized_dd_5d": realized_dd, "warned": lead_days is not None})
    return pd.DataFrame(rows)


# ============================================================================
# SECTION 7: VALIDATORS — Neighbor quality + Theoretical bound
# ============================================================================


@dataclass
class NeighborQuality:
    target: str
    target_idx: int
    n_neighbors: int
    weighted_corr_sum: float
    mean_corr: float
    effective_sample_size: float
    borrow_recommendation: str

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def compute_neighbor_quality(pipeline, target_idx, calib_days=252):
    if pipeline.corr_matrix is None:
        raise ValueError("Pipeline must be fitted.")
    nbr_idx = [i for i in pipeline.neighbors[target_idx] if i != target_idx]
    if not nbr_idx:
        return NeighborQuality(target=pipeline.codes[target_idx], target_idx=target_idx,
                               n_neighbors=0, weighted_corr_sum=0.0, mean_corr=0.0,
                               effective_sample_size=float(calib_days), borrow_recommendation="weak")
    p = pipeline.config.sharpness_p
    corrs = np.array([max(float(pipeline.corr_matrix[target_idx, i]), 0.0) for i in nbr_idx])
    weighted = float((corrs ** p).sum())
    mean_corr = float(corrs.mean())
    borrowed_days = calib_days * len(nbr_idx) * (weighted / max(len(nbr_idx), 1))
    ess = calib_days + borrowed_days
    rec = "strong" if weighted >= 5.0 else ("moderate" if weighted >= 2.0 else "weak")
    return NeighborQuality(target=pipeline.codes[target_idx], target_idx=target_idx,
                           n_neighbors=len(nbr_idx), weighted_corr_sum=weighted,
                           mean_corr=mean_corr, effective_sample_size=float(ess),
                           borrow_recommendation=rec)


def quality_dataframe(pipeline, calib_days_per_asset=None):
    if calib_days_per_asset is None:
        calib_days_per_asset = {c: 252 for c in pipeline.codes}
    rows = [compute_neighbor_quality(pipeline, i, calib_days_per_asset.get(pipeline.codes[i], 252)).to_dict()
            for i in range(len(pipeline.codes))]
    return pd.DataFrame(rows).set_index("target")


@dataclass
class TheoreticalBound:
    code: str
    n_neighbors: int
    max_tv: float
    weighted_tv_sum: float
    bound: float

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def total_variation_distance_ecdf(scores_p, scores_q):
    p = np.sort(np.asarray(scores_p, dtype=float))
    q = np.sort(np.asarray(scores_q, dtype=float))
    if len(p) == 0 or len(q) == 0:
        return 1.0
    all_pts = np.unique(np.concatenate([p, q]))
    F_p = np.searchsorted(p, all_pts, side="right") / len(p)
    F_q = np.searchsorted(q, all_pts, side="right") / len(q)
    return float(np.max(np.abs(F_p - F_q)))


def theoretical_coverage_bound(pipeline, scores_calib, alpha=0.05):
    if pipeline.corr_matrix is None:
        raise ValueError("Pipeline must be fitted.")
    p = pipeline.config.sharpness_p
    target_scores_arr = [scores_calib[c].dropna().values for c in pipeline.codes]
    rows = []
    for v_idx, code in enumerate(pipeline.codes):
        nbr_idx = [i for i in pipeline.neighbors[v_idx] if i != v_idx]
        if not nbr_idx:
            rows.append({"code": code, "n_neighbors": 0, "max_tv": 0.0, "weighted_tv_sum": 0.0, "bound": 0.0})
            continue
        target_s = target_scores_arr[v_idx]
        weighted_tv_sum, max_tv = 0.0, 0.0
        for src_idx in nbr_idx:
            tv = total_variation_distance_ecdf(target_s, target_scores_arr[src_idx])
            max_tv = max(max_tv, tv)
            corr_v = max(float(pipeline.corr_matrix[v_idx, src_idx]), 0.0)
            if corr_v <= 0:
                corr_v = 1e-6
            weighted_tv_sum += (corr_v ** p) * tv
        rows.append({"code": code, "n_neighbors": len(nbr_idx), "max_tv": max_tv,
                     "weighted_tv_sum": weighted_tv_sum, "bound": weighted_tv_sum / max(len(nbr_idx), 1)})
    return pd.DataFrame(rows).set_index("code")


def compare_bound_to_empirical(bounds, empirical_gaps):
    common = bounds.index.intersection(empirical_gaps.index)
    df = pd.DataFrame({"code": list(common), "bound": [float(bounds.loc[c, "bound"]) for c in common],
                        "empirical_gap": [float(empirical_gaps[c]) for c in common]})
    df["ratio"] = df["bound"] / df["empirical_gap"].replace(0, np.nan)
    df["bound_satisfied"] = df["bound"] >= df["empirical_gap"]
    return df.set_index("code")


# ============================================================================
# SECTION 8: SECTOR CLUSTERING
# ============================================================================


@dataclass
class SectorCAGCPResult:
    lower: pd.DataFrame
    upper: pd.DataFrame
    half_width: pd.DataFrame
    stress: pd.Series
    sector_stress: dict[str, pd.Series]


def load_sector_map(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df["code"].astype(str), df["sector"].astype(str)))


def build_sector_groups(codes, sector_map, min_size=2):
    sector_to_codes = {}
    for c in codes:
        sec = sector_map.get(c)
        if sec is not None:
            sector_to_codes.setdefault(sec, []).append(c)
    return {sec: cs for sec, cs in sector_to_codes.items() if len(cs) >= min_size}


def fit_sector_ca_gcp(returns_train, sectors, config=None, sector_overrides=None):
    cfg = config or CAGCPConfig()
    out = {}
    for sec, codes in sectors.items():
        sec_cfg = (sector_overrides or {}).get(sec, cfg)
        if sec_cfg.k >= len(codes):
            sec_cfg = CAGCPConfig(**{**sec_cfg.__dict__, "k": max(1, len(codes) - 1)})
        pipe = CAGCPipeline(sec_cfg)
        pipe.fit(returns_train[codes])
        out[sec] = pipe
    return out


def predict_sector_ca_gcp(pipelines, returns_calib, returns_test):
    sector_frames, sector_stress, seen_codes = {}, {}, set()
    for sec, pipe in pipelines.items():
        sec_codes = [c for c in pipe.codes if c in returns_calib.columns and c in returns_test.columns]
        new_codes = [c for c in sec_codes if c not in seen_codes]
        seen_codes.update(new_codes)
        if not new_codes:
            continue
        out = pipe.predict(returns_calib[new_codes], returns_test[new_codes])
        sector_frames[sec] = {"lower": out["lower"], "upper": out["upper"], "half_width": out["half_width"]}
        sector_stress[sec] = out["stress"]
    lower = pd.concat([v["lower"] for v in sector_frames.values()], axis=1).loc[:, ~pd.concat([v["lower"] for v in sector_frames.values()], axis=1).columns.duplicated()] if sector_frames else pd.DataFrame()
    upper = pd.concat([v["upper"] for v in sector_frames.values()], axis=1).loc[:, ~pd.concat([v["upper"] for v in sector_frames.values()], axis=1).columns.duplicated()] if sector_frames else pd.DataFrame()
    hw = pd.concat([v["half_width"] for v in sector_frames.values()], axis=1).loc[:, ~pd.concat([v["half_width"] for v in sector_frames.values()], axis=1).columns.duplicated()] if sector_frames else pd.DataFrame()
    stress_df = pd.concat(sector_stress, axis=1)
    stress = stress_df.mean(axis=1) if not stress_df.empty else pd.Series(dtype=float)
    return SectorCAGCPResult(lower=lower, upper=upper, half_width=hw, stress=stress, sector_stress=sector_stress)


def fit_sector_hybrid_ca_gcp(returns_train, sectors, config=None, cross_sector_threshold=0.5):
    cfg = config or CAGCPConfig()
    corr_full = returns_train.corr()
    out = {}
    for sec, codes in sectors.items():
        in_sec = list(codes)
        cross_codes = []
        for c in in_sec:
            for c2 in returns_train.columns:
                if c2 not in in_sec and c2 not in cross_codes and corr_full.at[c, c2] >= cross_sector_threshold:
                    cross_codes.append(c2)
        pipe = CAGCPipeline(cfg)
        pipe.fit(returns_train[in_sec + cross_codes])
        out[sec] = pipe
    return out


# ============================================================================
# SECTION 9: RISK FILTER
# ============================================================================


@dataclass
class RiskFilterRules:
    """Threshold rules for risk overlay."""
    width_z_yellow: float = 3.0
    width_z_red: float = 4.5
    stress_yellow: float = 0.92
    stress_red: float = 0.98
    yellow_scale: float = 0.85
    red_scale: float = 0.6
    panic_scale: float = 0.3
    stress_yellow_recovery: float = 0.80
    width_z_yellow_recovery: float = 2.0
    group_rules: dict[str, "RiskFilterRules"] | None = None
    asset_groups: dict[str, list[str]] | None = None


def experimental_rules():
    return RiskFilterRules(width_z_yellow=2.0, width_z_red=3.0, stress_yellow=0.6, stress_red=0.85)


def _compute_width_z(hw, history, today):
    hw_full = pd.concat([history, hw]).drop_duplicates() if history is not None else hw
    width_ts = (2.0 * hw_full).mean(axis=1)
    roll = width_ts.rolling(60, min_periods=10)
    width_z_full = (width_ts - roll.mean()) / roll.std().replace(0, np.nan)
    val = width_z_full.loc[today] if today in width_z_full.index else width_z_full.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def ca_gcp_risk_filter(weights, intervals, rules=None, today=None, history=None):
    """Apply CA-GCP risk filter to target weights. Returns (adjusted_weights, diagnostics)."""
    rules = rules or RiskFilterRules()
    if rules.group_rules and rules.asset_groups:
        return _ca_gcp_risk_filter_grouped(weights, intervals, rules, today, history)
    return _ca_gcp_risk_filter_global(weights, intervals, rules, today, history)


def _ca_gcp_risk_filter_global(weights, intervals, rules, today, history):
    hw, stress = intervals["half_width"], intervals["stress"]
    if today is None:
        today = hw.index[-1]
    width_z_today = _compute_width_z(hw, history, today)
    stress_today = float(stress.loc[today]) if today in stress.index else float(stress.iloc[-1])

    diag = {"width_z_today": width_z_today, "stress_today": stress_today, "alert_level": "green"}
    scale = 1.0
    if width_z_today > rules.width_z_red or stress_today > rules.stress_red:
        scale, diag["alert_level"] = rules.red_scale, "red"
    elif width_z_today > rules.width_z_yellow or stress_today > rules.stress_yellow:
        scale, diag["alert_level"] = rules.yellow_scale, "yellow"

    diag["applied_scale"] = scale
    adjusted = weights * scale
    if scale < 1.0:
        residual = (1.0 - adjusted.sum()) if adjusted.sum() < 1.0 else 0.0
        if residual > 0:
            adjusted[weights.abs().idxmin()] += residual
    return adjusted, diag


def _ca_gcp_risk_filter_grouped(weights, intervals, rules, today, history):
    hw, stress = intervals["half_width"], intervals["stress"]
    if today is None:
        today = hw.index[-1]
    stress_today = float(stress.loc[today]) if today in stress.index else float(stress.iloc[-1])

    group_alerts, group_scales, group_width_z = {}, {}, {}
    for group_name, group_assets in rules.asset_groups.items():
        group_hw_cols = [c for c in group_assets if c in hw.columns]
        if not group_hw_cols:
            continue
        group_history = history[[c for c in group_assets if c in history.columns]] if history is not None else None
        wz = _compute_width_z(hw[group_hw_cols], group_history, today)
        group_width_z[group_name] = wz
        grp_rules = rules.group_rules.get(group_name, rules) if rules.group_rules else rules
        alert, scale = "green", 1.0
        if wz > grp_rules.width_z_red or stress_today > grp_rules.stress_red:
            scale, alert = grp_rules.red_scale, "red"
        elif wz > grp_rules.width_z_yellow or stress_today > grp_rules.stress_yellow:
            scale, alert = grp_rules.yellow_scale, "yellow"
        group_alerts[group_name] = alert
        group_scales[group_name] = scale

    overall_alert, overall_scale = "green", 1.0
    worst = "red" if "red" in group_alerts.values() else ("yellow" if "yellow" in group_alerts.values() else "green")
    if worst != "green":
        overall_alert = worst
        total_w, weighted_s = 0.0, 0.0
        for grp, sc in group_scales.items():
            grp_w = sum(weights.get(a, 0.0) for a in rules.asset_groups.get(grp, []))
            if sc < 1.0:
                weighted_s += sc * grp_w
                total_w += grp_w
        if total_w > 0:
            overall_scale = weighted_s / total_w

    adjusted = weights * overall_scale
    if overall_scale < 1.0:
        residual = (1.0 - adjusted.sum()) if adjusted.sum() < 1.0 else 0.0
        if residual > 0:
            adjusted[weights.abs().idxmin()] += residual
    return adjusted, {"width_z_today": group_width_z, "stress_today": stress_today,
                      "alert_level": overall_alert, "applied_scale": overall_scale,
                      "group_alerts": group_alerts, "group_scales": group_scales}


def build_v10_2_pipeline(returns_history, config=None):
    pipe = CAGCPipeline(config)
    pipe.fit(returns_history)
    return pipe


# ============================================================================
# SECTION 10: DEMO — Quick start example
# ============================================================================


def demo():
    """Run a quick demo showing CA-GCP usage."""
    print("=" * 60)
    print("CA-GCP Standalone Demo")
    print("=" * 60)

    # Generate synthetic returns (10 assets, 500 days)
    rng = np.random.default_rng(42)
    n_assets, n_days = 10, 500
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    codes = [f"ASSET_{i}" for i in range(n_assets)]
    returns = pd.DataFrame(rng.normal(0, 0.01, (n_days, n_assets)), index=dates, columns=codes)

    # Inject correlation structure
    returns["ASSET_0"] = returns["ASSET_1"] * 0.8 + rng.normal(0, 0.005, n_days)
    returns["ASSET_2"] = returns["ASSET_3"] * 0.7 + rng.normal(0, 0.005, n_days)

    # Split: train / calib / test
    train = returns.iloc[:200]
    calib = returns.iloc[200:350]
    test = returns.iloc[350:]

    print(f"\nData: {n_assets} assets, {n_days} days")
    print(f"Train: {len(train)} days, Calib: {len(calib)} days, Test: {len(test)} days")

    # Fit pipeline
    config = CAGCPConfig(k=6, sensitivity_eta=0.5, recency_tau=20)
    pipe = CAGCPipeline(config)
    pipe.fit(train)
    print(f"\nFitted: k={config.k}, eta={config.sensitivity_eta}, tau={config.recency_tau}")
    print(f"  Neighbors per asset: {[len(v) for v in pipe.neighbors.values()][:3]}...")

    # Predict (fast)
    t0 = time.time()
    intervals = pipe.predict_fast(calib, test)
    elapsed = time.time() - t0
    print(f"\npredict_fast: {elapsed:.2f}s")
    print(f"  half_width mean: {intervals['half_width'].mean().mean():.6f}")
    print(f"  stress range: [{intervals['stress'].min():.4f}, {intervals['stress'].max():.4f}]")

    # Coverage
    m = compute_coverage_metrics(test, intervals["lower"], intervals["upper"])
    print(f"\nCoverage: marginal={m['marginal']:.3f}, pa_std={m['pa_std']:.4f}, "
          f"extreme={m['extreme']:.3f}")
    print(f"Width: {width_bps(intervals['half_width']):.0f} bps")

    # Risk filter
    weights = pd.Series(1.0 / n_assets, index=codes)
    rules = RiskFilterRules()
    adj_w, diag = ca_gcp_risk_filter(weights, intervals, rules, today=test.index[-1])
    print(f"\nRisk filter: alert={diag['alert_level']}, scale={diag['applied_scale']:.2f}")

    # Neighbor quality
    nq = quality_dataframe(pipe)
    print(f"\nNeighbor quality:")
    print(nq[["n_neighbors", "weighted_corr_sum", "borrow_recommendation"]].to_string())

    print("\n" + "=" * 60)
    print("Demo complete. CA-GCP standalone module ready for use.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
