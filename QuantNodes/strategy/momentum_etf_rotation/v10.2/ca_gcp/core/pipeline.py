"""CA-GCP end-to-end pipeline.

Encapsulates the four-step flow described in the paper:
  1. Build correlation KNN graph on training window.
  2. Volatility-normalized nonconformity scores.
  3. Proximity- and recency-weighted quantile across neighbor pool.
  4. Systemic-stress interval widening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .graph import build_knn_graph
from .modulator import apply_modulator, compute_systemic_stress
from .volatility import estimate_volatility
from .weighted_quantile import weighted_quantile


@dataclass
class CAGCPConfig:
    alpha: float = 0.05
    k: int = 8
    sharpness_p: float = 1.0
    recency_tau: float = 60.0
    sensitivity_eta: float = 0.5
    graph_method: str = "correlation"
    ewma_span: int = 60
    realized_window: int = 20
    pseudo_count_inf: bool = True


class CAGCPipeline:
    """Model-agnostic calibration layer producing prediction intervals."""

    def __init__(self, config: CAGCPConfig | None = None):
        self.config = config or CAGCPConfig()
        self.codes: list[str] = []
        self.neighbors: dict[int, list[int]] = {}
        self.A_norm: np.ndarray | None = None
        self.corr_matrix: np.ndarray | None = None
        self.train_sigma: pd.DataFrame | None = None

    def fit(self, returns_train: pd.DataFrame) -> "CAGCPipeline":
        self.A_norm, self.neighbors, self.codes = build_knn_graph(
            returns_train,
            k=self.config.k,
            method=self.config.graph_method,
        )
        self.corr_matrix = returns_train.corr().fillna(0.0).loc[self.codes, self.codes].values
        self.train_sigma = estimate_volatility(
            returns_train,
            ewma_span=self.config.ewma_span,
            realized_window=self.config.realized_window,
        )
        return self

    def _pool_scores(
        self,
        scores_calib: pd.DataFrame,
        target_idx: int,
        target_day: pd.Timestamp,
        calib_days: list[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray]:
        pool_scores: list[float] = []
        pool_weights: list[float] = []

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

    def predict(
        self,
        returns_calib: pd.DataFrame,
        returns_test: pd.DataFrame,
        point_forecasts: pd.DataFrame | None = None,
        volatility: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """Predict intervals on returns_test using returns_calib for calibration.

        point_forecasts default to zero (martingale baseline per paper Sec. 4.2).
        """
        if point_forecasts is None:
            point_forecasts = pd.DataFrame(0.0, index=returns_test.index, columns=self.codes)
        point_forecasts = point_forecasts.reindex(index=returns_test.index, columns=self.codes).fillna(0.0)
        point_forecasts_calib = pd.DataFrame(0.0, index=returns_calib.index, columns=self.codes)

        if volatility is None:
            full_returns = pd.concat([returns_calib, returns_test], axis=0)
            volatility = estimate_volatility(
                full_returns,
                ewma_span=self.config.ewma_span,
                realized_window=self.config.realized_window,
            )

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
                q = weighted_quantile(
                    pool_s,
                    pool_w,
                    level=1.0 - self.config.alpha,
                    pseudo_count_inf=self.config.pseudo_count_inf,
                )
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

        return {
            "lower": lower,
            "upper": upper,
            "half_width": half_width_adj,
            "thresholds": thresholds,
            "stress": stress,
        }

    def get_diagnostics(self) -> dict:
        return {
            "codes": self.codes,
            "neighbors": self.neighbors,
            "A_norm": self.A_norm,
            "corr_matrix": self.corr_matrix,
        }