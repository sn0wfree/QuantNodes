"""Platt scaling calibration for probability outputs."""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── PlattCalibrator ─────────────────────────────────────────────

class PlattCalibrator:
    """Probability calibration via Platt scaling (logistic regression).

    Fits: P(y=1|prob) = 1 / (1 + exp(A * logit(prob) + B))
    where A, B are estimated from recent 60d predictions vs actuals.
    """

    def __init__(self, window: int = 60, min_samples: int = 20):
        self.window = window
        self.min_samples = min_samples
        self.A = 1.0
        self.B = 0.0
        self._fitted = False

    def fit(self, prob: pd.Series, y_true: pd.Series):
        common = prob.dropna().index.intersection(y_true.dropna().index)
        if len(common) < self.min_samples:
            return
        recent = common[-self.window:]
        x = prob.loc[recent].values
        y = y_true.loc[recent].values
        if len(np.unique(y)) < 2:
            return
        from sklearn.linear_model import LogisticRegression
        # Clip to avoid log(0) / log(1)
        eps = 1e-6
        x_clip = np.clip(x, eps, 1 - eps)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x_clip.reshape(-1, 1), y)
        self.A = float(lr.coef_[0][0])
        self.B = float(lr.intercept_[0])
        self._fitted = True

    def transform(self, prob: pd.Series) -> pd.Series:
        if not self._fitted:
            return prob
        eps = 1e-6
        x = np.clip(prob.values, eps, 1 - eps)
        # logit → linear → sigmoid
        logit = np.log(x / (1 - x))
        calibrated = 1.0 / (1.0 + np.exp(-(self.A * logit + self.B)))
        return pd.Series(calibrated, index=prob.index)

    def fit_transform(self, prob: pd.Series, y_true: pd.Series) -> pd.Series:
        self.fit(prob, y_true)
        return self.transform(prob)
