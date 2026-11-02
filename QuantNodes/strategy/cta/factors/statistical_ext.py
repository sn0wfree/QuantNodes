# coding=utf-8
"""统计特征扩展因子.

为时序CTA方法论补充专用估计器 (区别于 statistical.py 的通用统计指标):
    - AR1Factor: 含截距 AR(1) 回归, 输出 (μ, ρ, σ) 三参数

输出形式说明:
    AR1Factor.raw_value() 返回 dict, keys = ('mu', 'rho', 'sigma'),
    每个 value 为 (T, N) DataFrame, 与 prices 同形.
    下游 (regime_classifier) 可按需取用任一参数.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class AR1Factor(Factor):
    """含截距 AR(1) 回归因子.

    模型: r_t = μ + ρ · r_{t-1} + ε_t,  ε_t ~ N(0, σ²).

    滚动 OLS 估计 (含截距):
        ρ > 0 → 短期动量 (趋势惯性)
        ρ < 0 → 短期反转 (均值回归)
        ρ ≈ 0 → 随机游走

    研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
    与 AutocorrelationFactor 的区别: AR1Factor 输出 (μ, ρ, σ) 三参数,
    供下游 TSI / Regime 分类器使用; AutocorrelationFactor 仅 ρ.

    参数:
        window:  回归窗口 (默认 60)
        rho_threshold: ρ 触发阈值 (默认 0.0, 即严格 >0 / <0)
    """

    name = "ar1"

    def __init__(self, window: int = 60, rho_threshold: float = 0.0):
        self.window = window
        self.threshold = rho_threshold

    def _fit(self, y: np.ndarray) -> tuple[float, float, float]:
        """OLS: y_t = mu + rho * y_{t-1} + eps. 返回 (mu, rho, sigma)."""
        n = len(y) - 1
        if n < 5:
            return np.nan, np.nan, np.nan
        x = np.column_stack([np.ones(n), y[:-1]])
        try:
            beta, *_ = np.linalg.lstsq(x, y[1:], rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        resid = y[1:] - x @ beta
        sigma = float(np.sqrt(np.mean(resid ** 2)))
        return float(beta[0]), float(beta[1]), sigma

    def raw_value(self, prices: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        """滚动估计 (μ, ρ, σ).

        Returns:
            dict[mu, rho, sigma] -> pd.DataFrame, 形状与 prices 对齐,
            前 (window-1) 行因样本不足为 NaN.
        """
        ret = prices.pct_change(fill_method=None)
        idx, cols = ret.index, ret.columns
        mu = pd.DataFrame(np.nan, index=idx, columns=cols)
        rho = pd.DataFrame(np.nan, index=idx, columns=cols)
        sigma = pd.DataFrame(np.nan, index=idx, columns=cols)

        arr = ret.to_numpy()
        for j, col in enumerate(cols):
            y = arr[:, j]
            # 寻找连续非 NaN 段 (rolling 在内部 NaN 时返回 NaN, 安全)
            for i in range(self.window, len(y) + 1):
                seg = y[i - self.window:i]
                if np.isnan(seg).any():
                    continue
                m, r, s = self._fit(seg)
                mu.iat[i - 1, j] = m
                rho.iat[i - 1, j] = r
                sigma.iat[i - 1, j] = s
        return {"mu": mu, "rho": rho, "sigma": sigma}

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """基于 ρ 的方向信号: ρ > +threshold → 多, ρ < -threshold → 空.

        单一信号列 'ar1', 与 prices 列一一对应.
        """
        params = self.raw_value(prices)
        rho = params["rho"]
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[rho > self.threshold] = 1.0
        sig[rho < -self.threshold] = -1.0
        return sig
