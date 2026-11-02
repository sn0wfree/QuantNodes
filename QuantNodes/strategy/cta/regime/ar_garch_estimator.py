# coding=utf-8
"""(μ, ρ, σ) 三参数估计器 + Kaufman 自适应窗口.

将 AR1Factor 升级为支持:
    1. Kaufman 效率比驱动的自适应窗口 (21-250 日, 文章推荐区间)
    2. 可选 GARCH(1,1) 增强 (当 `arch` 库可用时自动启用)

研报: 华泰金工《时序CTA方法论综述》(2026-06-12), §"市场状态的动态参数估计".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..factors.indicators import efficiency_ratio
from ..factors.statistical_ext import AR1Factor


def kaufman_adaptive_window(
    close: pd.Series,
    h_min: int = 21,
    h_max: int = 250,
    er_window: int = 30,
) -> pd.Series:
    """Kaufman 效率比驱动的自适应窗口.

    ER 高 (路径平滑, 趋势强) → 窗口缩短 → 灵敏捕捉转向.
    ER 低 (路径纠结, 噪声大) → 窗口拉长 → 平滑背景噪声.
    公式: h = h_min · ER + h_max · (1 - ER), 取整到 [h_min, h_max] 区间.

    Args:
        close: 收盘价序列
        h_min: 窗口下限 (默认 21, 文章 §"观测窗口选择")
        h_max: 窗口上限 (默认 250)
        er_window: Kaufman ER 计算窗口 (默认 30)

    Returns:
        pd.Series[int], 每个时点的建议滚动窗口长度.
    """
    er = efficiency_ratio(close, er_window)
    h = h_min * er + h_max * (1 - er)
    h = h.clip(h_min, h_max).round()
    return h.astype("Int64")


class ARGarchEstimator:
    """(μ, ρ, σ) 估计器.

    基础路径: 复用 AR1Factor 的 OLS 估计 (含截距), 见 factors.statistical_ext.
    增强路径: 当 `arch` 库可用且 use_garch=True, 滚动拟合 GARCH(1,1),
              用 GARCH 拟合得到的条件波动率替换 OLS 残差标准差, 吸收波动率聚集.

    自适应窗口: 可选 Kaufman 效率比驱动的变窗口 (默认 use_adaptive=True).

    输出:
        estimate() 返回 dict[str, DataFrame]:
            mu, rho, sigma, window_used  (key 缺失当未启用).
        每个 DataFrame 与 prices 等形, 滚动起始前的行填 NaN.

    参数:
        base_window:  基础/最小窗口 (默认 60)
        use_adaptive: 是否启用 Kaufman 自适应窗口 (默认 True)
        h_min/h_max:  自适应窗口上下限 (默认 21/250)
        er_window:    Kaufman ER 窗口 (默认 30)
        use_garch:    是否使用 GARCH(1,1) (默认 False, 需 `arch` 库)
        garch_p/q:    GARCH 阶数 (默认 1/1)
    """

    def __init__(
        self,
        base_window: int = 60,
        use_adaptive: bool = True,
        h_min: int = 21,
        h_max: int = 250,
        er_window: int = 30,
        use_garch: bool = False,
        garch_p: int = 1,
        garch_q: int = 1,
    ):
        self.base_window = base_window
        self.use_adaptive = use_adaptive
        self.h_min = h_min
        self.h_max = h_max
        self.er_window = er_window
        self.use_garch = use_garch
        self.garch_p = garch_p
        self.garch_q = garch_q

    def _ar1_params(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """用 AR1Factor 估计 (μ, ρ, σ), 固定窗口."""
        return AR1Factor(window=self.base_window).raw_value(prices)

    def _adaptive_windows(self, prices: pd.DataFrame) -> pd.DataFrame:
        """逐列计算 Kaufman 自适应窗口."""
        return prices.apply(lambda s: kaufman_adaptive_window(
            s, self.h_min, self.h_max, self.er_window,
        ))

    def _garch_sigma(self, prices: pd.DataFrame, base_sigma: pd.DataFrame) -> pd.DataFrame:
        """对每列用 GARCH(1,1) 估计条件波动率, 替换 OLS 残差 sigma.

        `arch` 不可用时返回 base_sigma 原值.
        """
        try:
            from arch import arch_model
        except ImportError:
            return base_sigma
        ret = prices.pct_change(fill_method=None).dropna()
        out = base_sigma.copy()
        for col in prices.columns:
            s = (ret[col].dropna() * 100).dropna()
            if len(s) < self.base_window + 10:
                continue
            try:
                am = arch_model(s, vol="GARCH", p=self.garch_p, q=self.garch_q, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                cv = res.conditional_volatility / 100
                out.loc[cv.index, col] = cv.values
            except Exception:
                continue
        return out

    def estimate(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """执行估计.

        Returns:
            dict 至少含 'mu', 'rho', 'sigma' (与 prices 等形的 DataFrame).
            启用自适应时额外含 'window_used' (每个时点实际使用的窗口长度).
        """
        if self.use_adaptive:
            windows = self._adaptive_windows(prices)
            params = {"mu": pd.DataFrame(np.nan, index=prices.index, columns=prices.columns),
                      "rho": pd.DataFrame(np.nan, index=prices.index, columns=prices.columns),
                      "sigma": pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)}
            arr = prices.pct_change(fill_method=None).to_numpy()
            win_arr = windows.to_numpy()
            for j, col in enumerate(prices.columns):
                y = arr[:, j]
                w = win_arr[:, j]
                # 防止初始 NaN: 在窗口有效时再计算
                for i in range(self.h_max, len(y) + 1):
                    h = int(w[i - 1]) if pd.notna(w[i - 1]) else self.base_window
                    h = max(h, self.base_window)
                    seg = y[i - h:i]
                    if np.isnan(seg).any() or len(seg) < 10:
                        continue
                    try:
                        n = len(seg) - 1
                        x = np.column_stack([np.ones(n), seg[:-1]])
                        beta, *_ = np.linalg.lstsq(x, seg[1:], rcond=None)
                        resid = seg[1:] - x @ beta
                        sig = float(np.sqrt(np.mean(resid ** 2)))
                        params["mu"].iat[i - 1, j] = float(beta[0])
                        params["rho"].iat[i - 1, j] = float(beta[1])
                        params["sigma"].iat[i - 1, j] = sig
                    except (np.linalg.LinAlgError, ValueError):
                        continue
            params["window_used"] = windows
        else:
            params = self._ar1_params(prices)

        if self.use_garch:
            params["sigma"] = self._garch_sigma(prices, params["sigma"])

        return params
