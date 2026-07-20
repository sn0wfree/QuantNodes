# coding=utf-8
"""v7.11 增强因子: 10 个新因子 (来自 comovement ARWS 40 维特征).

来源: ~/Public/comovement/resonance_warning/data/features.py (只读, 不修改)
新增因子 (不与现有 36 因子重叠):
  1. skew_60d — 60 日偏度 (尾部风险)
  2. kurt_60d — 60 日峰度 (极端事件)
  3. max_dd_60d — 60 日最大回撤 (近期风险)
  4. macd_hist — MACD 柱状图 (趋势动量)
  5. bollinger_pos — 布林带位置 (超买超卖)
  6. atr_14d — 平均真实波幅 (波动率)
  7. market_beta — 市场 beta (系统性风险暴露)
  8. return_dispersion — 截面离散度 (市场分化)
  9. tail_cooccurrence — 尾部共现 (极端事件同步性)
  10. corr_to_vix — 与 VIX 相关性 (风险偏好)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_skewness(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """60 日偏度 (尾部风险)."""
    return returns.rolling(window, min_periods=window // 2).skew()


def compute_kurtosis(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """60 日峰度 (极端事件)."""
    return returns.rolling(window, min_periods=window // 2).kurt()


def compute_max_drawdown(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """60 日最大回撤 (近期风险)."""
    rolling_max = close.rolling(window, min_periods=window // 2).max()
    return close / rolling_max - 1


def compute_macd_hist(close: pd.DataFrame) -> pd.DataFrame:
    """MACD 柱状图 (趋势动量)."""
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - macd_signal


def compute_bollinger_pos(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """布林带位置 (超买超卖)."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower + 1e-10)


def compute_atr(returns: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """平均真实波幅 (波动率). 用绝对收益近似."""
    return returns.abs().rolling(window, min_periods=window // 2).mean()


def compute_market_beta(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """市场 beta (系统性风险暴露)."""
    market_ret = returns.mean(axis=1)
    rolling_cov = returns.rolling(window, min_periods=window // 2).cov(market_ret)
    rolling_var = market_ret.rolling(window, min_periods=window // 2).var()
    return rolling_cov.div(rolling_var + 1e-10, axis=0)


def compute_return_dispersion(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """截面离散度 (市场分化)."""
    cross_std = returns.std(axis=1)
    # 广播到所有资产
    dispersion = pd.concat([cross_std] * returns.shape[1], axis=1)
    dispersion.columns = returns.columns
    return dispersion.rolling(window).mean()


def compute_tail_cooccurrence(returns: pd.DataFrame, window: int = 60, quantile: float = 0.90) -> pd.DataFrame:
    """尾部共现 (极端事件同步性)."""
    market_ret = returns.mean(axis=1)
    market_extreme = (market_ret.abs() > market_ret.abs().rolling(window).quantile(quantile)).astype(float)
    asset_extreme = (returns.abs() > returns.abs().rolling(window).quantile(quantile)).astype(float)
    joint_tail = (asset_extreme.multiply(market_extreme, axis=0)).rolling(window).sum()
    marginal_tail = asset_extreme.rolling(window).sum()
    return joint_tail / (marginal_tail + 1e-10)


def compute_corr_to_vix(returns: pd.DataFrame, vix: pd.Series, window: int = 60) -> pd.DataFrame:
    """与 VIX 相关性 (风险偏好)."""
    vix_aligned = vix.reindex(returns.index).ffill()
    return returns.rolling(window, min_periods=window // 2).corr(vix_aligned)


def compute_all_v7_11_factors(
    daily_returns: pd.DataFrame,
    daily_close: pd.DataFrame,
    vix: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    """计算全部 10 个 v7.11 增强因子 (日频).

    Parameters:
        daily_returns: 日频收益 DataFrame (T_daily, N_assets)
        daily_close: 日频收盘价 DataFrame (T_daily, N_assets)
        vix: VIX 日频数据 (可选)

    Returns:
        dict, 因子名 → 日频 DataFrame (T_daily, N_assets)
    """
    factors = {}

    factors["skew_60d"] = compute_skewness(daily_returns)
    factors["kurt_60d"] = compute_kurtosis(daily_returns)
    factors["max_dd_60d"] = compute_max_drawdown(daily_close)
    factors["macd_hist"] = compute_macd_hist(daily_close)
    factors["bollinger_pos"] = compute_bollinger_pos(daily_close)
    factors["atr_14d"] = compute_atr(daily_returns)
    factors["market_beta"] = compute_market_beta(daily_returns)
    factors["return_dispersion"] = compute_return_dispersion(daily_returns)
    factors["tail_cooccurrence"] = compute_tail_cooccurrence(daily_returns)

    if vix is not None:
        factors["corr_to_vix"] = compute_corr_to_vix(daily_returns, vix)
    else:
        # 用 VIX 数据文件
        from pathlib import Path
        vix_path = Path("data/high_freq_macro/macro_vix_daily.parquet")
        if vix_path.exists():
            vix_data = pd.read_parquet(vix_path)
            if isinstance(vix_data, pd.DataFrame):
                vix_series = vix_data.iloc[:, 0]
            else:
                vix_series = vix_data
            factors["corr_to_vix"] = compute_corr_to_vix(daily_returns, vix_series)
        else:
            # 无 VIX 数据, 用 0 填充
            factors["corr_to_vix"] = pd.DataFrame(0.0, index=daily_returns.index, columns=daily_returns.columns)

    return factors


def resample_factors_to_weekly(
    daily_factors: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """日频因子 → 周频 (周末值)."""
    weekly_factors = {}
    for name, df in daily_factors.items():
        weekly_factors[name] = df.resample("W-FRI").last()
    return weekly_factors


def get_factor_names() -> list[str]:
    """返回 10 个新因子名."""
    return [
        "skew_60d", "kurt_60d", "max_dd_60d",
        "macd_hist", "bollinger_pos", "atr_14d",
        "market_beta", "return_dispersion", "tail_cooccurrence", "corr_to_vix",
    ]
