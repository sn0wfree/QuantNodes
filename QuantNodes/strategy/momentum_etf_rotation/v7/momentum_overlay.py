# coding=utf-8
"""动量叠加模块 — 在 v7 宏观配置基础上叠加动量信号.

[设计原则]
- 不修改原始 v7+v2 代码 (macro_substrategy_v7_3.py)
- 独立模块, 可组合使用
- 动量只作用于 权益 + 商品 (债券动量效应弱且可能反向)

[两种整合方式]
Option A: 叠加混合 (overlay)
  w_final = (1-α) × w_FRP + α × w_momentum
  只混合 equity + commodity 部分, bond 权重不变

Option B: 第10因子 (factor)
  在 LASSO 中加 market_momentum 作为第10个宏观因子
  FRP 自然考虑动量风险贡献
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# === 资产分类 (与 macro_substrategy_v7_3.py BOND_INDICES 同级) ===
EQUITY_INDICES = ['沪深300指数', '中证500指数', '中证1000', '恒生指数']
COMMODITY_INDICES = ['南华工业品指数', '南华农产品指数', '期货结算价(连续):布伦特原油', '收盘价:沪金指数']
BOND_INDICES = [
    '中债10年期国债指数', '中债3-5年期国债指数', '中债1-3年国债财富指数',
    '中债国开行债券总指数', '中债企业债总指数',
]
# 参与动量计算的资产 (equity + commodity, 排除 bond)
MOMENTUM_UNIVERSE = EQUITY_INDICES + COMMODITY_INDICES


# ============================================================================
# 1. 动量计算
# ============================================================================

def _to_cumulative_prices(index_panel: pd.DataFrame) -> pd.DataFrame:
    """将日对数收益转为累积价格 (起点=1)."""
    return np.exp(index_panel.cumsum())


# 预计算累积价格缓存 (避免每次调用重复计算)
_price_cache: dict[int, pd.DataFrame] = {}

def _get_cumulative_prices(index_panel: pd.DataFrame) -> pd.DataFrame:
    """带缓存的累积价格计算."""
    key = id(index_panel)
    if key not in _price_cache:
        _price_cache[key] = _to_cumulative_prices(index_panel)
    return _price_cache[key]


def _price_momentum(
    index_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int,
) -> pd.Series:
    """纯价格动量: mom = P(T) / P(T-lookback) - 1."""
    prices = _get_cumulative_prices(index_panel)
    s = prices.loc[:as_of].iloc[-lookback - 1:]
    if len(s) < lookback + 1:
        return pd.Series(0.0, index=prices.columns)
    return s.iloc[-1] / s.iloc[0] - 1.0


def _slope_r2_momentum(
    index_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int,
    scale: float = 10000.0,
) -> pd.Series:
    """斜率×R² 动量: score = scale * slope * R² (90d OLS)."""
    prices = _get_cumulative_prices(index_panel)
    s = prices.loc[:as_of].iloc[-lookback:]
    if len(s) < lookback:
        return pd.Series(0.0, index=prices.columns)
    x = np.arange(len(s)).reshape(-1, 1)
    scores = {}
    for col in s.columns:
        y = (s[col] / s[col].iloc[0]).values
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            scores[col] = 0.0
            continue
        lr = LinearRegression().fit(x, y)
        slope = lr.coef_[0]
        r2 = lr.score(x, y)
        scores[col] = scale * slope * r2
    return pd.Series(scores)


def compute_momentum_score(
    index_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
    momentum_type: str = "hybrid",
    fused_weight: float = 0.5,
) -> pd.Series:
    """对 13 indices 算动量得分 (equity + commodity 有分, bond=0).

    Args:
        index_panel: 13 指数日对数收益 (累积净值或对数收益均可, 用 pct_change 前的原始值)
        as_of: 截至日期
        lookback: 回看天数
        momentum_type: "price" / "slope_r2" / "hybrid"
        fused_weight: hybrid 中 slope_r2 的权重 (0-1)

    Returns:
        pd.Series, index=13 列名, values=动量得分 (bond=0)
    """
    mom = _price_momentum(index_panel, as_of, lookback)
    sr2 = _slope_r2_momentum(index_panel, as_of, lookback)

    if momentum_type == "price":
        raw = mom
    elif momentum_type == "slope_r2":
        raw = sr2
    else:  # hybrid
        # 归一化到 [-1, 1]
        mom_max = mom.abs().max()
        sr2_max = sr2.abs().max()
        mom_norm = mom / mom_max if mom_max > 0 else mom * 0
        sr2_norm = sr2 / sr2_max if sr2_max > 0 else sr2 * 0
        raw = (1.0 - fused_weight) * mom_norm + fused_weight * sr2_norm

    # bond 动量设为 0 (不参与动量倾斜)
    result = raw.copy()
    for col in BOND_INDICES:
        if col in result.index:
            result[col] = 0.0
    return result


def scores_to_weights(scores: pd.Series, max_weight: float = 0.5) -> pd.Series:
    """动量得分 → 权重 (只对 equity+commodity, bond 保留原权重).

    正得分 → 超配, 负得分 → 低配.
    权重 = max(0, score) / sum(max(0, score)), 然后 clip 到 max_weight.
    """
    # 只取 equity + commodity
    active = scores[scores.index.isin(MOMENTUM_UNIVERSE)]
    # 正得分归一化
    pos = active.clip(lower=0)
    total = pos.sum()
    if total <= 0:
        # 所有得分非正 → 等权
        w = pd.Series(1.0 / len(active), index=active.index)
    else:
        w = pos / total
    # clip
    w = w.clip(upper=max_weight)
    # 重新归一化 (clip 后可能 <1)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    # bond 部分不参与 (返回时只返回 equity+commodity 权重)
    return w


# ============================================================================
# 2. Option A: 叠加混合 (overlay)
# ============================================================================

def apply_momentum_tilt_a(
    w: pd.Series,
    index_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
    momentum_type: str = "hybrid",
    alpha: float = 0.3,
    fused_weight: float = 0.5,
    max_weight: float = 0.5,
) -> pd.Series:
    """Option A: w_final = (1-α) × w_FRP + α × w_momentum (equity+commodity only).

    bond 权重不变, 只混合 equity+commodity 部分.
    """
    scores = compute_momentum_score(index_panel, as_of, lookback, momentum_type, fused_weight)
    w_mom = scores_to_weights(scores, max_weight)

    w_new = w.copy()
    # 只混合 equity+commodity
    for col in MOMENTUM_UNIVERSE:
        if col in w_new.index and col in w_mom.index:
            w_orig = w_new[col]
            w_blend = (1.0 - alpha) * w_orig + alpha * w_mom[col]
            w_new[col] = w_blend

    # 重新归一化 (确保 sum=1)
    total = w_new.sum()
    if total > 0:
        w_new = w_new / total
    return w_new


# ============================================================================
# 3. Option B: 第10因子 (market momentum)
# ============================================================================

def build_market_momentum_factor(
    index_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
) -> float:
    """PCA 简化版: equity+commodity 动量均值作为市场动量因子."""
    scores = compute_momentum_score(index_panel, as_of, lookback, "price")
    active = scores[scores.index.isin(MOMENTUM_UNIVERSE)]
    return float(active.mean())


def add_momentum_factor_to_panel(
    factor_panel: pd.DataFrame,
    index_panel: pd.DataFrame,
    lookback: int = 90,
) -> pd.DataFrame:
    """在 factor_panel 末尾加 market_momentum 列 (第10因子)."""
    # 对每个调仓日计算 market_momentum
    mom_values = {}
    for dt in factor_panel.index:
        mom_values[dt] = build_market_momentum_factor(index_panel, dt, lookback)

    factor_new = factor_panel.copy()
    factor_new['市场动量因子'] = pd.Series(mom_values, index=factor_new.index)
    # 填充缺失值 (ffill)
    factor_new['市场动量因子'] = factor_new['市场动量因子'].ffill().fillna(0)
    return factor_new
