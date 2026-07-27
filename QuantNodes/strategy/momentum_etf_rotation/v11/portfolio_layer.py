# coding=utf-8
"""Layer 5: 组合构建 (Portfolio Construction).

来源: v4 multi_strategy_v4 (RP 底仓 + tilt 合成) + v9 position (动态仓位)

功能:
    1. 底仓: 风险平价 (RP) 或等权 (EW)
    2. Layer 2A 行业调整 (sector_tilt)
    3. Layer 2C 因子选股 (factor_tilt)
    4. Layer 4 动态仓位 (position_size)
    5. 综合: w_final = position × base_rp × sector_tilt × factor_tilt

输出:
    weights: (T, N) DataFrame, sum=position_size (每行 ≤ position_size)
"""
from __future__ import annotations

import pandas as pd

from .config_v11 import PortfolioLayerConfig


def _compute_base_weights(
    returns_df: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: PortfolioLayerConfig,
) -> pd.Series:
    """计算底仓权重 (RP 或 EW)."""
    codes = returns_df.columns.tolist()
    n = len(codes)

    if cfg.base_method == 'equal_weight':
        return pd.Series(1.0 / n, index=codes)

    # 风险平价
    sub = returns_df.loc[:as_of].iloc[:-1]
    if len(sub) < cfg.rp_lookback + 5:
        return pd.Series(1.0 / n, index=codes)

    vol = sub.iloc[-cfg.rp_lookback:].std()
    inv_vol = 1.0 / (vol + 1e-10)
    w = inv_vol / inv_vol.sum()
    return w.fillna(1.0 / n)


def _apply_cap_floor(
    weights: pd.Series,
    cap: float,
    floor: float,
) -> pd.Series:
    """约束权重上下限 (迭代)."""
    if cap >= 1.0:
        return weights
    w = weights.copy()
    for _ in range(10):
        excess = 0.0
        for c in w.index:
            if w[c] > cap:
                excess += w[c] - cap
                w[c] = cap
        if excess <= 1e-6:
            break
        non_capped = [c for c in w.index if w[c] < cap]
        non_capped_sum = w[non_capped].sum()
        if non_capped_sum > 0 and non_capped:
            for c in non_capped:
                w[c] += excess * (w[c] / non_capped_sum)
    w = w.clip(lower=floor)
    return w / w.sum()


def build_final_weights_at(
    returns_df: pd.DataFrame,
    as_of: pd.Timestamp,
    sector_tilt: pd.Series,
    factor_tilt: pd.Series,
    position_size: float,
    cfg: PortfolioLayerConfig,
) -> pd.Series:
    """构造单期最终权重.

    参数:
        returns_df: ETF 收益
        as_of: 当前日期
        sector_tilt: (N,) 行业调整系数
        factor_tilt: (N,) 因子选股权重 (sum=1)
        position_size: 动态仓位 ∈ [0.2, 1.0]
        cfg: PortfolioLayerConfig

    返回:
        weights: (N,) Series, sum=position_size
    """
    codes = returns_df.columns.tolist()

    # 1. 底仓
    base_w = _compute_base_weights(returns_df, as_of, cfg)

    # 2. 行业调整 (sector_tilt 是相对 1/N 的系数, 例如 5x = 5.0)
    if sector_tilt is not None and not sector_tilt.empty:
        sector_tilt = sector_tilt.reindex(codes, fill_value=1.0)
    else:
        sector_tilt = pd.Series(1.0, index=codes)

    # 3. 因子选股 (factor_tilt 是 sum=1 的权重)
    if factor_tilt is not None and not factor_tilt.empty:
        factor_tilt = factor_tilt.reindex(codes, fill_value=0.0)
    else:
        factor_tilt = pd.Series(1.0 / len(codes), index=codes)

    # 4. 合成: base × sector_tilt × factor_tilt
    # factor_tilt 已 sum=1, base × sector_tilt 需要归一化
    raw = base_w * sector_tilt
    raw_sum = raw.sum()
    if raw_sum > 0:
        raw = raw / raw_sum
    else:
        raw = pd.Series(1.0 / len(codes), index=codes)

    # 5. 与 factor_tilt 加权 (默认 50/50, 简化: 直接相乘并归一化)
    # 注: factor_tilt 已经包含选股逻辑, base × sector_tilt 提供底仓结构
    combined = raw * factor_tilt
    combined_sum = combined.sum()
    if combined_sum > 0:
        combined = combined / combined_sum
    else:
        combined = pd.Series(1.0 / len(codes), index=codes)

    # 6. 上下限
    combined = _apply_cap_floor(combined, cfg.cap, cfg.floor)
    combined = combined / combined.sum()

    # 7. 应用动态仓位
    final = combined * position_size
    final = final.clip(lower=0)

    return final


def build_final_weights(
    returns_df: pd.DataFrame,
    rebal_dates: pd.DatetimeIndex,
    sector_tilt: pd.DataFrame,
    factor_tilt: pd.DataFrame,
    position_size: pd.Series,
    cfg: PortfolioLayerConfig,
) -> pd.DataFrame:
    """Layer 5 主入口: 构建最终权重时序.

    参数:
        returns_df: ETF 收益
        rebal_dates: 调仓日期
        sector_tilt: (T_rebal, N) Layer 2A
        factor_tilt: (T_rebal, N) Layer 2C
        position_size: (T,) Layer 4
        cfg: PortfolioLayerConfig

    返回:
        weights: (T_rebal, N) DataFrame
    """
    if not cfg.enabled:
        n = len(returns_df.columns)
        return pd.DataFrame(1.0 / n, index=rebal_dates, columns=returns_df.columns)

    weights_list = []
    for date in rebal_dates:
        s_tilt = sector_tilt.loc[date] if date in sector_tilt.index else pd.Series(dtype=float)
        f_tilt = factor_tilt.loc[date] if date in factor_tilt.index else pd.Series(dtype=float)
        pos = position_size.get(date, 1.0) if position_size is not None else 1.0

        w = build_final_weights_at(returns_df, date, s_tilt, f_tilt, pos, cfg)
        weights_list.append(w)

    return pd.DataFrame(weights_list, index=rebal_dates)


# ============================================================
# 类封装
# ============================================================
class PortfolioLayer:
    """Layer 5 组合构建封装."""

    def __init__(self, cfg: PortfolioLayerConfig | None = None):
        self.cfg = cfg or PortfolioLayerConfig()
        self.weights: pd.DataFrame | None = None

    def fit(self, returns_df: pd.DataFrame, rebal_dates: pd.DatetimeIndex,
            sector_tilt: pd.DataFrame, factor_tilt: pd.DataFrame,
            position_size: pd.Series) -> "PortfolioLayer":
        """构建最终权重时序."""
        self.weights = build_final_weights(
            returns_df, rebal_dates, sector_tilt, factor_tilt, position_size, self.cfg,
        )
        return self

    def get_weights(self, date: pd.Timestamp) -> pd.Series:
        """获取指定日期的权重."""
        if self.weights is None or date not in self.weights.index:
            n = len(self.weights.columns) if self.weights is not None else 0
            if n == 0:
                return pd.Series(dtype=float)
            return pd.Series(1.0 / n, index=self.weights.columns)
        return self.weights.loc[date]
