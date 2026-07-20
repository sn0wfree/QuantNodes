# coding=utf-8
"""回测工具: 换手成本, 调仓日生成, 权重约束, 日频 NAV 计算.

消除 v1-v7 中 10+ 个文件的重复代码.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_utils import (
        calculate_turnover,
        calculate_turnover_cost,
        generate_rebalance_dates,
        apply_max_weight,
        normalize_weights,
        compute_daily_nav_from_weights,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest_config import CostConfig


# ============================================================
# 1. 换手与成本 (替代 7+ 个文件中的重复定义)
# ============================================================
def calculate_turnover(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    """计算单边换手率.

    turnover = 0.5 * Σ|new_w[c] - old_w[c]| for all c in union

    Parameters:
        old_weights: 旧权重 {code: weight}
        new_weights: 新权重 {code: weight}

    Returns:
        单边换手率 (如 0.5 表示 50% 换手)
    """
    all_codes = set(old_weights) | set(new_weights)
    return sum(
        abs(new_weights.get(c, 0.0) - old_weights.get(c, 0.0))
        for c in all_codes
    ) / 2.0


def calculate_turnover_cost(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    cost_cfg: CostConfig,
) -> float:
    """计算换手成本.

    cost = turnover * cost_rate

    Parameters:
        old_weights: 旧权重
        new_weights: 新权重
        cost_cfg: 成本配置

    Returns:
        成本率 (如 0.001 表示 0.1%)
    """
    if not cost_cfg.enabled:
        return 0.0
    turnover = calculate_turnover(old_weights, new_weights)
    return turnover * cost_cfg.cost_rate()


# ============================================================
# 2. 调仓日生成 (替代 8 个文件中的重复定义)
# ============================================================
def generate_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str = "M",
    min_lookback: int | None = None,
) -> list[pd.Timestamp]:
    """从交易日索引生成调仓日.

    使用 groupby(period).max() 而非 resample().last(),
    确保输出是实际交易日 (避免日历月末非交易日问题, 见 v6_2 bug fix).

    Parameters:
        dates: 交易日索引
        freq: 调仓频率 ("M"=月度, "Q"=季度, "W"=周度, "W-FRI"=周五)
        min_lookback: 最小预热期 (如 252 表示需要 1 年数据)

    Returns:
        调仓日列表

    Raises:
        ValueError: 如果 min_lookback 过滤后无有效日期
    """
    if freq in ("M", "Q"):
        period = dates.to_period(freq)
        rebal = pd.Series(dates).groupby(period).max().tolist()
    elif freq.startswith("W"):
        rebal = dates.to_series().resample(freq).last().dropna().tolist()
        # 确保是实际交易日
        date_set = set(dates)
        rebal = [d for d in rebal if d in date_set]
    else:
        raise ValueError(f"Unsupported rebalance freq: {freq}")

    result = [pd.Timestamp(d) for d in rebal]

    if min_lookback is not None:
        valid = [d for d in result if dates.searchsorted(d) >= min_lookback]
        if not valid:
            raise ValueError(
                f"Insufficient data: need {min_lookback} periods, "
                f"got max index {dates.searchsorted(result[-1]) if result else 0}"
            )
        result = valid

    return result


# ============================================================
# 3. 权重约束 (替代 4 个文件中的重复定义)
# ============================================================
def apply_max_weight(
    weights: dict[str, float],
    max_w: float,
    max_iters: int = 50,
) -> dict[str, float]:
    """最大权重约束 (迭代法).

    超过 max_w 的权重按比例分配给未满的资产.

    Parameters:
        weights: 原始权重
        max_w: 最大权重上限
        max_iters: 最大迭代次数

    Returns:
        约束后的权重
    """
    w = dict(weights)
    for _ in range(max_iters):
        excess = {c: v - max_w for c, v in w.items() if v > max_w + 1e-10}
        if not excess:
            break
        total_excess = sum(excess.values())
        for c in excess:
            w[c] = max_w
        remaining = {c: v for c, v in w.items() if v < max_w - 1e-10}
        total_remaining = sum(remaining.values())
        if total_remaining > 0:
            for c in remaining:
                w[c] += total_excess * (w[c] / total_remaining)
    return w


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """权重归一化 (和为 1).

    Parameters:
        weights: 原始权重

    Returns:
        归一化后的权重
    """
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {c: v / total for c, v in weights.items()}


# ============================================================
# 4. 日频 NAV 计算 (统一 v1-v7)
# ============================================================
def compute_daily_nav_from_weights(
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]],
    daily_returns: pd.DataFrame,
    cost_cfg: CostConfig,
) -> pd.Series:
    """从权重历史 + 日频收益, 计算日频 NAV.

    统一逻辑 (v1-v7 通用):
    1. 权重在调仓日(t)生成, 从下一个交易日(t+1)开始生效
    2. 每个交易日: daily_ret = Σ(w * ret)
    3. 调仓日扣减换手成本
    4. 累积 NAV

    Parameters:
        weights_history: [(rebal_date, weights), ...] 权重历史
        daily_returns: (T_daily, N) 日频收益 DataFrame
        cost_cfg: 成本配置

    Returns:
        (T_daily,) 日频 NAV Series
    """
    all_dates = daily_returns.index
    nav = pd.Series(1.0, index=all_dates, dtype=float)

    if not weights_history:
        return nav

    # 构建: 生效日期 → 权重
    # 权重在 rebal_date 生成, 从 rebal_date 之后的第一个交易日开始生效
    date_to_weights: dict[pd.Timestamp, dict[str, float]] = {}
    for rebal_date, weights in weights_history:
        after = all_dates[all_dates > rebal_date]
        if len(after) > 0:
            date_to_weights[after[0]] = weights

    current_weights: dict[str, float] = {}

    for i in range(1, len(all_dates)):
        date = all_dates[i]

        # 检查是否有新权重生效
        cost = 0.0
        if date in date_to_weights:
            old_weights = current_weights
            current_weights = date_to_weights[date]
            cost = calculate_turnover_cost(old_weights, current_weights, cost_cfg)

        # 计算日收益
        daily_ret = 0.0
        for code, w in current_weights.items():
            if code in daily_returns.columns:
                ret = daily_returns.loc[date, code]
                if pd.notna(ret):
                    daily_ret += w * ret

        # 调仓日扣减成本
        daily_ret -= cost

        # 累积 NAV
        nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)

    return nav
