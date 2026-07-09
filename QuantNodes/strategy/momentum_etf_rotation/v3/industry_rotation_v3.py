# coding=utf-8
"""行业轮动子策略 (Stage 16A, v3.0).

思路: A 股行业 ETF 内部轮动, 选行业动量最高者
- 池: 仅 20 只 A 股行业 ETF (与 a_broad 区别)
- 信号: 60 日动量 (中短期, 行业轮动敏感)
- 调仓: 周度 (加快对行业轮动反应, 弥补月度调仓的滞后)
- top_n: 3 只 (与动量策略的 10 只 + 反转策略的 5 只 互补, 总持仓 18 只)

参数 (IndustryRotationConfig):
- industry_lookback: 行业动量窗口 (默认 60)
- rebalance_freq: 调仓频率 (默认 "W-FRI")
- top_n: 持仓数 (默认 3)
- min_history: 最少历史 (默认 60)

信号 (industry_rotation_score):
    score = rank_pct(60d_return) + 0.2 × rank_pct(20d_momentum_accel)
    (动量 + 动量加速度)

类别: 所有 a_sector 类别 ETF
A 股宽基 (a_broad) 不参与, 单独由动量策略处理

参考: reports/momentum_etf_rotation/v2/stage16a_plan.md §2.1.3
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..common.universe import Category, ETFPool
from .sub_strategy_v3 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
    validate_sub_strategy_result,
)


@dataclass
class IndustryRotationConfig(SubStrategyConfig):
    """行业轮动子策略配置.

    继承自 SubStrategyConfig, 增加行业轮动特有参数.
    """
    name: str = "industry_rotation"
    industry_lookback: int = 60        # 行业动量窗口
    accel_lookback: int = 20           # 动量加速度窗口
    accel_weight: float = 0.2          # 加速度权重
    rebalance_freq: str = "W-FRI"      # 周度调仓
    top_n: int = 3                     # 持仓数
    min_history: int = 60


def industry_rotation_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    industry_codes: list[str],
    lookback: int = 60,
    accel_lookback: int = 20,
    accel_weight: float = 0.2,
) -> pd.Series:
    """计算行业轮动信号得分 (越大越强).

    score = rank_pct(60d_return) + accel_weight × rank_pct(20d_momentum_accel)
    其中 20d_momentum_accel = (ret_20 - ret_60) / 60 (动量加速度)

    Args:
        nav_df: 价格面板
        as_of: 当前日期
        industry_codes: 行业 ETF code 列表
        lookback: 行业动量窗口
        accel_lookback: 加速度窗口
        accel_weight: 加速度权重

    Returns:
        pd.Series, index=code, values=score
    """
    if not industry_codes:
        return pd.Series(dtype=float)

    sub = nav_df.loc[:as_of, industry_codes]
    if len(sub) < lookback + 1:
        return pd.Series(dtype=float)

    # 1. 60日收益率
    ret_60 = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
    rank_60 = ret_60.rank(method="average", pct=True, na_option="bottom")

    # 2. 动量加速度 (20日 - 60日) / 60
    ret_20 = sub.iloc[-1] / sub.iloc[-accel_lookback - 1] - 1.0
    momentum_accel = (ret_20 - ret_60) / lookback
    rank_accel = momentum_accel.rank(method="average", pct=True, na_option="bottom")

    # 3. 综合得分
    score = rank_60 + accel_weight * rank_accel

    return score


def get_industry_codes(pool: ETFPool) -> list[str]:
    """从池中提取 A 股行业 ETF codes.

    Args:
        pool: ETF 池

    Returns:
        list[str]: A 股行业 ETF code 列表
    """
    return [
        m.code for m in pool.members
        if m.category == Category.A_SECTOR
    ]


def get_rebalance_dates(
    trading_dates: pd.DatetimeIndex,
    freq: str = "W-FRI",
) -> list[pd.Timestamp]:
    """从交易日中提取调仓日 (周度/月度).

    Args:
        trading_dates: 完整交易日索引
        freq: "W-FRI" (周五) / "ME" (月末) / "W" (周)

    Returns:
        list[pd.Timestamp]: 调仓日列表
    """
    if freq == "ME":
        # 月末: 用 groupby period max (避免 resample 标签错位)
        rebal = pd.Series(trading_dates).groupby(
            trading_dates.to_period("M")
        ).max().tolist()
    elif freq.startswith("W"):
        # 周度: 用 resample
        rebal = trading_dates.to_series().resample(freq).last().dropna().tolist()
    else:
        raise ValueError(f"Unsupported freq: {freq}")
    return [pd.Timestamp(d) for d in rebal]


class IndustryRotationSubStrategy(SubStrategy):
    """行业轮动子策略 (v3.0).

    选股逻辑:
        1. 限定 A 股行业 ETF (a_sector 类别)
        2. 按 industry_rotation_score 降序选 top_n
        3. 加权: 逆波动 (与 v2 一致)

    调仓频率: 周度 (rebalance_freq), 与主策略月度互补
    """

    def __init__(self, config: IndustryRotationConfig, pool: ETFPool):
        super().__init__(config, pool)
        self.config: IndustryRotationConfig = config
        self._industry_codes = get_industry_codes(pool)

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选 top_n 个 A 股行业 ETF."""
        if self.config.min_history > 0 and len(nav_df) < self.config.min_history:
            return []

        score = industry_rotation_score(
            nav_df, as_of, self._industry_codes,
            lookback=self.config.industry_lookback,
            accel_lookback=self.config.accel_lookback,
            accel_weight=self.config.accel_weight,
        )

        if score.empty:
            return []

        ranked = score.sort_values(ascending=False)
        chosen = [c for c in ranked.index if c in self.pool.codes][:self.config.top_n]
        return chosen

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """逆波动加权 (与 v2 动量策略一致)."""
        if not codes:
            return {}
        if nav_df is None or as_of is None:
            return {c: 1.0 / len(codes) for c in codes}

        sub = nav_df.loc[:as_of, [c for c in codes if c in nav_df.columns]]
        if len(sub) < 2:
            return {c: 1.0 / len(codes) for c in codes}

        log_ret = np.log(sub / sub.shift(1))
        # 逐列 std 避免跨列 NaN 污染
        vols = {}
        for c in log_ret.columns:
            valid = log_ret[c].dropna()
            vols[c] = valid.std() * np.sqrt(252) if len(valid) >= 2 else 1.0

        inv = {c: 1.0 / v if v > 0 else 0.0 for c, v in vols.items()}
        total = sum(inv.values())
        if total <= 0:
            return {c: 1.0 / len(codes) for c in codes}
        return {c: inv.get(c, 0.0) / total for c in codes}

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        """单次调仓: select + weight + 校验."""
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, signal_strength=0.0)

        weights = self.weight(nav_df, codes, as_of)

        # 计算 signal_strength
        score = industry_rotation_score(
            nav_df, as_of, self._industry_codes,
            lookback=self.config.industry_lookback,
            accel_lookback=self.config.accel_lookback,
            accel_weight=self.config.accel_weight,
        )
        valid_score = score.reindex(codes).dropna()
        signal_strength = float(valid_score.mean()) if len(valid_score) > 0 else 0.0

        result = SubStrategyResult(
            date=as_of,
            chosen=list(codes),
            weights=weights,
            signal_strength=signal_strength,
            meta={
                "strategy": "industry_rotation",
                "lookback": self.config.industry_lookback,
                "rebalance_freq": self.config.rebalance_freq,
            },
        )
        return validate_sub_strategy_result(result, self.pool)


__all__ = [
    "IndustryRotationConfig",
    "IndustryRotationSubStrategy",
    "industry_rotation_score",
    "get_industry_codes",
    "get_rebalance_dates",
]
