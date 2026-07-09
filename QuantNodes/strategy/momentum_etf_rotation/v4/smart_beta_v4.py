# coding=utf-8
"""Smart β 子策略 (Stage 17, v4.0).

思路:
- 7 只 Smart β 工具 ETF (红利低波/低波/质量/价值/现金流/红利100/红利低波100)
- 信号: 动量 (60d return) - 偏离 (close/MA60 - 1) × 0.3
- 选 Top-3
- 加权: 因子得分加权
- 调仓: 月度

为什么 "动量 - 偏离":
- 动量: 选近期强势因子
- 偏离: 选估值合理 (避免追高)
- 综合: 强势 + 不太贵
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .universe_v4 import (
    SMART_BETA_CODES,
    SMART_BETA_FACTOR_TYPE,
    SMART_BETA_METAS,
    SmartBetaFactor,
)
from .sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)


@dataclass
class SmartBetaConfig(SubStrategyConfig):
    """Smart β 子策略配置."""
    name: str = "smart_beta"
    lookback: int = 60                # 动量窗口
    deviation_lookback: int = 60      # 偏离窗口
    momentum_weight: float = 0.7      # 动量权重
    deviation_weight: float = 0.3     # 偏离权重
    top_n: int = 3                    # 持仓数
    min_history: int = 144
    max_weight: float = 0.20


def smart_beta_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    smart_beta_codes: Sequence[str],
    lookback: int = 60,
    deviation_lookback: int = 60,
    momentum_weight: float = 0.7,
    deviation_weight: float = 0.3,
) -> pd.Series:
    """计算 Smart β 因子得分.

    算法:
        raw_score = momentum_weight × rank_pct(60d_return) - deviation_weight × rank_pct(deviation)
        其中 deviation = (close - MA60) / MA60

    Args:
        nav_df: 价格面板
        as_of: 当前日期
        smart_beta_codes: Smart β ETF code 列表
        lookback: 动量窗口
        deviation_lookback: 偏离窗口
        momentum_weight: 动量权重
        deviation_weight: 偏离权重

    Returns:
        pd.Series, index=code, values=score (越大越强)
    """
    valid = [c for c in smart_beta_codes if c in nav_df.columns]
    if not valid:
        return pd.Series(dtype=float)

    sub = nav_df.loc[:as_of, valid]
    if len(sub) < max(lookback, deviation_lookback) + 1:
        return pd.Series(dtype=float)

    # 1. 动量
    ret = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
    ret = ret.dropna()
    if ret.empty:
        return pd.Series(dtype=float)
    rank_mom = ret.rank(method="average", pct=True, na_option="bottom")

    # 2. 偏离 (close / MA60 - 1)
    ma = sub.iloc[-deviation_lookback:].mean()
    dev = (sub.iloc[-1] / ma - 1.0).dropna()
    if dev.empty:
        return rank_mom
    rank_dev = dev.rank(method="average", pct=True, na_option="bottom")

    # 3. 综合得分 (动量 - 偏离)
    score = momentum_weight * rank_mom - deviation_weight * rank_dev
    return score.sort_values(ascending=False)


def select_top_smart_beta(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    smart_beta_codes: Sequence[str],
    lookback: int = 60,
    top_n: int = 3,
) -> list[str]:
    """选 Top-N Smart β ETF (按动量加权)."""
    scores = smart_beta_score(
        nav_df, as_of, smart_beta_codes,
        lookback=lookback,
    )
    if scores.empty:
        return []
    return scores.head(top_n).index.tolist()


def select_diversified_smart_beta(
    scores: pd.Series,
    factor_types: dict[SmartBetaFactor, str],
    top_n: int = 3,
) -> list[str]:
    """分散化选 Smart β (避免单一因子集中).

    规则: 按 factor_type 分类, 至少每类 1 只, 然后按得分排序.
    """
    if scores.empty:
        return []

    # 反查: code → factor
    code_to_factor: dict[str, SmartBetaFactor] = {}
    for f, meta in SMART_BETA_METAS.items():
        if meta.code in scores.index:
            code_to_factor[meta.code] = f

    # 按 factor_type 分桶
    buckets: dict[str, list[str]] = {}
    for code in scores.index:
        factor = code_to_factor.get(code)
        if factor is None:
            continue
        ft = factor_types.get(factor, "unknown")
        buckets.setdefault(ft, []).append(code)

    # 每桶取 Top-1 (按得分), 然后合并取 Top-N
    picks: list[str] = []
    for ft, codes in buckets.items():
        # 按 scores 排序
        sorted_codes = sorted(codes, key=lambda c: scores.get(c, -1), reverse=True)
        if sorted_codes:
            picks.append(sorted_codes[0])

    # 排序 picks 并取 top_n
    picks = sorted(picks, key=lambda c: scores.get(c, -1), reverse=True)
    return picks[:top_n]


class SmartBetaSubStrategy(SubStrategy):
    """Smart β 子策略 (v4.0).

    选股逻辑:
        1. 对 7 只 Smart β 工具 ETF 打分 (动量 - 偏离)
        2. 按因子类型 (defensive/value) 分散选 Top-3
        3. 加权: 因子得分加权
    """

    def __init__(self, config: SmartBetaConfig):
        super().__init__(config)
        self.config: SmartBetaConfig = config

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选 Top-N Smart β ETF."""
        if self.config.min_history > 0 and len(nav_df) < self.config.min_history:
            return []

        scores = smart_beta_score(
            nav_df, as_of, list(SMART_BETA_CODES.values()),
            lookback=self.config.lookback,
            deviation_lookback=self.config.deviation_lookback,
            momentum_weight=self.config.momentum_weight,
            deviation_weight=self.config.deviation_weight,
        )
        if scores.empty:
            return []

        # 分散化选 (避免单一因子集中)
        return select_diversified_smart_beta(
            scores, SMART_BETA_FACTOR_TYPE, top_n=self.config.top_n,
        )

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """加权: 因子得分加权."""
        if not codes:
            return {}

        scores = smart_beta_score(
            nav_df, as_of, list(SMART_BETA_CODES.values()),
            lookback=self.config.lookback,
            deviation_lookback=self.config.deviation_lookback,
            momentum_weight=self.config.momentum_weight,
            deviation_weight=self.config.deviation_weight,
        )

        # 移到非负 + softmax 化
        weights: dict[str, float] = {}
        for code in codes:
            s = scores.get(code, 0.0)
            weights[code] = max(0.0, 1.0 + s)  # [0, 2] 范围

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        """单次调仓."""
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, meta={"strategy": self.config.name})

        weights = self.weight(nav_df, codes, as_of)
        weights = self._apply_max_weight(weights, self.config.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # signal_strength: 平均因子得分
        scores = smart_beta_score(
            nav_df, as_of, list(SMART_BETA_CODES.values()),
            lookback=self.config.lookback,
            deviation_lookback=self.config.deviation_lookback,
            momentum_weight=self.config.momentum_weight,
            deviation_weight=self.config.deviation_weight,
        )
        valid_score = scores.reindex(codes).dropna()
        signal = float(valid_score.mean()) if len(valid_score) > 0 else 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": self.config.name,
                "lookback": self.config.lookback,
                "momentum_weight": self.config.momentum_weight,
                "top_n": self.config.top_n,
            },
        )


__all__ = [
    "SmartBetaConfig",
    "SmartBetaSubStrategy",
    "smart_beta_score",
    "select_top_smart_beta",
    "select_diversified_smart_beta",
]
