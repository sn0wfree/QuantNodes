# coding=utf-8
"""Smart β 子策略 (Stage 17 + Stage 27 重构).

Stage 17 思路:
- 7 只 Smart β 工具 ETF (红利低波/低波/质量/价值/现金流/红利100/红利低波100)
- 信号: 动量 (60d return) - 偏离 (close/MA60 - 1) × 0.3

Stage 27 重构:
- 43 ETF 中没有专门的 Smart β ETF
- 改为从行业 ETF 中筛选 Smart β 代理
- 筛选条件: 价值得分 + 质量得分 + 低波得分 综合选 Top-K
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .universe_v4 import (
    SMART_BETA_CODES,
    SMART_BETA_FACTOR_TYPE,
    SMART_BETA_METAS,
    SECTOR_CODES,
    DEFENSIVE_SECTOR_CODES,
    select_smart_beta_proxy,
    SmartBetaFactor,
)
from .sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)


@dataclass
class SmartBetaConfig(SubStrategyConfig):
    """Smart β 子策略配置 (Stage 27 重构: 用代理).

    Stage 27: 43 ETF 中没有专门的 Smart β ETF, 改为从行业 ETF 筛选代理.
    Stage 28: 加入动量因子, 网格搜索最优权重 (value=0.2, quality=0.3, low_vol=0.2, momentum=0.3).
    """
    name: str = "smart_beta"
    lookback: int = 60                # 动量窗口
    deviation_lookback: int = 60      # 偏离窗口
    momentum_weight: float = 0.7      # 动量权重
    deviation_weight: float = 0.3     # 偏离权重
    top_n: int = 5                    # 持仓数 (Stage 27: 5 个代理 ETF)
    min_history: int = 144
    max_weight: float = 0.20

    # Stage 27: 代理筛选参数
    proxy_lookback: int = 60          # 代理筛选窗口
    use_proxy: bool = True            # 使用代理 (默认)
    defensive_only: bool = False      # 只选防御型代理

    # Stage 28: 4 因子权重 (网格搜索最优)
    proxy_value_weight: float = 0.20    # 价值权重
    proxy_quality_weight: float = 0.30  # 质量权重
    proxy_low_vol_weight: float = 0.20   # 低波权重
    proxy_momentum_weight: float = 0.30  # 动量权重
    proxy_zscore_norm: bool = True      # z-score 标准化
    proxy_winsorize_sigma: float = 3.0  # 去极值化阈值

    # Stage 29: 多窗口动量加权 (短/中/长综合)
    proxy_momentum_windows: tuple[int, ...] = ()  # 空 = 单一短期窗口
    proxy_momentum_window_weights: tuple[float, ...] = ()

    # Stage 29: 相关性约束
    proxy_corr_constraint: bool = False        # 启用相关性约束
    proxy_corr_threshold: float = 0.7         # 相关性阈值
    proxy_corr_window: int = 60                # 相关性窗口


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

    # 2. 偏离
    ma = sub.iloc[-deviation_lookback:].mean()
    dev = (sub.iloc[-1] / ma - 1.0).dropna()
    if dev.empty:
        return rank_mom
    rank_dev = dev.rank(method="average", pct=True, na_option="bottom")

    # 3. 综合得分
    score = momentum_weight * rank_mom - deviation_weight * rank_dev
    return score.sort_values(ascending=False)


def select_top_smart_beta(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    smart_beta_codes: Sequence[str],
    lookback: int = 60,
    top_n: int = 3,
) -> list[str]:
    """选 Top-N Smart β ETF."""
    scores = smart_beta_score(
        nav_df, as_of, smart_beta_codes, lookback=lookback,
    )
    if scores.empty:
        return []
    return scores.head(top_n).index.tolist()


def select_diversified_smart_beta(
    scores: pd.Series,
    factor_types: dict[SmartBetaFactor, str],
    top_n: int = 3,
) -> list[str]:
    """分散化选 Smart β (兼容旧版, 43 ETF 上可能不工作)."""
    if scores.empty:
        return []

    code_to_factor: dict[str, SmartBetaFactor] = {}
    for f, meta in SMART_BETA_METAS.items():
        if meta.code in scores.index:
            code_to_factor[meta.code] = f

    buckets: dict[str, list[str]] = {}
    for code in scores.index:
        factor = code_to_factor.get(code)
        if factor is None:
            continue
        ft = factor_types.get(factor, "unknown")
        buckets.setdefault(ft, []).append(code)

    picks: list[str] = []
    for ft, codes in buckets.items():
        sorted_codes = sorted(codes, key=lambda c: scores.get(c, -1), reverse=True)
        if sorted_codes:
            picks.append(sorted_codes[0])

    picks = sorted(picks, key=lambda c: scores.get(c, -1), reverse=True)
    return picks[:top_n]


class SmartBetaSubStrategy(SubStrategy):
    """Smart β 子策略 (Stage 27 重构: 用代理).

    Stage 27 改进:
    - 从 43 ETF 行业 ETF 中筛选 Smart β 代理
    - 筛选条件: 价值 + 质量 + 低波 综合得分
    - 兼容旧版接口
    """

    def __init__(self, config: SmartBetaConfig):
        super().__init__(config)
        self.config: SmartBetaConfig = config

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选 Top-N Smart β 代理 ETF."""
        if self.config.min_history > 0 and len(nav_df) < self.config.min_history:
            return []

        if self.config.use_proxy:
            # Stage 27: 用代理筛选
            returns = nav_df.pct_change().loc[:as_of].fillna(0)
            codes = DEFENSIVE_SECTOR_CODES if self.config.defensive_only else SECTOR_CODES

            # Stage 28: 4 因子权重 (网格搜索最优)
            weights = {
                "value": self.config.proxy_value_weight,
                "quality": self.config.proxy_quality_weight,
                "low_vol": self.config.proxy_low_vol_weight,
                "momentum": self.config.proxy_momentum_weight,
            }

            # Stage 29: 多窗口动量参数
            momentum_windows = (
                tuple(self.config.proxy_momentum_windows)
                if self.config.proxy_momentum_windows else None
            )
            momentum_window_weights = (
                tuple(self.config.proxy_momentum_window_weights)
                if self.config.proxy_momentum_window_weights else None
            )

            return select_smart_beta_proxy(
                returns,
                lookback=self.config.proxy_lookback,
                top_k=self.config.top_n,
                codes=codes,
                weights=weights,
                zscore_norm=self.config.proxy_zscore_norm,
                winsorize_sigma=self.config.proxy_winsorize_sigma,
                momentum_windows=momentum_windows,
                momentum_window_weights=momentum_window_weights,
                corr_constraint=self.config.proxy_corr_constraint,
                corr_threshold=self.config.proxy_corr_threshold,
                corr_window=self.config.proxy_corr_window,
            )
        else:
            # 兼容旧版: 用 7 只 Smart β ETF
            scores = smart_beta_score(
                nav_df, as_of, list(SMART_BETA_CODES.values()),
                lookback=self.config.lookback,
                deviation_lookback=self.config.deviation_lookback,
                momentum_weight=self.config.momentum_weight,
                deviation_weight=self.config.deviation_weight,
            )
            if scores.empty:
                return []
            return select_diversified_smart_beta(
                scores, SMART_BETA_FACTOR_TYPE, top_n=self.config.top_n,
            )

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """等权 + max_weight 约束."""
        if not codes:
            return {}
        w = 1.0 / len(codes)
        w = min(w, self.config.max_weight)
        weights = {code: w for code in codes}
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
        if not weights:
            return SubStrategyResult(date=as_of, meta={"strategy": self.config.name})

        # signal_strength: 等权 (代理模式下没有 scores)
        signal = 0.5 if codes else 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": self.config.name,
                "use_proxy": self.config.use_proxy,
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
