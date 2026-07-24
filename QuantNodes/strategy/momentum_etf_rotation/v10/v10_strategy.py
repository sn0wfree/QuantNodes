# coding=utf-8
"""v10 主入口 — 5 层架构串联.

V10Strategy: 集成所有 Layer, 一次性输出最终权重时序.
run_v10: 便捷函数.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_v10 import V10Config
from .macro_layer import MacroLayer
from .industry_layer import IndustryLayer
from .style_layer import StyleLayer
from .factor_layer import FactorLayer
from .risk_layer import RiskLayer
from .position_layer import PositionLayer
from .portfolio_layer import PortfolioLayer


def _get_rebal_dates(index: pd.DatetimeIndex, freq: str = 'W') -> pd.DatetimeIndex:
    """获取调仓日期."""
    if freq == 'W':
        # 每周调仓: 所有周
        return index
    elif freq == 'M':
        # 月频调仓: 每月最后一周
        return pd.Series(index).groupby(index.to_period('M')).max().tolist()
    return index


class V10Strategy:
    """v10 5 层架构主入口.

    使用示例:
        strategy = V10Strategy(V10Config())
        weights = strategy.run(etf_returns, macro_df)
    """

    def __init__(self, cfg: V10Config | None = None):
        self.cfg = cfg or V10Config()
        self.macro_layer = MacroLayer(self.cfg.macro)
        self.industry_layer = IndustryLayer(self.cfg.industry)
        self.style_layer = StyleLayer(self.cfg.style)
        self.factor_layer = FactorLayer(self.cfg.factor)
        self.risk_layer = RiskLayer(self.cfg.risk)
        self.position_layer = PositionLayer(self.cfg.position)
        self.portfolio_layer = PortfolioLayer(self.cfg.portfolio)

        # 中间结果
        self.weights: pd.DataFrame | None = None
        self.macro_score: pd.Series | None = None
        self.regime_state: pd.Series | None = None
        self.bear_prob: pd.Series | None = None
        self.position_size: pd.Series | None = None

    def run(
        self,
        returns_df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """运行 v10 完整策略, 输出最终权重时序.

        参数:
            returns_df: (T, N) ETF 收益
            macro_df: (T, K) 宏观因子 (Layer 1 可选)

        返回:
            weights: (T, N) DataFrame, 调仓日 sum <= position_size
        """
        rebal_dates = _get_rebal_dates(returns_df.index, self.cfg.rebal_freq)

        # === Layer 1: 宏观择时 ===
        print("[v10] Layer 1: 宏观择时 (5 因子 + 熵权 + TV-PR)")
        self.macro_layer.fit(macro_df, returns_df)
        self.macro_score = self.macro_layer.macro_score
        self.regime_state = self.macro_layer.regime_state

        # === Layer 3: 风险控制 (Jump Model) ===
        print("[v10] Layer 3: Jump Model 风险控制")
        self.risk_layer.fit(returns_df)
        self.bear_prob = self.risk_layer.bear_prob

        # === Layer 2A: 行业轮动 ===
        print("[v10] Layer 2A: 行业轮动 (regime 条件)")
        self.industry_layer.fit(returns_df, rebal_dates, self.regime_state)

        # === Layer 2B: 风格轮动 ===
        print("[v10] Layer 2B: 风格轮动 (IC 驱动)")
        self.style_layer.fit(returns_df, rebal_dates, self.regime_state)

        # === Layer 2C: 因子选股 ===
        print("[v10] Layer 2C: 因子选股 (5 因子 + K=10)")
        self.factor_layer.fit(returns_df, rebal_dates, self.regime_state)

        # === Layer 4: 动态仓位 ===
        print("[v10] Layer 4: 动态仓位 (pos + bear_prob)")
        self.position_layer.fit(
            macro_score=self.macro_score,
            sector_tilt=self.industry_layer.tilt,
            style_weights=self.style_layer.weights,
            bear_prob=self.bear_prob,
        )
        self.position_size = self.position_layer.position_size

        # === Layer 5: 组合构建 ===
        print("[v10] Layer 5: 组合构建 (RP × tilt × pos)")
        self.portfolio_layer.fit(
            returns_df, rebal_dates,
            sector_tilt=self.industry_layer.tilt,
            factor_tilt=self.factor_layer.weights,
            position_size=self.position_size,
        )
        self.weights = self.portfolio_layer.weights

        print(f"[v10] 完成. 权重时序: {self.weights.shape}")
        return self.weights


def run_v10(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
    cfg: V10Config | None = None,
) -> pd.DataFrame:
    """便捷函数: 跑 v10 完整策略.

    参数:
        returns_df: ETF 收益
        macro_df: 宏观因子 (可选)
        cfg: V10Config

    返回:
        weights: 权重时序
    """
    strategy = V10Strategy(cfg)
    return strategy.run(returns_df, macro_df)


__all__ = ["V10Strategy", "run_v10"]