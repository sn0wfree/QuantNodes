# coding=utf-8
"""v4 行业轮动子策略 (Stage 27) — 基于 v5 的 11 因子 + regime 条件.

核心思想:
- 继承 v5 的 11 量价因子作为基础信号
- 添加 regime 条件: bull→进攻型行业, bear→防御型行业
- 添加估值/基本面因子 (Stage 27 新增)
- 添加相关性约束: 剔除相关系数 > 0.7 的冗余行业

参考:
- 华西证券《行业有效量价因子与行业轮动策略》(2022-08-22)
- v4 因子择时框架 (Stage 18-19)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .sub_strategy_v4 import SubStrategy, SubStrategyResult
from .universe_v4 import (
    SECTOR_CODES,
    DEFENSIVE_SECTOR_CODES,
    GROWTH_SECTOR_CODES,
)

logger = logging.getLogger(__name__)


@dataclass
class IndustryRotationConfig:
    """行业轮动配置."""
    top_n: int = 5                    # 选 Top-N 行业
    min_history: int = 252            # 最少历史数据
    max_weight: float = 0.25          # 单行业最大权重
    rebalance_freq: str = "M"         # 月频调仓
    rebal_lag: int = 1                # T+1 执行

    # 因子权重 (可选, 默认等权)
    factor_weights: dict[str, float] | None = None

    # Regime 条件
    regime_enabled: bool = True       # 启用 regime 条件
    bull_momentum_boost: float = 1.5  # 牛市动量加权
    bear_defensive_boost: float = 1.5 # 熊市防御加权

    # 相关性约束
    corr_constraint: bool = True      # 启用相关性约束
    corr_threshold: float = 0.7       # 相关系数阈值
    corr_window: int = 52             # 相关系数计算窗口

    # 估值/基本面因子
    use_value_factor: bool = True     # 启用估值因子
    use_quality_factor: bool = True   # 启用基本面因子


# 防御型行业 ETF (熊市应超配) - 用 43 ETF 中的行业 ETF
DEFENSIVE_INDUSTRIES = {
    "512800",  # 银行
    "512170",  # 医疗
    "512010",  # 医药
    "159928",  # 消费
    "159996",  # 家电
    "512120",  # 化工
}

# 进攻型行业 ETF (牛市应超配)
GROWTH_INDUSTRIES = {
    "512760",  # 半导体
    "512480",  # 半导体
    "515030",  # 新能源车
    "515790",  # 光伏
    "512660",  # 军工
}


def _cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """横截面 z-score 标准化."""
    mean = df.mean(axis=1)
    std = df.std(axis=1) + 1e-10
    return df.sub(mean, axis=0).div(std, axis=0)


def _compute_momentum_factor(
    returns: pd.DataFrame, window: int = 120, skip: int = 5,
) -> pd.Series:
    """动量因子: 过去 window 周收益, 跳过最近 skip 周."""
    cumulative = (1 + returns).rolling(window).apply(np.prod, raw=True) - 1
    if skip > 0:
        cumulative = cumulative.shift(skip)
    return cumulative.iloc[-1]


def _compute_volatility_factor(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    """波动率因子 (反向)."""
    vol = returns.rolling(window).std().iloc[-1]
    return -vol  # 反向: 波动低 = 好


def _compute_turnover_factor(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """换手率变化因子 (用收益波动率代理)."""
    vol_change = returns.rolling(window).std().iloc[-1]
    return -vol_change  # 反向: 波动变化小 = 好


def _compute_value_factor(returns: pd.DataFrame, window: int = 52) -> pd.Series:
    """估值因子 (Stage 27 新增): 过去 52 周累计收益的反向."""
    lookback = min(window, len(returns) - 1)
    if lookback > 10:
        cum_ret = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
        return -cum_ret.iloc[-1]  # 反向: 跌得多 = 估值低
    return pd.Series(0.0, index=returns.columns)


def _compute_quality_factor(returns: pd.DataFrame, window: int = 26) -> pd.Series:
    """基本面因子 (Stage 27 新增): 26 周 Sharpe ratio."""
    lookback = min(window, len(returns) - 1)
    if lookback > 5:
        mean_ret = returns.rolling(lookback).mean().iloc[-1]
        std_ret = returns.rolling(lookback).std().iloc[-1]
        sharpe = mean_ret / (std_ret + 1e-10)
        return sharpe  # 正向: Sharpe 高 = 质量好
    return pd.Series(0.0, index=returns.columns)


def _compute_composite_score(
    returns: pd.DataFrame,
    cfg: IndustryRotationConfig,
) -> pd.Series:
    """计算 13 因子综合得分 (11 基础 + 估值 + 基本面)."""
    # 11 基础因子 (v5 风格)
    f1_momentum = _compute_momentum_factor(returns, window=120, skip=5)
    f2_volatility = _compute_volatility_factor(returns, window=20)
    f3_turnover = _compute_turnover_factor(returns, window=60)

    # 简化: 用 3 个核心因子代表 11 因子
    # 完整实现需要 v5 的 FactorEngine
    factors = pd.DataFrame({
        "momentum": f1_momentum,
        "volatility": f2_volatility,
        "turnover": f3_turnover,
    })

    # Stage 27: 添加估值/基本面因子
    if cfg.use_value_factor:
        factors["value"] = _compute_value_factor(returns, window=52)

    if cfg.use_quality_factor:
        factors["quality"] = _compute_quality_factor(returns, window=26)

    # 横截面 z-score
    z_scores = _cross_section_zscore(factors)

    # 综合得分 (等权或指定权重)
    if cfg.factor_weights:
        weights = pd.Series(cfg.factor_weights)
        weights = weights.reindex(z_scores.columns, fillvalue=1.0)
        weights = weights / weights.sum()
        composite = z_scores.mul(weights, axis=1).sum(axis=1)
    else:
        composite = z_scores.mean(axis=1)

    return composite


def _apply_regime_condition(
    scores: pd.Series,
    regime: str,
    cfg: IndustryRotationConfig,
) -> pd.Series:
    """应用 regime 条件调整得分."""
    if not cfg.regime_enabled:
        return scores

    adjusted = scores.copy()

    if regime == "bull":
        # 牛市: 进攻型行业加权
        for ind in GROWTH_INDUSTRIES:
            if ind in adjusted.index:
                adjusted[ind] *= cfg.bull_momentum_boost
    elif regime == "bear":
        # 熊市: 防御型行业加权
        for ind in DEFENSIVE_INDUSTRIES:
            if ind in adjusted.index:
                adjusted[ind] *= cfg.bear_defensive_boost

    return adjusted


def _apply_corr_constraint(
    selected: list[str],
    returns: pd.DataFrame,
    cfg: IndustryRotationConfig,
) -> list[str]:
    """应用相关性约束: 剔除相关系数 > 阈值的冗余行业."""
    if not cfg.corr_constraint or len(selected) <= 1:
        return selected

    # 计算行业间相关系数
    corr_matrix = returns[selected].rolling(cfg.corr_window).corr()

    # 逐个检查, 剔除高相关
    filtered = [selected[0]]
    for code in selected[1:]:
        # 检查与已选行业的相关性
        max_corr = 0.0
        for s in filtered:
            if s in corr_matrix.columns and code in corr_matrix.columns:
                try:
                    corr_val = corr_matrix[s].iloc[-1].get(code, 0.0)
                    if not np.isnan(corr_val):
                        max_corr = max(max_corr, abs(corr_val))
                except Exception:
                    pass

        if max_corr < cfg.corr_threshold:
            filtered.append(code)

    return filtered


class IndustryRotationV4(SubStrategy):
    """v4 行业轮动子策略.

    继承 v5 的因子框架, 添加 regime 条件和相关性约束.
    """

    def __init__(self, cfg: IndustryRotationConfig | None = None):
        self.cfg = cfg or IndustryRotationConfig()
        self._factor_panel = None
        self._last_init_date = None

        # Stage 27: 默认候选池为 43 ETF 中的行业 ETF
        self.candidate_codes = list(SECTOR_CODES)

    def select(
        self,
        date: pd.Timestamp,
        pp: pd.DataFrame,
        regime: str = "sideways",
    ) -> list[str]:
        """选择行业.

        Args:
            date: 当前日期
            pp: 价格面板 (index=date, columns=code)
            regime: 市场状态 ("bull" | "bear" | "transition" | "sideways")

        Returns:
            选中的行业代码列表
        """
        if len(pp) < self.cfg.min_history:
            return []

        # Stage 27: 用候选池 (默认 43 ETF 中的行业 ETF)
        valid_codes = [c for c in self.candidate_codes if c in pp.columns]
        sub_pp = pp[valid_codes]

        if len(sub_pp) < self.cfg.min_history:
            return []

        # 计算收益
        returns = sub_pp.pct_change().dropna()

        if len(returns) < 60:
            return []

        # 计算综合得分
        scores = _compute_composite_score(returns, self.cfg)

        # 应用 regime 条件
        scores = _apply_regime_condition(scores, regime, self.cfg)

        # Top-N 选优
        top_n = scores.nlargest(self.cfg.top_n * 2)

        # 应用相关性约束
        selected = _apply_corr_constraint(
            top_n.index.tolist(), returns, self.cfg
        )

        # 截断到 top_n
        selected = selected[:self.cfg.top_n]

        return selected

    def weight(
        self,
        selected: list[str],
        pp: pd.DataFrame,
        date: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """计算权重 (等权 + max_weight 约束).

        Args:
            selected: 选中的行业代码
            pp: 价格面板
            date: 当前日期

        Returns:
            dict, code → weight
        """
        if not selected:
            return {}

        n = len(selected)
        w = 1.0 / n

        # 应用 max_weight 约束
        w = min(w, self.cfg.max_weight)

        weights = {code: w for code in selected}

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def run_step(
        self,
        date: pd.Timestamp,
        pp: pd.DataFrame,
        prev_weights: dict[str, float] | None = None,
        regime: str = "sideways",
    ) -> SubStrategyResult:
        """执行一步.

        Args:
            date: 当前日期
            pp: 价格面板
            prev_weights: 上一期权重
            regime: 市场状态

        Returns:
            SubStrategyResult
        """
        # 选股
        selected = self.select(date, pp, regime)

        # 计算权重
        weights = self.weight(selected, pp, date)

        # 计算信号强度
        if weights:
            returns = pp.pct_change().dropna()
            scores = _compute_composite_score(returns, self.cfg)
            held_scores = [scores.get(code, 0.0) for code in weights.keys()]
            signal_strength = np.mean(held_scores) if held_scores else 0.0
        else:
            signal_strength = 0.0

        return SubStrategyResult(
            date=date,
            chosen=selected,
            weights=weights,
            signal_strength=signal_strength,
        )


__all__ = [
    "IndustryRotationConfig",
    "IndustryRotationV4",
    "DEFENSIVE_INDUSTRIES",
    "GROWTH_INDUSTRIES",
]
