# coding=utf-8
"""Preprocess strategies (Phase 2.2, Strategy pattern).

将原 factor_preprocess_node.py::_preprocess_vectorized 中 3 类硬编码 if
分支 (missing fill / de-extreme / normalise) 抽象为独立的 Strategy 类:

  MissingFillStrategy (ABC)
    ├── PassThroughMissing        # 默认无操作
    └── IndustryAverageMissing     # ind_avg: 行业内均值填充

  DeExtremeStrategy (ABC)
    ├── PassThroughExtreme         # 默认无操作
    ├── MedianAbsoluteDeviationExtreme  # median ± n * MAD
    └── PercentileShrinkExtreme    # quantile clip

  NormStrategy (ABC)
    ├── PassThroughNorm            # 默认无操作
    ├── ZScoreNorm                 # (x - mean) / std
    └── RankToNormalNorm           # rank → scipy.stats.norm.ppf

  build_preprocess_strategies(missing, extreme, norm) -> (Missing, Extreme, Norm)

每个 strategy 的 apply() 接受 in-place 输入 + 必要 kwargs, 返回处理后的
DataFrame. _preprocess_vectorized 退化为简单的 3 行 dispatch.

新增预处理类型 (如 winsorize 自定义 quantile / rank_to_uniform) 只需新增
一个 Strategy 子类, _preprocess_vectorized 无需修改.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm


# ============================================================================
# MissingFillStrategy
# ============================================================================

class MissingFillStrategy(ABC):
    """缺失值填充策略. apply() 返回处理后的 DataFrame."""

    name: str = ""

    @abstractmethod
    def apply(
        self, result: pd.DataFrame, industry: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class PassThroughMissing(MissingFillStrategy):
    """默认: 不做缺失值填充."""

    name = "passthrough"

    def apply(
        self, result: pd.DataFrame, industry: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return result


class IndustryAverageMissing(MissingFillStrategy):
    """ind_avg: 用 (date, industry) 组内的均值填充缺失值.

    行为与原 _preprocess_vectorized line 120-141 一致.
    """

    name = "ind_avg"

    def apply(
        self, result: pd.DataFrame, industry: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if industry is None:
            return result
        industry_aligned = industry.reindex(
            index=result.index, columns=result.columns,
        )
        # Stack to long format for groupby transform on (date, industry).
        long = pd.DataFrame({
            "factor": result.stack(),
            "ind": industry_aligned.stack(),
        }).dropna(subset=["ind"])
        long = long[long["ind"] > 0]
        if long.empty:
            return result
        date_level = long.index.get_level_values(0)
        group_mean = long.groupby([date_level, "ind"])["factor"].transform("mean")
        filled = long["factor"].fillna(group_mean)
        if isinstance(filled.index, pd.MultiIndex):
            filled_wide = filled.unstack(level=1)
        else:
            filled_wide = filled.unstack(level=-1)
        result.loc[filled_wide.index, filled_wide.columns] = filled_wide.values
        return result


# ============================================================================
# DeExtremeStrategy
# ============================================================================

class DeExtremeStrategy(ABC):
    """去极值策略. apply() 返回处理后的 DataFrame."""

    name: str = ""

    @abstractmethod
    def apply(
        self, result: pd.DataFrame, *,
        mad_n: float = 3.0, pct_low: float = 0.01, pct_high: float = 0.99,
    ) -> pd.DataFrame:
        raise NotImplementedError


class PassThroughExtreme(DeExtremeStrategy):
    """默认: 不去极值."""

    name = "passthrough"

    def apply(
        self, result: pd.DataFrame, *,
        mad_n: float = 3.0, pct_low: float = 0.01, pct_high: float = 0.99,
    ) -> pd.DataFrame:
        return result


class MedianAbsoluteDeviationExtreme(DeExtremeStrategy):
    """median: 用 median ± n * MAD 截断.

    行为与原 _preprocess_vectorized line 144-151 一致.
    """

    name = "median"

    def apply(
        self, result: pd.DataFrame, *,
        mad_n: float = 3.0, pct_low: float = 0.01, pct_high: float = 0.99,
    ) -> pd.DataFrame:
        median_per_row = result.median(axis=1)
        mad_per_row = (result.sub(median_per_row, axis=0)).abs().median(axis=1)
        lower = median_per_row - mad_n * mad_per_row
        upper = median_per_row + mad_n * mad_per_row
        return result.clip(lower=lower, upper=upper, axis=0)


class PercentileShrinkExtreme(DeExtremeStrategy):
    """pct_shrink: 用 [pct_low, pct_high] 分位数截断.

    行为与原 _preprocess_vectorized line 152-155 一致.
    """

    name = "pct_shrink"

    def apply(
        self, result: pd.DataFrame, *,
        mad_n: float = 3.0, pct_low: float = 0.01, pct_high: float = 0.99,
    ) -> pd.DataFrame:
        q1 = result.quantile(pct_low, axis=1)
        q2 = result.quantile(pct_high, axis=1)
        return result.clip(lower=q1, upper=q2, axis=0)


# ============================================================================
# NormStrategy
# ============================================================================

class NormStrategy(ABC):
    """标准化策略. apply() 返回处理后的 DataFrame."""

    name: str = ""

    @abstractmethod
    def apply(self, result: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PassThroughNorm(NormStrategy):
    """默认: 不标准化."""

    name = "passthrough"

    def apply(self, result: pd.DataFrame) -> pd.DataFrame:
        return result


class ZScoreNorm(NormStrategy):
    """zscore: (x - mean) / std. ddof=1 与原代码一致 (line 160)."""

    name = "zscore"

    def apply(self, result: pd.DataFrame) -> pd.DataFrame:
        mean_per_row = result.mean(axis=1)
        std_per_row = result.std(axis=1, ddof=1)
        std_per_row = std_per_row.replace(0, np.nan)
        return result.sub(mean_per_row, axis=0).div(std_per_row, axis=0)


class RankToNormalNorm(NormStrategy):
    """norm: 排名 → scipy.stats.norm.ppf. 边界 rank=0/1 clip 到 (min/2, (max+1)/2).

    行为与原 _preprocess_vectorized line 163-184 一致.
    """

    name = "norm"

    def apply(self, result: pd.DataFrame) -> pd.DataFrame:
        ranks = result.rank(axis=1, pct=True)
        # Clip ranks away from 0 and 1 (avoid -inf/+inf from ppf).
        min_rank = ranks.where(ranks > 0).min(axis=1) * 0.5
        max_rank = (ranks.where(ranks < 1).max(axis=1) + 1) * 0.5
        # Default fallback for rows where min_rank/max_rank is NaN.
        min_rank = min_rank.fillna(0.01)
        max_rank = max_rank.fillna(0.99)
        lower = pd.DataFrame(
            np.broadcast_to(min_rank.values[:, None], ranks.shape),
            index=ranks.index, columns=ranks.columns,
        )
        upper = pd.DataFrame(
            np.broadcast_to(max_rank.values[:, None], ranks.shape),
            index=ranks.index, columns=ranks.columns,
        )
        ranks = ranks.where(ranks.notna(), np.nan)  # preserve NaN
        ranks = ranks.clip(lower=lower, upper=upper, axis=1)
        return pd.DataFrame(
            scipy_norm.ppf(ranks.values, 0, 1),
            index=ranks.index, columns=ranks.columns,
        )


# ============================================================================
# Factory
# ============================================================================

def build_missing_strategy(missing: str) -> MissingFillStrategy:
    """根据配置名构造 MissingFillStrategy. 未知值 → PassThroughMissing."""
    table = {
        "ind_avg": IndustryAverageMissing,
    }
    cls = table.get(missing, PassThroughMissing)
    return cls()


def build_extreme_strategy(extreme: str) -> DeExtremeStrategy:
    """根据配置名构造 DeExtremeStrategy. 未知值 → PassThroughExtreme."""
    table = {
        "median": MedianAbsoluteDeviationExtreme,
        "pct_shrink": PercentileShrinkExtreme,
    }
    cls = table.get(extreme, PassThroughExtreme)
    return cls()


def build_norm_strategy(norm: str) -> NormStrategy:
    """根据配置名构造 NormStrategy. 未知值 → PassThroughNorm."""
    table = {
        "zscore": ZScoreNorm,
        "norm": RankToNormalNorm,
    }
    cls = table.get(norm, PassThroughNorm)
    return cls()


def build_preprocess_strategies(
    missing: str, extreme: str, norm: str,
) -> Tuple[MissingFillStrategy, DeExtremeStrategy, NormStrategy]:
    """一次性构造 3 个 strategy.

    Phase 2.2: 替代 _preprocess_vectorized 内的硬编码 if 链.
    """
    return (
        build_missing_strategy(missing),
        build_extreme_strategy(extreme),
        build_norm_strategy(norm),
    )
