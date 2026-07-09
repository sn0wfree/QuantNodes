# coding=utf-8
"""风格轮动子策略 (Stage 17, v4.0).

思路 (金融街证券《风格轮动与因子择时》):
- 5 个风格组: 大盘/中盘/成长/科创/红利
- 月度调仓, 选 Top-K 风格组
- 每个风格组选代表 ETF (现版本每个组 1 只)
- 加权: 风格得分加权

信号 (style_rotation_score):
    group_score = rank_pct(60d_return) + 0.3 × rank_pct(trend_strength)
    trend_strength = (close - MA60) / MA60  (距离 60 日均线的偏离)

参考: reports/momentum_etf_rotation/v4/STAGE17_PLAN.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .universe_v4 import (
    STYLE_GROUP_CODES,
    STYLE_GROUP_METAS,
    StyleGroup,
    load_smartbeta_panel,
)
from .sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)


@dataclass
class StyleRotationConfig(SubStrategyConfig):
    """风格轮动子策略配置."""
    name: str = "style_rotation"
    lookback: int = 60                # 动量窗口
    trend_lookback: int = 60          # 趋势窗口
    trend_weight: float = 0.3         # 趋势权重
    top_n_styles: int = 3             # 选 Top-N 风格
    top_n_per_style: int = 1          # 每风格选 ETF 数
    min_history: int = 144
    max_weight: float = 0.20


def style_rotation_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_codes: dict[StyleGroup, tuple[str, ...]],
    lookback: int = 60,
    trend_lookback: int = 60,
    trend_weight: float = 0.3,
) -> pd.Series:
    """计算风格组得分 (越大越强).

    算法:
        group_score = rank_pct(60d_return) + trend_weight × rank_pct(trend_strength)
        trend_strength = (close - MA_trend_lookback) / MA_trend_lookback
                         (用组内最强 ETF 的 trend_strength)

    Args:
        nav_df: 价格面板 (index=date, columns=code)
        as_of: 当前日期
        group_codes: 风格组 → 代表 ETF codes
        lookback: 动量窗口
        trend_lookback: 趋势窗口
        trend_weight: 趋势权重

    Returns:
        pd.Series, index=StyleGroup, values=score (越大越强)
    """
    sub = nav_df.loc[:as_of]
    if len(sub) < max(lookback, trend_lookback) + 1:
        return pd.Series(dtype=float)

    # 1. 动量 (60 日收益率, 组内最强)
    group_momentum: dict[StyleGroup, float] = {}
    for group, codes in group_codes.items():
        valid_codes = [c for c in codes if c in sub.columns]
        if not valid_codes:
            continue
        sub_codes = sub[valid_codes].iloc[-1] / sub[valid_codes].iloc[-lookback - 1] - 1.0
        sub_codes = sub_codes.dropna()
        if len(sub_codes) > 0:
            group_momentum[group] = float(sub_codes.max())

    if not group_momentum:
        return pd.Series(dtype=float)

    # 排名分位
    mom_series = pd.Series(group_momentum)
    rank_mom = mom_series.rank(method="average", pct=True, na_option="bottom")

    # 2. 趋势强度 (组内最强 ETF)
    group_trend: dict[StyleGroup, float] = {}
    for group, codes in group_codes.items():
        valid_codes = [c for c in codes if c in sub.columns]
        if not valid_codes:
            continue
        sub_codes = sub[valid_codes]
        if len(sub_codes) < trend_lookback:
            continue
        ma = sub_codes.iloc[-trend_lookback:].mean()
        latest = sub_codes.iloc[-1]
        trend = (latest / ma - 1.0).dropna()
        if len(trend) > 0:
            group_trend[group] = float(trend.max())

    if group_trend:
        trend_series = pd.Series(group_trend)
        rank_trend = trend_series.rank(method="average", pct=True, na_option="bottom")
    else:
        rank_trend = pd.Series(0.0, index=rank_mom.index)

    # 3. 综合得分
    score = rank_mom + trend_weight * rank_trend
    # 用 values 排序 (避免 Enum 比较错误)
    return score.iloc[score.values.argsort()[::-1]]


def select_top_styles(
    scores: pd.Series,
    top_n: int = 3,
) -> list[StyleGroup]:
    """从风格得分选 Top-N 风格."""
    if scores.empty:
        return []
    ranked = scores.sort_values(ascending=False)
    return [g for g in ranked.index if isinstance(g, StyleGroup)][:top_n]


def style_etf_picks(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_codes: dict[StyleGroup, tuple[str, ...]],
    top_styles: Sequence[StyleGroup],
    top_n_per_style: int = 1,
) -> list[str]:
    """对每个选中的风格, 选其代表 ETF 中动量最强."""
    sub = nav_df.loc[:as_of]
    if len(sub) < 61:
        return []

    picks: list[str] = []
    for group in top_styles:
        codes = group_codes.get(group, ())
        valid_codes = [c for c in codes if c in sub.columns]
        if not valid_codes:
            continue
        ret = sub[valid_codes].iloc[-1] / sub[valid_codes].iloc[-61] - 1.0
        ret = ret.dropna().sort_values(ascending=False)
        picks.extend(ret.index[:top_n_per_style].tolist())

    return picks


class StyleRotationSubStrategy(SubStrategy):
    """风格轮动子策略 (v4.0).

    选股逻辑:
        1. 对 5 个风格组打分 (动量 + 趋势强度)
        2. 选 Top-N 风格 (默认 3)
        3. 每个风格选 top_n_per_style (默认 1) ETF
        4. 加权: 风格得分加权

    调仓: 月度
    """

    def __init__(self, config: StyleRotationConfig):
        super().__init__(config)
        self.config: StyleRotationConfig = config

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        """选 top_n_styles × top_n_per_style 个 ETF."""
        if self.config.min_history > 0 and len(nav_df) < self.config.min_history:
            return []

        scores = style_rotation_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            lookback=self.config.lookback,
            trend_lookback=self.config.trend_lookback,
            trend_weight=self.config.trend_weight,
        )
        if scores.empty:
            return []

        top_styles = select_top_styles(scores, self.config.top_n_styles)
        picks = style_etf_picks(
            nav_df, as_of, STYLE_GROUP_CODES,
            top_styles, self.config.top_n_per_style,
        )
        return picks

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """加权: 风格得分加权 (等权 fallback)."""
        if not codes:
            return {}
        # 反查每个 code 的风格组得分
        scores = style_rotation_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            lookback=self.config.lookback,
            trend_lookback=self.config.trend_lookback,
            trend_weight=self.config.trend_weight,
        )

        weights: dict[str, float] = {}
        for code in codes:
            grp = None
            for g, cs in STYLE_GROUP_CODES.items():
                if code in cs:
                    grp = g
                    break
            if grp and grp in scores.index:
                weights[code] = float(scores[grp])
            else:
                weights[code] = 1.0  # fallback

        # 归一化
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

        # signal_strength: 平均风格得分
        scores = style_rotation_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            lookback=self.config.lookback,
            trend_lookback=self.config.trend_lookback,
            trend_weight=self.config.trend_weight,
        )
        signal = float(scores.iloc[:self.config.top_n_styles].mean()) if len(scores) > 0 else 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": self.config.name,
                "lookback": self.config.lookback,
                "trend_weight": self.config.trend_weight,
                "top_n_styles": self.config.top_n_styles,
            },
        )


__all__ = [
    "StyleRotationConfig",
    "StyleRotationSubStrategy",
    "style_rotation_score",
    "select_top_styles",
    "style_etf_picks",
]
