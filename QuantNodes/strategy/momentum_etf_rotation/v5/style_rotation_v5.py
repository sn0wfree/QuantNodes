# coding=utf-8
"""v5 风格轮动 (StyleRotation v5.0).

基于 Stage 17 v4 诊断 (SUB_STRATEGY_DIAGNOSTIC.md) 4 大改进:

1. **强制 dividend 底仓 20%** (5 风格相关 0.86-0.90, dividend 是唯一分散器)
2. **多窗口 Long-biased 5/20/120/180** (权重 0.10/0.20/0.30/0.40, 单窗口 L=120 Calmar 仅 0.016)
3. **Top-2 选择** (Top-1 准确率 34.5%, Top-2 53.4%)
4. **Sideways regime filter** (sideways 70% 时间亏钱 -2.50% ann, 此时 50% 仓位, 50% cash)

诊断基础:
- reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md §1.1-1.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.style_rotation_v4 import (
    StyleGroup,
    style_etf_picks,
)
from ..v4.universe_v4 import (
    STYLE_GROUP_CODES,
    load_smartbeta_panel,
)
from ..v4.sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)


@dataclass
class StyleRotationV5Config(SubStrategyConfig):
    """v5 风格轮动配置 (4 改进)."""
    name: str = "style_rotation_v5"

    # 改进 2: 多窗口 Long-biased (权重 0.1/0.2/0.3/0.4)
    windows: tuple[int, ...] = (5, 20, 120, 180)
    window_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)

    # 改进 1: 强制 dividend 底仓
    dividend_floor: float = 0.20

    # 改进 3: Top-N 选择
    top_n: int = 2
    top_n_per_style: int = 1

    # 改进 4: Sideways regime filter
    regime_lookback_short: int = 60
    regime_lookback_long: int = 252
    bull_threshold: float = 0.05
    bear_threshold: float = -0.05
    long_threshold: float = 0.10
    sideways_style_exposure: float = 0.50

    rebalance_freq: str = "M"
    min_history: int = 252
    max_weight: float = 0.40


def multi_window_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_codes: dict[StyleGroup, tuple[str, ...]],
    windows: Sequence[int],
    window_weights: Sequence[float],
) -> pd.Series:
    """多窗口 Long-biased 风格得分.

    算法:
        per_window_score[L] = rank_pct(60d_return)  (用组内最强 ETF)
        combined = Σ w[L] × per_window_score[L]
        (高分 = 强势)

    Args:
        nav_df: 价格面板
        as_of: 当前日期
        group_codes: 风格组 → ETF codes
        windows: 动量窗口列表 (e.g. (5, 20, 120, 180))
        window_weights: 对应权重 (和=1)

    Returns:
        pd.Series, index=StyleGroup, values=combined score
    """
    if len(windows) != len(window_weights):
        raise ValueError("windows 和 window_weights 长度必须一致")
    if abs(sum(window_weights) - 1.0) > 1e-6:
        raise ValueError("window_weights 之和必须为 1")

    min_window = max(windows)
    sub = nav_df.loc[:as_of]
    if len(sub) < min_window + 1:
        return pd.Series(dtype=float)

    combined = pd.Series(0.0, dtype=float)
    init = False

    for L, w in zip(windows, window_weights):
        group_momentum: dict[StyleGroup, float] = {}
        for group, codes in group_codes.items():
            valid_codes = [c for c in codes if c in sub.columns]
            if not valid_codes:
                continue
            recent = sub[valid_codes].iloc[-1]
            past = sub[valid_codes].iloc[-L - 1]
            ret = (recent / past - 1.0).dropna()
            if len(ret) > 0:
                group_momentum[group] = float(ret.max())

        if not group_momentum:
            continue

        mom_s = pd.Series(group_momentum)
        rank_mom = mom_s.rank(method="average", pct=True, na_option="bottom")

        if not init:
            combined = rank_mom * 0.0
            init = True
        combined = combined.add(rank_mom * w, fill_value=0.0)

    if not init or combined.empty:
        return pd.Series(dtype=float)

    return combined.sort_values(ascending=False)


def classify_regime(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    short_window: int = 60,
    long_window: int = 252,
    bull_short: float = 0.05,
    bear_short: float = -0.05,
    long_th: float = 0.10,
    long_neg_th: float = -0.10,
) -> str:
    """用 HS300 (510300) 动量分类 regime.

    Returns:
        "bull" | "bear" | "sideways"
    """
    if "510300" not in nav_df.columns:
        return "sideways"
    sub = nav_df.loc[:as_of, "510300"]
    if len(sub) < long_window + 1:
        return "sideways"

    mom_short = float(sub.iloc[-1] / sub.iloc[-short_window - 1] - 1.0)
    mom_long = float(sub.iloc[-1] / sub.iloc[-long_window - 1] - 1.0)

    if mom_short > bull_short and mom_long > long_th:
        return "bull"
    if mom_short < bear_short and mom_long < long_neg_th:
        return "bear"
    return "sideways"


class StyleRotationV5SubStrategy(SubStrategy):
    """v5 风格轮动子策略.

    选股逻辑:
        1. 计算多窗口 Long-biased 风格得分 (改进 2)
        2. 选 Top-N 风格 (改进 3, default Top-2)
        3. 每个风格选 top_n_per_style ETF
        4. 强制 dividend 底仓 (改进 1, default 20%)
        5. Sideways 时降低总仓位 (改进 4, 50% 风格 + 50% cash)

    调仓: 月度 (rebalance_freq="M")
    """

    def __init__(self, config: StyleRotationV5Config):
        super().__init__(config)
        self.config: StyleRotationV5Config = config

    def _compute_weights(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
        top_codes: list[str],
        top_groups: list[StyleGroup],
    ) -> dict[str, float]:
        """加权: dividend 底仓 + score 加权 + max_weight."""
        cfg = self.config

        scores = multi_window_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            cfg.windows, cfg.window_weights,
        )

        weights: dict[str, float] = {}
        for code, grp in zip(top_codes, top_groups):
            if grp == StyleGroup.DIVIDEND:
                weights[code] = cfg.dividend_floor
            else:
                if grp in scores.index:
                    weights[code] = float(scores[grp])
                else:
                    weights[code] = 1.0

        if not weights:
            return {}

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        weights = self._apply_max_weight(weights, cfg.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        cfg = self.config
        if cfg.min_history > 0 and len(nav_df) < cfg.min_history:
            return []

        scores = multi_window_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            cfg.windows, cfg.window_weights,
        )
        if scores.empty:
            return []

        top_groups = [g for g in scores.index if isinstance(g, StyleGroup)][:cfg.top_n]
        top_codes: list[str] = []
        for grp in top_groups:
            codes = STYLE_GROUP_CODES.get(grp, ())
            valid = [c for c in codes if c in nav_df.columns]
            if not valid:
                continue
            ret = nav_df.loc[:as_of, valid].iloc[-1] / nav_df.loc[:as_of, valid].iloc[-61] - 1.0
            ret = ret.dropna().sort_values(ascending=False)
            if len(ret) > 0:
                top_codes.append(ret.index[0])

        if StyleGroup.DIVIDEND not in top_groups:
            div_codes = STYLE_GROUP_CODES.get(StyleGroup.DIVIDEND, ())
            valid_div = [c for c in div_codes if c in nav_df.columns]
            if valid_div:
                top_codes.append(valid_div[0])
                top_groups.append(StyleGroup.DIVIDEND)

        return top_codes

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        cfg = self.config
        if not codes:
            return {}

        code_to_group: dict[str, StyleGroup] = {}
        for grp, cs in STYLE_GROUP_CODES.items():
            for c in cs:
                code_to_group[c] = grp

        top_groups = [code_to_group.get(c) for c in codes if c in code_to_group]
        return self._compute_weights(nav_df, as_of, list(codes), top_groups)

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        cfg = self.config
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, meta={"strategy": self.config.name})

        weights = self.weight(nav_df, codes, as_of)
        if not weights:
            return SubStrategyResult(date=as_of, meta={"strategy": self.config.name})

        regime = classify_regime(
            nav_df, as_of,
            cfg.regime_lookback_short, cfg.regime_lookback_long,
            cfg.bull_threshold, cfg.bear_threshold, cfg.long_threshold,
        )
        if regime == "sideways":
            scale = cfg.sideways_style_exposure
        elif regime == "bull":
            scale = 1.0
        else:
            scale = 0.7

        weights = {k: v * scale for k, v in weights.items()}
        cash_weight = 1.0 - sum(weights.values())
        if cash_weight < 0:
            cash_weight = 0.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        scores = multi_window_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            cfg.windows, cfg.window_weights,
        )
        signal = float(scores.iloc[0]) if len(scores) > 0 else 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": self.config.name,
                "regime": regime,
                "style_exposure_scale": scale,
                "cash_weight": cash_weight,
                "windows": list(cfg.windows),
                "window_weights": list(cfg.window_weights),
                "dividend_floor": cfg.dividend_floor,
                "top_n": cfg.top_n,
            },
        )


__all__ = [
    "StyleRotationV5Config",
    "StyleRotationV5SubStrategy",
    "multi_window_score",
    "classify_regime",
]
