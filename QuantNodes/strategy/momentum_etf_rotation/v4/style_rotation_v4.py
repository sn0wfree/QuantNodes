# coding=utf-8
"""v4 风格轮动子策略 (Stage 18, 合并 v5 4 改进).

思路 (金融街证券《风格轮动与因子择时》):
- 5 个风格组: 大盘/中盘/成长/科创/红利
- 月度调仓, 选 Top-N 风格组
- 每个风格组选代表 ETF
- 加权: 风格得分加权 + 强制 dividend 底仓 + regime filter

Stage 18 升级 (基于 v4 诊断研究 SUB_STRATEGY_DIAGNOSTIC.md):
1. **多窗口 Long-biased** (默认 5/20/120/180, 权重 0.1/0.2/0.3/0.4)
   - 诊断: 单窗口 L=120 Calmar 0.016, 多窗口 0.439 (27x 提升)
2. **强制 dividend 底仓 20%**
   - 诊断: 5 风格组 0.86-0.90 高度相关, dividend 是唯一分散器
3. **Top-N 选择** (默认 Top-2)
   - 诊断: Top-1 准确率 34.5%, Top-2 53.4%
4. **Regime filter** (Sideways 50% 仓位 + cash)
   - 诊断: Sideways 70% 时间亏钱 -2.50% ann

向后兼容:
- `lookback` 字段保留 (单窗口模式, 默认 None = 用多窗口)
- `trend_weight` 字段保留 (单窗口 trend filter)
- 旧配置 (单窗口 L=60, top_n=3, 无 regime) 仍可工作

参考: reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md §1
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    """风格轮动子策略配置 (Stage 18 升级版).

    Stage 18 新增字段 (defaults = v5 优化值):
    - windows / window_weights: 多窗口 Long-biased
    - dividend_floor: 强制 dividend 底仓 (默认 0.20)
    - sideways_style_exposure: Sideways regime 仓位缩放
    - regime_lookback_* / *_threshold: regime 分类参数

    向后兼容字段 (默认单窗口):
    - lookback: 单窗口 (默认 None = 走多窗口路径)
    - trend_lookback / trend_weight: 单窗口 trend filter
    """
    name: str = "style_rotation"

    # 单窗口模式 (向后兼容, 默认 None = 走多窗口)
    lookback: int | None = None
    trend_lookback: int = 60
    trend_weight: float = 0.3

    # Stage 18 #2: 多窗口 Long-biased
    windows: tuple[int, ...] = (5, 20, 120, 180)
    window_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)

    # 选股
    top_n_styles: int = 2              # Stage 18 #3: Top-2
    top_n_per_style: int = 1

    # Stage 18 #1: dividend 底仓
    dividend_floor: float = 0.20

    # Stage 18 #4: regime filter
    regime_lookback_short: int = 60
    regime_lookback_long: int = 252
    bull_threshold: float = 0.05
    bear_threshold: float = -0.05
    long_threshold: float = 0.10
    long_neg_threshold: float = -0.10
    sideways_style_exposure: float = 0.50
    bear_style_exposure: float = 0.70

    # 通用
    min_history: int = 252
    max_weight: float = 0.40
    rebalance_freq: str = "M"

    cash_proxy_code: str = ""          # 如果想用某 ETF 当 cash proxy, 留空则用真实 cash


def _rank_pct(s: pd.Series) -> pd.Series:
    """百分位排名 (NaN-safe, NaN 排最后)."""
    return s.rank(method="average", pct=True, na_option="bottom")


def _group_max_return(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_codes: dict[StyleGroup, tuple[str, ...]],
    lookback: int,
) -> dict[StyleGroup, float]:
    """组内最强 ETF 的 L 日收益率."""
    sub = nav_df.loc[:as_of]
    if len(sub) < lookback + 2:
        return {}
    out: dict[StyleGroup, float] = {}
    for group, codes in group_codes.items():
        valid = [c for c in codes if c in sub.columns]
        if not valid:
            continue
        recent = sub[valid].iloc[-1]
        past = sub[valid].iloc[-lookback - 1]
        ret = (recent / past - 1.0).dropna()
        if len(ret) > 0:
            out[group] = float(ret.max())
    return out


def style_rotation_score(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    group_codes: dict[StyleGroup, tuple[str, ...]] = STYLE_GROUP_CODES,
    lookback: int | None = None,
    trend_lookback: int = 60,
    trend_weight: float = 0.3,
    windows: tuple[int, ...] | None = None,
    window_weights: tuple[float, ...] | None = None,
) -> pd.Series:
    """计算风格组得分 (越大越强).

    模式 A (向后兼容, 单窗口):
        group_score = rank_pct(L-day-group-max-return)
                      + trend_weight × rank_pct(trend_strength)
        trend_strength = (close - MA_trend_lookback) / MA_trend_lookback

    模式 B (Stage 18, 多窗口 Long-biased, 默认):
        per_window_score[L] = rank_pct(L-day-group-max-return)
        combined = Σ w[L] × per_window_score[L]

    Args:
        nav_df: 价格面板
        as_of: 当前日期
        group_codes: 风格组 → ETF codes
        lookback: 单窗口动量窗口 (模式 A, 缺省走模式 B)
        trend_lookback / trend_weight: 模式 A trend filter 参数
        windows / window_weights: 模式 B 多窗口 (e.g. (5,20,120,180))

    Returns:
        pd.Series, index=StyleGroup, values=score (越大越强, 已排序)
    """
    use_multi = windows is not None and len(windows) > 0

    if use_multi:
        if window_weights is None or len(window_weights) != len(windows):
            raise ValueError("windows 和 window_weights 长度必须一致")
        if abs(sum(window_weights) - 1.0) > 1e-6:
            raise ValueError("window_weights 之和必须为 1")

        combined = pd.Series(dtype=float)
        init = False
        for L, w in zip(windows, window_weights):
            gm = _group_max_return(nav_df, as_of, group_codes, L)
            if not gm:
                continue
            mom_s = pd.Series(gm)
            rank_mom = _rank_pct(mom_s)
            if not init:
                combined = rank_mom * 0.0
                init = True
            combined = combined.add(rank_mom * w, fill_value=0.0)

        if not init or combined.empty:
            return pd.Series(dtype=float)
        return combined.sort_values(ascending=False)

    if lookback is None:
        raise ValueError("必须提供 lookback (单窗口) 或 windows (多窗口)")

    gm = _group_max_return(nav_df, as_of, group_codes, lookback)
    if not gm:
        return pd.Series(dtype=float)
    mom_s = pd.Series(gm)
    rank_mom = _rank_pct(mom_s)

    if trend_weight <= 0:
        return rank_mom.sort_values(ascending=False)

    sub = nav_df.loc[:as_of]
    if len(sub) < trend_lookback + 1:
        return rank_mom.sort_values(ascending=False)

    trend_dict: dict[StyleGroup, float] = {}
    for group, codes in group_codes.items():
        valid = [c for c in codes if c in sub.columns]
        if not valid:
            continue
        sub_codes = sub[valid]
        if len(sub_codes) < trend_lookback:
            continue
        ma = sub_codes.iloc[-trend_lookback:].mean()
        latest = sub_codes.iloc[-1]
        trend = (latest / ma - 1.0).dropna()
        if len(trend) > 0:
            trend_dict[group] = float(trend.max())

    if trend_dict:
        trend_s = pd.Series(trend_dict)
        rank_trend = _rank_pct(trend_s)
    else:
        rank_trend = pd.Series(0.0, index=rank_mom.index)

    score = rank_mom + trend_weight * rank_trend
    return score.sort_values(ascending=False)


def select_top_styles(
    scores: pd.Series,
    top_n: int = 2,
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
    """对每个选中的风格, 选其代表 ETF 中 60d 动量最强."""
    sub = nav_df.loc[:as_of]
    if len(sub) < 61:
        return []
    picks: list[str] = []
    for group in top_styles:
        codes = group_codes.get(group, ())
        valid = [c for c in codes if c in sub.columns]
        if not valid:
            continue
        ret = sub[valid].iloc[-1] / sub[valid].iloc[-61] - 1.0
        ret = ret.dropna().sort_values(ascending=False)
        picks.extend(ret.index[:top_n_per_style].tolist())
    return picks


def classify_regime(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    short_window: int = 60,
    long_window: int = 252,
    bull_short: float = 0.05,
    bear_short: float = -0.05,
    long_th: float = 0.10,
    long_neg_th: float = -0.10,
    reference_code: str = "510300",
) -> str:
    """用 reference_code (默认 HS300) 动量分类 regime.

    Returns:
        "bull" | "bear" | "sideways"
    """
    if reference_code not in nav_df.columns:
        return "sideways"
    sub = nav_df.loc[:as_of, reference_code]
    if len(sub) < long_window + 1:
        return "sideways"
    mom_short = float(sub.iloc[-1] / sub.iloc[-short_window - 1] - 1.0)
    mom_long = float(sub.iloc[-1] / sub.iloc[-long_window - 1] - 1.0)
    if mom_short > bull_short and mom_long > long_th:
        return "bull"
    if mom_short < bear_short and mom_long < long_neg_th:
        return "bear"
    return "sideways"


class StyleRotationSubStrategy(SubStrategy):
    """v4 风格轮动子策略 (Stage 18 升级版).

    选股逻辑 (Stage 18 升级):
        1. 计算风格得分 (多窗口 Long-biased 5/20/120/180) — 改进 1
        2. 选 Top-2 风格 (默认) — 改进 3
        3. 每个风格选 top_n_per_style ETF
        4. 强制 dividend 底仓 20% — 改进 2
        5. Regime filter: sideways 50% 风格, bear 70% 风格 — 改进 4

    调仓: 月度
    """

    def __init__(self, config: StyleRotationConfig):
        super().__init__(config)
        self.config: StyleRotationConfig = config

    def _use_multi_window(self) -> bool:
        return self.config.windows is not None and len(self.config.windows) > 0

    def _compute_score(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        cfg = self.config
        if self._use_multi_window():
            return style_rotation_score(
                nav_df, as_of, STYLE_GROUP_CODES,
                windows=cfg.windows, window_weights=cfg.window_weights,
            )
        return style_rotation_score(
            nav_df, as_of, STYLE_GROUP_CODES,
            lookback=cfg.lookback or 60,
            trend_lookback=cfg.trend_lookback,
            trend_weight=cfg.trend_weight,
        )

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        cfg = self.config
        if cfg.min_history > 0 and len(nav_df) < cfg.min_history:
            return []

        scores = self._compute_score(nav_df, as_of)
        if scores.empty:
            return []

        top_styles = select_top_styles(scores, cfg.top_n_styles)
        picks = style_etf_picks(
            nav_df, as_of, STYLE_GROUP_CODES,
            top_styles, cfg.top_n_per_style,
        )

        if cfg.dividend_floor > 0 and StyleGroup.DIVIDEND not in top_styles:
            div_codes = STYLE_GROUP_CODES.get(StyleGroup.DIVIDEND, ())
            valid_div = [c for c in div_codes if c in nav_df.columns]
            if valid_div and valid_div[0] not in picks:
                picks.append(valid_div[0])

        return picks

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

        scores = self._compute_score(nav_df, as_of)

        weights: dict[str, float] = {}
        for code in codes:
            grp = code_to_group.get(code)
            if grp is None:
                weights[code] = 1.0
                continue
            if grp == StyleGroup.DIVIDEND and cfg.dividend_floor > 0:
                weights[code] = cfg.dividend_floor
                continue
            if grp in scores.index:
                weights[code] = float(scores[grp])
            else:
                weights[code] = 1.0

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        weights = self._apply_max_weight(weights, cfg.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        cfg = self.config
        codes = self.select(nav_df, as_of)
        if not codes:
            return SubStrategyResult(date=as_of, meta={"strategy": cfg.name})

        weights = self.weight(nav_df, codes, as_of)
        if not weights:
            return SubStrategyResult(date=as_of, meta={"strategy": cfg.name})

        regime = classify_regime(
            nav_df, as_of,
            cfg.regime_lookback_short, cfg.regime_lookback_long,
            cfg.bull_threshold, cfg.bear_threshold,
            cfg.long_threshold, cfg.long_neg_threshold,
        )
        if regime == "sideways":
            scale = cfg.sideways_style_exposure
        elif regime == "bear":
            scale = cfg.bear_style_exposure
        else:
            scale = 1.0

        weights = {k: v * scale for k, v in weights.items()}
        cash_weight = max(0.0, 1.0 - sum(weights.values()))
        if cash_weight >= 1.0:
            weights = {}
            cash_weight = 1.0
        elif sum(weights.values()) > 1.0:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
            cash_weight = 0.0

        scores = self._compute_score(nav_df, as_of)
        signal = float(scores.iloc[0]) if len(scores) > 0 else 0.0

        return SubStrategyResult(
            date=as_of,
            chosen=codes,
            weights=weights,
            signal_strength=signal,
            meta={
                "strategy": cfg.name,
                "regime": regime,
                "style_exposure_scale": scale,
                "cash_weight": cash_weight,
                "windows": list(cfg.windows) if self._use_multi_window() else None,
                "window_weights": list(cfg.window_weights) if self._use_multi_window() else None,
                "single_lookback": cfg.lookback if not self._use_multi_window() else None,
                "dividend_floor": cfg.dividend_floor,
                "top_n_styles": cfg.top_n_styles,
            },
        )


__all__ = [
    "StyleRotationConfig",
    "StyleRotationSubStrategy",
    "style_rotation_score",
    "select_top_styles",
    "style_etf_picks",
    "classify_regime",
]
