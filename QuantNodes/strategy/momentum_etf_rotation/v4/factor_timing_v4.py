# coding=utf-8
"""因子择时 (Stage 17, v4.0) — IC 驱动版 (HMM 部分后续补).

核心思想 (金融街证券《风格轮动与因子择时》):
- 把 6 因子作为"可交易品种"
- 用滚动 IC 监控哪个因子最近有效
- 给子策略动态分配权重: IC 高的因子权重大

因子 → 子策略映射 (默认):
- momentum  → style_rotation  (动量风格组)
- reversal  → style_rotation  (反转风格组)
- value     → smart_beta       (价值因子 ETF)
- low_vol   → smart_beta       (低波因子 ETF)
- dividend  → smart_beta       (红利因子 ETF)
- quality   → smart_beta       (质量因子 ETF)

权重公式:
    raw_weight[name] = max(0, IC[name] + base) ** 2
    weight[name] = raw_weight / sum(raw_weight)

参数:
- ic_window: 60 天滚动 IC 窗口
- forward_window: 20 天 (用于 IC 计算)
- base: 0.05 (基础权重, 防止全 0)
- warmup_period: 120 天 (冷启动期, 等权)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .factor_ic import (
    FACTOR_NAMES,
    factor_ic_rolling_mean,
    rolling_factor_ic,
)
from .universe_v4 import ALL_V4_CODES


@dataclass
class FactorTimingConfig:
    """因子择时配置 (Stage 17, v4.0)."""
    ic_window: int = 60               # 滚动 IC 窗口
    forward_window: int = 20         # 预测窗口
    ic_step: int = 5                  # IC 采样步长
    lookback: int = 60                # 因子 lookback
    base: float = 0.05                # 基础权重 (避免全 0)
    power: float = 2.0                # IC → 权重 的幂 (越大越极化)
    smooth_window: int = 12           # IC 平滑窗口 (60d @ 5d step)
    warmup_period: int = 120          # 冷启动期 (等权)
    min_weight: float = 0.05          # 最小权重 (任何因子都保留 5%)

    # 因子 → 子策略映射
    factor_to_strategy: dict[str, str] = field(default_factory=lambda: {
        "momentum": "style_rotation",
        "reversal": "style_rotation",
        "value":    "smart_beta",
        "low_vol":  "smart_beta",
        "dividend": "smart_beta",
        "quality":  "smart_beta",
    })


def compute_factor_weights(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig | None = None,
) -> dict[str, float]:
    """根据最新一期 IC 计算因子权重.

    算法:
        raw_w[name] = max(0, IC[name] + cfg.base) ** cfg.power
        weight[name] = raw_w / sum(raw_w)
        应用 min_weight 兜底

    Args:
        ic_history: 滚动 IC DataFrame (index=date, columns=factor)
        cfg: 配置

    Returns:
        dict, factor name → weight (sum=1)
    """
    cfg = cfg or FactorTimingConfig()

    if ic_history.empty:
        return {n: 1.0 / len(FACTOR_NAMES) for n in FACTOR_NAMES}

    latest = ic_history.iloc[-1].fillna(0.0)

    raw = {}
    for name in FACTOR_NAMES:
        ic = float(latest.get(name, 0.0))
        raw[name] = max(0.0, ic + cfg.base) ** cfg.power

    total = sum(raw.values())
    if total <= 0:
        return {n: 1.0 / len(FACTOR_NAMES) for n in FACTOR_NAMES}

    weights = {k: v / total for k, v in raw.items()}

    # 应用 min_weight (避免某因子被 0 权重, 失去分散化)
    for name in weights:
        if weights[name] < cfg.min_weight:
            weights[name] = cfg.min_weight

    # 重新归一化
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def compute_strategy_weights(
    factor_weights: dict[str, float],
    factor_to_strategy: dict[str, str],
) -> dict[str, float]:
    """聚合因子权重到子策略权重.

    多个因子映射到同一子策略 → 权重累加

    Args:
        factor_weights: factor → weight
        factor_to_strategy: factor → strategy name

    Returns:
        strategy name → weight (sum=1)
    """
    out: dict[str, float] = {}
    for f, w in factor_weights.items():
        s = factor_to_strategy.get(f, "unknown")
        out[s] = out.get(s, 0.0) + w

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def backtest_factor_timing(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """回测因子择时的 IC 时序.

    Args:
        nav_df: 价格面板
        all_codes: 候选 ETF codes
        cfg: 因子择时配置
        start/end: 范围

    Returns:
        pd.DataFrame, index=date, columns=IC × 6
    """
    cfg = cfg or FactorTimingConfig()
    ic_raw = rolling_factor_ic(
        nav_df, list(all_codes),
        start=start, end=end,
        window=cfg.ic_window,
        forward_window=cfg.forward_window,
        step=cfg.ic_step,
        lookback=cfg.lookback,
    )
    if ic_raw.empty:
        return ic_raw

    # 平滑
    ic_smooth = factor_ic_rolling_mean(ic_raw, smooth_window=cfg.smooth_window)
    return ic_smooth


def backtest_factor_weights_history(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """回测因子权重时序.

    Returns:
        pd.DataFrame, index=date, columns=factor name (weights)
    """
    cfg = cfg or FactorTimingConfig()
    ic_history = backtest_factor_timing(
        nav_df, all_codes, cfg, start, end,
    )
    if ic_history.empty:
        return pd.DataFrame()

    # 每个采样日算一次权重
    weights_list = []
    dates = []
    for ts, row in ic_history.iterrows():
        ic_dict = row.to_dict()
        w = compute_factor_weights(pd.DataFrame([ic_dict], index=[ts]), cfg)
        weights_list.append(w)
        dates.append(ts)

    return pd.DataFrame(weights_list, index=dates)


__all__ = [
    "FactorTimingConfig",
    "compute_factor_weights",
    "compute_strategy_weights",
    "backtest_factor_timing",
    "backtest_factor_weights_history",
]
